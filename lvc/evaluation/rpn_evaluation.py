import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import torch
from collections import OrderedDict
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import BoxMode
from detectron2.utils.logger import create_small_table

from lvc.evaluation.evaluator import DatasetEvaluator
import pickle
from .testing import print_csv_format


class RPNEvaluator(DatasetEvaluator):
    """
    Evaluate instance detection outputs using COCO's metrics and APIs.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                    so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True):
                if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path
        self._is_splits = "all" in dataset_name or "base" in dataset_name \
            or "novel" in dataset_name
        self._base_classes = [
            8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
            81, 82, 84, 85, 86, 87, 88, 89, 90,
        ]
        self._novel_classes = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21,
                               44, 62, 63, 64, 67, 72]
        self._all_classes = sorted(self._novel_classes + self._base_classes)
        self._novel_idx = [i for i, cat in enumerate(self._all_classes)
                           if cat in self._novel_classes]
        self._base_idx = [i for i, cat in enumerate(self._all_classes)
                          if cat in self._base_classes]

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset
        self._training_set = True if "trainval" in dataset_name else False
        if 'voc' in self._dataset_name:
            self._do_evaluation = False

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "proposals" in output:
                instances = output["proposals"].to(self._cpu_device)
                prediction["proposals"] = instances_to_coco_json(
                    instances, input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(
                self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "proposals" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _get_proposal_form(self):
        from collections import defaultdict
        from tqdm.auto import tqdm
        imgToAnns = defaultdict(list)
        for ann in self._coco_results:
            imgToAnns[ann['image_id']].append(ann)

        save_dict = {}
        scores = []
        bboxs = []
        iids = []
        for iid, anns_dt in tqdm(imgToAnns.items(),
                                 total=len(imgToAnns)):
            iids.append(iid)
            anns_dt = sorted(anns_dt, key=lambda x: x['score'], reverse=True)
            bbox = np.array([ann['bbox'] for ann in anns_dt])
            bbox[:, -2] = bbox[:, 0] + bbox[:, -2]
            bbox[:, -1] = bbox[:, 1] + bbox[:, -1]
            score = np.array([ann['score'] for ann in anns_dt])
            bboxs.append(bbox)
            scores.append(score)
        save_dict['ids'] = iids
        save_dict['boxes'] = bboxs
        save_dict['objectness_logits'] = scores
        return save_dict

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(
            itertools.chain(*[x["proposals"] for x in self._predictions]))

        # unmap the category ids for COCO
        # if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
        #     reverse_id_mapping = {
        #         v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        #     }
        for result in self._coco_results:
            result["category_id"] = 1

        proposal_form = self._get_proposal_form()

        if self._output_dir:
            file_path = os.path.join(
                self._output_dir,
                "coco_proposals_{}{}_results.pkl".format(
                    'trainval' if 'trainval' in self._dataset_name else 'test',
                    '_2007' if '2007' in self._dataset_name else '_2012' if '2012' in self._dataset_name else ''))
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "wb") as f:
                pickle.dump(proposal_form, f, protocol=pickle.HIGHEST_PROTOCOL)
                # f.write(json.dumps(self._coco_results))
                # f.flush()
        # self._logger.info("No evaluation for RPN proposals, saving only")
        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        if self._training_set:
            self._logger.info("Do not run rpn eval on training set (takes a long time).")
            return

        self._logger.info("Evaluating proposals ...")
        if self._is_splits:
            self._results["bbox"] = {}
            for split, classes, names in [
                    ("all", None, self._metadata.get("thing_classes")),
                    ("base", self._base_classes, self._metadata.get("base_classes")),
                    ("novel", self._novel_classes, self._metadata.get("novel_classes"))]:
                if "all" not in self._dataset_name and \
                        split not in self._dataset_name:
                    continue
                if split == "all":
                    coco_eval = (
                        _evaluate_proposals_on_coco(
                            self._coco_api, self._coco_results, "bbox", classes,
                        )
                        if len(self._coco_results) > 0
                        else None  # cocoapi does not handle empty results very well
                    )
                res_ = self._derive_coco_results(
                    coco_eval, "bbox", class_names=names,
                )
                res = {}
                for metric in res_.keys():
                    # if len(metric) <= 4:
                    if split == "all":
                        res[metric] = res_[metric]
                    elif split == "base":
                        res["b"+metric] = res_[metric]
                    elif split == "novel":
                        res["n"+metric] = res_[metric]
                self._results["bbox"].update(res)

            print(self._results["bbox"])
            # add "AP" if not already in
            # if "AR" not in self._results["bbox"]:
            #     if "nAR (100)" in self._results["bbox"]:
            #         self._results["bbox"]["AR"] = self._results["bbox"]["nAR"]
            #     else:
            #         self._results["bbox"]["AP"] = self._results["bbox"]["bAR"]
        else:
            coco_eval = (
                _evaluate_proposals_on_coco(
                    self._coco_api, self._coco_results, "bbox",
                )
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            res = self._derive_coco_results(
                coco_eval, "bbox",
                class_names=self._metadata.get("thing_classes")
            )
            self._results["bbox"] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        from collections import defaultdict
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        coco_gt = coco_eval.cocoGt
        metrics = ["AR", "AR50", "AR75", "AR50s", "AR50m", "AR50l"]
        cat_ids = coco_gt.loadCats(coco_gt.getCatIds())
        cat_ids = [c['id'] for c in cat_ids if c['name'] in class_names]

        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        results = {}
        for m in coco_eval.params.maxDets:
            results[m] = {}
            for a in coco_eval.params.areaRngLbl:
                results[m][a] = {}
                for c in cat_ids:
                    results[m][a][c] = defaultdict(list)
        t = 0
        for img in coco_eval.evalImgs:
            if img is None:
                t += 1
                continue
            a = coco_eval.params.areaRng.index(img['aRng'])
            a = coco_eval.params.areaRngLbl[a]
            for m in coco_eval.params.maxDets:
                for i, dt in enumerate(img['dtMatches'][:, :m]):
                    iou = coco_eval.params.iouThrs[i]
                    rec_gtIds = dt[dt != 0.0]
                    gtAnns = coco_gt.loadAnns(rec_gtIds)
                    for ann in gtAnns:
                        if ann['iscrowd']:
                            continue
                        c = ann['category_id']
                        if c not in cat_ids:
                            continue
                        results[m][a][c][iou].append(ann['id'])
        sh = list(coco_eval.eval['recall'].shape)
        sh[1] = len(class_names)
        recalls = np.zeros(sh)
        for i, iou in enumerate(coco_eval.params.iouThrs):
            for m, md in enumerate(coco_eval.params.maxDets):
                for c, cid in enumerate(cat_ids):
                    for a, albl in enumerate(coco_eval.params.areaRngLbl):
                        rec = results[md][albl][cid][iou]
                        tot_gt = coco_gt.getAnnIds(
                            catIds=cid,
                            areaRng=coco_eval.params.areaRng[a],
                            iscrowd=False)
                        recalls[i, c, a, m] = (len([r for r in tot_gt if r in rec])/len(tot_gt)
                                               if len(tot_gt) != 0 else -1)
        # the standard metrics
        results = []
        # AR (100)
        rec = recalls[:, :, 0, 0]
        results.append(rec[rec != -1].mean())
        # AR50 (100)
        rec = recalls[0, :, 0, 0]
        results.append(rec[rec != -1].mean())
        # AR75 (100)
        rec = recalls[5, :, 0, 0]
        results.append(rec[rec != -1].mean())
        # AR50s (100)
        rec = recalls[0, :, 1, 0]
        results.append(rec[rec != -1].mean())
        # AR50m (100)
        rec = recalls[0, :, 2, 0]
        results.append(rec[rec != -1].mean())
        # AR50l (100)
        rec = recalls[0, :, 3, 0]
        results.append(rec[rec != -1].mean())
        met_names = [s + ' (100)' for s in metrics]
        if 1000 in coco_eval.params.maxDets:
            # AR (1000)
            rec = recalls[:, :, 0, 1]
            results.append(rec[rec != -1].mean())
            # AR50 (1000)
            rec = recalls[0, :, 0, 1]
            results.append(rec[rec != -1].mean())
            # AR75 (1000)
            rec = recalls[5, :, 0, 1]
            results.append(rec[rec != -1].mean())
            # AR50s (1000)
            rec = recalls[0, :, 1, 1]
            results.append(rec[rec != -1].mean())
            # AR50m (1000)
            rec = recalls[0, :, 2, 1]
            results.append(rec[rec != -1].mean())
            # AR50l (1000)
            rec = recalls[0, :, 3, 1]
            results.append(rec[rec != -1].mean())
            met_names += [s + ' (1000)' for s in metrics]

        results = {m: float(r)*100. for m, r in zip(met_names, results)}
        self._logger.info(
            ("Evaluation results for {}: \n".format(iou_type)
             + create_small_table(results))
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # recall has dims (ious, cls, area range, max dets)
        assert len(class_names) == recalls.shape[1]
        results_per_category_rec = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            # iou thrs index 0: IoU 0.5
            recall = recalls[0, idx, 0, 0]
            results_per_category_rec.append(
                ("{}".format(name), float(recall * 100)))
        # tabulate it
        N_COLS = min(6, len(results_per_category_rec) * 2)
        results_flatten = list(itertools.chain(*results_per_category_rec))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "recall"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} Recall 100: \n".format(iou_type)
                          + table)

        results_per_category_dict_rec = OrderedDict(
            results_per_category_rec)
        print_csv_format(OrderedDict((['bbox-cat-recall 100',
                                       results_per_category_dict_rec],)))
        if 1000 in coco_eval.params.maxDets:
            results_per_category_rec = []
            for idx, name in enumerate(class_names):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                # iou thrs index 0: IoU 0.5
                recall = recalls[0, idx, 0, -1]
                results_per_category_rec.append(
                    ("{}".format(name), float(recall * 100)))
            # tabulate it
            N_COLS = min(6, len(results_per_category_rec) * 2)
            results_flatten = list(itertools.chain(*results_per_category_rec))
            results_2d = itertools.zip_longest(
                *[results_flatten[i::N_COLS] for i in range(N_COLS)])
            table = tabulate(
                results_2d,
                tablefmt="pipe",
                floatfmt=".3f",
                headers=["category", "recall"] * (N_COLS // 2),
                numalign="left",
            )
            self._logger.info("Per-category {} Recall 1000: \n".format(iou_type)
                              + table)

            results_per_category_dict_rec = OrderedDict(
                results_per_category_rec)
            print_csv_format(OrderedDict((['bbox-cat-recall 1000',
                                           results_per_category_dict_rec],)))

        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.proposal_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.objectness_logits.tolist()
    classes = instances.objectness_logits.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results


def _evaluate_proposals_on_coco(coco_gt, coco_results, iou_type, catIds=None):
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.maxDets = [100, 1000]
    coco_eval.params.useCats = 0
    coco_eval.evaluate()
    coco_eval.accumulate()

    return coco_eval


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, catIds=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if catIds is not None:
        coco_eval.params.catIds = catIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
