"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in FsDet.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

from lvc.config import get_cfg, set_global_cfg
from lvc.engine import DefaultTrainer, default_argument_parser, default_setup

import detectron2.utils.comm as comm
import os
import json
import logging
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import launch
from detectron2.evaluation import inference_context
from lvc.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    verify_results,
    RPNEvaluator,
    # ImageContextEvaluator,
    )
import time
import datetime
from collections import OrderedDict
from tqdm.auto import tqdm
from typing import Any, Dict, List, Set
import torch
import numpy as np
from detectron2.config import CfgNode
import itertools
from collections import defaultdict
from lvc.config import set_global_cfg

import resource

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_optimizer(cls, cfg, model):
        from detectron2.solver import build_optimizer

        return build_optimizer(cfg, model)

    @classmethod
    def process_outputs(cls, dict_out, dict_store):
        pres_classes = list(set(dict_out['gt_classes']))
        input_ious = np.array(dict_out['input_ious'])
        output_ious = np.array(dict_out['output_ious'])
        gt_classes = np.array(dict_out['gt_classes'])
        for k in pres_classes:
            dict_store['input_ious'][k] += np.sum(input_ious[gt_classes == k])
            dict_store['output_ious'][k] += np.sum(output_ious[gt_classes == k])
            dict_store['total'][k] += np.sum(gt_classes == k)
        dict_store['input_ious_all'] = sum(
            dict_store['input_ious'][k]
            for k in dict_store['input_ious'].keys()
        )
        dict_store['output_ious_all'] = sum(
            dict_store['output_ious'][k]
            for k in dict_store['output_ious'].keys()
        )
        dict_store['total_all'] = sum(
            dict_store['total'][k]
            for k in dict_store['total'].keys()
        )
        return dict_store

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        # if isinstance(evaluators, DatasetEvaluator):
        #     evaluators = [evaluators]
        # if evaluators is not None:
        #     assert len(cfg.DATASETS.TEST) == len(
        #         evaluators
        #     ), "{} != {}".format(len(cfg.DATASETS.TEST), len(evaluators))

        results = OrderedDict()
        for i, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            # if evaluators is not None:
            #     evaluator = evaluators[idx]
            # else:
            #     try:
            #         evaluator = cls.build_evaluator(cfg, dataset_name)
            #     except NotImplementedError:
            #         logger.warn(
            #             "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
            #             "or implement its `build_evaluator` method."
            #         )
            #         results[dataset_name] = {}
            #         continue
            all_results = {
                'input_ious': defaultdict(float),
                'output_ious': defaultdict(float),
                'total': defaultdict(float),
            }
            total = len(data_loader)
            total_compute_time = 0.0
            logging_interval = 500
            num_warmup = 10
            logger = logging.getLogger('lvc.evaluator')
            with inference_context(model), torch.no_grad():
                for idx, data in enumerate(data_loader):
                    if idx == num_warmup:
                        start_time = time.time()
                        total_compute_time = 0.0
                    start_compute_time = time.time()
                    dict_out = model(data)
                    # torch.cuda.synchronize()
                    dict_out['input_ious'] = dict_out['input_ious'].cpu().tolist()
                    dict_out['output_ious'] = dict_out['output_ious'].cpu().tolist()
                    dict_out['gt_classes'] = dict_out['gt_classes'].cpu().tolist()
                    all_results = cls.process_outputs(dict_out, all_results)
                    total_compute_time += time.time() - start_compute_time
                    if (idx + 1) % logging_interval == 0:
                        duration = time.time() - start_time
                        seconds_per_img = duration / (idx + 1 - num_warmup)
                        eta = datetime.timedelta(
                            seconds=int(seconds_per_img * (total - num_warmup) - duration)
                        )
                        logger.info(
                            "Inference done {}/{}. {:.4f} s / img. ETA={}, input_ious={:.4f}, output_ious={:.4f}".format(
                                idx + 1, total, seconds_per_img, str(eta),
                                float(all_results['input_ious_all']/all_results['total_all']),
                                float(all_results['output_ious_all']/all_results['total_all'])
                            )
                        )
            comm.synchronize()
            all_results = comm.gather(all_results, dst=0)
            if comm.is_main_process():
                pres_cats = list({c for a in all_results for c in a['total'].keys()})
                gathered_results = {}
                for k in all_results[0].keys():
                    if isinstance(all_results[0][k], dict):
                        gathered_results[k] = {c: sum([d[k][c] for d in all_results])
                                               for c in pres_cats}
                    else:
                        gathered_results[k] = sum([d[k] for d in all_results])
                gathered_results['input_ious'] = {
                    k: v/gathered_results['total'][k]
                    for k, v in gathered_results['input_ious'].items()
                }
                gathered_results['output_ious'] = {
                    k: v/gathered_results['total'][k]
                    for k, v in gathered_results['output_ious'].items()
                }
                gathered_results['input_ious_all'] =\
                    gathered_results['input_ious_all']/gathered_results['total_all']
                gathered_results['output_ious_all'] =\
                    gathered_results['output_ious_all']/gathered_results['total_all']
                results[dataset_name] = gathered_results

            else:
                results[dataset_name] = {}
        return results

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluator_list = []
        if cfg.MODEL.PROPOSAL_GENERATOR.NAME == 'RBG':
            pass
        elif cfg.MODEL.META_ARCHITECTURE == 'ProposalNetwork':
            evaluator_list.append(
                RPNEvaluator(dataset_name, cfg, True, output_folder))
            return DatasetEvaluators(evaluator_list)
        # elif cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNN_Context' and cfg.MODEL.IMAGES_ONLY:
        #     evaluator_list.append(
        #         ImageContextEvaluator(dataset_name, cfg, True, output_folder))
            return DatasetEvaluators(evaluator_list)
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(
                COCOEvaluator(dataset_name, cfg, True, output_folder)
            )
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            # verify_results(cfg, res)
            save_dir, weight_path = os.path.split(cfg.MODEL.WEIGHTS)
            os.makedirs(os.path.join(save_dir, 'inference'), exist_ok=True)
            save_path = weight_path.replace('model', 'results')
            save_path = save_path.replace('.pth', '.json')
            with open(os.path.join(save_dir, 'inference', save_path), 'w') as f:
                json.dump(res, f)
        return res

    if cfg.QUERY_EXPAND.ENABLED:
        from lvc.data.utils import register_results
        register_results(cfg)

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
