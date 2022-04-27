# import sys
import os
import json
# import clip
import resource
# from types import SimpleNamespace
from lvc.engine import default_setup, DefaultTrainer, default_argument_parser
from lvc.config import get_cfg, set_global_cfg
# from lvc.modeling.meta_arch.build import META_ARCH_REGISTRY
from lvc.data.dataset_mapper import DatasetMapperQE
from lvc.data.utils import register_results, print_precision_per_class
from lvc.data.build import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from lvc.evaluation.evaluator import inference_context
import torch
import torch.nn.functional as F
# from detectron2.structures import ImageList
# from detectron2.structures.boxes import BoxMode
# from detectron2.data import transforms as T
from detectron2.engine import launch
# import torchvision
# import torchvision.models as models
from torchvision.ops import box_iou
# from torchvision.ops.boxes import batched_nms
# import numpy as np
# from PIL import Image
# from detectron2.layers.batch_norm import FrozenBatchNorm2d
from tqdm.auto import tqdm
# import random
import copy
from collections import defaultdict  # , OrderedDict
from tabulate import tabulate
from detectron2.data.catalog import MetadataCatalog
import itertools
from pycocotools.coco import COCO

import detectron2.utils.comm as comm
# from lvc.evaluation import verify_results

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


UNSEEN_IDS = [
    0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62]
UNSEEN_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair',
    'couch', 'potted plant', 'dining table', 'tv']
SEEN_IDS = [
    7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 59, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 78, 79]
SEEN_NAMES = [
    'truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'bed', 'toilet', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush']
ALL_IDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79]
ALL_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']


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


def preprocess_crops(data, mean, std):
    crops = torch.cat([x['instances'].crops for x in data])
    crops = (crops.float() - mean) / std
    return crops


def get_descriptors(cfg, model, dset):
    mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1)
    std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1)
    shot_features = []
    dataloader_shot = build_detection_test_loader(
        cfg, dset, mapper=DatasetMapperQE(cfg, False))

    with inference_context(model), torch.no_grad():
        for i, data in tqdm(enumerate(dataloader_shot),
                            total=len(dataloader_shot)):
            crops = preprocess_crops(data, mean, std)
            crops = crops.to(model.device)
            crop_features = model(crops)

            del data[0]['image']
            data[0]['instances'].remove('crops')
            data[0]['instances'].set('crop_feats', crop_features)
            data[0]['instances'] = data[0]['instances'].to('cpu')
            shot_features.append(copy.deepcopy(data[0]))

    return shot_features


def assemble_tensors(shot_features):
    classes = torch.cat([x['instances'].gt_classes for x in shot_features])
    sorter = classes.argsort()
    shot_descriptors = torch.cat(
        [x['instances'].crop_feats for x in shot_features])
    shot_descriptors = shot_descriptors[sorter]
    classes = classes[sorter]

    return classes, shot_descriptors


def run_nearest_neighbours(
        shot_classes, shot_descriptors, query_features, cosine=True):
    crop_mean = shot_descriptors.mean(dim=0, keepdim=True)

    for i, data in tqdm(enumerate(query_features), total=len(query_features)):
        query_feats = data['instances'].get('crop_feats')
        if cosine:
            sim_crop = \
                F.cosine_similarity(
                    shot_descriptors.sub(crop_mean).unsqueeze(0).float(),
                    query_feats.sub(crop_mean).unsqueeze(1).float(),
                    dim=-1)
        else:
            sim_crop = \
                torch.cdist(
                    shot_descriptors.unsqueeze(0).float(),
                    query_feats.unsqueeze(0).float()
                ).squeeze(0).t().mul(-1.0)
        top10_shots = sim_crop.topk(10, dim=-1)[1]
        data['instances'].set('top10_shots', shot_classes[top10_shots])
    return query_features


def get_gt_instances(cfg, query_features):
    dataloader_all = build_detection_test_loader(cfg, 'coco_trainval_all')
    print('gt data loaded')
    query_ids = [v['image_id'] for v in query_features]
    print('query ids compiled')
    imgid2idxs_all = {
        v1['image_id']: i
        for i, v1 in enumerate(dataloader_all.dataset._dataset)}
    print('iid2idxs done')
    print(comm.get_rank(), len(dataloader_all.dataset._dataset))
    # gt_instances = []
    # for i in tqdm(query_ids, total=len(query_ids)):
    #     gt_instances.append(
    #         dataloader_all.dataset[imgid2idxs_all[i]]
    #     )

    gt_instances = []
    for i, dt in tqdm(enumerate(query_features), total=len(query_ids)):
        gt = dataloader_all.dataset[imgid2idxs_all[dt['image_id']]]
        dt_bbox = dt['instances'].gt_boxes.tensor
        gt_bbox = gt['instances'].gt_boxes.tensor
        gt_classes = gt['instances'].gt_classes
        ious = box_iou(gt_bbox, dt_bbox)
        iou_max, gt_class = ious.max(dim=0)
        gt_class = torch.LongTensor(
            [gt_classes[gtc].item() if gt_iou > 0.5 else -1
             for gt_iou, gtc in zip(iou_max, gt_class)])
        dt['instances'].set('gt_iou_class', gt_class)
        gt.pop("image")
        gt_instances.append(gt)
    return gt_instances


def get_before_stats(query_features):
    stats = defaultdict(list)
    for d in query_features:
        for dt_cls, gt_cls in zip(
             d['instances'].gt_classes.tolist(),
             d['instances'].gt_iou_class.tolist()):
            stats[dt_cls].append(gt_cls)
    table = []
    for k, name in zip(UNSEEN_IDS, UNSEEN_NAMES):
        t = len(stats[k])
        mean = (torch.Tensor(stats[k]) == k).float().mean()
        table.append([name, t, mean])
    print(tabulate(
        table, headers=['Category', 'NUM', 'Precision'], tablefmt='orgtbl'))


def get_nn_class_confirmatory(query_features, k):
    # stats = defaultdict(list)
    for d in query_features:
        keep = torch.zeros(len(d['instances'])).long()
        # gt_classes = d['instances'].gt_iou_class
        dt_classes = d['instances'].gt_classes
        votes = d['instances'].get('top10_shots')
        nn_class = torch.mode(votes[:, :k], dim=1)[0]
        for i, (dt_cls, nn_cls) in enumerate(
             zip(dt_classes.tolist(), nn_class.tolist())):
            if dt_cls == nn_cls:
                # stats[dt_cls].append(gt_cls)
                keep[i] = 1
        d['instances'].set('keep', keep)


def save_coco(cfg, keep_ids, qe_dset):
    coco_json = json.load(open(qe_dset, 'r'))
    aid2ann = {x['id']: x for x in coco_json["annotations"]}
    iid2img = {x['id']: x for x in coco_json["images"]}
    # new_anns = [x for x in coco_json["annotations"] if x['id'] in keep_ids]
    new_anns = [aid2ann[v] for v in keep_ids]
    new_iids = list(set([x['image_id'] for x in new_anns]))
    # new_imgs = [x for x in coco_json["images"] if x['id'] in new_iids]
    new_imgs = [iid2img[v] for v in new_iids]
    coco_json["annotations"] = new_anns
    coco_json["images"] = new_imgs
    filename = qe_dset
    filename = filename.replace(
        '.json', '_{}_{}_{}.json'.format(
            cfg.QUERY_EXPAND.NN_MODEL.replace('/', ''),
            str(cfg.QUERY_EXPAND.KNN).zfill(2),
            'cosine' if cfg.QUERY_EXPAND.COSINE_SIM else 'euclid'))
    print(filename)
    json.dump(coco_json, open(filename, 'w'))
    return filename


def remove_gt_seen_overlap(query_features, gt_instances):
    # stats = defaultdict(list)
    seen_ids = torch.Tensor(SEEN_IDS).view(1, -1)
    for i, (dt, gt) in tqdm(
         enumerate(zip(query_features, gt_instances)),
         total=len(query_features)):
        assert gt["image_id"] == dt["image_id"]
        gt_classes = gt['instances'].gt_classes
        gt_boxes = gt['instances'].gt_boxes.tensor
        dt_boxes = dt['instances'].gt_boxes.tensor
        val_boxes = (gt_classes.view(-1, 1) == seen_ids).any(dim=-1)
        gt_boxes = gt_boxes[val_boxes]
        if gt_boxes.size(0):
            ious = box_iou(gt_boxes, dt_boxes)
            keep_iou = (ious.max(dim=0)[0] < 0.5).long()
            dt['instances'].keep = dt['instances'].keep * keep_iou
        # keep = dt['instances'].keep
        # dt_classes = dt['instances'].gt_classes
        # gt_classes = dt['instances'].gt_iou_class
    #     for i, (dt_cls, k, gt_cls) in enumerate(
    #          zip(dt_classes.tolist(), keep.tolist(), gt_classes.tolist())):
    #         if k:
    #             stats[dt_cls].append(gt_cls)
    # table = []
    # for k, name in zip(UNSEEN_IDS, UNSEEN_NAMES):
    #     t = len(stats[k])
    #     mean = (torch.Tensor(stats[k]) == k).float().mean()
    #     table.append([name, t, mean])
    # print(tabulate(
    #     table, headers=['Category', 'NUM', 'Precision'], tablefmt='orgtbl'))
    return


def main(args):
    cfg = setup(args)
    register_results(cfg)

    if args.eval_only:
        gt_path = MetadataCatalog.get('coco_trainval_all').json_file

        model = torch.hub.load(
            'facebookresearch/dino:main', cfg.QUERY_EXPAND.NN_MODEL)
        model = model.to(cfg.MODEL.DEVICE)
        model.device = list(model.parameters())[0].device

        for nn_dset, qe_dset, train_dset in zip(
             cfg.QUERY_EXPAND.NN_DSET,
             cfg.DATASETS.DT_PATH,
             cfg.DATASETS.TRAIN):
            shot_features = get_descriptors(cfg, model, nn_dset)
            shot_classes, shot_descriptors = assemble_tensors(shot_features)
            if comm.get_world_size() > 1:
                shot_classes_all = comm.all_gather(shot_classes)
                shot_descriptors_all = comm.all_gather(shot_descriptors)
                print(type(shot_classes_all))
                print(type(shot_classes_all[0]))
                shot_classes = torch.cat(shot_classes_all)
                shot_descriptors = torch.cat(shot_descriptors_all)
            print(len(shot_classes))
            query_features = get_descriptors(cfg, model, qe_dset)
            query_features = run_nearest_neighbours(
                shot_classes, shot_descriptors, query_features,
                cosine=cfg.QUERY_EXPAND.COSINE_SIM)
            # if 'unlabeled' not in cfg.DATASETS.DT_PATH[0]:
            #     gt_instances = get_gt_instances(cfg, query_features)
            #     # stats = get_before_stats(query_features)
            get_nn_class_confirmatory(query_features, cfg.QUERY_EXPAND.KNN)
            # if 'unlabeled' not in cfg.DATASETS.DT_PATH[0]:
            #     remove_gt_seen_overlap(query_features, gt_instances)
            comm.synchronize()
            print('before: ', len(query_features))
            if comm.get_world_size() > 1:
                query_features_all = comm.gather(query_features)
                query_features = list(itertools.chain(*query_features_all))
            print('after: ', len(query_features))

            if comm.is_main_process():
                from lvc.data.utils import iou_check_gt
                all_ids = torch.cat(
                    [x['instances'].ids for x in query_features]).long()
                keep_ids = torch.cat(
                    [x['instances'].keep for x in query_features]).bool()
                keep_ids = all_ids[keep_ids].long().tolist()
                coco_dt = COCO(qe_dset)
                coco_gt = COCO(
                    MetadataCatalog.get(train_dset).json_file)
                seen_coco_ids = [k for k, v in coco_gt.cats.items() if v['name'] in SEEN_NAMES]
                print(len(keep_ids))
                keep_ids = [aid for aid in keep_ids
                            if not iou_check_gt(aid, coco_dt, coco_gt, gt_cids=seen_coco_ids)]
                print(len(keep_ids))
                import time
                t = time.time()
                filename = save_coco(cfg, keep_ids, qe_dset)
                print(time.time() - t)
                if "unlabeled" not in os.path.basename(qe_dset):
                    print_precision_per_class(filename, gt_path)
                # res = DefaultTrainer.test(cfg, model)
                # if comm.is_main_process():
                #     verify_results(cfg, res)
        return
    else:
        raise NotImplementedError

    if cfg.QUERY_EXPAND.ENABLED:
        register_results(cfg)

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = DefaultTrainer(cfg)
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
