import argparse
import lvc.data
from lvc.data.utils import combine_datasets
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.build import get_detection_dataset_dicts
from lvc.data.utils import print_precision_per_class
from lvc.config.defaults import _CC
from pycocotools.coco import COCO
from collections import defaultdict
import numpy as np
import json


class COCO_PK(COCO):
    def __init__(self, annotation_file=None):
        super().__init__(annotation_file)

    def createIndex(self):
        super().createIndex()

        print('creating area Ratio index...')
        # add in area portion
        for iid in self.getImgIds():
            ann_ids = super().getAnnIds(imgIds=iid)
            img = self.imgs[iid]
            h, w = img['height'], img['width']
            tot_area = float(h)*float(w)
            anns = [self.anns[i] for i in ann_ids]
            for ann in anns:
                ann['area_ratio'] = ann['area']/tot_area

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], areaRatioRng=[],
                  iscrowd=None):
        ids = super().getAnnIds(imgIds=imgIds, catIds=catIds, areaRng=areaRng,
                                iscrowd=iscrowd)
        if len(areaRatioRng) == 0:
            return ids
        else:
            assert len(areaRatioRng) == 2
            assert areaRatioRng[0] >= 0.0 and areaRatioRng[1] <= 1.0
            anns = [self.anns[i] for i in ids]
            anns = [ann for ann in anns if ann['area_ratio'] > areaRatioRng[0]
                    and ann['area_ratio'] < areaRatioRng[1]]
            return [ann['id'] for ann in anns]

    def loadRes(self, resFile, full_dataset=False):
        if full_dataset:
            return COCO_PK(resFile)
        res = super().loadRes(resFile)
        res1 = COCO_PK()
        res1.dataset = res.dataset
        res1.createIndex()
        return res1


UNSEEN_CLASSES = _CC.DATASETS.UNSEEN_CLASSES
SEEN_CLASSES = _CC.DATASETS.SEEN_CLASSES
AREA_RNG = [0 ** 2, 1e5 ** 2]

parser = argparse.ArgumentParser(
        description='Create coco dataset file from detections')
parser.add_argument('--json-data', default='coco_trainval_all')
parser.add_argument('--gt-data', help='what is the starting groundtruth data',
                    required=True)
parser.add_argument('--top', action='store_true',
                    help='method to select detections',
                    )
parser.add_argument('--full', action='store_true',
                    help='retain all dets in a selected image',
                    )
parser.add_argument('--full-dataset', action='store_true',
                    help='dt_path is in COCO() format',
                    )
parser.add_argument('--K-min', type=float,
                    help='min value for select detections',
                    required=True)
parser.add_argument('--K-max', type=float,
                    help='max value for select detections',
                    required=True)
parser.add_argument('--ar', type=float,
                    help='lower bound for area ratio range',
                    default=0.0)
parser.add_argument('--dt-path', type=str, required=True)
parser.add_argument('--all-cats', action='store_true')


def get_ids_names(coco_gt):
    unseen_classes = [
        "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "dining table", "dog", "horse", "motorcycle",
        "person", "potted plant", "sheep", "couch", "train", "tv"]

    seen_classes = [
        'truck', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
        'pizza', 'donut', 'cake', 'bed', 'toilet', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    seen_ids = [
        i for i, cid in enumerate(coco_gt.cats.keys())
        if coco_gt.cats[cid]['name'] in seen_classes]
    unseen_ids = [
        i for i, cid in enumerate(coco_gt.cats.keys())
        if coco_gt.cats[cid]['name'] in unseen_classes]
    unseen_coco_ids = [
        cid for i, cid in enumerate(coco_gt.cats.keys())
        if coco_gt.cats[cid]['name'] not in seen_classes]
    seen_coco_ids = [
        cid for i, cid in enumerate(coco_gt.cats.keys())
        if coco_gt.cats[cid]['name'] in seen_classes]
    unseen_names = [coco_gt.cats[i]['name'] for i in unseen_coco_ids]
    seen_names = [coco_gt.cats[i]['name'] for i in seen_coco_ids]
    all_coco_ids = sorted(seen_coco_ids + unseen_coco_ids)
    all_ids = sorted(seen_ids + unseen_ids)

    return (unseen_ids, seen_ids, all_ids,
            unseen_coco_ids, seen_coco_ids, all_coco_ids,
            unseen_names, seen_names)


def get_ret_anns(coco_dt, train_imgs, args, unseen_coco_ids):
    all_anns = []

    for cid in unseen_coco_ids:
        invalid_imgs = coco_dt.getImgIds()
        valid_imgs = [i for i in invalid_imgs if i not in train_imgs[cid]]
        ann_ids = coco_dt.getAnnIds(
            catIds=cid, imgIds=valid_imgs, areaRng=AREA_RNG,
            areaRatioRng=[args.ar, 1.0], iscrowd=False)
        anns = coco_dt.loadAnns(ann_ids)
        anns = sorted(anns, key=lambda x: x['score'], reverse=True)
        pres_img_ids = list(set(a['image_id'] for a in anns))
        assert not any(i in train_imgs[cid] for i in pres_img_ids)
        if args.top:
            # use top K
            K_min = int(args.K_min)
            K_max = int(args.K_max)
            if args.full:
                anns = anns[K_max:K_min]
                for ann in anns:
                    ann['ignore_qe'] = 0
                    ann['iscrowd'] = 0
                pres_img_ids = list(set(a['image_id'] for a in anns))
                keep_ids = [a['id'] for a in anns]
                new_ann_ids = coco_dt.getAnnIds(
                    catIds=cid, imgIds=pres_img_ids,
                    areaRng=AREA_RNG, areaRatioRng=[args.ar, 1.0],
                    iscrowd=False)
                new_ann_ids = list(set(new_ann_ids) - set(keep_ids))
                new_anns = coco_dt.loadAnns(new_ann_ids)
                # new_anns = [a for a in new_anns if a['score'] > 0.9]
                for ann in new_anns:
                    ann['ignore_qe'] = 1
                    ann['iscrowd'] = 1
                all_anns.extend(new_anns)
                all_anns.extend(anns)
            else:
                all_anns.extend(anns[K_max:K_min])
        else:
            # use minimum score
            K_min = float(args.K_min)
            K_max = float(args.K_max)
            scores = np.array([x['score'] for x in anns])
            ind_min = np.searchsorted(-scores, -K_min)
            ind_max = np.searchsorted(-scores, -K_max)
            keep_anns = anns[ind_max:ind_min]
            for ann in keep_anns:
                ann['ignore_qe'] = 0
                ann['iscrowd'] = 0
            if args.full:
                pres_img_ids = list(set(a['image_id'] for a in keep_anns))
                keep_ids = [a['id'] for a in keep_anns]
                new_ann_ids = coco_dt.getAnnIds(
                    catIds=cid, imgIds=pres_img_ids,
                    areaRng=AREA_RNG, areaRatioRng=[args.ar, 1.0],
                    iscrowd=False)
                new_ann_ids = list(set(new_ann_ids) - set(keep_ids))
                # new_ann_ids = [i for i in new_ann_ids if i not in keep_ids]
                new_anns = coco_dt.loadAnns(new_ann_ids)
                for ann in new_anns:
                    ann['ignore_qe'] = 1
                    ann['iscrowd'] = 1
                all_anns.extend(new_anns)
            all_anns.extend(keep_anns)
    return all_anns


def save_coco(args, coco_gt, coco_dt, return_anns, return_imgs):
    save_dict = {}
    if args.full_dataset:
        for k in coco_dt.dataset.keys():
            if k not in ['annotations', 'images']:
                save_dict[k] = coco_dt.dataset[k]
    else:
        for k in coco_gt.dataset.keys():
            if k not in ['annotations', 'images']:
                save_dict[k] = coco_gt.dataset[k]
    print("Saving {} new annotations across {} images".format(
        len(return_anns), len(return_imgs)))
    for ann in return_anns:
        if 'segmentation' in ann:
            del ann['segmentation']
        if 'top2_scores' in ann:
            del ann['top2_scores']
        if 'top2_inds' in ann:
            del ann['top2_inds']
    save_dict['annotations'] = return_anns
    save_dict['images'] = return_imgs
    save_name = args.dt_path
    if args.ar:
        s = '_ar{}'.format(str(args.ar).replace('.', ''))
    else:
        s = ''
    if args.top:
        s += '_top_max{}_min{}{}_all.json'.format(str(int(args.K_max)).zfill(4),
                                              str(int(args.K_min)).zfill(4),
                                              '_full' if args.full else '')
    else:
        s += '_score_max{}_min{}{}_all.json'.format(str(args.K_max).replace('.', ''),
                                                str(args.K_min).replace('.', ''),
                                                '_full' if args.full else '')
    save_name = save_name.replace('.json', s)
    if args.all_cats:
        print('here')
        save_name = save_name.replace('.json', '_allcats.json')
    print(save_name)
    with open(save_name, 'w') as fp:
        s = json.dumps(save_dict, indent=4, sort_keys=True)
        fp.write(s)
    return save_name


def main(args):
    gt_path = MetadataCatalog.get(args.json_data).json_file
    coco_gt = COCO_PK(gt_path)
    coco_gt_cats = COCO_PK(MetadataCatalog.get('coco_test_all').json_file)
    coco_gt.dataset['categories'] = coco_gt_cats.dataset['categories']
    ids = get_ids_names(coco_gt_cats)
    (unseen_ids, seen_ids, all_ids,
     unseen_coco_ids, seen_coco_ids, all_coco_ids,
     unseen_names, seen_names) = ids
    train_imgs = defaultdict(list)
    if len(args.gt_data):
        unseen_data = get_detection_dataset_dicts(
            (args.gt_data,))
        unseen_data = combine_datasets([unseen_data, ])
        for d in unseen_data:
            pres_cats = list(set([a['category_id'] for a in d['annotations']]))
            for c in pres_cats:
                train_imgs[all_coco_ids[c]].append(d['image_id'])
    # print(sorted(train_imgs.keys()))
    coco_dt = coco_gt.loadRes(args.dt_path, args.full_dataset)
    return_anns = get_ret_anns(coco_dt, train_imgs, args, unseen_coco_ids if not args.all_cats else all_coco_ids)
    return_img_ids = list(set([a['image_id'] for a in return_anns]))
    return_imgs = coco_gt.loadImgs(return_img_ids)
    save_name = save_coco(args, coco_gt_cats, coco_dt, return_anns, return_imgs)
    if ("unlabeled" not in args.json_data):
        print_precision_per_class(save_name, gt_path)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
