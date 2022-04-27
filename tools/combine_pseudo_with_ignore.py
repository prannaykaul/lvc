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

UNSEEN_CLASSES = _CC.DATASETS.UNSEEN_CLASSES
SEEN_CLASSES = _CC.DATASETS.SEEN_CLASSES
AREA_RNG = [64 ** 2, 1e5 ** 2]

parser = argparse.ArgumentParser(
    description='Combine pseudo-annotations with a ignore dataset')

parser.add_argument('--ps-data', required=True, help='path to the pseudo annotation data')
parser.add_argument('--ig-data', required=True, help='path to the ignore data')

def save_coco(args, coco_qe, anns_all):
    save_dict = {}
    for k in coco_qe.dataset.keys():
        if k != 'annotations':
            save_dict[k] = coco_qe.dataset[k]
    save_dict['annotations'] = anns_all
    save_name = args.ps_data
    save_name = save_name.replace('.json', '_ignore.json')
    print(save_name)
    with open(save_name, 'w') as fp:
        s = json.dumps(save_dict, indent=4, sort_keys=True)
        fp.write(s)
    return save_name


def main(args):
    coco_ig = COCO(args.ig_data)
    try:
        coco_qe = COCO(args.ps_data)
        filt = True
    except AssertionError:
        coco_qe = coco_ig.loadRes(args.ps_data)
        filt = False

    qe_ids = coco_qe.getAnnIds()
    qe_anns = coco_qe.loadAnns(qe_ids)
    if filt:
        qe_anns = [ann for ann in qe_anns if not ann['ignore_qe']]
    qe_ids = [ann['id'] for ann in qe_anns]
    qe_imgs = list(set([ann['image_id'] for ann in qe_anns]))
    # qe_imgs = coco_qe.getImgIds()
    ig_ids = coco_ig.getAnnIds(imgIds=qe_imgs)
    if filt:
        ig_ids = list(set(ig_ids) - set(qe_ids))
    # ig_ids = [i for i in ig_ids if i not in qe_ids]
    anns_ig = coco_ig.loadAnns(ig_ids)
    max_id = max(ann['id'] for ann in anns_ig)
    for ann in anns_ig:
        ann['iscrowd'] = 0
        ann['ignore_qe'] = 1
    anns_qe = coco_qe.loadAnns(qe_ids)
    if not filt:
        for ann in anns_qe:
            ann['id'] += max_id
    anns_all = anns_ig + anns_qe
    assert (len(set([ann['id'] for ann in anns_all])) == len(anns_all)), (len(anns_all),
                                                                         len(set([ann['id'] for ann
                                                                                  in anns_all])))
    # for ann in anns_all:
    #     ann['ignore_reg'] = 1
    save_coco(args, coco_qe, anns_all)
    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
