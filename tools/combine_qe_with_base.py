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
    description='Combine pseudo-annotations with a base dataset')

parser.add_argument('--ps-data', required=True, help='path to the pseudo annotation data')
parser.add_argument('--bs-data', required=True, help='path to the base data')
parser.add_argument('--base-ignore', action='store_true')


def save_coco(args, coco_qe, anns_all):
    save_dict = {}
    for k in coco_qe.dataset.keys():
        if k != 'annotations':
            save_dict[k] = coco_qe.dataset[k]
    save_dict['annotations'] = anns_all
    save_name = args.ps_data
    save_name = save_name.replace('.json', '_wbase.json')
    if args.base_ignore:
        save_name = save_name.replace('.json', '_base_ig.json')
    print(save_name)
    with open(save_name, 'w') as fp:
        json.dump(save_dict, fp, indent=4, sort_keys=True)
    return save_name


def main(args):
    coco_qe = COCO(args.ps_data)
    coco_bs = COCO(args.bs_data)
    base_classes = [k for k, v in coco_bs.cats.items() if v['name'] in SEEN_CLASSES]
    qe_ids = coco_qe.getAnnIds()
    qe_imgs = coco_qe.getImgIds()
    bs_ids = coco_bs.getAnnIds(imgIds=qe_imgs)
    # bs_ids = [i for i in bs_ids if i not in qe_ids]
    anns_bs = coco_bs.loadAnns(bs_ids)
    anns_bs = [ann for ann in anns_bs if ann['category_id'] in base_classes]
    for ann in anns_bs:
        ann['iscrowd'] = 0
        ann['ignore_qe'] = float(args.base_ignore)
        ann['ignore_reg'] = float(args.base_ignore)

    anns_all = anns_bs + coco_qe.loadAnns(qe_ids)
    save_coco(args, coco_qe, anns_all)
    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
