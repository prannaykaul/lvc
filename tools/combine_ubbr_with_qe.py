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
import copy


parser = argparse.ArgumentParser(
    description='Combine ubbr with pseudo-annotations')

parser.add_argument('--ubbr-data', required=True, help='path to the pseudo annotation data')
parser.add_argument('--qe-data', required=True, help='path to the base data')


def save_coco(args, coco_qe, anns_all):
    save_dict = {}
    for k in coco_qe.dataset.keys():
        if k != 'annotations':
            save_dict[k] = coco_qe.dataset[k]
    save_dict['annotations'] = anns_all
    save_name = args.ubbr_data
    save_name = save_name.replace('.json', '_id.json')
    print(save_name)
    with open(save_name, 'w') as fp:
        json.dump(save_dict, fp, indent=4, sort_keys=True)
    return save_name


def main(args):
    coco_qe = COCO(args.qe_data)
    json_ubbr = json.load(open(
        args.ubbr_data, 'r'))
    uaid2anns = {a['id']: a for a in json_ubbr}
    anns_old = copy.deepcopy(coco_qe.loadAnns(
        list(uaid2anns.keys())))
    anns_new = []
    for a in anns_old:
        assert a['id'] in uaid2anns
        a['bbox'] = uaid2anns[a['id']]['bbox']
        anns_new.append(a)
    save_coco(args, coco_qe, anns_new)
    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
