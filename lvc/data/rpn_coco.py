from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO

import contextlib
import io
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import pickle


def load_coco_rpn_pkl(pickle_file, image_root, dataset_name):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection.
    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        metadata: meta data associated with dataset_name
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    pickle_file = PathManager.get_local_path(pickle_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO()
        with open(pickle_file, 'rb') as fp:
            dataset_dt = pickle.load(fp)
        coco_api.dataset = dataset_dt
        coco_api.createIndex()
    # sort indices for reproducible results
    img_ids = sorted(list(coco_api.imgs.keys()))
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))
#     id_map = metadata["thing_dataset_id_to_contiguous_id"]

    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id", "id"]

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(
            image_root, img_dict["file_name"]
        )
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["category_id"] = obj["category_id"]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_rpn_coco(name, metadata, imgdir, annofile):
    DatasetCatalog.register(
        name,
        lambda: load_coco_rpn_pkl(annofile, imgdir, name),
    )
    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
        dirname="datasets/coco",
        **metadata,
    )
