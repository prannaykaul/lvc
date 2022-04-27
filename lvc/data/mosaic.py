import copy
import logging
import numpy as np
import pickle
import random
import torch.utils.data as data

from typing import List, Optional, Union
import torch

from detectron2.utils.serialize import PicklableWrapper
from detectron2.config import configurable
from detectron2.structures import BoxMode
from detectron2.data import DatasetMapper

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from .detection_utils import annotations_to_instances_ignore

__all__ = ["DatasetMapperMosaic", "MapDatasetMosaic"]


def get_mosaic(dataset_dicts, images):
    h0, w0, n = images[0].shape
    s = max(h0, w0)
    image_out = np.full((s*2, s*2, n), 114, dtype=np.uint8)
    yc, xc = s, s
    maxx2, minx1 = 0, 1000000000
    maxy2, miny1 = 0, 1000000000

    for i, img in enumerate(images):
        h, w = img.shape[:-1]
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        if i == 0 or i == 2:
            minx1 = min((minx1, x1a))
        if i == 0 or i == 1:
            miny1 = min(miny1, y1a)
        if i == 1 or i == 3:
            maxx2 = max(maxx2, x2a)
        if i == 2 or i == 3:
            maxy2 = max(maxy2, y2a)
        image_out[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        for ann in dataset_dicts[i]['annotations']:
            ann['bbox'][0] += x1a
            ann['bbox'][1] += y1a

    dataset_dict_out = dataset_dicts[i]
    dataset_dict_out['annotations'] = [ann for d in dataset_dicts for ann in d['annotations']]
    image_out = image_out[miny1:maxy2, minx1:maxx2]
    for ann in dataset_dict_out['annotations']:
        ann['bbox'][0] -= minx1
        ann['bbox'][1] -= miny1
    return image_out, dataset_dict_out


def get_mosaic9(dataset_dicts, images):
    h0, w0, n = images[0].shape
    s = max(h0, w0)
    image_out = np.full((s*3, s*3, n), 114, dtype=np.uint8)  # base image with 9 tiles
    H, W = image_out.shape[:-1]
    yc, xc = (3*s)//2, (3*s)//2
    # maxx3, minx1 = 0, 1000000000
    # maxy3, miny1 = 0, 1000000000
    ltrb = []
    for i, img in enumerate(images):
        h, w = img.shape[:-1]
        if i == 0:
            x1a, y1a, x2a, y2a = xc - int(np.floor(w/2)), yc - int(np.floor(h/2)), xc + int(np.ceil(w/2)), yc + int(np.ceil(h/2))
            x1b, y1b, x2b, y2b = 0, 0, w, h
        elif i == 1:  # top
            x1a, y1a, x2a, y2a = max(0, xc - int(np.floor(w/2))), max(0, ltrb[0][1] - h), min(W, xc + int(np.ceil(w/2))), ltrb[0][1]
            x1b, y1b, x2b, y2b = w//2 - int(np.floor((x2a-x1a)/2)), h - (y2a - y1a), w//2 + int(np.ceil((x2a-x1a)/2)), h
        elif i == 2:  # top left
            x1a, y1a, x2a, y2a = max(0, ltrb[1][0] - w), max(0, ltrb[1][3] - h), ltrb[1][0], ltrb[1][3]
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 3:  # left
            x1a, y1a, x2a, y2a = max(0, ltrb[0][0] - w), ltrb[2][3], ltrb[0][0], min(ltrb[0][3], ltrb[2][3] + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, y2a-y1a
        elif i == 4:  # bottom left
            x1a, y1a, x2a, y2a = max(0, ltrb[0][0] - w), ltrb[3][3], ltrb[0][0], min(H, ltrb[3][3] + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, (y2a - y1a)
        elif i == 5:  # bottom
            x1a, y1a, x2a, y2a = ltrb[0][0], ltrb[0][3], min(W, ltrb[0][0] + w), min(H, ltrb[0][3] + h)
            x1b, y1b, x2b, y2b = 0, 0, (x2a - x1a), (y2a - y1a)
        elif i == 6:  # bottom right
            x1a, y1a, x2a, y2a = ltrb[5][2], ltrb[0][3], min(H, ltrb[5][2] + w), min(H, ltrb[0][3] + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, (y2a - y1a)
        elif i == 7:  # right
            x1a, y1a, x2a, y2a = ltrb[0][2], ltrb[2][3], min(W, ltrb[0][2] + w), min(ltrb[0][3], ltrb[2][3] + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h//2 - int(np.floor((y2a-y1a)/2)), w, h//2 + int(np.ceil((y2a-y1a)/2))
        elif i == 8:  # top right
            x1a, y1a, x2a, y2a = ltrb[1][2], max(0, ltrb[1][3] - h), min(ltrb[1][2] + w, W), ltrb[1][3]
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        ltrb.append([x1a, y1a, x2a, y2a])
        for ann in dataset_dicts[i]['annotations']:
            x1, y1, w, h = ann['bbox']
            x2, y2 = x1 + w, y1 + h
            y2, x2 = min(y2, y2b), min(x2, x2b)
            y1, x1 = max(y1, y1b), max(x1, x1b)
            h, w, = max(0.0, y2 - y1), max(0.0, x2 - x1)
            x1 += x1a - x1b
            y1 += y1a - y1b
            ann['bbox'] = [x1, y1, w, h]

        image_out[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
    dataset_dict_out = dataset_dicts[i]
    dataset_dict_out['annotations'] = [ann for d in dataset_dicts for ann in d['annotations']]
    x1s, y1s, x2s, y2s = list(zip(*ltrb))
    miny1, minx1, maxy2, maxx2 = min(y1s), min(x1s), max(y2s), max(x2s)
    image_out = image_out[miny1:maxy2, minx1:maxx2]
    for ann in dataset_dict_out['annotations']:
        ann['bbox'][0] -= minx1
        ann['bbox'][1] -= miny1

    return image_out, dataset_dict_out


class MapDatasetMosaic(data.Dataset):
    """
    Map a function over the elements in a dataset.

    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, map_func_mosaic, map_func, cfg):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work
        self._map_func_mosaic = PicklableWrapper(map_func_mosaic)  # wrap so that a lambda will work


#         self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))
        self._mos = cfg.INPUT.MOSAIC
        self._mos49 = cfg.INPUT.MOSAIC49SPLIT

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)
        if random.random() < self._mos:
            if random.random() < self._mos49:
                idxs = [cur_idx] + random.sample(range(len(self)), k=3)
                return self._map_func_mosaic([self._dataset[cur_idx] for cur_idx in idxs])
            else:
                idxs = [cur_idx] + random.sample(range(len(self)), k=8)
                return self._map_func_mosaic([self._dataset[cur_idx] for cur_idx in idxs])
        else:
            return self._map_func(self._dataset[cur_idx])


class DatasetMapperMosaic:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = augmentations
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(
                0,
                T.RandomCrop(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(augs[0])
            )
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }
        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __call__(self, dataset_dicts):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dicts = copy.deepcopy(dataset_dicts)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        images = [utils.read_image(d["file_name"], format=self.image_format)
                  for d in dataset_dicts]
        for d, image in zip(dataset_dicts, images): utils.check_image_size(d, image)

        if len(dataset_dicts) == 4:
            image, dataset_dict = get_mosaic(dataset_dicts, images)
        elif len(dataset_dicts) == 9:
            image, dataset_dict = get_mosaic9(dataset_dicts, images)
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt, boxes=boxes)
        transforms = aug_input.apply_augmentations(self.augmentations)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances_ignore(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
