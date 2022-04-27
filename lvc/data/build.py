# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
import pickle
from detectron2.structures import BoxMode
from collections import defaultdict

from detectron2.utils.file_io import PathManager

from detectron2.data.catalog import (
    MetadataCatalog
    )
from detectron2.data.common import (
    DatasetFromList, MapDataset
    )
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    RepeatFactorTrainingSampler, TrainingSampler
    )

from lvc.data.samplers import CategoryAwareSampler, CategoryAreaAwareSampler

from detectron2.data.build import (
    get_detection_dataset_dicts,
    build_batch_data_loader,
    )

from .utils import (
    filter_image_annotations,
    combine_datasets,
    print_instances_class_histogram_force,
    unseen_sample,
    filter_annotations,
    remove_ignore_overlap,
    )

from detectron2.data.build import trivial_batch_collator
from detectron2.data.samplers import InferenceSampler

"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_detection_train_loader",
]


def load_proposals_into_dataset(dataset_dicts, proposal_files):
    """
    Load precomputed object proposals into the dataset.

    The proposal file should be a pickled dict with the following keys:

    - "ids": list[int] or list[str], the image ids
    - "boxes": list[np.ndarray], each is an Nx4 array of boxes corresponding to the image id
    - "objectness_logits": list[np.ndarray], each is an N sized array of objectness scores
      corresponding to the boxes.
    - "bbox_mode": the BoxMode of the boxes array. Defaults to ``BoxMode.XYXY_ABS``.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        proposal_file (str): file path of pre-computed proposals, in pkl format.

    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """
    logger = logging.getLogger(__name__)
    proposals = defaultdict(list)
    for i, proposal_file in enumerate(proposal_files):
        logger.info("Loading proposals from: {}".format(proposal_file))
        with PathManager.open(proposal_file, "rb") as f:
            proposals_ = pickle.load(f, encoding="latin1")
        for k, v in proposals_.items():
            proposals[k].extend(v)

    # Rename the key names in D1 proposal files
    rename_keys = {"indexes": "ids", "scores": "objectness_logits"}
    for key in rename_keys:
        if key in proposals:
            proposals[rename_keys[key]] = proposals.pop(key)

    # Fetch the indexes of all proposals that are in the dataset
    # Convert image_id to str since they could be int.
    img_ids = set({str(record["image_id"]) for record in dataset_dicts})
    id_to_index = {str(id): i for i, id in enumerate(proposals["ids"]) if str(id) in img_ids}

    # Assuming default bbox_mode of precomputed proposals are 'XYXY_ABS'
    bbox_mode = BoxMode(proposals["bbox_mode"]) if "bbox_mode" in proposals else BoxMode.XYXY_ABS

    for record in dataset_dicts:
        # Get the index of the proposal
        i = id_to_index[str(record["image_id"])]

        boxes = proposals["boxes"][i]
        objectness_logits = proposals["objectness_logits"][i]
        # Sort the proposals in descending order of the scores
        inds = objectness_logits.argsort()[::-1]
        record["proposal_boxes"] = boxes[inds]
        record["proposal_objectness_logits"] = objectness_logits[inds]
        record["proposal_bbox_mode"] = bbox_mode

    return dataset_dicts


def get_dataset_dicts_all(cfg):
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        # proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        proposal_files=None,
    )

    if 'all' in cfg.DATASETS.TRAIN[0]:
        # filter out novel class annotations from large scale data
        dataset_dicts = filter_image_annotations(
            dataset_dicts,
            cfg.DATASETS.TRAIN[0],
            cfg.DATASETS.UNSEEN_CLASSES)
        fs_dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.FS_TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            # proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,)
            proposal_files=None,)

        fs_dataset_dicts = filter_image_annotations(
            fs_dataset_dicts,
            cfg.DATASETS.FS_TRAIN[0],
            cfg.DATASETS.SEEN_CLASSES)

        dataset_dicts_new = combine_datasets([fs_dataset_dicts, dataset_dicts])
    else:
        dataset_dicts_new = combine_datasets([dataset_dicts, ])
    if cfg.QUERY_EXPAND.ENABLED:
        dataset_dicts_det = get_detection_dataset_dicts(
            cfg.DATASETS.DT_PATH,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            # proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            proposal_files=None,
        )
#         print_instances_class_histogram_force(dataset_dicts_det, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)
        dataset_dicts_new = combine_datasets([dataset_dicts_new, dataset_dicts_det])
        print(sum(len(d['annotations']) for d in dataset_dicts_new))
        dataset_dicts_new = remove_ignore_overlap(dataset_dicts_new)
        print(sum(len(d['annotations']) for d in dataset_dicts_new))
#         print(len(dataset_dicts_new))
    if cfg.DATASETS.SUBSET:
        dataset_dicts_new = unseen_sample(dataset_dicts_new)
    if cfg.MODEL.LOAD_PROPOSALS:
        # from detectron2.data.build import load_proposals_into_dataset
        dataset_dicts_new = load_proposals_into_dataset(
            dataset_dicts_new, cfg.DATASETS.PROPOSAL_FILES_TRAIN)
    return dataset_dicts_new


def build_detection_train_loader(cfg, mapper=None):
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be ``DatasetMapper(cfg, True)``.

    Returns:
        an infinite iterator of training data
    """
    dataset_dicts_new = get_dataset_dicts_all(cfg)

    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    if cfg.MODEL.PROPOSAL_GENERATOR.NAME == 'RBG':
        dataset_dicts_new = filter_annotations(
            dataset_dicts_new,
            area_rng=cfg.DATALOADER.SHOTS.AREA_RNG,
            rel_area_rng=cfg.DATALOADER.SHOTS.REL_AREA_RNG,
            x_rng=cfg.DATALOADER.SHOTS.X_RNG,
            y_rng=cfg.DATALOADER.SHOTS.Y_RNG,
            check_longest_side_only=cfg.DATALOADER.SHOTS.LONGEST_SIDE_ONLY
            )
    print_instances_class_histogram_force(dataset_dicts_new, class_names)

    dataset = DatasetFromList(dataset_dicts_new, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts_new, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    elif sampler_name == "CategoryAwareSampler":
        c2i = CategoryAwareSampler.category_to_images_from_dict(dataset_dicts_new)
        sampler = CategoryAwareSampler(c2i)
    elif sampler_name == "CategoryAreaAwareSampler":
        ca2i = CategoryAreaAwareSampler.category_to_images_from_dict(dataset_dicts_new)
        sampler = CategoryAreaAwareSampler(ca2i)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_detection_train_mosaic_loader(cfg, mapper=None):
    from .mosaic import DatasetMapperMosaic, MapDatasetMosaic
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be ``DatasetMapper(cfg, True)``.

    Returns:
        an infinite iterator of training data
    """
    dataset_dicts_new = get_dataset_dicts_all(cfg)

    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    print_instances_class_histogram_force(dataset_dicts_new, class_names)

    dataset = DatasetFromList(dataset_dicts_new, copy=False)

    mapper_mosaic = DatasetMapperMosaic(cfg, True)
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDatasetMosaic(dataset, mapper_mosaic, mapper, cfg)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts_new, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    elif sampler_name == "CategoryAwareSampler":
        c2i = CategoryAwareSampler.category_to_images_from_dict(dataset_dicts_new)
        sampler = CategoryAwareSampler(c2i)
    elif sampler_name == "CategoryAreaAwareSampler":
        ca2i = CategoryAreaAwareSampler.category_to_images_from_dict(dataset_dicts_new)
        sampler = CategoryAreaAwareSampler(ca2i)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_detection_test_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=(cfg.MODEL.PROPOSAL_GENERATOR.NAME == 'RGB'),
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset_dicts = combine_datasets([dataset_dicts])
    if cfg.MODEL.PROPOSAL_GENERATOR.NAME == 'RBG':
        dataset_dicts = filter_annotations(
            dataset_dicts,
            area_rng=cfg.DATALOADER.SHOTS.AREA_RNG,
            rel_area_rng=cfg.DATALOADER.SHOTS.REL_AREA_RNG,
            x_rng=cfg.DATALOADER.SHOTS.X_RNG,
            y_rng=cfg.DATALOADER.SHOTS.Y_RNG,
            check_longest_side_only=cfg.DATALOADER.SHOTS.LONGEST_SIDE_ONLY
            )
        dataset_dicts = [d for d in dataset_dicts if len(d['annotations'])]
    class_names = MetadataCatalog.get(dataset_name).thing_classes
    print_instances_class_histogram_force(dataset_dicts, class_names)
    # dataset_dicts = dataset_dicts[:100]

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader
