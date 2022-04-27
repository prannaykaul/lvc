# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from torchvision.ops import box_iou

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

from lvc.data.dataset_mapper import DatasetMapperCrop

from detectron2.data.build import (
    get_detection_dataset_dicts,
    build_batch_data_loader,
    )

from .utils import (
    filter_image_annotations,
    combine_datasets,
    print_instances_class_histogram_force,
    unseen_sample,
    filter_proposal_boxes,
    filter_annotations,
    remove_overlap_proposals,
    )

from detectron2.structures import BoxMode, Boxes
from detectron2.data.build import trivial_batch_collator
from detectron2.data.samplers import InferenceSampler
from .common_rpn import MapDatasetCrop
from .build import get_dataset_dicts_all


def build_crop_shots_loader(cfg, mapper=None):

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.FS_TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        # proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        proposal_files=None,
    )

    dataset_dicts = filter_annotations(
        dataset_dicts,
        area_rng=cfg.DATALOADER.SHOTS.AREA_RNG,
        rel_area_rng=cfg.DATALOADER.SHOTS.REL_AREA_RNG,
        x_rng=cfg.DATALOADER.SHOTS.X_RNG,
        y_rng=cfg.DATALOADER.SHOTS.Y_RNG,
        check_longest_side_only=cfg.DATALOADER.SHOTS.LONGEST_SIDE_ONLY
        )

    class_names = MetadataCatalog.get(cfg.DATASETS.FS_TRAIN[0]).thing_classes
    print_instances_class_histogram_force(dataset_dicts, class_names)

    cumsums = torch.cumsum(
        torch.Tensor([len(dd["annotations"]) for dd in dataset_dicts]), dim=0)

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapperCrop(cfg, False, True)
    dataset = MapDatasetCrop(dataset, cumsums, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        drop_last=False,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def build_crop_test_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_test_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size as defined.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given proposal crops
        dataset, with test-time transformation and batching.
    """

    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset_dicts = filter_proposal_boxes(
        dataset_dicts,
        area_rng=cfg.DATALOADER.PROPOSALS.AREA_RNG,
        rel_area_rng=cfg.DATALOADER.PROPOSALS.REL_AREA_RNG,
        x_rng=cfg.DATALOADER.PROPOSALS.X_RNG,
        y_rng=cfg.DATALOADER.PROPOSALS.Y_RNG,
        topk=cfg.DATALOADER.PROPOSALS.TOPK)

    dataset_dicts = filter_image_annotations(
        dataset_dicts,
        dataset_name,
        cfg.DATASETS.UNSEEN_CLASSES,
        test=True)

    dataset_dicts = remove_overlap_proposals(
        dataset_dicts,
        cfg.DATALOADER.PROPOSALS.IOU_THRESH,
        )

    dataset_dicts = [dd for dd in dataset_dicts if len(dd["proposal_boxes"])]

    cumsums = torch.cumsum(
        torch.Tensor([len(dd["proposal_boxes"]) for dd in dataset_dicts]), dim=0)
    logging.getLogger(__name__).log(
        logging.INFO,
        "Total proposals: {}, Total images: {}".format(cumsums[-1].item(), len(cumsums)))

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapperCrop(cfg, False, False)
    dataset = MapDatasetCrop(dataset, cumsums, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        drop_last=False,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader
