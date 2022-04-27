import logging
import random
import torch
import operator
import os

from detectron2.data.catalog import MetadataCatalog
from detectron2.data.common import (
    DatasetFromList, MapDataset
    )

from detectron2.data.dataset_mapper import DatasetMapper  # TODO
from detectron2.data.samplers import TrainingSampler
from detectron2.utils.comm import get_world_size, is_main_process


from detectron2.data.build import (
    get_detection_dataset_dicts,
    worker_init_reset_seed,)

from .utils import (
    filter_image_annotations,
    combine_datasets)


class AspectRatioGroupedDatasetExem(torch.utils.data.IterableDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size, fg_buckets):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)]
        self._fg_buckets = fg_buckets

        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    @classmethod
    def get_fg_buckets(cls, dataset_dicts, no_bkg_inds):
        fg_buckets = [[] for _ in range(2)]
        for i in no_bkg_inds:
            d = dataset_dicts[i]
            w, h = d["width"], d["height"]
            bucket_id = 0 if w > h else 1
            fg_buckets[bucket_id].append(i)
        return fg_buckets

    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            bucket_id = 0 if w > h else 1
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size-1:
                rand_id = random.choice(self._fg_buckets[bucket_id])
                bucket.append(self.dataset.dataset[rand_id])
                yield bucket[:]
                del bucket[:]


def build_batch_data_loader(
    dataset, sampler, total_batch_size, fg_buckets,
        aspect_ratio_grouping=False, num_workers=0):
    """
    Build a batched dataloader for training.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size (int): total batch size across GPUs.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    if aspect_ratio_grouping:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedDatasetExem(data_loader, batch_size, fg_buckets)
    else:
        raise NotImplementedError


def build_detection_train_loader(cfg, mapper=None):

    dataset_dicts_ls = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    # filter out novel class annotations from large scale data
    dataset_dicts_ls = filter_image_annotations(
        dataset_dicts_ls,
        cfg.DATASETS.TRAIN[0],
        cfg.DATASETS.UNSEEN_CLASSES)

    dataset_dicts_fs = get_detection_dataset_dicts(
        cfg.DATASETS.FS_TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,)

    dataset_dicts_fs = filter_image_annotations(
        dataset_dicts_fs,
        cfg.DATASETS.FS_TRAIN[0],
        cfg.DATASETS.SEEN_CLASSES)

    dataset_dicts_ls = combine_datasets([dataset_dicts_ls])
    dataset_dicts_fs = combine_datasets([dataset_dicts_fs])

    # get_cid_id list
    id_cid = [[a['category_id'], a['id']]
              for d in dataset_dicts_fs
              for a in d['annotations']]
    # mapping from id to eid
    id_cid = sorted(id_cid)
    id_eid_map = {a[1]: i for i, a in enumerate(id_cid)}
    if is_main_process():
        id_mapping = {i: a for i, a in enumerate(id_cid)}
        torch.save(id_mapping,
                   os.path.join(cfg.OUTPUT_DIR, 'dset_id_mapping.pth'))

    for d in dataset_dicts_fs:
        for a in d['annotations']:
            s = a['category_id']
            a['category_id'] = id_eid_map[a['id']]
            a['og_category_id'] = s

    bkg = len(id_eid_map)
    for d in dataset_dicts_ls:
        for a in d['annotations']:
            s = a['category_id']
            a['category_id'] = bkg
            a['og_category_id'] = s

    dataset_dicts = combine_datasets([dataset_dicts_ls, dataset_dicts_fs])
    no_bkg_inds = []
    for i, d in enumerate(dataset_dicts):
        keep = any([a['category_id'] != bkg for a in d['annotations']])
        if keep:
            no_bkg_inds.append(i)

    fg_buckets = AspectRatioGroupedDatasetExem.get_fg_buckets(
        dataset_dicts, no_bkg_inds)

    dataset = DatasetFromList(dataset_dicts, copy=False)
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler = TrainingSampler(len(dataset))
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        fg_buckets,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
