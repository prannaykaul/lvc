import itertools
import math
from collections import defaultdict
from typing import Optional, Dict, List
import torch
from torch.utils.data.sampler import Sampler

from detectron2.utils import comm


class CategoryAwareSampler(Sampler):
    """
    Here we sample categories first and then sample an image from a list of
    images containing this catgory
    """
    def __init__(self,
                 cats2imgs: Dict[int, List],
                 shuffle: bool = True,
                 seed: Optional[int] = None):
        """
        Args:
            cats2imgs (dict): dictionary of images containing each cat_id
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._c2i = cats2imgs

    @staticmethod
    def category_to_images_from_dict(dataset_dicts):
        """
        Compute the category to image ids dict from the dataset_dicts list

        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.

        Returns:
            dict[list]: category dict with indices list in each value
        """

        c2i = defaultdict(list)
        for i, dataset_dict in enumerate(dataset_dicts):
            cat_ids = {ann["category_id"]
                       for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                c2i[cat_id].append(i)

        return c2i

    def repeat_for_balanced_sampling(self, g):
        most_frequent = len(max(self._c2i.values(), key=lambda x: len(x)))
        inds = []
        for k, v in self._c2i.items():
            reps = most_frequent // len(v)
            inds.extend(v*reps)
            rem = most_frequent % len(v)
            sub_samp = torch.randperm(len(v), generator=g)[:rem]
            sub_samp = torch.tensor(v, dtype=torch.int64)[sub_samp].tolist()
            inds.extend(sub_samp)

        return torch.tensor(inds, dtype=torch.int64)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            # Get long list of indices with images balanced
            indices = self.repeat_for_balanced_sampling(g=g)
            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm]
            else:
                yield from indices


class CategoryAreaAwareSampler(Sampler):
    """
    Here we sample categories first and then sample an image from a list of
    images containing this catgory
    """
    def __init__(self,
                 ca2imgs: Dict[int, List],
                 shuffle: bool = True,
                 seed: Optional[int] = None):
        """
        Args:
            cats2imgs (dict): dictionary of images containing each cat_id
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._ca2i = ca2imgs

    @staticmethod
    def size_labels(dataset_dict):
        size_label = []
        for ann in dataset_dict['annotations']:
            if ann['area'] < 32**2:
                lbl = 's'
            elif ann['area'] > 96**2:
                lbl = 'l'
            else:
                lbl = 'm'
            size_label.append(lbl)
        return size_label

    @staticmethod
    def category_to_images_from_dict(dataset_dicts):
        """
        Compute the category to image ids dict from the dataset_dicts list

        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.

        Returns:
            dict[list]: category dict with indices list in each value
        """

        ca2i = defaultdict(list)
        for i, dataset_dict in enumerate(dataset_dicts):
            a_labels = CategoryAreaAwareSampler.size_labels(dataset_dict)
            cat_ids = [ann["category_id"]
                       for ann in dataset_dict["annotations"]]
            comb_ids = {str(c)+lbl for c, lbl in zip(cat_ids, a_labels)}
            for cat_id in comb_ids:
                ca2i[cat_id].append(i)

        return ca2i

    def repeat_for_balanced_sampling(self, g):
        most_frequent = len(max(self._ca2i.values(), key=lambda x: len(x)))
        inds = []
        for k, v in self._ca2i.items():
            reps = most_frequent // len(v)
            inds.extend(v*reps)
            rem = most_frequent % len(v)
            sub_samp = torch.randperm(len(v), generator=g)[:rem]
            sub_samp = torch.tensor(v, dtype=torch.int64)[sub_samp].tolist()
            inds.extend(sub_samp)

        return torch.tensor(inds, dtype=torch.int64)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            # Get long list of indices with images balanced
            indices = self.repeat_for_balanced_sampling(g=g)
            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm]
            else:
                yield from indices
