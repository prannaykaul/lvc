# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import torch.utils.data as data

from detectron2.utils.serialize import PicklableWrapper
import torch


class MapDatasetCrop(data.Dataset):
    """
    Map a function over the elements in a dataset.

    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, cumsums, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))
        self._cumsums = cumsums.int()

    def __len__(self):
        return self._cumsums[-1].item()

    def __getitem__(self, idx):
        cur_idx = int(idx)
        img_idx = torch.searchsorted(self._cumsums, cur_idx, right=True).item()
        if img_idx != 0:
            box_idx = idx % self._cumsums[img_idx - 1].item()
        else:
            box_idx = idx

        data = self._map_func(self._dataset[img_idx], box_idx)
        return data
