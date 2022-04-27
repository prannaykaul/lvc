import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from detectron2.modeling.backbone import build_backbone
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n

from .build import META_ARCH_REGISTRY

__all__ = ["RPNComp"]


@META_ARCH_REGISTRY.register()
class RPNComp(nn.Module):
    """
    RPN Comparison between a cropped patch and set of NK shots
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        if cfg.MODEL.RPNCOMP.POOLER == 'avg':
            self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        elif cfg.MODEL.RPNCOMP.POOLER == 'max':
            self.pooler = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pooler = nn.Identity()

    def forward(self, batched_inputs):

        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = features['res5']
        features = self.pooler(features).squeeze(-1).squeeze(-1)

        return features

    def register_shots(self, shots, classes):
        self.shot_features = shots.to(self.device)
        self.shot_classes = classes.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images
