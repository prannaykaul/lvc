"""Implement ROI_heads with extra class agnostic bounding box regression."""
import numpy as np
import torch
from torch import nn
from fvcore.nn import smooth_l1_loss, giou_loss

import logging
from detectron2.layers import ShapeSpec, cat, nonzero_tuple
from detectron2.config import configurable
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from typing import Dict

from .box_head import build_box_head
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, FastRCNNOutputLayers, FastRCNNOutputs
from .roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY


@ROI_HEADS_OUTPUT_REGISTRY.register()
class BoxOnlyLayers(nn.Module):

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        cls_agnostic_bbox_reg: bool = False,
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        box_dim = len(box2box_transform.weights)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.num_classes = num_classes
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else 1
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            # fmt: on
        }

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)
        return self.num_classes, proposal_deltas


@ROI_HEADS_OUTPUT_REGISTRY.register()
class BoxOnlyLayersCascade(nn.Module):

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        lambda_: float,
        roi_heads_name: str,
        cls_agnostic_bbox_reg: bool = False,
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(BoxOnlyLayersCascade, self).__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        box_dim = len(box2box_transform.weights)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.num_classes = num_classes
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else 1
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
        self.lambda_ = lambda_
        self.box2box_transform = box2box_transform
        self.iterate = roi_heads_name != 'CascadeROIHeads'

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "lambda_"               : cfg.MODEL.UBBR.LAMBDA,
            "roi_heads_name"        : cfg.MODEL.ROI_HEADS.NAME
            # fmt: on
        }

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)
        return self.num_classes, proposal_deltas

    def losses(self, predictions, proposals):
        gt_classes = (
            cat([p.gt_classes
                 for p in proposals], dim=0)
            if len(proposals) else torch.empty(0)
        )

        scores, proposal_deltas = predictions
        if len(proposals):
            proposal_boxes = cat(
                [p.proposal_boxes.tensor for p in proposals],
                dim=0)  # Nx4])
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )

        losses = {
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes)
        }

        return losses

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):

        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for
        # foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple(
            (gt_classes >= 0)
            & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )

        loss_after = giou_loss(fg_pred_boxes, gt_boxes[fg_inds])
        if not self.iterate:
            return loss_after.mean()
        loss_before = giou_loss(proposal_boxes[fg_inds], gt_boxes[fg_inds])
        # print('before:', loss_before)
        # print('after', loss_after)
        loss_diff = torch.maximum(
            loss_after - loss_before.mul(self.lambda_),
            torch.zeros_like(loss_after)
        )
        # print(len(loss_diff))
        return loss_diff.mean()

    def predict_boxes(self, predictions, proposals):
        _, pred_proposal_deltas = predictions
        num_preds_per_image = [len(p) for p in proposals]
        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        proposals_stack = box_type.cat([p.proposal_boxes for p in proposals])

        num_pred = len(proposals_stack)
        B = proposals_stack.tensor.shape[1]
        K = pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            pred_proposal_deltas.view(num_pred * K, B),
            proposals_stack.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B))

        return boxes.view(num_pred, K * B).split(num_preds_per_image, dim=0)


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsCascadeBBox(StandardROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.

    Added a class agnostic bounding box regressor on the Fast-RCNN outputs
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeadsCascadeBBox, self).__init__(cfg, input_shape)

        self._init_regressor_head(cfg)

    def _init_regressor_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k]
                              for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.regressor_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )

        self.regressor_predictor = \
            ROI_HEADS_OUTPUT_REGISTRY.get('BoxOnlyLayers')(
                cfg,
                self.regressor_head.output_size,
                True,
            )

    def forward(self, images, features, proposals, targets=None):
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            losses = self._forward_box(features, proposals, targets)
            return proposals, losses
        else:
            pred_instances = self.inference_debug(features, proposals)
            return pred_instances, {}

    def inference_debug(self, features, proposals):
        features = [features[f] for f in self.in_features]
        box_features_proposals = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )

        box_features = self.box_head(box_features_proposals)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            box_features)

        outputs_reg = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta)

        reg_instances, _ = outputs_reg.inference(
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_detections_per_img)
        return reg_instances

        outputs_no_delta = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            torch.zeros_like(pred_proposal_deltas),
            proposals,
            self.smooth_l1_beta)

        box_features_agn = self.regressor_head(box_features_proposals)
        pred_proposal_deltas_agn = self.regressor_predictor(box_features_agn)

        outputs_agn = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas_agn,
            proposals,
            self.smooth_l1_beta)

        no_delta_instances, _ = outputs_no_delta.inference(
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_detections_per_img)

        agn_instances, _ = outputs_agn.inference(
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_detections_per_img)

        for inst in reg_instances:
            inst.set('proposal_boxes', inst.pred_boxes)

        box_features_cascade = self.box_pooler(
            features, [x.pred_boxes for x in reg_instances])
        box_features_cascade = self.regressor_head(box_features_cascade)
        pred_proposal_deltas_cascade = self.regressor_predictor(
            box_features_cascade)

        outputs_cascade = FastRCNNOutputs(
            self.box2box_transform,
            None,
            pred_proposal_deltas_cascade,
            reg_instances,
            self.smooth_l1_beta)

        cascade_pred_boxes = outputs_cascade.predict_boxes()

        for r_inst, c_pred_b in zip(
             reg_instances,
             cascade_pred_boxes):
            r_inst.set('cas_pred_boxes', Boxes(c_pred_b))

        print(reg_instances)
        print(agn_instances)
        print(no_delta_instances)

    def _forward_box(self, features, proposals, targets=None):
        features = [features[f] for f in self.in_features]

        box_features_ = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features_)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            box_features)

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta)

        if self.training:
            losses = outputs.losses()

        box_features = self.regressor_head(box_features_)
        pred_proposal_deltas_new = self.regressor_predictor(
            box_features)
        outputs_new = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas_new,
            proposals,
            self.smooth_l1_beta)

        if self.training:
            losses["loss_box_reg"] *= 2.0
            losses.update(
                {"loss_box_reg_ag": 2.0*outputs_new.smooth_l1_loss(
                    min_area=200.**2)})
            return losses
        else:
            pred_instances, _ = outputs_new.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances

    def _create_proposals_from_boxes(self, boxes_in, probs, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)

        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        boxes = []
        # print(boxes, probs, image_sizes)
        for b, s, im_s in zip(boxes_in, probs, image_sizes):
            # print('here')
            s = s[:, :-1]
            num_bbox_reg_classes = b.shape[1] // 4
            # Convert to Boxes to use the `clip` function ...
            b = Boxes(b.reshape(-1, 4))
            b.clip(im_s)
            b = b.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

            s_fm, fm = s.flatten().topk(100)
            # print(s_fm.max(), s_fm.min(), s_fm.size())
            fm = fm[s_fm > 0.5]
            s_fm = s_fm[s_fm > 0.5]
            # print(s_fm.size())
            c = fm % num_bbox_reg_classes
            r = fm // num_bbox_reg_classes
            # Filter results based on detection scores
            # R' x 2. First column contains indices of the R predictions;
            # Second column contains indices of classes.
            # filter_inds = filter_mask.nonzero(as_tuple=False)
            # print(filter_inds.size())
            fm = torch.stack((r, c)).t()
            # print(fm.size())
            # print(b.size())
            # print(filter_inds)
            # print(fm)
            if num_bbox_reg_classes == 1:
                b = b[fm[:, 0], 0]
            else:
                b = b[fm[:, 0], fm[:, 1]]
            # print(b.size())
            boxes.append(Boxes(b.detach().view(-1, 4)))
        # boxes = [Boxes(b.detach().view(-1, 4)) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            if self.training:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals

    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, targets):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.

        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances

        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matcher(match_quality_matrix)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "bb_head/num_fg_samples",
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "bb_head/roi_head/num_bg_samples",
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals
