"""Implement ROI_heads."""
import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function

import inspect


import logging
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from lvc.modeling.sampling import subsample_labels
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from typing import Dict

from .box_head import build_box_head
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference
# from .fast_rcnn2 import FastRCNNOutputs2
from .fast_rcnn_debug import FastRCNNOutputsDebug

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self.box_reg_loss_type        = cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE
        self.loss_weight              = {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT}
        self.ignore_reg = cfg.MODEL.ROI_HEADS.IGNORE_REG
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes, inference=False):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
            inference,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets, inference=False):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix
            )
            ig_tog = targets_per_image.has('gt_ignores')
            if ig_tog:
                ignores = targets_per_image.gt_ignores.bool()
                if ignores.sum():
                    max_ig = match_quality_matrix[ignores].max(dim=0)[0]
                    needs_toggling = max_ig > self.proposal_matcher.thresholds[1]
                    matched_labels[needs_toggling.bool()] = -1
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes,
                inference=inference,
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith(
                        "gt_"
                    ) and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets]
                        )
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4)
                    )
                )
                proposals_per_image.gt_boxes = gt_boxes
                # gt_ignores_reg = targets_per_image.gt_ignores_reg.new_zeros(
                #     (len(sampled_idxs)))
                # proposals_per_image.gt_ignores_reg = gt_ignores_reg

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item()
            )
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        if not inference:
            storage = get_event_storage()
            storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
            storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled
        )
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class ParallelStandardROIHeads(nn.Module):
    def __init__(self, cfg, input_shape):
        super(ParallelStandardROIHeads, self).__init__()
        cfg_seen = cfg.clone()
        cfg_unseen = cfg.clone()
        cfg_seen.defrost()
        cfg_unseen.defrost()
        cfg_seen.MODEL.ROI_HEADS.NUM_CLASSES = \
            len(cfg.DATASETS.SEEN_IDS)
        cfg_unseen.MODEL.ROI_HEADS.NUM_CLASSES = \
            len(cfg.DATASETS.UNSEEN_IDS)
        cfg_seen.freeze()
        cfg_unseen.freeze()
        self.roi_heads = nn.ModuleDict(
            {'base': StandardROIHeads(cfg_seen, input_shape),
             'novel': StandardROIHeads(cfg_unseen, input_shape)}
        )

        self.seen = torch.Tensor(cfg.DATASETS.SEEN_IDS).long().to(
            cfg.MODEL.DEVICE)
        self.unseen = torch.Tensor(cfg.DATASETS.UNSEEN_IDS).long().to(
            cfg.MODEL.DEVICE)
        self.split = torch.Tensor(cfg.DATASETS.SPLIT_IDS).long().to(
            cfg.MODEL.DEVICE)

    def forward(self, images, features, proposals, targets=None):
        if self.training:
            for gt_inst in targets:
                seen_tog = gt_inst.gt_classes.unsqueeze(1).eq(
                    self.seen.unsqueeze(0)).any(dim=1)
                gt_inst.set('seen', seen_tog)
            gt_instances_split = []
            for gt_inst in targets:
                seen_gt = gt_inst[gt_inst.seen]
                unseen_gt = gt_inst[gt_inst.seen.logical_not()]
                seen_gt.gt_classes = self.split[seen_gt.gt_classes]
                unseen_gt.gt_classes = self.split[unseen_gt.gt_classes]
                gt_instances_split.append([seen_gt, unseen_gt])
            gt_instances_split = list(zip(*gt_instances_split))
            gt_instances_split = {
                'base': gt_instances_split[0],
                'novel': gt_instances_split[1]
            }
            losses = {
                k: v(images, features, proposals, gt_instances_split[k])[1]
                for k, v in self.roi_heads.items()
            }
            return (None, {
                k+'_'+k1: v1
                for k, v in losses.items()
                for k1, v1 in v.items()})

        else:
            return self.inference(images, features, proposals, targets)

    def inference(self, images, features, proposals, targets=None):
        pred_inst_base, _ = \
            self.roi_heads['base'](images, features, proposals, targets)
        pred_inst_novel, _ = \
            self.roi_heads['novel'](images, features, proposals, targets)
        combined_instances = []
        for inst_bas, inst_nov in zip(pred_inst_base, pred_inst_novel):
            inst_bas.pred_classes = self.seen[inst_bas.pred_classes]
            inst_nov.pred_classes = self.unseen[inst_nov.pred_classes]
            inst_new = Instances.cat([inst_bas, inst_nov])
            combined_instances.append(inst_new)

        return combined_instances, None


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)
        if cfg.DEBUG:
            self.debug = True
        else:
            self.debug = False

        if cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNN_Context':
            self.second_highest = True
        else:
            self.second_highest = False
        if cfg.MODEL.PROPOSAL_GENERATOR.NAME == 'RBG':
            self.rbg = True
        else:
            self.rbg = False
        self.reg_off = cfg.MODEL.ROI_HEADS.REG_OFF

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            # self.num_classes,
            # self.cls_agnostic_bbox_reg,
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        elif self.rbg:
            proposals = self.label_and_sample_proposals(proposals, targets, inference=True)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        # import pdb; pdb.set_trace()
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            box_features
        )
        if self.reg_off:
            pred_proposal_deltas = torch.zeros_like(pred_proposal_deltas)
        del box_features
        if self.debug:
            outputs = FastRCNNOutputsDebug(
                self.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
            )

        elif not self.second_highest:
            outputs = FastRCNNOutputs(
                self.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
                self.box_reg_loss_type,
                self.loss_weight,
            )
        else:
            raise NotImplementedError
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class UBBRROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super(UBBRROIHeads, self).__init__(cfg, input_shape)

    def forward(self, images, features, proposals, targets=None):
        if self.training:
            return super(UBBRROIHeads, self).forward(
                images, features, proposals, targets=targets)

        features_list = [features[f] for f in self.in_features]

        pred_instances = self._forward_box(features_list, proposals, targets)
        return pred_instances, {}

    def _forward_box(self, features, proposals, targets):
        # convert targets to proposal_boxes
        for x in targets:
            x.set('proposal_boxes', x.gt_boxes)
        tar = targets
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in tar]
        )
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            box_features
        )
        assert isinstance(pred_class_logits, int)

        image_shapes = [x.image_size for x in tar]
        boxes = self._predict_boxes(pred_proposal_deltas, tar)
        scores = tuple(b.new_ones(len(b), 2) for b in boxes)

        out = fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            -1.e20,
            1.0,
            10**10
        )

        out_sorted = tuple(
            (out[0][i][out[1][i].argsort()] for i in range(len(out[0]))))

        return out_sorted[0], None

    def _predict_boxes(self, pred_proposal_deltas, proposals):
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
class CascadeUBBRROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super(CascadeUBBRROIHeads, self).__init__(cfg, input_shape)

        self.steps = cfg.MODEL.UBBR.CASCADE_STEPS
        self.qe = cfg.QUERY_EXPAND.ENABLED

    def forward(self, images, features, proposals, targets=None):
        del images
        if not self.qe:
            proposals = self.label_and_sample_proposals(
                proposals, targets, inference=(not self.training))
        else:
            for x in targets:
                x.set('proposal_boxes', x.gt_boxes)
        if self.training:
            losses = self._forward_box(features, proposals, targets)
            return proposals, losses

        # features_list = [features[f] for f in self.in_features]

        pred_instances, proposals = self._forward_box(features, proposals, targets)
        return pred_instances, proposals

    def _forward_box(self, features, proposals, targets):
        if self.qe:
            proposals = targets
        # convert targets to proposal_boxes
        features = [features[f] for f in self.in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.steps):
            if k > 0:
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes, image_sizes)
                if self.training:
                    proposals = self._match_and_label_boxes(
                        proposals, k, targets)
            predictions = self._run_step(features, proposals, k)
            prev_pred_boxes = self._predict_boxes(predictions, proposals)
            head_outputs.append([predictions, proposals, prev_pred_boxes])

        if self.training:
            losses = {}
            storage = get_event_storage()
            for step, (predictions, proposals, _) in enumerate(head_outputs):
                with storage.name_scope("step{}".format(step)):
                    step_losses = self.box_predictor.losses(predictions, proposals)
                losses.update({k + "_step{}".format(step): v for k, v in step_losses.items()})
            return losses

        else:
            (num_classes, _), _, pred_boxes = head_outputs[-1]
#             print(pred_boxes)
#             print(len(pred_boxes[0]))
            proposals = head_outputs[0][1]
#             print(proposals)
#             print(len(proposals[0]))
            scores = list(b.new_zeros(len(p), num_classes+1) for b, p in zip(pred_boxes, proposals))
            for j, (prop, sc) in enumerate(zip(proposals, scores)):
                sc[torch.arange(len(prop)).to(sc.device), prop.gt_classes] = 1.0
                prop = prop[prop.gt_classes < num_classes]
                proposals[j] = prop
            out = fast_rcnn_inference(
                pred_boxes,
                scores,
                image_sizes,
                0.1,
                1.0,
                10**10
            )
            return tuple(o[0][o[1].argsort()] for o in zip(*out)), proposals
#             return tuple((out[0][i][out[1][i].argsort()] for i in range(len(out[0])))), proposals

    def _run_step(self, features, proposals, k):
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals])

        box_features = _ScaleGradient.apply(box_features, 1.0 / self.steps)
        box_features = self.box_head(box_features)
        return self.box_predictor(box_features)

    def _predict_boxes(self, predictions, proposals):
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

    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
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
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)

        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        boxes = [Boxes(b.detach()) for b in boxes]
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


@ROI_HEADS_REGISTRY.register()
class CascadeStandardROIHeads(StandardROIHeads):

    def __init__(self, cfg, input_shape):
        super(CascadeStandardROIHeads, self).__init__(cfg, input_shape)

    def _forward_box(self, features, proposals):
        # print(proposals)
        if self.training:
            return super(CascadeStandardROIHeads, self)._forward_box(features, proposals)
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            box_features
        )
        del box_features
        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        init_instances, _ = outputs.inference(
            self.test_score_thresh,
            1.5,
            # self.test_nms_thresh,
            int(1e10),
        )
        box_features = self.box_pooler(
            features, [x.pred_boxes for x in init_instances]
        )
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            box_features
        )
        del box_features
        init_instances[0].set('proposal_boxes', init_instances[0].pred_boxes)
        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            init_instances,
            self.smooth_l1_beta,
        )
        pred_instances, _ = outputs.inference(
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_detections_per_img,
        )
        return pred_instances
