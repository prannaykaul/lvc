import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import box_iou

from typing import Optional, Tuple, Dict, List

from lvc.modeling.roi_heads import build_roi_heads

import logging
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
import detectron2.utils.comm as comm
from lvc.modeling.proposal_generator.rbg import RBG

# avoid conflicting with the existing GeneralizedRCNN module in Detectron2
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

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

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.BACKBONE.FREEZE_BOTTOM_UP:
            for p in self.backbone.bottom_up.parameters():
                p.requires_grad = False
            print("froze backbone bottom up parameters")

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            if self.proposal_generator:
                for p in self.proposal_generator.parameters():
                    p.requires_grad = False
                print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            if (cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG):
                for n, p in self.roi_heads.box_predictor.named_parameters():
                    if 'bbox_pred' in n:
                        p.requires_grad = False
                print("class agnostic regression froze box_pred parameters")
            print("froze roi_box_head parameters")
        if (cfg.MODEL.ROI_HEADS.FREEZE_BBOX_PRED):
            for n, p in self.roi_heads.box_predictor.named_parameters():
                if 'bbox_pred' in n:
                    p.requires_grad = False
            print("froze box_pred parameters")

        if cfg.MODEL.PROPOSAL_GENERATOR.UNFREEZE_FIN:
            for p in self.proposal_generator.rpn_head.objectness_logits.parameters():
                p.requires_grad = True
            for p in self.proposal_generator.rpn_head.anchor_deltas.parameters():
                p.requires_grad = True
            print("unfroze final rpn conv parameters")
        assert not cfg.MODEL.IMAGES_ONLY
        self.roi_heads_name = cfg.MODEL.ROI_HEADS.NAME
        self.output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        # print(images.tensor.size(), images.tensor.device)
        # print(len(features), type(features))
        # for k, v in features.items():
        #     print(k, v.size())

        if isinstance(self.proposal_generator, RBG):
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}
            proposals, _ = self.proposal_generator(proposals, gt_instances)

        elif self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.output_layer == "BoxOnlyLayersCascade":
                if "instances" in batched_inputs[0]:
                    gt_instances = [
                        x["instances"].to(self.device) for x in batched_inputs
                    ]
                if "proposals" in batched_inputs[0]:
                    proposals = [
                        x["proposals"].to(self.device) for x in batched_inputs
                    ]
                out_results, in_proposals = self.roi_heads(
                    images, features, proposals, gt_instances)
                # print(out_results)
                # print(in_proposals)
                input_ious = []
                output_ious = []
                gt_classes = []
                for out, inp in zip(out_results, in_proposals):
                    pred_boxes = out.pred_boxes.tensor
                    gt_boxes = inp.gt_boxes.tensor
                    prop_boxes = inp.proposal_boxes.tensor
                    input_ious.append(torch.diag(
                        box_iou(prop_boxes, gt_boxes)))
                    output_ious.append(torch.diag(
                        box_iou(pred_boxes, gt_boxes)))
                    gt_classes.append(inp.gt_classes)
                return {
                    'input_ious': torch.cat(input_ious), 'output_ious': torch.cat(output_ious),
                    'gt_classes': torch.cat(gt_classes)
                }
                    
                # processed_results = []
                # for results_per_image, input_per_image, image_size in zip(
                #     out_results, batched_inputs, images.image_sizes
                # ):
                #     height = input_per_image.get("height", image_size[0])
                #     width = input_per_image.get("width", image_size[1])
                #     r = detector_postprocess(input_per_image["instances"], height, width)
                #     input_per_image["instances"] = r
                #     del input_per_image["image"]
                #     processed_results.append(input_per_image)
                # return processed_results

            elif self.roi_heads_name == "UBBRROIHeads":
                if "instances" in batched_inputs[0]:
                    gt_instances = [
                        x["instances"].to(self.device) for x in batched_inputs
                    ]

                do_postprocess = True
                results, _ = self.roi_heads(images, features, None, gt_instances)
                for res_per_image, inp_per_image in zip(results, batched_inputs):
                    inp_per_image["instances"].set('pred_boxes', res_per_image.pred_boxes)
                    inp_per_image["instances"].set('pred_classes', inp_per_image["instances"].gt_classes)
                if not do_postprocess:
                    return results
                else:
                    processed_results = []
                    for results_per_image, input_per_image, image_size in zip(
                        results, batched_inputs, images.image_sizes
                    ):
                        height = input_per_image.get("height", image_size[0])
                        width = input_per_image.get("width", image_size[1])
                        r = detector_postprocess(input_per_image["instances"], height, width)
                        input_per_image["instances"] = r
                        del input_per_image["image"]
                        processed_results.append(input_per_image)
                    return processed_results

            elif isinstance(self.proposal_generator, RBG):
                if "instances" in batched_inputs[0]:
                    gt_instances = [
                        x["instances"].to(self.device) for x in batched_inputs
                    ]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]
                proposals, _ = self.proposal_generator(proposals, gt_instances)
                do_postprocess = self.roi_heads_name == "UBBRROIHeads"
                results, _ = self.roi_heads(images, features, proposals, gt_instances)
                if not do_postprocess:
                    return results
                else:
                    processed_results = []
                    for results_per_image, input_per_image, image_size in zip(
                        results, batched_inputs, images.image_sizes
                    ):
                        height = input_per_image.get("height", image_size[0])
                        width = input_per_image.get("width", image_size[1])
                        r = detector_postprocess(results_per_image, height, width)
                        processed_results.append({"instances": r})
                    return processed_results

            elif self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

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


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNRegOnly(GeneralizedRCNN):
    def __init__(self, cfg):
        super(GeneralizedRCNNRegOnly, self).__init__(cfg)

    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if "instances" in batched_inputs[0]:
                gt_instances = [
                    x["instances"].to(self.device) for x in batched_inputs
                ]

            do_postprocess = True
            results, _ = self.roi_heads(images, features, None, gt_instances)
            for res_per_image, inp_per_image in zip(results, batched_inputs):
                inp_per_image["instances"].set('pred_boxes', res_per_image.pred_boxes)
                inp_per_image["instances"].set('pred_classes', inp_per_image["instances"].gt_classes)
            if not do_postprocess:
                return results
            else:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(input_per_image["instances"], height, width)
                    input_per_image["instances"] = r
                    del input_per_image["image"]
                    processed_results.append(input_per_image)
                return processed_results

        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )

        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs, no_post=False):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )
        if no_post:
            return proposals, images
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_Context(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.build_context(cfg)

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

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.BACKBONE.FREEZE_BOTTOM_UP:
            for p in self.backbone.bottom_up.parameters():
                p.requires_grad = False
            print("froze backbone bottom up parameters")

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            if self.proposal_generator:
                for p in self.proposal_generator.parameters():
                    p.requires_grad = False
                print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

        if cfg.MODEL.PROPOSAL_GENERATOR.UNFREEZE_FIN:
            for p in self.proposal_generator.rpn_head.objectness_logits.parameters():
                p.requires_grad = True
            for p in self.proposal_generator.rpn_head.anchor_deltas.parameters():
                p.requires_grad = True
            print("unfroze final rpn conv parameters")

    def build_context(self, cfg):
        in_features = self.backbone.output_shape()
        in_levels = cfg.MODEL.ROI_HEADS.IN_FEATURES
        channels = [in_features[f] for f in in_levels]
        in_channels = [s.channels for s in channels]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]
        self.cont_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.cont_avg_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.cont_cls = nn.Linear(in_channels, self.num_classes, bias=False)
        self.cont_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.cont_in_feature = cfg.MODEL.ROI_HEADS.IN_FEATURES[-1]
        nn.init.normal_(self.cont_conv.weight, std=0.01)
        nn.init.normal_(self.cont_cls.weight, std=0.01)
        nn.init.constant_(self.cont_conv.bias, 0)

        self.image_only = cfg.MODEL.IMAGES_ONLY

    def context_forward_train(self, features, gt_instances):
        feat = features[self.cont_in_feature]
        pres_classes = [inst.gt_classes.unique() for inst in gt_instances]

        mask = torch.zeros(len(feat), self.num_classes).to(feat)
        for m, g in zip(mask, pres_classes):
            m[g] = 1.0

        x = self.cont_conv(feat)
        x = self.cont_avg_pool(feat)
        x = x.flatten(start_dim=1)
        x = F.relu(x)
        x = self.cont_cls(x)
        loss = self.cont_loss(x, mask)
        return {'loss_context': loss}

    def context_forward_test(self, features):
        feat = features[self.cont_in_feature]
        x = self.cont_conv(feat)
        x = self.cont_avg_pool(feat)
        x = x.flatten(start_dim=1)
        x = F.relu(x)
        x = self.cont_cls(x)
        x = x.sigmoid()
        filter_inds = (x > -0.01).nonzero(as_tuple=False)
        if filter_inds.size(0):
            return (filter_inds[:, -1].tolist(),
                    x[filter_inds[:, 0], filter_inds[:, 1]].tolist())
        else:
            return [], []

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        # print(images.tensor.size())
        # print(len(features), type(features))
        # for k, v in features.items():
        #     print(k, v.size())

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        context_losses = self.context_forward_train(features, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(context_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        img_cls, img_sc = self.context_forward_test(features)

        if not self.image_only:
            if detected_instances is None:
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [
                        x["proposals"].to(self.device) for x in batched_inputs
                    ]

                results, _ = self.roi_heads(images, features, proposals, None)
            else:
                detected_instances = [
                    x.to(self.device) for x in detected_instances
                ]
                results = self.roi_heads.forward_with_given_boxes(
                    features, detected_instances
                )
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r,
                                              "image_classes": img_cls,
                                              "image_scores": img_sc})
                return processed_results
            else:
                return [{'instances': results[0],
                         'image_classes': torch.LongTensor(img_cls),
                         'image_scores': torch.Tensor(img_sc)}]
        else:
            processed_results = []
            for _ in range(len(batched_inputs)):
                processed_results.append({"image_classes": img_cls,
                                          "image_scores": img_sc})
            return processed_results

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
