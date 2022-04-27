from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY
from detectron2.config import configurable
from torch import nn
from torchvision.ops import box_iou
import torch
import numpy as np
from detectron2.structures import BoxMode, Boxes, Instances


@PROPOSAL_GENERATOR_REGISTRY.register()
class RBG(nn.Module):

    @configurable
    def __init__(
         self,
         *,
         alpha: float,
         beta: float,
         t: float,
         batch_size_per_image: int,
         positive_fraction: float,
         # device: str,
    ):

        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.t = t
        self.positive_num_per_image = int(batch_size_per_image * positive_fraction)
        # self.tx = torch.ones((1000000,)).uniform_(
        #         -self.alpha, self.alpha)
        # self.ty = torch.ones((1000000,)).uniform_(
        #         -self.alpha, self.alpha)
        # self.tw = torch.ones((1000000,)).uniform_(
        #         np.log(1-self.beta), np.log(1+self.beta))
        # self.th = torch.ones((1000000,)).uniform_(
        #         np.log(1-self.beta), np.log(1+self.beta))

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            "alpha": cfg.MODEL.RBG.ALPHA,
            "beta": cfg.MODEL.RBG.BETA,
            "t": cfg.MODEL.RBG.T,
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            # "device": cfg.MODEL.DEVICE,
        }

        return ret

    def forward(self, proposals, targets):
        if self.training:
            filtered_proposals = self.filter_proposals(proposals, targets)
            new_proposals = self.generate_proposals(targets)

            prop_list = [
                Instances.cat([f, n])
                for f, n in zip(filtered_proposals, new_proposals)
                ]
            return prop_list, {}
        else:
            return proposals, {}
            filtered_proposals = self.filter_proposals(proposals, targets)
            new_proposals = self.generate_proposals(targets)

            prop_list = [
                Instances.cat([f, n])
                for f, n in zip(filtered_proposals, new_proposals)
                ]
            return prop_list, {}

    def filter_proposals(self, proposals, targets):
        new_proposals = []
        for i, (tar, prop) in enumerate(zip(targets, proposals)):

            bbox_gt = tar.gt_boxes.tensor
            bbox_proposals = prop.proposal_boxes.tensor
            if len(bbox_proposals) and len(bbox_gt):
                ious = box_iou(bbox_gt, bbox_proposals)  # gt x proposals
                keep = ious.max(dim=0)[0] > self.t
                new_proposals.append(prop[keep])
            else:
                empty_proposals = Instances(
                    tar.image_size,
                    proposal_boxes=Boxes(torch.zeros((0, 4)).to(bbox_gt)),
                    objectness_logits=torch.zeros((0), device=bbox_gt.device))
                new_proposals.append(empty_proposals)
        return new_proposals

    def generate_proposals(self, targets):
        new_instances_all = []

        for tar in targets:
            if len(tar):
                N = (2 * self.positive_num_per_image) // len(tar)
                H, W = tar.image_size
                # num = N * len(tar)
                # selected_indices = torch.LongTensor(
                #     np.random.choice(1000000, (num, 4), replace=False))
                # tx = self.tx[selected_indices[:, 0]].view(N, len(tar)).to(tar.gt_boxes.tensor.device)
                # ty = self.ty[selected_indices[:, 1]].view(N, len(tar)).to(tar.gt_boxes.tensor.device)
                # tw = self.tw[selected_indices[:, 2]].view(N, len(tar)).to(tar.gt_boxes.tensor.device)
                # th = self.th[selected_indices[:, 3]].view(N, len(tar)).to(tar.gt_boxes.tensor.device)
                tx = torch.ones((N, len(tar))).to(
                    tar.gt_boxes.tensor.device).uniform_(
                        -self.alpha, self.alpha)
                ty = torch.ones((N, len(tar))).to(
                    tar.gt_boxes.tensor.device).uniform_(
                        -self.alpha, self.alpha)
                tw = torch.ones((N, len(tar))).to(
                    tar.gt_boxes.tensor.device).uniform_(
                        np.log(1-self.beta), np.log(1+self.beta))
                th = torch.ones((N, len(tar))).to(
                    tar.gt_boxes.tensor.device).uniform_(
                        np.log(1-self.beta), np.log(1+self.beta))
                gt_boxes = BoxMode.convert(
                    tar.gt_boxes.tensor,
                    from_mode=BoxMode.XYXY_ABS,
                    to_mode=BoxMode.XYWH_ABS)
                new_x = (gt_boxes[:, 0].view(-1, 1)
                         + gt_boxes[:, 2].view(-1, 1)*tx.t())
                new_y = (gt_boxes[:, 1].view(-1, 1)
                         + gt_boxes[:, 3].view(-1, 1)*ty.t())
                new_w = gt_boxes[:, 2].view(-1, 1) * torch.exp(tw.t())
                new_h = gt_boxes[:, 3].view(-1, 1) * torch.exp(th.t())
                new_boxes_xywh = torch.stack(
                    [new_x, new_y, new_w, new_h]).permute(1, 2, 0)
                new_boxes_xyxy = Boxes(BoxMode.convert(
                    new_boxes_xywh.view(-1, 4),
                    from_mode=BoxMode.XYWH_ABS,
                    to_mode=BoxMode.XYXY_ABS))
                new_boxes_xyxy.clip(tar.image_size)
                ious = box_iou(tar.gt_boxes.tensor, new_boxes_xyxy.tensor)
                try:
                    keep = (ious.max(dim=0)[0] > self.t)
                except RuntimeError:
                    dev = tar.gt_boxes.tensor.device
                    new_instances = Instances(
                        tar.image_size,
                        proposal_boxes=Boxes(torch.zeros(len(tar), 4).to(dev)),
                        objectness_logits=torch.zeros(len(tar)).to(dev))
                    new_instances_all.append(new_instances)
                    continue

                new_boxes_xyxy = new_boxes_xyxy[keep]
                new_instances = Instances(
                    tar.image_size,
                    proposal_boxes=new_boxes_xyxy,
                    objectness_logits=torch.ones(len(new_boxes_xyxy)).to(th))
                new_instances_all.append(new_instances)
            else:
                dev = tar.gt_boxes.tensor.device
                new_instances = Instances(
                    tar.image_size,
                    proposal_boxes=Boxes(torch.zeros(len(tar), 4).to(dev)),
                    objectness_logits=torch.zeros(len(tar)).to(dev))
                new_instances_all.append(new_instances)

        return new_instances_all
