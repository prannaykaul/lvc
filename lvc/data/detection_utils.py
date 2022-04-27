from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
import torch

__all__ = ["annotations_to_instances_ignore"]


def annotations_to_instances_ignore(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes
    ignores = [obj.get("ignore_qe", 0) for obj in annos]
    ignores = torch.tensor(ignores, dtype=torch.int64)
    target.gt_ignores = ignores
    ids = [obj.get("id", -1) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.ids = ids
    # ignores_reg = [obj.get("ignore_reg", 0) for obj in annos]
    # ignores_reg = torch.tensor(ignores_reg, dtype=torch.int64)
    # target.gt_ignores_reg = ignores_reg

    return target
