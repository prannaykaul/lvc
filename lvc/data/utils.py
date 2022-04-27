import logging
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from tabulate import tabulate
from termcolor import colored
import numpy as np
import itertools
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools._mask import iou
from detectron2.structures.boxes import BoxMode
import contextlib
import io
import os
import copy
from detectron2.utils.file_io import PathManager
import pickle
import torch
import random
from collections import defaultdict
from .builtin_meta import _get_builtin_metadata, COCO_UNSEEN_IDS
from tqdm.auto import tqdm
from detectron2.structures import BoxMode, Boxes
from torchvision.ops import box_iou
from .meta_coco import load_coco_json


def remove_ignore_overlap(dataset_dicts_new):
    for i, data_dict in tqdm(enumerate(dataset_dicts_new), total=len(dataset_dicts_new)):
        annos_ig = [ann for ann in data_dict['annotations']
                    if ann.get('ignore_qe', 0)]
        if len(annos_ig):
            annos_real = [ann for ann in data_dict['annotations']
                          if not ann.get('ignore_qe', 0)]
            bbox_raw_ig = np.array(
                [ann['bbox'] for ann in annos_ig]
                )
            bbox_xyxy_ig = BoxMode.convert(
                bbox_raw_ig,
                from_mode=annos_ig[0]['bbox_mode'],
                to_mode=BoxMode.XYWH_ABS)
            bbox_raw_real = np.array(
                [ann['bbox'] for ann in annos_real]
                )
            bbox_xyxy_real = BoxMode.convert(
                bbox_raw_real,
                from_mode=annos_real[0]['bbox_mode'],
                to_mode=BoxMode.XYWH_ABS)
            ious = iou(
                bbox_xyxy_real,
                bbox_xyxy_ig,
                [0 for _ in range(len(bbox_xyxy_ig))])
            keep_ig = (ious.max(axis=0) < 0.5).nonzero()[0].tolist()
            annos_ig_new = [annos_ig[i] for i in keep_ig]
            comb_anns = annos_ig_new + annos_real
            data_dict["annotations"] = comb_anns
    return dataset_dicts_new


def filter_proposal_boxes(
     dataset_dicts,
     area_rng=(0., 1.e10),
     rel_area_rng=(0., 1.),
     x_rng=(0., 1.e10),
     y_rng=(0., 1.e10),
     topk=int(1e10)):

    for data in dataset_dicts:
        boxes = data['proposal_boxes']
        logits = data['proposal_objectness_logits']
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        rel_areas = areas / float(data["height"]*data["width"])
        ws = (boxes[:, 2] - boxes[:, 0])
        hs = (boxes[:, 3] - boxes[:, 1])
        areas_val = (areas <= area_rng[1]) * (areas > area_rng[0])
        rel_areas_val = (
            (rel_areas <= rel_area_rng[1]) * (rel_areas > rel_area_rng[0])
            )
        x_val = (ws <= x_rng[1]) * (ws > x_rng[0])
        y_val = (hs <= y_rng[1]) * (hs > y_rng[0])
        keep = areas_val * x_val * y_val * rel_areas_val
        data['proposal_boxes'] = boxes[:topk][keep[:topk]]
        data['proposal_objectness_logits'] = logits[:topk][keep[:topk]]
    return dataset_dicts


def filter_annotations(
     dataset_dicts,
     area_rng=(0., 1.e10),
     rel_area_rng=(0., 1.),
     x_rng=(0., 1.e10),
     y_rng=(0., 1.e10),
     check_longest_side_only=True):

    for data in dataset_dicts:
        if not len(data['annotations']):
            continue
        annos = data["annotations"]
        bbox_raw = torch.Tensor(
            [ann['bbox'] for ann in annos]
            )
        bbox_xyxy = BoxMode.convert(
            bbox_raw,
            from_mode=annos[0]['bbox_mode'],
            to_mode=BoxMode.XYXY_ABS)
        bbox_xyxy = Boxes(bbox_xyxy)
        areas = bbox_xyxy.area()
        rel_areas = areas.div(data["width"]*data["height"])
        areas_val = (areas <= area_rng[1]) * (areas > area_rng[0])
        rel_areas_val = (
            (rel_areas <= rel_area_rng[1]) * (rel_areas > rel_area_rng[0])
            )
        if check_longest_side_only:
            assert (x_rng[0] == y_rng[0]) and (x_rng[1] == y_rng[1])
            bbox_max_dim = torch.max(bbox_raw[:, 2], bbox_raw[:, 3])
            dim_val = (bbox_max_dim > x_rng[0]) * (bbox_max_dim <= x_rng[1])
        else:
            x_val = (bbox_raw[:, 2] > x_rng[0]) * (bbox_raw[:, 2] <= x_rng[1])
            y_val = (bbox_raw[:, 3] > y_rng[0]) * (bbox_raw[:, 3] <= y_rng[1])
            dim_val = x_val * y_val
        keep = (areas_val * rel_areas_val * dim_val).nonzero(as_tuple=True)[0]
        data["annotations"] = [ann for i, ann in enumerate(annos) if i in keep]
    return dataset_dicts


def remove_overlap_proposals(dataset_dicts, iou_thresh):
    for data in dataset_dicts:
        if not (len(data["annotations"]) and len(data["proposal_boxes"])):
            continue
        annos = data["annotations"]
        bbox_proposals = torch.from_numpy(data["proposal_boxes"]).float()
        bbox_gt_raw = torch.Tensor(
            [ann['bbox'] for ann in annos]
            )
        bbox_xyxy = BoxMode.convert(
            bbox_gt_raw,
            from_mode=annos[0]['bbox_mode'],
            to_mode=BoxMode.XYXY_ABS)
        bbox_ious = box_iou(bbox_xyxy, bbox_proposals)
        keep = (bbox_ious.max(dim=0)[0] < iou_thresh).numpy()
        data["proposal_boxes"] = data["proposal_boxes"][keep]
        data["proposal_objectness_logits"] = data["proposal_objectness_logits"][keep]

    return dataset_dicts


def register_results(cfg, nn_dset=False):
    for idx, results_file in enumerate(cfg.DATASETS.DT_PATH):
        print(results_file)
        if not os.path.isfile(results_file):
            continue
        tv_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        image_root = tv_metadata.image_root
    #     _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    #     image_root = os.path.join(_root, 'coco/trainval2014/')
        dataset = "coco_fewshot"
        metadata = _get_builtin_metadata(dataset)
        print(metadata)
        print(results_file)
        if results_file in DatasetCatalog._REGISTERED.keys():
            print("Removed old file: {}".format(results_file))
            DatasetCatalog.remove(results_file)
        # if detections_path in MetadataCatalog._REGISTERED.keys():
        #     DatasetCatalog.remove(detections_path)
        register_individual(
            "query_expand{}".format(str(idx).zfill(3)),
            metadata,
            image_root,
            results_file)
    K = idx
    for idx, results_file in enumerate(cfg.QUERY_EXPAND.NN_DSET, start=K+1):
        print(results_file)
        if not os.path.isfile(results_file):
            continue
        tv_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        image_root = tv_metadata.image_root
    #     _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    #     image_root = os.path.join(_root, 'coco/trainval2014/')
        dataset = "coco_fewshot"
        metadata = _get_builtin_metadata(dataset)
        print(metadata)
        print(results_file)
        if results_file in DatasetCatalog._REGISTERED.keys():
            print("Removed old file: {}".format(results_file))
            DatasetCatalog.remove(results_file)
        # if detections_path in MetadataCatalog._REGISTERED.keys():
        #     DatasetCatalog.remove(detections_path)
        register_individual(
            "query_expand{}".format(str(idx).zfill(3)),
            metadata,
            image_root,
            results_file)
    return


def register_individual(name, metadata, imgdir, annofile):
    if 'unlabeled' in os.path.basename(annofile):
        _root = os.getenv("DETECTRON2_DATASETS", "datasets")
        imgdir = os.path.join(_root, 'coco/unlabeled2017/')
    DatasetCatalog.register(
        annofile,
        lambda: load_coco_json(annofile, imgdir, metadata, name,
                               extra_annotation_keys=['id', 'score', 'ignore_qe', 'ignore_reg']),
    )

    MetadataCatalog.get(annofile).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
        dirname="datasets/coco",
        **metadata,
    )
    print(MetadataCatalog.get(annofile))


# def register_results(cfg):
#     func = load_coco_json
#
#     for results_file in cfg.DATASETS.DT_PATH:
#         tv_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
#         image_root = tv_metadata.image_root
#     #     _root = os.getenv("DETECTRON2_DATASETS", "datasets")
#     #     image_root = os.path.join(_root, 'coco/trainval2014/')
#         dataset = "coco_fewshot"
#         metadata = _get_builtin_metadata(dataset)
#         print(results_file)
#         if results_file in DatasetCatalog._REGISTERED.keys():
#             DatasetCatalog.remove(results_file)
#         # if detections_path in MetadataCatalog._REGISTERED.keys():
#         #     DatasetCatalog.remove(detections_path)
#
#         DatasetCatalog.register(results_file, lambda: func(
#             results_file, image_root, metadata, "query_expand",
#             extra_annotation_keys=['score', 'id']))
#         MetadataCatalog.get(results_file).set(
#                 json_file=results_file, image_root=image_root,
#                 evaluator_type="coco", **metadata
#             )


def register_results_voc(cfg):
    gt_dsets = cfg.DATASETS.TRAIN
    if len(gt_dsets) != cfg.DATASETS.DT_PATH:
        gt_dsets = [gt_dsets[0]]*len(cfg.DATASETS.DT_PATH)
    for idx, (results_file, gt_dataset) in enumerate(zip(cfg.DATASETS.DT_PATH, gt_dsets)):
        func = copy.deepcopy(load_coco_json)
        tv_metadata = MetadataCatalog.get(gt_dataset)
        image_root = tv_metadata.image_root

        dataset = "pascal_voc_fewshot"
        metadata = _get_builtin_metadata(dataset)
        print(results_file)
        if results_file in DatasetCatalog._REGISTERED.keys():
            DatasetCatalog.remove(results_file)
        root = os.getenv("DETECTRON2_DATASETS", "datasets")
        register_individual_voc(
            "query_expand{}".format(str(idx).zfill(3)),
            tv_metadata,
            root,
            results_file)


def register_individual_voc(name, metadata, imgdir, annofile):
    DatasetCatalog.register(
        annofile,
        lambda: load_coco_json(annofile, imgdir, metadata, name,
                               extra_annotation_keys=['id', 'score', 'max_score']),
    )

    MetadataCatalog.get(annofile).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
    )
    MetadataCatalog.get(annofile).set(
        thing_classes=metadata.thing_classes,
        novel_classes=metadata.novel_classes,
        base_classes=metadata.base_classes,
    )


def unseen_sample(dataset_dicts):
    rng = random.Random(20000)
    cat2imgs = defaultdict(list)
    for i, im_dict in enumerate(dataset_dicts):
        anns = im_dict['annotations']
        cat_ids, cat_counts = np.unique([a['category_id'] for a in anns], return_counts=True)
        # img_id = im_dict['image_id']
        for cat_id, count in zip(cat_ids, cat_counts):
            cat2imgs[cat_id].append([i, count])
    samp_num = max(len(v) for k, v in cat2imgs.items() if k in COCO_UNSEEN_IDS)
    keep_ids = []
    for k, v in cat2imgs.items():
        if len(v) <= samp_num:
            keep_ids.extend([v1[0] for v1 in v])
        else:
            inds = rng.sample(v, len(v))
            keep_id = []
            tot = 0
            for iid, count in inds:
                tot += count
                if tot > samp_num:
                    break
                keep_id.append(iid)
            keep_ids.extend(keep_id)
    keep_ids = list(set(keep_ids))
    dataset_dicts_samp = [dataset_dicts[i] for i in keep_ids]
    return dataset_dicts_samp


def save_filtered_dataset(cfg, cfg_lin, results_ids, scores, classes):
    pickle_file = PathManager.get_local_path(cfg.QUERY_EXPAND.SEED_MODEL)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO()
        with open(pickle_file, 'rb') as fp:
            dataset_dt = pickle.load(fp)
        coco_api.dataset = dataset_dt
        coco_api.createIndex()
    anns = coco_api.loadAnns(results_ids)
    cats = coco_api.loadCats(coco_api.getCatIds())
    cats = {k: v['id'] for k, v in enumerate(cats)}
    # print(cats)
    pres_imgs = sorted(set([ann['image_id'] for ann in anns]))
    imgs = coco_api.loadImgs(pres_imgs)
    # print(len(anns), len(scores), len(classes))
    for ann, s, cl in zip(anns, scores, classes):
        # print(ann, s, cl)
        ann.update({'category_id': cats[cl], 'score': s})
        # print(ann)
    # print(anns[0])
    dataset_dt['annotations'] = anns
    dataset_dt['images'] = imgs
    save_filename = os.path.basename(cfg.QUERY_EXPAND.SEED_MODEL)
    save_filename = save_filename.replace('.pkl', '_qe.pkl')
    save_filename = os.path.join(
        os.path.dirname(cfg.QUERY_EXPAND.SEED_MODEL),
        save_filename)
    print(save_filename)
    with open(save_filename, 'wb') as fp:
        pickle.dump(dataset_dt, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return


def print_instances_class_histogram_force(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int64)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = [x["category_id"] for x in annos if not x.get("iscrowd", 0)]
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    logging.getLogger(__name__).log(
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan"))


def filter_image_annotations(
        dataset_dicts, dataset_name, unseen_class_names, test=False
        ):
    orig_num = len(dataset_dicts)
    metadata = MetadataCatalog.get(dataset_name)
    class_names = metadata.thing_classes
    assert all([cls_name in class_names for cls_name in unseen_class_names]), \
        "unseen class does not exist"
    unseen_class_ids = [class_names.index(cls_name)
                        for cls_name in unseen_class_names]
    for annos in dataset_dicts:
        annos_new = [
            ann for ann in annos['annotations']
            if ann['category_id'] not in unseen_class_ids]
        annos['annotations'] = annos_new
    if not test:
        dataset_dicts_new = [
            annos for annos in dataset_dicts
            if len(annos['annotations']) != 0]
    else:
        dataset_dicts_new = dataset_dicts
    new_num = len(dataset_dicts_new)
    logger = logging.getLogger(__name__)
    logger.info(
        "After filtering out unseen annotations {} images removed".format(
            orig_num-new_num))
    return dataset_dicts_new


def combine_datasets(list_of_dataset_dicts):
    sep_img2dataset_dicts = []
    for i, d in enumerate(list_of_dataset_dicts):
        img2dataset_dicts = {}
        for ann in d:
            if ann['image_id'] in img2dataset_dicts:
                img2dataset_dicts[ann['image_id']]['annotations'].extend(ann['annotations'])
            else:
                img2dataset_dicts[ann['image_id']] = ann
        sep_img2dataset_dicts.append(img2dataset_dicts)

    dict_lens = [len(v) for v in sep_img2dataset_dicts]
    base_dict = sep_img2dataset_dicts.pop(dict_lens.index(max(dict_lens)))

    for rem_dict in sep_img2dataset_dicts:
        for img_id, anns in rem_dict.items():
            if img_id in base_dict:
                base_dict[img_id]['annotations'].extend(anns['annotations'])
            else:
                base_dict[img_id] = anns
    dataset_dicts_new = list(base_dict.values())

    return dataset_dicts_new


def get_crops(img, proposal_boxes):
    crops = []
    for bb in proposal_boxes:
        # print(img.size())
        if (bb[0] < bb[2]).item() and (bb[1] < bb[3]).item():
            crop = img[:, bb[1]:bb[3], bb[0]:bb[2]]
            crops.append(crop)
    return crops


def resize_shortest_pk(img, l, max_size):
    h, w = img.size()[-2:]
    scale = l * 1.0 / min(h, w)
    if h < w:
        newh, neww = l, scale * w
    else:
        newh, neww = scale * h, l
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    # h, w, newh, neww
    assert img.size()[-2:] == (h, w)
    assert len(img.size()) <= 4
#     print(img.size())
    img = F.interpolate(img.unsqueeze(0).float(), (newh, neww),
                        mode='bilinear', align_corners=False).squeeze(0)
    return img


def convert_crops(crops):
    out_crops = []
    for crop in crops:
        # print(crop.size())
        crop = resize_shortest_pk(crop, 224, 224)
        # print(crop.size())
        # _ = crop.apply_augmentations(tfm)
        # crop = crop.image
        # crop = torch.as_tensor(np.ascontiguousarray(crop.transpose(2, 0, 1)))
        out_crops.append(crop)
    return out_crops


def get_padding(H, W):
    max_d = max(H, W)

    imsize = (H, W)
    h_padding = (max_d - imsize[1]) / 2
    v_padding = (max_d - imsize[0]) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5

    padding = (int(l_pad), int(r_pad), int(t_pad), int(b_pad))

    return padding


def get_crops_qe(imgs, gt_boxes, operation='pad'):
    crops = []
    for i in range(len(gt_boxes)):
        box = gt_boxes[i]
        x1, y1, x2, y2 = box.gt_boxes.tensor.long().squeeze(0).tolist()
        if operation == 'pad':
            l_p, r_p, t_p, b_p = get_padding(y2-y1+1, x2-x1+1)
            crop = imgs[:, :, y1:y2+1, x1:x2+1]
            crop = F.pad(crop, (l_p, r_p, t_p, b_p))
        elif operation == 'context':
            l_p, r_p, t_p, b_p = get_padding(y2-y1+1, x2-x1+1)
            y1n, x1n = max(0, y1-t_p), max(0, x1-l_p)
            y2n, x2n = min(imgs.size(-2), y2+b_p), min(imgs.size(-1), x2+r_p)
            l_p, r_p, t_p, b_p = get_padding(y2n-y1n+1, x2n-x1n+1)
            crop = imgs[:, :, y1n:y2n+1, x1n:x2n+1]
            crop = F.pad(crop, (l_p, r_p, t_p, b_p))
        crop = F.interpolate(crop, (224, 224), mode='nearest')
        crops.append(crop)
    return torch.cat(crops)


def filter_out_invalid_anns(dataset_dicts, wh=10):
    for j, img_annos in enumerate(dataset_dicts):
        annos = img_annos['annotations']
        val_annos = [
            ann for ann in annos
            if (ann['bbox'][-2] > wh
                and ann['bbox'][-1] > wh
                and ann['iscrowd'] == 0)]
        img_annos['annotations'] = val_annos
        dataset_dicts[j] = img_annos
    return dataset_dicts


def iou_check_gt(dt_id, coco_dt, coco_gt, gt_cids, thresh=0.5):
    ann = coco_dt.loadAnns(dt_id)[0]
    iid = ann['image_id']
    anns_gt = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=iid, catIds=gt_cids))
    if len(anns_gt):
        bbox_dt = BoxMode.convert(
            np.array([ann['bbox']]),
            from_mode=BoxMode.XYWH_ABS,
            to_mode=BoxMode.XYWH_ABS)
        bbox_gt = BoxMode.convert(
            np.array([a['bbox'] for a in anns_gt]),
            from_mode=BoxMode.XYWH_ABS,
            to_mode=BoxMode.XYWH_ABS)
        ious = iou(
            bbox_dt, bbox_gt, [0 for _ in range(len(bbox_gt))]).squeeze()
        return 1 if ious.max() > thresh else 0
    else:
        return 0


def iou_check(dt_id, coco_dt, coco_gt, thresh=0.5):
    ann = coco_dt.loadAnns(dt_id)[0]
    if 'ignore_qe' in ann:
        if ann['ignore_qe']:
            return -1
    iid, cid = ann['image_id'], ann['category_id']
    anns_gt = coco_gt.loadAnns(
        coco_gt.getAnnIds(imgIds=iid, catIds=cid, iscrowd=False))
    if len(anns_gt):
        bbox_dt = BoxMode.convert(
            np.array([ann['bbox']]),
            from_mode=BoxMode.XYWH_ABS,
            to_mode=BoxMode.XYXY_ABS)
        bbox_gt = BoxMode.convert(
            np.array([a['bbox'] for a in anns_gt]),
            from_mode=BoxMode.XYWH_ABS,
            to_mode=BoxMode.XYXY_ABS)
        ious = iou(
            bbox_dt, bbox_gt, [0 for _ in range(len(bbox_gt))]).squeeze()
        return 1 if ious.max() > thresh else 0
    else:
        return 0


def print_results_table(precisions, coco_gt):
    table = []
    cat_ids = sorted(precisions.keys())
    for cid in cat_ids:
        val = np.array(precisions[cid])
        val = val[val != -1]
        name = coco_gt.cats[cid]['name']
        table.append(
            [name, len(val), np.array(val).mean()])
    print(tabulate(
        table, headers=['Category', 'NUM', 'Precision'], tablefmt='orgtbl'))
    return table


def print_precision_per_class(filename_dt, filename_gt, iou_thresh=0.5):
    with contextlib.redirect_stdout(io.StringIO()):
        if isinstance(filename_gt, str):
            coco_gt = COCO(filename_gt)
        else:
            coco_gt = filename_gt
        if isinstance(filename_dt, str):
            print(filename_dt)
            coco_dt = COCO(filename_dt)
        else:
            coco_dt = filename_dt
    # print(filename_dt)
    precisions = defaultdict(list)
    dt_ids = coco_dt.getAnnIds()
    for i, dt_id in tqdm(enumerate(dt_ids), total=len(dt_ids)):
        cid = coco_dt.loadAnns(dt_id)[0]['category_id']
        tp = iou_check(dt_id, coco_dt, coco_gt, thresh=iou_thresh)
        precisions[cid].append(tp)
    table = print_results_table(precisions, coco_gt)
    return table
