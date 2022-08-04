from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C

# adding additional default values built on top of the default values in detectron2

_CC = _C

_CC.DEBUG = False

_CC.INPUT.COLOR_JITTER = False
_CC.INPUT.MOSAIC = 0.0
_CC.INPUT.MOSAIC49SPLIT = 0.0


_CC.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.MODEL.MOBILENET = False
_CC.MODEL.FCOS = CN()

# This is the number of foreground classes.
_CC.MODEL.FCOS.NUM_CLASSES = 80
_CC.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_CC.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_CC.MODEL.FCOS.PRIOR_PROB = 0.01
_CC.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_CC.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_CC.MODEL.FCOS.NMS_TH = 0.6
_CC.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_CC.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_CC.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_CC.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_CC.MODEL.FCOS.TOP_LEVELS = 2
_CC.MODEL.FCOS.NORM = "GN"  # Support GN or none
_CC.MODEL.FCOS.USE_SCALE = True

# The options for the quality of box prediction
# It can be "ctrness" (as described in FCOS paper) or "iou"
# Using "iou" here generally has ~0.4 better AP on COCO
# Note that for compatibility, we still use the term "ctrness" in the code
_C.MODEL.FCOS.BOX_QUALITY = "ctrness"

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_CC.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_CC.MODEL.FCOS.LOSS_ALPHA = 0.25
_CC.MODEL.FCOS.LOSS_GAMMA = 2.0

# The normalizer of the classification loss
# The normalizer can be "fg" (normalized by the number of the foreground samples),
# "moving_fg" (normalized by the MOVING number of the foreground samples),
# or "all" (normalized by the number of all samples)
_C.MODEL.FCOS.LOSS_NORMALIZER_CLS = "fg"
_C.MODEL.FCOS.LOSS_WEIGHT_CLS = 1.0

_CC.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_CC.MODEL.FCOS.USE_RELU = True
_CC.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_CC.MODEL.FCOS.NUM_CLS_CONVS = 4
_CC.MODEL.FCOS.NUM_BOX_CONVS = 4
_CC.MODEL.FCOS.NUM_SHARE_CONVS = 0
_CC.MODEL.FCOS.CENTER_SAMPLE = True
_CC.MODEL.FCOS.POS_RADIUS = 1.5
_CC.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
_CC.MODEL.FCOS.YIELD_PROPOSAL = False
_CC.MODEL.FCOS.DROPOUT = 0.0

_CC.MODEL.FCOS.CS_CLS = False

_CC.MODEL.FCOS.FREEZE = False
_CC.MODEL.FCOS.UNFREEZE_REG = False
_CC.MODEL.FCOS.UNFREEZE_TOWERS = False

_CC.MODEL.FCOS.REG_ONLY = False

_CC.MODEL.UBBR = CN()
_CC.MODEL.UBBR.LAMBDA = 0.6
_CC.MODEL.UBBR.CASCADE_STEPS = 3

# FREEZE Parameters
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.BACKBONE.FREEZE_BOTTOM_UP = False
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False
_CC.MODEL.ROI_HEADS.REG_OFF = False
_CC.MODEL.ROI_HEADS.FREEZE_BBOX_PRED = False
_CC.MODEL.ROI_HEADS.IGNORE_REG = False
_CC.MODEL.PROPOSAL_GENERATOR.UNFREEZE_FIN = False
_CC.MODEL.IMAGES_ONLY = False

# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0

_CC.MODEL.ROI_BOX_HEAD.DROPOUT = 0.0

_CC.MODEL.RBG = CN()
_CC.MODEL.RBG.ALPHA = 0.35
_CC.MODEL.RBG.BETA = 0.5
_CC.MODEL.RBG.T = 0.3

_CC.MODEL.RPNCOMP = CN()
_CC.MODEL.RPNCOMP.POOLER = ''

_CC.MODEL.SWIN = CN()
_CC.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
_CC.MODEL.SWIN.PATCH_SIZE = 4
_CC.MODEL.SWIN.SWIN_SIZE = 'tiny'
_CC.MODEL.SWIN.WINDOW_SIZE = 7
_CC.MODEL.SWIN.MLP_RATIO = 4.0
_CC.MODEL.SWIN.QKV_BIAS = True
_CC.MODEL.SWIN.QK_SCALE = None
_CC.MODEL.SWIN.DROP_RATE = 0.0
_CC.MODEL.SWIN.ATTN_DROP_RATE = 0.0
_CC.MODEL.SWIN.DROP_PATH_RATE = 0.2
_CC.MODEL.SWIN.NORM_LAYER = 'LayerNorm'
_CC.MODEL.SWIN.APE = False
_CC.MODEL.SWIN.PATCH_NORM = True
_CC.MODEL.SWIN.OUT_INDICES = (0, 1, 2, 3)
_CC.MODEL.SWIN.FROZEN_STAGES = -1

# Backward Compatible options.
_CC.MUTE_HEADER = True

_CC.QUERY_EXPAND = CN()
_CC.QUERY_EXPAND.GET_CROPS = False
_CC.QUERY_EXPAND.ENABLED = False
_CC.QUERY_EXPAND.NN_MODEL = ''
_CC.QUERY_EXPAND.NN_DSET = ()
_CC.QUERY_EXPAND.KNN = 10
_CC.QUERY_EXPAND.COSINE_SIM = True
# _CC.QUERY_EXPAND.DIFF = CN()
# _CC.QUERY_EXPAND.DIFF.VALUE = 0.2
# _CC.QUERY_EXPAND.DIFF.ENABLED = True
# _CC.QUERY_EXPAND.SCORE = CN()
# _CC.QUERY_EXPAND.SCORE.VALUE = 0.65
# _CC.QUERY_EXPAND.SCORE.ENABLED = True
# _CC.QUERY_EXPAND.AREA = CN()
# _CC.QUERY_EXPAND.AREA.VALUE = 0.0
# _CC.QUERY_EXPAND.AREA.ENABLED = False
# _CC.QUERY_EXPAND.SEED_MODEL = ''
# _CC.QUERY_EXPAND.IGNORE = False
# _CC.QUERY_EXPAND.IGNORE_REG = False
# _CC.QUERY_EXPAND.METHOD = '81way'
# _CC.QUERY_EXPAND.START_NUM = 1000
# _CC.QUERY_EXPAND.CONFIG_FILE = '../osssod/OSSSOD/configs/LIN_COCO_81/lin_seed0.yaml'
# _CC.QUERY_EXPAND.MODEL_WEIGHTS_160 = ''

_CC.TEMPLATE = CN()
_CC.TEMPLATE.SIZE = 224
_CC.TEMPLATE.FROZEN = ["fc", "layer4"]
_CC.TEMPLATE.MLP = False
_CC.TEMPLATE.WEIGHTS = "checkpoints/swav_800ep_pretrain.pth"
_CC.TEMPLATE.ARCH = "resnet50"
_CC.TEMPLATE.PIXEL_MEAN = [123.675, 116.280, 103.530]
_CC.TEMPLATE.PIXEL_STD = [58.395, 57.120, 57.375]
_CC.TEMPLATE.CLASSIFIER = True

_CC.DATASETS.FINETUNE_SEED = 0
_CC.DATASETS.FINETUNE_SHOTS = 30
_CC.DATASETS.UNSEEN_CLASSES = [
    "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "dining table", "dog", "horse", "motorcycle",
    "person", "potted plant", "sheep", "couch", "train", "tv"]
_CC.DATASETS.SEEN_CLASSES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "bed", "toilet", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"]

_CC.DATASETS.UNSEEN_IDS = [
    0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62]

_CC.DATASETS.SEEN_IDS = [
    7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 59, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 78, 79]

_CC.DATASETS.ALL_IDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
    73, 74, 75, 76, 77, 78, 79]

_CC.DATASETS.SPLIT_IDS = [
    0, 1, 2, 3, 4, 5, 6, 0, 7, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 14, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 15, 16, 17, 41, 18, 42,
    19, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]


_CC.DATASETS.FS_TRAIN = ()
_CC.DATASETS.SUBSET = False
_CC.DATASETS.DT_PATH = ()

_CC.DATALOADER.PROPOSALS = CN()
_CC.DATALOADER.PROPOSALS.AREA_RNG = [0.0, 1.e10]
_CC.DATALOADER.PROPOSALS.REL_AREA_RNG = [0.0, 2.0]
_CC.DATALOADER.PROPOSALS.X_RNG = [0.0, 1.e10]
_CC.DATALOADER.PROPOSALS.Y_RNG = [0.0, 1.e10]
_CC.DATALOADER.PROPOSALS.TOPK = 1000
_CC.DATALOADER.PROPOSALS.IOU_THRESH = 0.3

_CC.DATALOADER.SHOTS = CN()
_CC.DATALOADER.SHOTS.AREA_RNG = [0.0, 1.e10]
_CC.DATALOADER.SHOTS.REL_AREA_RNG = [0.0, 2.0]
_CC.DATALOADER.SHOTS.X_RNG = [0.0, 1.e10]
_CC.DATALOADER.SHOTS.Y_RNG = [0.0, 1.e10]
_CC.DATALOADER.SHOTS.LONGEST_SIDE_ONLY = False

_CC.SOLVER.CLIP_LR = _CC.SOLVER.BASE_LR
