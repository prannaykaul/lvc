# LVC Training Instructions

LVC is trained in multiple stages some of which have sub-stages. In this file, training an LVC model for the COCO 30-shot task will be used as an example.
For a full, single file with all the below command contained, see [coco_full_run.sh](../scripts/coco_full_run.sh)

## Part 1: Training a baseline detector

### Part 1, Step 1: Base Training
First train a base model. For the MS-COCO, run:
```bash
python -m tools.train_net \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_base.yaml \
    --num-gpus 4
```

### Part 1, Step 2: Checkpoint Surgery - remove the final classifier and box regressor
Here we use ```tools/ckpt_surgery.py``` as follows:
```bash
python -m tools.ckpt_surgery \
    --method remove \
    --coco \
    --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base/model_final.pth \
    --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base/
```
This creates a model ```checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base/model_reset_remove.pth```.

### Part 1, Step 3: Novel data only training
Train a new final classifier and box regressor for the novel classes in the MS-COCO 30-shot case:
```bash
python -m tools.train_net \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_novel_30shot.yaml \
    --num-gpus 4
```

### Part 1, Step 4: Combine the base only and novel only models
We combine the two models into a single model for further finetuning:
```bash
python -m tools.ckpt_surgery \
    --method combine \
    --coco \
    --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base/model_final.pth \
    --src2 checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_novel_30shot/model_final.pth \
    --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_novel_30shot/
```
This creates a model ```checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base/model_reset_combine.pth```.

*Up to now the process is largely the same as TFA/Fsdet with some simple training improvements.*

### Part 1, Step 5: Few-shot finetuning for improved baseline detector
Train the improved baseline few-shot detector using the given novel data and balanced base data
with the training improvements from the paper (augmentations plus fine-tune more layers):
```bash
python -m tools.train_net \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout.yaml \
    --num-gpus 4
```

*Now we have our improved baseline few-shot object detector and Part 1 of training is complete!*


## Part 2: Training the box correction model

### Part 2, Step 1: Extracting proposals using the Base Only RPN
When we *base* train the box corrector, we use the proposals from the our baseline detector after base training [Part 1, Step 1](#part-1-step-1-base-training) as sample boxes to correct,
so lets extract these proposals now (just extract on all training data for simplicity):
```bash
python -m tools.train_net \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_base.yaml \
    --num-gpus 4 \
    --resume \
    --eval-only \
    DATASETS.TEST "('coco_test_all', 'coco_trainval_all',)" \
    MODEL.META_ARCHITECTURE "ProposalNetwork"
```

### Part 2, Step 2: Extracting proposals using the Finetuned Detector RPN
When we *finetune* the box corrector, we use the proposals from the our baseline detector after finetuning
[Part 1, Step 5](#part-1-step-5-few-shot-finetuning-for-improved-baseline-detector) as sample boxes to correct,
so lets extract these proposals now (just extract on all training data for simplicity):
```bash
python -m tools.train_net \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout.yaml \
    --num-gpus 4 \
    --eval-only \
    --resume \
    DATASETS.TEST "('coco_test_all', 'coco_trainval_all',)" \
    MODEL.META_ARCHITECTURE "ProposalNetwork"
```

### Part 2, Step 3: Box Corrector Base Training
Train a box corrector on base data using the base detector proposals:
```bash
python -m tools.train_net_reg \
    --config-file configs/COCO-detection/cascade_ubbr_R_50_FPN_base.yaml \
    --num-gpus 4 \
    DATASETS.PROPOSAL_FILES_TRAIN "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base/inference/coco_proposals_trainval_results.pkl',)" \
    DATASETS.PROPOSAL_FILES_TEST "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base/inference/coco_proposals_test_results.pkl',)"
```

### Part 2, Step 4: Box Corrector Finetuning
Finetune the box corrector on base and novel data using the finetuned detector proposals:
```bash
python -m tools.train_net_reg \
    --config-file configs/COCO-detection/cascade_ubbr_R_50_FPN_ft_all_30shot_aug_ftmore.yaml \
    --num-gpus 4 \
    DATASETS.PROPOSAL_FILES_TRAIN "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_proposals_trainval_results.pkl',)" \
    DATASETS.PROPOSAL_FILES_TEST "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_proposals_test_results.pkl',)"
```

*Now we have our box corrector model trained for use during the pseudo-annotation process*


## Part 3: Running the pseudo-annotation process

### Part 3, Step 1: Candidate Sourcing
To begin the pseudo-annotation process we need a list of initial detections from the
finetuned detector on the training sets (and unlabelled set):
```bash
python -m tools.train_net \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout.yaml \
    --resume \
    --eval-only \
    DATASETS.TEST "('coco_trainval_all', 'coco_unlabeled_all')"
```

### Part 3, Step 2: Filter Candidates
To reduce the number of initial pseudo-annotations, remove all detections with a score
below ```--K-min``` and convert list of detections into a "COCO-style" json:
```bash
python -m tools.create_coco_dataset_from_dets_all \
    --json-data 'coco_trainval_all' \
    --gt-data 'coco_trainval_all_30shot' \
    --full \
    --K-min 0.8 \
    --K-max 1.0 \
    --dt-path "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results.json"

# Unlabeled set
python -m tools.create_coco_dataset_from_dets_all \
    --json-data 'coco_unlabeled_all' \
    --gt-data 'coco_trainval_all_30shot' \
    --full \
    --K-min 0.8 \
    --K-max 1.0 \
    --dt-path "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results.json"
```
We need an extra run for unlabelled data which contains base class predictions
```bash
python -m tools.create_coco_dataset_from_dets_all \
    --json-data 'coco_unlabeled_all' \
    --gt-data 'coco_trainval_all_30shot' \
    --full \
    --K-min 0.8 \
    --K-max 1.0 \
    --dt-path "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results.json" \
    --all-cats
```

### Part 3, Step 3: Label Verification
After filtering we run label verification using a DINO self-supervised model:
```bash
python -m tools.run_nearest_neighbours \
    --config-file configs/LABEL-Verification/dino_label_verification.yaml \
    --num-gpus 4 \
    --eval-only \
    --opts \
    DATASETS.DT_PATH "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all.json',)" \
    QUERY_EXPAND.NN_DSET "('coco_trainval_all_30shot',)" \
    QUERY_EXPAND.KNN 10 \
    OUTPUT_DIR "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout"

# Unlabelled set
python -m tools.run_nearest_neighbours \
    --config-file configs/LABEL-Verification/dino_label_verification.yaml \
    --num-gpus 4 \
    --eval-only \
    --opts \
    DATASETS.DT_PATH "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all.json',)" \
    QUERY_EXPAND.NN_DSET "('coco_trainval_all_30shot',)" \
    QUERY_EXPAND.KNN 10 \
    OUTPUT_DIR "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout"
```

### Part 3, Step 4: Box Correction
Using the model we learnt in [Part 2, Step 4](#part-2-step-4-box-corrector-finetuning),
correct the boxes returned from [Label Verification](#part-3-step-3-label-verification)
```bash
python -m tools.train_net_reg_qe \
    --config-file configs/COCO-detection/cascade_ubbr_R_50_FPN_ft_all_30shot_aug_ftmore.yaml \
    --num-gpus 4 \
    --opts \
    --resume \
    --eval-only \
    DATASETS.DT_PATH "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all_dino_vits8_10_cosine.json',)" \
    MODEL.META_ARCHITECTURE "GeneralizedRCNNRegOnly" \
    QUERY_EXPAND.ENABLED True \
    MODEL.LOAD_PROPOSALS False

python -m tools.train_net_reg_qe \
    --config-file configs/COCO-detection/cascade_ubbr_R_50_FPN_ft_all_30shot_aug_ftmore.yaml \
    --num-gpus 4 \
    --opts \
    --resume \
    --eval-only \
    DATASETS.DT_PATH "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all_dino_vits8_10_cosine.json',)" \
    MODEL.META_ARCHITECTURE "GeneralizedRCNNRegOnly" \
    QUERY_EXPAND.ENABLED True \
    MODEL.LOAD_PROPOSALS False
```

### Part 3, Step 5: Misc pseudo-annotation manipulation (e.g. including ignore regions)
Convert results from box correcion to a "COCO-style" json:
```bash
python -m tools.combine_ubbr_with_qe \
    --ubbr-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr.json" \
    --qe-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all_dino_vits8_10_cosine.json"

python -m tools.combine_ubbr_with_qe \
    --ubbr-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr.json" \
    --qe-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all_dino_vits8_10_cosine.json"
```

Include all rejected detections as "ignore regions":
```bash
python -m tools.combine_pseudo_with_ignore \
    --ps-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr_id.json" \
    --ig-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all.json"

python -m tools.combine_pseudo_with_ignore \
    --ps-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr_id.json" \
    --ig-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all.json"
```

Combine verified and corrected pseudo-annotations for novel classes with (known)
base annotations to boost base class performance:
```bash
python -m tools.combine_qe_with_base \
    --ps-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr_id_ignore.json" \
    --bs-data "datasets/cocosplit/datasplit/trainvalno5k.json"

python -m tools.combine_qe_with_base \
    --ps-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr_id_ignore.json" \
    --bs-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all_allcats.json" \
    --base-ignore
```

*Now we have completed the pseudo-annotation process and have the data in the correct format*


## Part 4: End-to-end training with pseudo-annotations

### Part 4, Step 1:
Perform end-to-end training on pseudo-annotations
(note even in traditional object detection, initial
layers of the pretrained backbone are kept frozen for training speed):
```bash
python -m tools.train_net_qe_ig \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout.yaml \
    --num-gpus 4 \
    --opts \
    OUTPUT_DIR checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/end_to_end_pseudo_annotations \
    MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/model_final.pth \
    DATASETS.DT_PATH "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr_id_ignore_wbase.json', \
                       'checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr_id_ignore_wbase_base_ig.json')" \
    QUERY_EXPAND.ENABLED True \
    MODEL.BACKBONE.FREEZE False \
    MODEL.BACKBONE.FREEZE_AT 2
```
