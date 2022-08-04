#!/usr/bin/env python

##### OUR BASELINE TRAINING

# First we perform base training
python -m tools.train_net \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_base.yaml \
    --num-gpus 4

# We remove the final classifier and regressor only from the base detector for NOVEL finetuning (only for COCO, same as TFA)
python -m tools.ckpt_surgery \
    --method remove \
    --coco \
    --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base/model_final.pth \
    --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base/

# We learn a final classifier and regressor only from the NOVEL data (only for COCO, same as TFA)
python -m tools.train_net \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_novel_30shot.yaml \
    --num-gpus 4

# We combine the classifiers from base training and from novel finetuning (only for COCO, same as TFA)
# Save in the directory for novel data!
python -m tools.ckpt_surgery \
    --method combine \
    --coco \
    --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base/model_final.pth \
    --src2 checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_novel_30shot/model_final.pth \
    --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_novel_30shot/

# Finetune on novel and (balanced) base data (same as TFA)
# Finetune more layers and apply augmentations and dropout (NEW)
python -m tools.train_net \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout.yaml \
    --num-gpus 4


##### BOX CORRECTOR TRAINING

# When we base train the box corrector, we use the proposals from the our baseline detector after base training as samples to correct,
# so lets extract these proposals now (just extract on all training data for simplicity)
python -m tools.train_net \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_base.yaml \
    --num-gpus 4 \
    --resume \
    --eval-only \
    DATASETS.TEST "('coco_test_all', 'coco_trainval_all',)" \
    MODEL.META_ARCHITECTURE "ProposalNetwork"

# When we finetune the box corrector, we use the proposals from the our baseline detector after finetuning as samples to correct,
# so lets extract these proposals now (just extract on all training data for simplicity)
python -m tools.train_net \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout.yaml \
    --num-gpus 4 \
    --eval-only \
    --resume \
    DATASETS.TEST "('coco_test_all', 'coco_trainval_all',)" \
    MODEL.META_ARCHITECTURE "ProposalNetwork"

# Let's train the box corrector on base data
python -m tools.train_net_reg \
    --config-file configs/COCO-detection/cascade_ubbr_R_50_FPN_base.yaml \
    --num-gpus 4 \
    DATASETS.PROPOSAL_FILES_TRAIN "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base/inference/coco_proposals_trainval_results.pkl',)" \
    DATASETS.PROPOSAL_FILES_TEST "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_base/inference/coco_proposals_test_results.pkl',)"


python -m tools.train_net_reg \
    --config-file configs/COCO-detection/cascade_ubbr_R_50_FPN_ft_all_30shot_aug_ftmore.yaml \
    --num-gpus 4 \
    DATASETS.PROPOSAL_FILES_TRAIN "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_proposals_trainval_results.pkl',)" \
    DATASETS.PROPOSAL_FILES_TEST "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_proposals_test_results.pkl',)"


# ##### CANDIDATE SOURCING

# To start candidate sourcing we need to extract detections from the training (and unlabelled set)
python -m tools.train_net \
    --config-file configs/COCO-detection/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout.yaml \
    --resume \
    --eval-only \
    DATASETS.TEST "('coco_trainval_all', 'coco_unlabeled_all')"

# We only want detections with score > x and we want to create a new COCO-style dataset for the candidates
python -m tools.create_coco_dataset_from_dets_all \
    --json-data 'coco_trainval_all' \
    --gt-data 'coco_trainval_all_30shot' \
    --full \
    --K-min 0.8 \
    --K-max 1.0 \
    --dt-path "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results.json"

# We only want detections with score > x and we want to create a new COCO-style dataset for the candidates
python -m tools.create_coco_dataset_from_dets_all \
    --json-data 'coco_unlabeled_all' \
    --gt-data 'coco_trainval_all_30shot' \
    --full \
    --K-min 0.8 \
    --K-max 1.0 \
    --dt-path "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results.json"

python -m tools.create_coco_dataset_from_dets_all \
    --json-data 'coco_unlabeled_all' \
    --gt-data 'coco_trainval_all_30shot' \
    --full \
    --K-min 0.8 \
    --K-max 1.0 \
    --dt-path "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results.json" \
    --all-cats

# Run label verification on the detections with score > x
python -m tools.run_nearest_neighbours \
    --config-file configs/LABEL-Verification/dino_label_verification.yaml \
    --num-gpus 4 \
    --eval-only \
    --opts \
    DATASETS.DT_PATH "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all.json',)" \
    QUERY_EXPAND.NN_DSET "('coco_trainval_all_30shot',)" \
    QUERY_EXPAND.KNN 10 \
    OUTPUT_DIR "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout"

# Run label verification on the detections with score > x
python -m tools.run_nearest_neighbours \
    --config-file configs/LABEL-Verification/dino_label_verification.yaml \
    --num-gpus 4 \
    --eval-only \
    --opts \
    DATASETS.DT_PATH "('checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all.json',)" \
    QUERY_EXPAND.NN_DSET "('coco_trainval_all_30shot',)" \
    QUERY_EXPAND.KNN 10 \
    OUTPUT_DIR "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout"

# Run bounding box correction on verified candidate detections using the finetuned bounding box regressor
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

# Some dataset sundries
# Make the ubbr results into a coco data
python -m tools.combine_ubbr_with_qe \
    --ubbr-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr.json" \
    --qe-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all_dino_vits8_10_cosine.json"

python -m tools.combine_ubbr_with_qe \
    --ubbr-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr.json" \
    --qe-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all_dino_vits8_10_cosine.json"

# Include candidates as ignore regions
python -m tools.combine_pseudo_with_ignore \
    --ps-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr_id.json" \
    --ig-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all.json"

python -m tools.combine_pseudo_with_ignore \
    --ps-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr_id.json" \
    --ig-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all.json"

# Combine pseudo-annotations for novel classes with (known) base annotations to boost base class performance
python -m tools.combine_qe_with_base \
    --ps-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_trainval_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr_id_ignore.json" \
    --bs-data "datasets/cocosplit/datasplit/trainvalno5k.json"

python -m tools.combine_qe_with_base \
    --ps-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all_dino_vits8_10_cosine_ubbr_id_ignore.json" \
    --bs-data "checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot_aug_ftmore_dropout/inference/coco_instances_unlabeled_results_score_max10_min08_full_all_allcats.json" \
    --base-ignore

## NOW WE HAVE COMPLETED THE PSEUDO-ANNOTATIONS PROCESS AND IT IS TIME TO END-TO-END TRAIN
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