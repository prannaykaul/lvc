# Label, Verify, Correct: A Simple Few-Shot Object Detection Method
Code for our paper: [Label, Verify, Correct: A Simple Few-Shot Object Detection Method](https://arxiv.org/abs/2112.05749)

> **Abstract:** *The objective of this paper is few-shot object detection (FSOD)
> – the task of expanding an object detector for a new category given only a few instances for training.
> We introduce a simple pseudo-labelling method to source high-quality pseudo-annotations from the training set,
> for each new category, vastly increasing the number of training instances and reducing class imbalance;
> our method finds previously unlabelled instances.
> Naïvely training with model predictions yields sub-optimal performance;
> we present two novel methods to improve the precision of the pseudo-labelling process:
> first, we introduce a verification technique to remove candidate detections with incorrect class labels;
> second, we train a specialised model to correct poor quality bounding boxes.
> After these two novel steps, we obtain a large set of high-quality pseudo-annotations that allow our final detector to be trained end-to-end. Additionally, we demonstrate our method maintains base class performance,
> and the utility of simple augmentations in FSOD.
> While benchmarking on PASCAL VOC and MS-COCO,
> our method achieves state-of-the-art or second-best performance compared to existing approaches across all number of shots.*

![Alt text](/assets/main_img.png)

## Code For COCO Experiments now available!

If you find this repository useful for your own research, please consider citing our paper.
```angular2html
@InProceedings{Kaul22,
  author       = "Prannay Kaul and Weidi Xie and Andrew Zisserman",
  title        = "Label, Verify, Correct: A Simple Few-Shot Object Detection Method",
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
  year         = "2022",
}
```

This repo began from the excellent [Fsdet](https://github.com/ucbdrive/few-shot-object-detection) repo.

## Updates
- (Apr 2022) Initial public code release containing code for MS-COCO experiments, Pascal VOC to follow!

## Table of Contents
- [Label, Verify, Correct (LVC)](#label-verify-correct-a-simple-few-shot-object-detection-method)
    - [Updates](#updates)
    - [Table of Contents](#table-of-contents)
    - [Installation](#installation)
    - [Code Structure](#code-structure)
    - [Data Preparation](#data-preparation)
    - [Getting Started](#getting-started)
        - [Full Training & Evaluation in Command Line](#full-training--evaluation-in-command-line)

## Installation

**Requirements**

* Linux with Python >= 3.8
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.7
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* CUDA 9.2, 10.0, 10.1, 10.2, 11.0
* GCC >= 4.9

**Build lvc**
We recommend using conda to create an enviroment for this project
```bash
conda create --name lvc
conda activate lvc
```
* Install PyTorch. You can choose the PyTorch and CUDA version according to your machine.
Just make sure your PyTorch version matches the prebuilt Detectron2 version (next step).
For example for PyTorch v1.7.1:
```bash
pip install torch==1.7.1 torchvision==0.8.2
```
This code uses [Detectron2 v0.2.1](https://github.com/facebookresearch/detectron2/releases/tag/v0.2.1), however for ease
we have modified this code and a self-contained version exists in this repo.

* To install our Detectron2 code use pip locally, ensuring our `detectron2/` directory is in the present working directory
```bash
python -m pip install -e .
```
The installation of our Detectron2 code is identical to the official Detectron2 code,
so if you run into errors at the above step please see the installation docs for Detectron2,
specifically the [common issues](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#common-installation-issues).

* Install all other dependencies
```bash
python -m pip install -r requirements.txt
```

## Code Structure
- **configs**: Configuration files
- **datasets**: Dataset files (see [Data Preparation](#data-preparation) for more details)
- **lvc**
  - **checkpoint**: Checkpoint code.
  - **config**: Configuration code and default configurations.
  - **data**: Contains code for the few-shot object detection specific datasets.
  - **engine**: Contains training and evaluation loops and hooks.
  - **modeling**: Code for models, including backbones, proposal networks, and prediction heads.
  - **evaluation**: Code for evaluating RPNs, detectors and saving files as needed.
- **scripts**
  - **coco_full_run.sh**: Example full script for COCO 30shot experiments with baseline and pseudo-annotations.
- **tools**
  - **train_net.py**: Training script for baseline detector.
  - **train_net_reg.py**: Training script for box corrector.
  - **train_net_reg_qe.py**: Script to run box corrector.
  - **create_coco_dataset_from_dets_all.py**: Get candidates from baseline detections.
  - **run_nearest_neighbours.py**: Run label verification.
  - **combine.py**: Various tools for doing .json file manipulation.
  - **ckpt_surgery.py**: Surgery on checkpoints.


## Data Preparation
We evaluate our models on three datasets:
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/): We use the train/val sets of PASCAL VOC 2007+2012 for training and the test set of PASCAL VOC 2007 for evaluation. We randomly split the 20 object classes into 15 base classes and 5 novel classes, and we consider 3 random splits. The splits can be found in [lvc/data/builtin_meta.py](lvc/data/builtin_meta.py).
- [COCO](http://cocodataset.org/): We use COCO 2014 and extract 5k images from the val set for evaluation and use the rest for training. We use the 20 object classes that are the same with PASCAL VOC as novel classes and use the rest as base classes.

See [datasets/README.md](datasets/README.md) for more details.

## Getting Started
### Full Training & Evaluation in Command Line
LVC is conceptually simple but the full run involves many commands.
For a full explanation of how to train for LVC, see [TRAIN_FULL.md](docs/TRAIN_FULL.md).
