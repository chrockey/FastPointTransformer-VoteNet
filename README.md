# Fast Point Transformer (Detection)
### [Project Page](http://cvlab.postech.ac.kr/research/FPT/) | [Paper](https://arxiv.org/abs/2112.04702)
This repository contains the official code of Fast Point Transformer for 3D object detection experiments below:
#### 3D object detection with [VoteNet](https://arxiv.org/abs/1904.09664) using [Torch-Points3D](https://github.com/torch-points3d/torch-points3d)
| Backbone                          | Voxel Size   | mAP@0.25 | mAP@0.5 | Reference |
|:----------------------------------|:------------:|:--------:|:-------:|:---------:|
| MinkowskiNet42<sup>&dagger;</sup> | 5cm | 55.3 | 32.8 | [Checkpoint](https://postechackr-my.sharepoint.com/:u:/g/personal/p0125ch_postech_ac_kr/Ecod112ZRnlPp97NNu77N0oBfPtgwzmPxr-tvLvs3eFkwA?download=1) |
| FastPointTransformer              | 5cm | 59.1 | 35.4 | [Checkpoint](https://postechackr-my.sharepoint.com/:u:/g/personal/p0125ch_postech_ac_kr/EZkpqNi9elVFohE4Xmx66GEBQSjys0ED_h1vUBnJwUz72g?download=1) |

## Installation
This repository is developed and tested on

- Ubuntu 20.04
- Conda 4.12.0
- CUDA 11.1
- Python 3.8.13
- PyTorch 1.7.1
- MinkowskiEngine 0.5.4

### Environment Setup
Since this repo is forked from the [Torch-Points3D repo](https://github.com/torch-points3d/torch-points3d), you can setup the environment by following the the [Torch-Points3D repo](https://github.com/torch-points3d/torch-points3d).
We also provide a docker image to ease the environment setup.
You can pull and run the docker image via the following commands:
```bash
~$ docker pull chrockey/fpt-votenet:v0.1.0
~$ docker run {docker_arguments} chrockey/fpt-votenet:v0.1.0 # interactive mode
```
Within the docker container, you may find a conda environment named `tp3d-fpt`:
```bash
~$ conda activate tp3d-fpt
(tp3d-fpt) ~$ python -c "import torch; import cuda_sparse_ops"
```

### Dataset Preparation
First, you need to make a symbolic link for raw ScanNet V2 dataset via the following command:
```bash
~/FastPointTransformer-VoteNet$ ln -s {dir_to_scannet_v2_dataset} data/scannet-sparse/raw
```
And then, your data directory should look like the structure below:
```
~/FastPointTransformer-VoteNet/data/scannet-sparse
└── raw
    ├── metadata
    ├── scans
    ├── scans_test
    └── scannetv2-labels.combined.tsv
```

### Training & Evaluation
After linking the raw dataset, run the provided training script (`train_scripts/train_votenet_fpt.sh`).
The training outputs will be saved in the `outputs` directory.
```bash
~/FastPointTransformer-VoteNet$ conda activate tp3d-fpt
(tp3d-fpt) ~/FastPointTransformer-VoteNet$ sh train_scripts/train_votenet_fpt.sh
```
And then, you can evaluate the model as:
```bash
(tp3d-fpt) ~/FastPointTransformer-VoteNet$ sh eval_scripts/eval_votenet_fpt.sh
```
Note that you may need to modify the checkpoint directory within the script (`eval_scripts/eval_votenet_fpt.sh`).

### LICENSE
For [Torch-Points3D](https://github.com/torch-points3d/torch-points3d) repo, please check [the license](https://github.com/chrockey/FastPointTransformer-VoteNet/blob/main/LICENSE).

## Acknowledgement

This repo is forked from [Torch-Points3D](https://github.com/torch-points3d/torch-points3d) repo.
If you use our model, please consider citing [Torch-Points3D](https://github.com/torch-points3d/torch-points3d) and [VoteNet](https://arxiv.org/abs/1904.09664) as well.
