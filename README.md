# AdaGSFormer: Complexity-Adaptive Gaussian Evolution for 3D Semantic Occupancy Prediction

<div align="center">

% [![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://your-project-page.github.io)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/your-username/AdaGSFormer?style=social)](https://github.com/your-username/AdaGSFormer)

[Author 1](https://github.com/), [Author 2](https://github.com/)\*, [Author 3](https://github.com/)

*Institution / Laboratory Name*

</div>

> **Abstract:** Gaussian-based representations offer an efficient alternative to dense volumetric features for 3D semantic occupancy prediction. Existing methods adapt the Gaussian population mainly through progressive densification or allocation under a predefined final budget, limiting their ability to increase capacity in structurally complex regions while removing redundancy elsewhere.
We propose **AdaGSFormer**, a complexity-adaptive Gaussian evolution framework that adjusts the population size according to local scene structure without prescribing its final cardinality. AdaGSFormer predicts Gaussian-wise complexity scores using supervision derived from local semantic heterogeneity and uses scene-relative thresholds to guide population restructuring. It splits Gaussians in complex regions and removes redundant primitives through contribution-aware pruning and moment-preserving merging. To decode the resulting variable-size population, we formulate Gaussian contributions and Gaussian-to-voxel splatting in additive optical-density space. On NuScenes, AdaGSFormer achieves state-of-the-art semantic occupancy performance, while providing approximately $2.5\times$ faster inference and 57.8\% lower GPU memory than GaussianFormer-2. These results demonstrate the effectiveness of complexity-adaptive Gaussian evolution for accurate and efficient 3D semantic occupancy prediction.

---

## 📢 News
* **[2026/08/25]** Code, configs, and pre-trained checkpoints are released!
% * **[2026/xx/xx]** AdaGSFormer is available on arXiv: [arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx).

---

## 💡 Overview

<div align="center">
  <img src="assets/framework.png" width="95%" alt="AdaGSFormer Framework"/>
</div>

### Highlights:
* **Complexity-Adaptive Gaussian Allocation:** Dynamically distributes 3D Gaussian primitives based on regional geometric intricacies (dense for fine details/small objects and compact for homogeneous flat regions).
* **Evolutionary Gaussian Refinement:** Employs an evolutionary updating mechanism across Transformer layers to iteratively optimize Gaussian geometry and semantic features.
* **Efficient & SOTA Performance:** Delivers state-of-the-art mIoU on the **SurroundOcc** benchmark while significantly saving GPU memory.

---

## 🛠️ Getting Started

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-username/AdaGSFormer.git
cd AdaGSFormer

# Create conda environment
conda create -n adagsformer python=3.11.7
conda activate adagsformer

# Install PyTorch & dependencies
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install OpenMMLab & 3D packages
pip install openmim
mim install mmcv==2.1.0 mmdet==3.2.0 mmdet3d==1.3.0
pip install spconv-cu120
```

### 2. Dataset Preparation

Download the [nuScenes](https://www.nuscenes.org/download) dataset and the [Occ3D-nuScenes](https://github.com/tusen-ai/Occ3D) annotations. Organize the directory as follows:

```text
AdaGSFormer
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-trainval/
│   ├── surround_occ/
│   │   ├── samples/
│   │   │   ├── xxxx_occupancy.npy/
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
├── configs/
├── tools/
└── models/
```

---

## 📊 Benchmark & Pre-trained Models

| Benchmark | Modality | Backbone | Resolution | IoU (%) | mIoU (%) | Checkpoint | Config |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SurroundOcc | Camera (6-View) | ResNet-101 | 200x200x16 | --.- | --.- | [Download](https://github.com/) | [config](configs/adagsformer_r50_occ3d.py) |
% | Occ3D-nuScenes | Camera (6-View) | ResNet-101 | 200x200x16 | --.- | --.- | [Download](https://github.com/) | [config](configs/adagsformer_r101_occ3d.py) |
% | Occ3D-nuScenes | LiDAR + Camera | ResNet-50 | 200x200x16 | --.- | --.- | [Download](https://github.com/) | [config](configs/adagsformer_fusion_occ3d.py) |

---

## 🚀 Training & Evaluation

### Evaluation
To evaluate a pre-trained model on Occ3D-nuScenes validation set:

```bash
# 8-GPU Evaluation
bash tools/dist_test.sh configs/adagsformer_r50_occ3d.py ckpts/adagsformer_r50.pth 8 --eval mIoU
```

### Training
To train AdaGSFormer on 8 GPUs:

```bash
# Distributed Training
bash tools/dist_train.sh configs/adagsformer_r50_occ3d.py 8 --work-dir work_dirs/adagsformer_r50
```

### Visualization
To visualize predicted 3D Gaussians and voxel occupancy:

```bash
python tools/visualize.py \
    --config configs/adagsformer_r50_occ3d.py \
    --checkpoint ckpts/adagsformer_r50.pth \
    --sample-idx 0 \
    --save-dir vis_outputs/
```

---

## 🔗 Related Projects

Our project builds upon and is inspired by the following excellent repositories:
* [GaussianFormer](https://github.com/wzzheng/GaussianFormer)
* [Occ3D](https://github.com/tusen-ai/Occ3D)
* [SurroundOcc](https://github.com/weiyithu/SurroundOcc)
* [BEVFormer](https://github.com/fundamentalvision/BEVFormer)

---

## 📝 Citation

If you find this code or research helpful, please consider citing:

```bibtex
@article{adagsformer2026,
  title   = {AdaGSFormer: Complexity-Adaptive Gaussian Evolution for 3D Semantic Occupancy Prediction},
  author  = {Author One and Author Two and Author Three},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year    = {2026}
}
```
