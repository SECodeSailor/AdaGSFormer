# AdaGSFormer: Complexity-Adaptive Gaussian Evolution for 3D Semantic Occupancy Prediction

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://your-project-page.github.io)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

[Author 1](https://github.com/)*, [Author 2](https://github.com/), [Author 3](https://github.com/), [Corresponding Author](https://github.com/)<sup>†</sup>

*Institution / University Name*

</div>

> **Abstract:** 3D semantic occupancy prediction is crucial for fine-grained 3D scene perception in autonomous driving. While continuous 3D Gaussian representations offer significant memory savings over dense voxel grids, standard Gaussian-based methods often suffer from rigid Gaussian distribution and redundant computations in homogeneous regions. In this work, we propose **AdaGSFormer**, a novel framework featuring **complexity-adaptive Gaussian evolution** for 3D semantic occupancy prediction. By dynamically evolving 3D Gaussian queries according to local geometric and semantic complexity, AdaGSFormer achieves superior occupancy accuracy with enhanced computational efficiency and reduced memory consumption.

---

## 📢 News
* **[2026/xx/xx]** Code, pretrained models, and benchmark configs are released!
* **[2026/xx/xx]** AdaGSFormer is submitted to IEEE TPAMI / arXiv preprint is available at [arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx).

---

## 💡 Overview

<div align="center">
  <img src="assets/framework.png" width="95%"/>
</div>

Accurate and efficient 3D scene perception requires balancing fine geometric detail with low memory overhead. In this paper, we propose **AdaGSFormer**, which introduces:
1. **Complexity-Adaptive Gaussian Allocation:** Dynamically allocating Gaussian primitives based on regional geometric complexity (e.g., dense Gaussians for intricate boundaries/small objects and sparse Gaussians for flat surfaces/free space).
2. **Evolutionary Gaussian Refinement:** A progressive refinement mechanism that evolves Gaussian attributes across Transformer layers for precise semantic query and occupancy rendering.
3. **State-of-the-Art Performance:** Extensive evaluations on the **Occ3D-nuScenes** (and SurroundOcc) benchmarks demonstrate that AdaGSFormer delivers leading mIoU performance while maintaining high efficiency.

---

## 🛠️ Getting Started

### Installation
Follow [installation instructions](docs/install.md) to set up the Python environment, CUDA toolkit, and custom Gaussian rasterization/CUDA ops.

```bash
# Clone the repository
git clone https://github.com/your-username/AdaGSFormer.git
cd AdaGSFormer

# Create conda environment
conda create -n adagsformer python=3.8 -y
conda activate adagsformer

# Install PyTorch & dependencies
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python setup.py develop
