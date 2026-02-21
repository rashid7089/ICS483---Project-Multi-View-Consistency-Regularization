# MV-SAX: Multi-View Consistency Regularization for Sparse-View CT Reconstruction

[![Course](https://img.shields.io/badge/Course-ICS483-blue)](https://github.com/rashid7089/ICS483---Project-Multi-View-Consistency-Regularization)
[![Reference](https://img.shields.io/badge/Reference-SAX--NeRF%20CVPR%202024-green)](https://openaccess.thecvf.com/content/CVPR2024/papers/Cai_Structure-Aware_Sparse-View_X-ray_3D_Reconstruction_CVPR_2024_paper.pdf)
[![Base Repo](https://img.shields.io/badge/Base%20Repo-SAX--NeRF-orange)](https://github.com/caiyuanhao1998/SAX-NeRF)

## Course Project Information

| Field | Details |
|-------|---------|
| **Course** | ICS483 |
| **Team** | Rashed Jafry, Marwan Khayat, Mohammed Ali |
| **Student IDs** | 202157670, 202265080, 202252120 |
| **Reference Paper** | [Structure-Aware Sparse-View X-ray 3D Reconstruction (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Cai_Structure-Aware_Sparse-View_X-ray_3D_Reconstruction_CVPR_2024_paper.pdf) |
| **Reference Repo** | [github.com/caiyuanhao1998/SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF) |
| **Dataset** | [Google Drive](https://drive.google.com/drive/folders/1SlneuSGkhk0nvwPjxxnpBCO59XhjGGJX?usp=sharing) |

---

## Overview

**MV-SAX** enhances the SAX-NeRF (Structure-Aware Sparse-View X-ray 3D Reconstruction) model with **Multi-View Consistency Regularization (MVCR)**, a novel training-time constraint that improves CT reconstruction quality from sparse X-ray projections.

### The Problem

Sparse-view CT reconstruction is an ill-posed inverse problem: given only K ≪ N X-ray projections (e.g., 50 out of 720 full-rotation views), reconstruct the 3D density volume. NeRF-based methods like SAX-NeRF can overfit to the sparse training angles, producing artifacts and inconsistencies at novel viewpoints.

### Our Innovation: Multi-View Consistency Regularization (MVCR)

During training, in addition to the standard reconstruction loss, we enforce **geometric consistency** between views:

1. **Virtual View Sampling**: At each training step, sample M "virtual" intermediate views at angles θ_v between adjacent training views θ_a and θ_b.

2. **Pseudo Ground Truth**: Construct an interpolated pseudo ground truth for each virtual view:
   ```
   I_pseudo(θ_v) = (1 − α) · I_gt(θ_a) + α · I_gt(θ_b)
   ```
   where α = (θ_v − θ_a) / (θ_b − θ_a) ∈ (0, 1).

3. **Consistency Loss**: Penalize discrepancy between the model's rendering and the pseudo GT:
   ```
   L_vc = ||render(net, θ_v) − I_pseudo(θ_v)||²
   ```

4. **Total Training Loss**:
   ```
   L_total = L_recon + λ(t) · L_vc
   ```
   where λ(t) is a curriculum-scheduled weight that ramps up linearly over `warmup_epochs`.

### Why It Works

- **Fills angular gaps**: Forces the 3D representation to generalize between training views, not just memorize them.
- **Curriculum scheduling**: The gradual warm-up prevents the consistency loss from destabilizing early training.
- **Computationally efficient**: Each virtual view only requires a small subsample of rays (default: 512), adding minimal overhead.

---

## Architecture

```
Input: K sparse X-ray projections {I_1, ..., I_K} at angles {θ_1, ..., θ_K}
         ↓
   Hash Grid Encoder (3D coordinates → features)
         ↓
   Lineformer (Line-based Transformer)
   ┌─ Linear Layer
   ├─ LineAttentionBlock × N (attention along projection lines)
   └─ Linear Layer → density σ(x)
         ↓
   Volume Rendering (Beer-Lambert integration)
         ↓
   Loss = L_recon + λ · L_MVCR
         ↑
   Virtual view rendering (MVCR)
```

---

## Repository Structure

```
MV-SAX/
├── train_mv_sax.py           # Main training script for MV-SAX
├── train_sax_nerf.py         # Baseline SAX-NeRF training (for comparison)
├── test.py                    # Evaluation script
├── requirements.txt
├── config/
│   ├── MV_SAX/               # MV-SAX configs (with MVCR settings)
│   │   ├── chest_50.yaml
│   │   ├── foot_50.yaml
│   │   └── ... (15 scenes)
│   └── Lineformer/           # Baseline SAX-NeRF configs
│       ├── chest_50.yaml
│       └── ...
├── src/
│   ├── config/configloading.py
│   ├── dataset/
│   │   ├── tigre.py          # Standard TIGRE dataset
│   │   └── tigre_mlg.py      # MLG window-sampling dataset
│   ├── encoder/
│   │   ├── freqencoder.py    # Frequency encoder
│   │   └── hashencoder/      # Hash grid encoder (requires CUDA compilation)
│   ├── loss/
│   │   ├── loss.py           # Standard losses (MSE, TV)
│   │   └── mv_consistency.py # ★ Multi-View Consistency Regularization (NEW)
│   ├── network/
│   │   ├── network.py        # MLP density network
│   │   └── Lineformer.py     # SAX-NeRF Lineformer architecture
│   ├── render/render.py      # Volume rendering
│   ├── utils/util.py         # Metrics (PSNR, SSIM), logging
│   ├── trainer_mlg.py        # Base trainer (SAX-NeRF)
│   └── trainer_mv_sax.py     # ★ MV-SAX trainer with MVCR (NEW)
└── tests/
    └── test_mv_consistency.py # Unit tests for MVCR components
```

---

## Setup

### 1. Create Environment

```bash
conda create -n mv_sax python=3.9
conda activate mv_sax

# Install PyTorch (hash encoder requires CUDA 11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies
pip install -r requirements.txt
```

### 2. Compile Hash Grid Encoder

The hash encoder requires CUDA compilation. It compiles automatically on first use:

```bash
python -c "from src.encoder import get_encoder; enc = get_encoder('hashgrid')"
```

> **Note**: Requires CUDA 11.3 and PyTorch 1.11.0. For CPU-only debugging, use `encoding: frequency` in the config.

### 3. Prepare Dataset

Download the datasets from [Google Drive](https://drive.google.com/drive/folders/1SlneuSGkhk0nvwPjxxnpBCO59XhjGGJX?usp=sharing) and place them in `data/`:

```
data/
├── chest_50.pickle
├── foot_50.pickle
└── ... (15 scenes)
```

---

## Training

### Train MV-SAX (our method)

```bash
# Train on chest CT with Multi-View Consistency Regularization
python train_mv_sax.py --config config/MV_SAX/chest_50.yaml --gpu_id 0
```

### Train Baseline (SAX-NeRF without MVCR)

```bash
python train_sax_nerf.py --config config/Lineformer/chest_50.yaml --gpu_id 0
```

### Monitor Training

```bash
tensorboard --logdir logs/
```

Key metrics: `train/loss_mse`, `train/loss_mvcr`, `eval/psnr_3d`, `eval/ssim_3d`

---

## Evaluation

```bash
python test.py \
    --config config/MV_SAX/chest_50.yaml \
    --weights logs/MV_SAX/chest_50/<timestamp>/ckpt_best.tar \
    --output_path output/chest_mvsax \
    --gpu_id 0
```

---

## MVCR Configuration

```yaml
mvcr:
  lambda_mvcr: 0.1        # Weight for consistency loss (recommended: 0.05–0.2)
  n_virtual_views: 2      # Virtual views per training step (recommended: 1–4)
  n_virtual_rays: 512     # Rays per virtual view (recommended: 256–1024)
  warmup_epochs: 150      # Linear ramp-up period (~10% of total epochs)
  use_model_renders: false # false = GT interpolation, true = model renders
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Method Details

At each training step:

```python
# For each of n_virtual_views:
θ_a, θ_b = adjacent_training_angles
α ~ Uniform(0.2, 0.8)
θ_v = θ_a + α * (θ_b - θ_a)              # virtual angle
I_pseudo = (1 - α) * I_gt(θ_a) + α * I_gt(θ_b)  # pseudo GT
I_pred = render(net, θ_v, n_rays=512)     # virtual view render
L_vc += MSE(I_pred, I_pseudo)

L_total = L_recon + λ(epoch) * L_vc / n_virtual_views
```

Curriculum scheduling: `λ(t) = λ_mvcr × min(1, t / warmup_epochs)`

---

## Citation

```bibtex
@inproceedings{sax_nerf,
  title={Structure-Aware Sparse-View X-ray 3D Reconstruction},
  author={Yuanhao Cai and Jiahao Wang and Alan Yuille and Zongwei Zhou and Angtian Wang},
  booktitle={CVPR},
  year={2024}
}

@misc{mv_sax_2025,
  title={MV-SAX: Multi-View Consistency Regularization for Sparse-View CT Reconstruction},
  author={Rashed Jafry and Marwan Khayat and Mohammed Ali},
  note={ICS483 Course Project, 2025},
  year={2025}
}
```

---

## Acknowledgements

This project builds upon [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF) by Cai et al. (CVPR 2024). We thank the authors for their excellent codebase and publicly available datasets.