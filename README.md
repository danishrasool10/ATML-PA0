# ATML PA 0 — Deep Learning Foundations

Programming assignment exploring four core model families in modern computer vision and generative modeling: **ResNet**, **Vision Transformers (ViT)**, **CLIP**, and **Variational Autoencoders (VAE)**.

All experiments are implemented as Jupyter notebooks under `task1_resnet/`, `task2_vit/`, `task3_clip/`, and `task4_vae/`.

---

## Tasks Overview

### Task 1 — ResNet
Transfer learning with pretrained ResNet-152 on CIFAR-10. Notebooks cover:
- Freezing the backbone and training only the classification head
- Ablating residual skip connections and comparing training dynamics
- Partial vs. full fine-tuning (and training from random initialization)
- Feature hierarchy visualization (t-SNE / UMAP) and comparison with ResNet-18

### Task 2 — Vision Transformer (ViT)
Analysis of `google/vit-base-patch16-224` using Hugging Face Transformers:
- Baseline predictions and CLS attention maps on Imagenette
- Patch-masking robustness (random vs. center occlusion)
- Frozen ViT feature extraction on STL-10 with linear probes (CLS vs. mean-pooled patches)

### Task 3 — CLIP
OpenAI CLIP (`ViT-B/32`) for vision–language alignment on STL-10:
- Zero-shot classification under different text prompt styles
- Exploring the modality gap with t-SNE and similarity statistics
- Bridging the gap via orthogonal Procrustes alignment and measuring the effect on accuracy

### Task 4 — VAE
A basic fully connected VAE trained on MNIST:
- Encoder / decoder with reparameterization trick
- ELBO training (reconstruction + KL)
- Latent-space visualization, reconstructions, and random sample generation

---

## Repository Structure

```
.
├── requirements.txt
├── task1_resnet/
│   ├── resnet152_frozen_backbone_cifar10.ipynb
│   ├── resnet152_partial_and_full_finetune.ipynb
│   └── resnet152_vs_resnet18_comparison.ipynb
├── task2_vit/
│   ├── vit_baseline_imagenette_predictions.ipynb
│   ├── vit_patch_masking_robustness.ipynb
│   └── vit_stl10_feature_extraction.ipynb
├── task3_clip/
│   └── clip_zeroshot_and_modality_gap.ipynb
└── task4_vae/
    └── basic_vae.ipynb
```

Datasets (CIFAR-10, STL-10, Imagenette, MNIST) download automatically into a local `./data` directory when a notebook is first run.

---

## Setup

**Requirements:** Python 3.10+, pip, and (recommended) a CUDA-capable GPU.

```bash
# From the repository root
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
pip install jupyter
```

> **Note:** CLIP is installed from OpenAI’s GitHub (`git+https://github.com/openai/CLIP.git`). Git must be available on your `PATH`.

---

## How to Run

### Option A — Jupyter Notebook / Lab

```bash
jupyter notebook
# or
jupyter lab
```

Open any notebook under the `task*_*/` folders and run cells top to bottom (`Run All`).

### Option B — Command line (nbconvert)

Execute a notebook non-interactively from the repo root:

```bash
# Example: Task 4 VAE
jupyter nbconvert --to notebook --execute task4_vae/basic_vae.ipynb --output basic_vae_executed.ipynb

# Example: Task 3 CLIP
jupyter nbconvert --to notebook --execute task3_clip/clip_zeroshot_and_modality_gap.ipynb --output clip_executed.ipynb
```

### Suggested order

| Order | Notebook | Notes |
|------:|----------|--------|
| 1 | `task1_resnet/resnet152_frozen_backbone_cifar10.ipynb` | Baseline ResNet transfer learning |
| 2 | `task1_resnet/resnet152_partial_and_full_finetune.ipynb` | Fine-tuning strategies |
| 3 | `task1_resnet/resnet152_vs_resnet18_comparison.ipynb` | Depth / representation comparison |
| 4 | `task2_vit/vit_baseline_imagenette_predictions.ipynb` | ViT predictions & attention |
| 5 | `task2_vit/vit_patch_masking_robustness.ipynb` | Occlusion robustness |
| 6 | `task2_vit/vit_stl10_feature_extraction.ipynb` | Linear probing on STL-10 |
| 7 | `task3_clip/clip_zeroshot_and_modality_gap.ipynb` | Zero-shot + modality gap |
| 8 | `task4_vae/basic_vae.ipynb` | Generative modeling on MNIST |

---

## Notes

- Most notebooks auto-detect CUDA (`torch.cuda.is_available()`); CPU works but training/inference will be slower.
- Multi-GPU setups use `nn.DataParallel` where available.
- A fixed random seed (`42`) is used in several notebooks for reproducibility.
- Large pretrained weights (ResNet-152, ViT, CLIP) are downloaded on first use and may require a network connection.
