# DSCSP Project Context

## Required Post-Clone Artifact Restore

Large artifacts are intentionally not tracked in git. After cloning this repository, restore checkpoints and outputs from Hugging Face before training or evaluation.

1. Install dependencies (`pip install -r requirements.txt`)
2. Run `bash pull_artifacts_from_hf.sh .` from repo root
  - Windows PowerShell alternative: `./pull_artifacts_from_hf.ps1 .`
3. Verify key folders are populated (`checkpoints_v5/`, `checkpoints_v5_clean/`, `checkpoints_v5_quality/`, `outputs_v5_ablation/`)

Hugging Face artifact source:
- Dataset repo: `OneAutumnLeef/dscsp-full-backup`
- Fallback repo: `OneAutumnLeef/dscsp-checkpoints`
- Repo type: `dataset`

If private access is needed, authenticate first with `hf auth login`.

## Project Overview
DSCSP (Diffusion-Stego with Conditional Sampling Paths) - Deep steganography framework using diffusion-generated cover images. Building toward IEEE TIFS journal publication.

## Architecture Versions
- **v1/v2**: Tiled 8x8 grid Controller (nn.Parameter lookup) + ResNet-18 Extractor + SRM Discriminator
- **v4 (current)**: Content-adaptive U-Net Controller + Global-pool Extractor + SpectralNorm Discriminator

## Target Metrics (Publication-Ready)
| Metric | Target | v2 Result | v3 Result | Gap |
|--------|--------|-----------|-----------|-----|
| Clean BER | 0.0000 | 0.0003 | 0.0004 | Small |
| JPEG Q75 BER | < 0.01 | ~0.29 | ~0.30 | MASSIVE |
| JPEG Q50 BER | < 0.05 | 0.33 | 0.33 | MASSIVE |
| PSNR | >= 35 dB | 40.87 | 44.31 | Met |
| SSIM | >= 0.97 | 0.9617 | 0.9825 | Close |
| LPIPS | <= 0.05 | 0.0377 | 0.0341 | Met |
| FID increase | <= 3 | N/A | N/A | Not measured |
| AUC | <= 0.60 | 0.9986 | 1.0 | MASSIVE |
| Test images | >= 1000 | 500 | 500 | Need more |

## Root Cause Analysis
1. **AUC=1.0**: Tiled grid creates periodic 32px boundary artifacts; content-independent delta
2. **JPEG BER=0.29**: Signal in high-freq DCT coefficients; STE gradient noisy at low Q
3. **SSIM<0.97**: Fixed with lower max_perturbation (v3 achieved 0.9825)

## Key Architectural Decisions for v4
1. **Content-Adaptive Controller**: U-Net encoder-decoder taking BOTH image + payload → delta adapted to image content, no grid artifacts
2. **Global-Pool Extractor**: ResNet-18 with GAP+FC instead of rigid 1x1 conv head
3. **JPEG-Heavy Training**: 60% JPEG weight in attack layer, quality range [50, 95]
4. **Strong Adversarial**: lambda_adv=0.5, from step 1000 (ramped over 4K steps), 3 disc steps per gen step, spectral norm
5. **Spectral Loss**: Penalize frequency-domain differences (FFT magnitude MSE, weight=0.01)

## v4 Training Config (Current Run)
```python
CONFIG = {
    "max_perturbation": 0.08, "base_channels": 48, "batch_size": 4,
    "total_steps": 50000, "lr_controller": 1e-4, "lr_extractor": 3e-4,
    "lambda_perc": 0.3, "robustness_start_step": 500,
    "robustness_ramp_steps": 2000, "robustness_max_prob": 0.85,
    "jpeg_weight": 0.6, "lambda_adv": 0.5, "adv_start_step": 1000,
    "adv_ramp_steps": 4000, "disc_steps_per_gen_step": 3,
}
```

## Hardware Constraints
- GPU: RTX 3060 Laptop 6GB VRAM
- Environment: conda base, PyTorch 2.5.1 + CUDA
- Batch size: 4 (content-adaptive controller uses more memory than parameter lookup)
- Training time: ~7.5 hours for 50K steps at ~0.55s/step

## File Structure
```
dscsp/
  models/
    controller.py          # v1/v2 tiled grid (legacy)
    controller_v4.py       # v4 content-adaptive U-Net
    extractor.py           # ResNet-18 extractor
    discriminator.py       # SRM discriminator
    diffusion_wrapper.py   # DDPM wrapper
  losses/
    composite_loss.py      # Combined loss function
  augmentations/
    differentiable.py      # DiffJPEG, blur, noise, crop, BC
  utils/
    metrics.py, visualization.py, redundant_coding.py
train_v2.py               # v2 training (tiled controller)
train_v4.py               # v4 training (content-adaptive)
evaluate.py               # Basic evaluation
evaluate_v4.py            # Full eval (1000+ imgs, FID, capacity)
ablation.py               # Ablation study framework
paper_ieee.tex            # IEEE TIFS paper
```

## Important Notes
- The diffusion model is FROZEN - only Controller, Extractor, Discriminator are trained
- Image pool: 128 pre-generated diffusion images, refreshed every 2000 steps
- Images are in [-1, 1] range (not [0, 1])
- DDIM with 20 inference steps for image generation
- Ablation study PID 1468 died — 3 remaining variants need re-running
- V4 training: PID 23560, started 2026-03-13, ~0.55s/step
