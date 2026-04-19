# Graph Report - .  (2026-04-19)

## Corpus Check
- 39 files · ~157,681 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 268 nodes · 523 edges · 24 communities detected
- Extraction: 67% EXTRACTED · 33% INFERRED · 0% AMBIGUOUS · INFERRED: 175 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]

## God Nodes (most connected - your core abstractions)
1. `DiffusionWrapper` - 36 edges
2. `ContentAdaptiveController` - 29 edges
3. `train_v4.py: Main Training Script` - 24 edges
4. `train()` - 24 edges
5. `DiffJPEG` - 22 edges
6. `GlobalExtractor` - 21 edges
7. `SRMFilterLayer` - 20 edges
8. `AttackLayer` - 19 edges
9. `DSCSPLoss` - 19 edges
10. `evaluate_v4.py: Core Evaluation Pipeline` - 16 edges

## Surprising Connections (you probably didn't know these)
- `TIFS-oriented evaluation for DSCSP.  Focus areas: - Detectability (adaptive s` --uses--> `DiffJPEG`  [INFERRED]
  evaluate_tifs.py → dscsp\augmentations\differentiable.py
- `Modified DDIM sampler for DSCSP.  Simplified architecture: the entire DDIM denoi` --uses--> `DiffusionWrapper`  [INFERRED]
  dscsp\sampling\stego_ddim.py → dscsp\models\diffusion_wrapper.py
- `DDIM sampler that adds a learned perturbation to the final denoised image.` --uses--> `DiffusionWrapper`  [INFERRED]
  dscsp\sampling\stego_ddim.py → dscsp\models\diffusion_wrapper.py
- `Generate stego images encoding the payload.          Args:             payload_b` --uses--> `DiffusionWrapper`  [INFERRED]
  dscsp\sampling\stego_ddim.py → dscsp\models\diffusion_wrapper.py
- `_eval_jpeg_series()` --calls--> `DiffJPEG`  [INFERRED]
  evaluate_tifs.py → dscsp\augmentations\differentiable.py

## Hyperedges (group relationships)
- **training_pipeline_stack** — run_logged_sh, train_v4_py, tifs_autopilot_py, evaluate_tifs_py, training_log_md [EXTRACTED]
- **model_architecture_group** — controller_module, extractor_module, discriminator_module [EXTRACTED]
- **metric_suite** — metric_clean_ber, metric_jpeg_q75_ber, metric_crop_q80_ber, metric_adaptive_auc, metric_psnr, metric_ssim, metric_lpips [EXTRACTED]
- **training_phase_variants** — phase_ber_recovery_clean, phase_ber_recovery_curriculum, phase_robust_balance, phase_detectability_harden [EXTRACTED]
- **sprint_execution_record** — record_initial_probe, record_deps_install, record_probe_after_deps, record_warmup_2000, record_bootstrap_logging, record_sprint_plan [EXTRACTED]
- **dependency_framework** — torch_lib, diffusers_lib, requirements_txt [EXTRACTED]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.0
Nodes (39): DSCSPLoss, Combined training loss for the DSCSP steganography system., ContentAdaptiveController, Args:             image: (B, 3, 256, 256) cover image in [-1, 1]             p, Content-adaptive steganographic encoder.      Takes cover image + payload bits, AttackLayer, DiffJPEG, Apply 8x8 block DCT to a single-channel image. (+31 more)

### Community 1 - "Community 1"
Cohesion: 0.0
Nodes (34): _adaptive_attack(), _adaptive_steganalysis(), _bootstrap_ci(), _eval_clean_series(), _eval_crop_series(), _eval_jpeg_series(), evaluate_checkpoint(), _load_dataset_images() (+26 more)

### Community 2 - "Community 2"
Cohesion: 0.0
Nodes (34): Crop Robustness Blocker: Near-Random ~0.50 BER, JPEG Q75 BER Blocker: 0.136973 vs Target 0.20, major_cycle_011_robust_balance.pt: Best Checkpoint from Cycle 11, PROJECT_CONTEXT_MANUAL_2026-04-18.md: Full Handover Document, 2026-04-18: Sprint Execution Date, deps_install: Successful Dependency Installation, Diffusers 0.25+: Diffusion Model Library, NVIDIA GeForce RTX 5090: Training Hardware (31.4GB VRAM) (+26 more)

### Community 3 - "Community 3"
Cohesion: 0.0
Nodes (11): ConvBlock, DownBlock, _group_norm(), _infer_payload_grid(), Content-adaptive steganographic controller for DSCSP v4.  Replaces the tiled-g, Conv + GroupNorm + ReLU block., Downsample by 2x with two conv layers., Upsample by 2x with ConvTranspose + conv. (+3 more)

### Community 4 - "Community 4"
Cohesion: 0.0
Nodes (20): Controller: Payload-to-Sampling-Path Mapping Network, Frozen Diffusion Model: Base Image Generator, Discriminator, Discriminator: Stego Detection Adversary, Args:             x: (B, 3, H, W) image in [-1, 1]          Returns:, Steganalysis discriminator for adversarial training.      Classifies images as c, DSCSP: Diffusion-Stego with Conditional Sampling Paths, Extractor: Bit Recovery from Stego Images (+12 more)

### Community 5 - "Community 5"
Cohesion: 0.0
Nodes (15): brightness_contrast(), _build_dct_matrix(), crop_resize(), gaussian_blur(), gaussian_noise(), Differentiable attack layers for robustness training.  All attacks are different, Scale quantization tables by JPEG quality factor., Args:             x: (B, 3, H, W) in [-1, 1]             quality: JPEG quality f (+7 more)

### Community 6 - "Community 6"
Cohesion: 0.0
Nodes (10): Redundant encoding/decoding for crop robustness.  Maps M message bits to N = M *, Expand M message bits to N physical bits via replication.          Args:, Aggregate N physical logits to M message logits via mean-pooling.          Diffe, Majority voting for hard bit decisions at evaluation time.          Args:, Redundant bit encoding/decoding for crop robustness.      Args:         message_, Map a physical slot index to its cell index., Map a cell index to (row, col) in the spatial grid., Manhattan distance between two cells in the spatial grid. (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.0
Nodes (10): Args:             x: (B, 3, H, W) stego image             attack_prob: probabili, predict_noise(), Frozen diffusion model wrapper for DSCSP.  Loads a pre-trained HuggingFace DDPM, Get alpha_cumprod for a given timestep value., Get alpha_cumprod for the previous timestep in the DDIM schedule.          Args:, sample_clean(), Modified DDIM sampler for DSCSP.  Simplified architecture: the entire DDIM denoi, DDIM sampler that adds a learned perturbation to the final denoised image. (+2 more)

### Community 8 - "Community 8"
Cohesion: 0.0
Nodes (16): _append_major_log(), _build_cycle_config(), _is_significant(), _load_json(), main(), _meets_required(), _safe_get_step(), _save_json() (+8 more)

### Community 9 - "Community 9"
Cohesion: 0.0
Nodes (3): Visualization utilities for DSCSP., Save side-by-side comparison of clean vs stego images, plus an amplified     dif, save_comparison()

### Community 10 - "Community 10"
Cohesion: 0.0
Nodes (2): Compute adversarial loss ramp factor based on current step., Compute the composite loss with curriculum warmup.          Args:             bi

### Community 11 - "Community 11"
Cohesion: 0.0
Nodes (1): Composite loss for DSCSP training.  Combines:   1. Message reconstruction loss (

### Community 12 - "Community 12"
Cohesion: 0.0
Nodes (1): DSCSP v4 Config Override for RTX 5090 (32GB VRAM).  The RTX 5090 has ample VRA

### Community 13 - "Community 13"
Cohesion: 0.0
Nodes (0): 

### Community 14 - "Community 14"
Cohesion: 0.0
Nodes (1): Build the NxN DCT-II transform matrix.

### Community 15 - "Community 15"
Cohesion: 0.0
Nodes (1): Convert RGB [0, 255] to YCbCr.

### Community 16 - "Community 16"
Cohesion: 0.0
Nodes (1): Convert YCbCr to RGB [0, 255].

### Community 17 - "Community 17"
Cohesion: 0.0
Nodes (0): 

### Community 18 - "Community 18"
Cohesion: 0.0
Nodes (1): Run the frozen UNet to predict noise. Output is FP32, fully detached.          A

### Community 19 - "Community 19"
Cohesion: 0.0
Nodes (1): Generate a clean (unperturbed) image using standard DDIM sampling.         Usefu

### Community 20 - "Community 20"
Cohesion: 0.0
Nodes (1): SSIM: Structural Similarity Index

### Community 21 - "Community 21"
Cohesion: 0.0
Nodes (1): LPIPS: Learned Perceptual Image Patch Similarity

### Community 22 - "Community 22"
Cohesion: 0.0
Nodes (1): 20260418_bootstrap_logging_setup.md: Logging Infrastructure Record

### Community 23 - "Community 23"
Cohesion: 0.0
Nodes (1): training_ops/chat_history_archive: Copilot Transcript Archive

## Knowledge Gaps
- **56 isolated node(s):** `DSCSP v4 Config Override for RTX 5090 (32GB VRAM).  The RTX 5090 has ample VRA`, `Iterated final training orchestration for DSCSP v4.  This script runs multi-st`, `Differentiable attack layers for robustness training.  All attacks are different`, `Add zero-mean Gaussian noise. Gradient: identity.`, `Gaussian blur via depthwise convolution. Fully differentiable.` (+51 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 12`** (2 nodes): `train_v4_5090.py`, `DSCSP v4 Config Override for RTX 5090 (32GB VRAM).  The RTX 5090 has ample VRA`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `Build the NxN DCT-II transform matrix.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `Convert RGB [0, 255] to YCbCr.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `Convert YCbCr to RGB [0, 255].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `Run the frozen UNet to predict noise. Output is FP32, fully detached.          A`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `Generate a clean (unperturbed) image using standard DDIM sampling.         Usefu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `SSIM: Structural Similarity Index`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `LPIPS: Learned Perceptual Image Patch Similarity`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `20260418_bootstrap_logging_setup.md: Logging Infrastructure Record`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `training_ops/chat_history_archive: Copilot Transcript Archive`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.