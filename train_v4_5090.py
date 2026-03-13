"""
DSCSP v4 Config Override for RTX 5090 (32GB VRAM).

The RTX 5090 has ample VRAM but the proven optimization dynamics from the
RTX 3060 base config must be preserved.  The previous run (batch=16,
base_channels=64, 2× LR) failed to escape the random-guess regime at step
7200 (BER ≈ 0.50).  Lesson 13 says: only use extra VRAM for a safe batch-size
increase — do NOT simultaneously scale model width, learning rates, and attack
intensity.

Key changes from the failed v1 5090 config:
  - batch_size: 16 → 8  (moderate increase; proven dynamics preserved)
  - base_channels: 64 → 48  (restored to proven base setting)
  - LR: 2e-4/6e-4 → 1e-4/3e-4  (restored to proven base rates)
  - disc_lr: 2e-4 → 1e-4  (restored)
  - jpeg_weight: 0.7 → 0.6  (restored to proven base value)
  - robustness_max_prob: 0.9 → 0.85  (restored)
  - checkpoint_interval: 2000 → 1000  (more frequent; reduces progress lost on interrupt)
  - save_image_interval: 500 → 250  (more frequent preview images)
  - Larger pool (256) and faster refresh (1500) are kept — safe with 32 GB VRAM

Usage:
    python train_v4_5090.py
    python train_v4_5090.py --resume checkpoints_v4_5090/dscsp_v4_step_006000.pt
"""

import sys
import train_v4

# ─── 5090 Optimized Overrides ───────────────────────────────────────────────
OVERRIDES = {
    # Moderate batch increase — 5090 has 32 GB VRAM; keeps proven dynamics
    "batch_size": 8,

    # Keep proven architecture width (do NOT increase — Lesson 13)
    "base_channels": 48,

    # Keep proven learning rates (do NOT scale — Lesson 13)
    "lr_controller": 1e-4,
    "lr_extractor": 3e-4,
    "disc_lr": 1e-4,

    # Keep proven attack schedule (Lesson 13: don't tighten simultaneously)
    "jpeg_weight": 0.6,
    "robustness_max_prob": 0.85,

    # Larger image pool — safe with 32 GB VRAM
    "pool_size": 256,

    # Pool refresh more often — 5090 generates images faster
    "pool_refresh_interval": 1500,

    # Adversarial settings (same as proven base)
    "lambda_adv": 0.5,
    "disc_steps_per_gen_step": 3,

    # More frequent checkpoints — reduces progress lost if interrupted (Lesson 14)
    "checkpoint_interval": 1000,
    "save_image_interval": 250,

    # Output dirs (separate from local training)
    "output_dir": "outputs_v4_5090",
    "checkpoint_dir": "checkpoints_v4_5090",
}

if __name__ == "__main__":
    import torch

    # Enable TF32 for matmul and cuDNN on Ampere / Hopper / Blackwell GPUs.
    # This has no effect on output dtype but speeds up FP32 matrix ops ~2×
    # with negligible accuracy loss.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = {**train_v4.CONFIG, **OVERRIDES}

    # Handle --resume argument
    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        if idx + 1 < len(sys.argv):
            config["resume_path"] = sys.argv[idx + 1]

    print("=" * 60)
    print("DSCSP v4 Training — RTX 5090 Config (conservative)")
    print("=" * 60)
    for k, v in OVERRIDES.items():
        orig = train_v4.CONFIG.get(k, "N/A")
        if orig != v:
            print(f"  {k}: {orig} → {v}")
    if config.get("resume_path"):
        print(f"  resume_path: {config['resume_path']}")
    print()
    train_v4.train(config)
