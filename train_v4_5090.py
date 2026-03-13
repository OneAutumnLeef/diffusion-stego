"""
DSCSP v4 Config Override for RTX 5090 (32GB VRAM).

The RTX 5090 allows 4x larger batch and wider model compared to RTX 3060.
This module imports train_v4, applies 5090-optimized overrides, and runs training.

Key changes from base config:
  - batch_size: 4 → 16 (32GB VRAM vs 6GB)
  - base_channels: 48 → 64 (more model capacity)
  - LR scaled up (linear scaling rule with batch size)
  - JPEG weight 0.7 (more JPEG-heavy for Q75 target)
  - Larger image pool (256 vs 128) for diversity

Usage:
    python train_v4_5090.py
    python train_v4_5090.py --resume checkpoints_v4/dscsp_v4_step_012000.pt
"""

import sys
import train_v4

# ─── 5090 Optimized Overrides ───────────────────────────────────────────────
OVERRIDES = {
    # Larger batch = better gradient estimates = faster convergence
    "batch_size": 16,

    # Wider U-Net = more capacity for content-adaptive patterns
    "base_channels": 64,

    # Higher LR with larger batch (linear scaling rule: 4x batch → 2x LR)
    "lr_controller": 2e-4,
    "lr_extractor": 6e-4,
    "disc_lr": 2e-4,

    # More JPEG training (key for Q75 target)
    "jpeg_weight": 0.7,
    "robustness_max_prob": 0.9,

    # Larger image pool (more VRAM allows bigger pool)
    "pool_size": 256,

    # Pool refresh more often (faster generation on 5090)
    "pool_refresh_interval": 1500,

    # Stronger adversarial (key for AUC target)
    "lambda_adv": 0.5,
    "disc_steps_per_gen_step": 3,

    # Save more frequently (training is faster)
    "checkpoint_interval": 2000,
    "save_image_interval": 500,

    # Output dirs (separate from local training)
    "output_dir": "outputs_v4_5090",
    "checkpoint_dir": "checkpoints_v4_5090",
}

if __name__ == "__main__":
    config = {**train_v4.CONFIG, **OVERRIDES}

    # Handle --resume argument
    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        if idx + 1 < len(sys.argv):
            config["resume_path"] = sys.argv[idx + 1]

    print("=" * 60)
    print("DSCSP v4 Training — RTX 5090 Config")
    print("=" * 60)
    for k, v in OVERRIDES.items():
        orig = train_v4.CONFIG.get(k, "N/A")
        if orig != v:
            print(f"  {k}: {orig} → {v}")
    if config.get("resume_path"):
        print(f"  resume_path: {config['resume_path']}")
    print()
    train_v4.train(config)
