"""Instruction 4 driver — detectability hardening.

Resumes from the Instruction 3 crop-robust checkpoint and engages the four
detectability components required to drop the adaptive ensemble AUC below
0.70 without regressing BER or crop:

  Component A  mode-spread regularizer (SVD top-k concentration penalty)
  Component B  SRM feature-matching loss (via the discriminator's own
               frozen SRM filter bank)
  Component C  WGAN-GP gradient penalty on the discriminator, 6 D steps
               per G step
  Component D  epsilon_max = 0.06 (tighter perturbation budget)

Target gate: adaptive_auc_ens <= 0.70 while holding clean BER <= 0.01
and crop Q80 BER <= 0.20.

Usage:
    python train_inst4_detect_harden.py
    python train_inst4_detect_harden.py --resume <inst3_ckpt>
    python train_inst4_detect_harden.py --lambda-spread 0.20 --lambda-gp 20
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pprint

from train_v4 import CONFIG as V4_CONFIG, train


INST4_OVERRIDES = {
    "resume_path": "checkpoints_tifs/inst3_crop_robust/dscsp_v4_final.pt",
    "resume_load_optimizer": False,
    "resume_load_scheduler": False,
    "resume_load_disc_optimizer": False,
    "override_lrs_on_resume": True,

    # Keep Instruction 3 architecture
    "num_bits": 256,
    "carrier_type": "spread_spectrum",
    "spread_seed": 20260424,
    "extractor_mode": "shift_pool",
    "shift_pool_mode": "mean",
    "shift_pool_kernel": 5,
    "base_channels": 48,

    # ── Component D: tighter perturbation budget ─────────────────────────
    "max_perturbation": 0.06,
    "lambda_perc": 0.15,
    "lambda_spec": 0.006,

    # ── Component C: strong adversary + GP ───────────────────────────────
    "use_adversarial": True,
    "lambda_adv": 0.55,
    "adv_start_step": 0,          # resuming; adversary already meaningful
    "adv_ramp_steps": 1500,
    "disc_steps_per_gen_step": 6,
    "lambda_gp": 10.0,            # WGAN-GP 1-centred penalty

    # ── Component A: mode-spread (ramps in at step 1000) ────────────────
    "lambda_spread": 0.10,
    "lambda_spread_start_step": 1000,
    "lambda_spread_k": 8,

    # ── Component B: SRM feature-matching ───────────────────────────────
    "lambda_srm_match": 0.05,

    # Retain JPEG and crop pressure from Inst 2 + 3
    "jpeg_weight": 0.55,
    "jpeg_quality_range": (50, 85),
    "real_jpeg_quality_range": (50, 85),
    "real_jpeg_prob_start": 0.50,
    "real_jpeg_prob_end": 0.50,
    "real_jpeg_ramp_step": 0,
    "real_jpeg_prob": 0.0,
    "robustness_start_step": 0,
    "robustness_ramp_steps": 500,
    "robustness_max_prob": 0.70,
    "crop_always_prob": 0.5,
    "crop_ratio_range": (0.65, 0.95),

    # Skip bootstraps (already trained)
    "extractor_bootstrap_steps": 0,
    "controller_bootstrap_steps": 0,
    "bit_curriculum_start_bits": 256,
    "bit_curriculum_end_bits": 256,
    "bit_curriculum_steps": 0,

    "use_combined_attacks": False,

    "lr_controller": 5e-5,
    "lr_extractor": 1.5e-4,
    "disc_lr": 1e-4,
    "lr_min": 1e-6,

    "total_steps": 8000,
    "batch_size": 4,
    "checkpoint_dir": "checkpoints_tifs/inst4_detect_harden",
    "output_dir": "outputs_tifs/inst4_detect_harden",
    "checkpoint_interval": 1500,
    "log_interval": 25,
}


def build_config(overrides):
    cfg = copy.deepcopy(V4_CONFIG)
    cfg.update(overrides)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--lambda-spread", type=float, default=None)
    parser.add_argument("--lambda-gp", type=float, default=None)
    parser.add_argument("--max-perturbation", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = build_config(INST4_OVERRIDES)
    if args.resume is not None:
        cfg["resume_path"] = args.resume
    if args.total_steps is not None:
        cfg["total_steps"] = args.total_steps
    if args.lambda_spread is not None:
        cfg["lambda_spread"] = args.lambda_spread
    if args.lambda_gp is not None:
        cfg["lambda_gp"] = args.lambda_gp
    if args.max_perturbation is not None:
        cfg["max_perturbation"] = args.max_perturbation

    print("=" * 72)
    print("Instruction 4 — detectability hardening (L_spread + L_srm_match + GP + ε=0.06)")
    print("=" * 72)
    highlights = [
        "resume_path", "total_steps", "max_perturbation",
        "lambda_spread", "lambda_spread_start_step", "lambda_spread_k",
        "lambda_srm_match",
        "lambda_adv", "disc_steps_per_gen_step", "lambda_gp",
        "jpeg_quality_range", "real_jpeg_prob_start", "real_jpeg_prob_end",
        "crop_always_prob", "crop_ratio_range",
        "lr_controller", "lr_extractor", "disc_lr",
        "checkpoint_dir", "output_dir",
    ]
    pprint.pprint({k: cfg[k] for k in highlights if k in cfg}, width=100)
    print("=" * 72, flush=True)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "inst4_config.json"), "w") as f:
        serialisable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
        json.dump(serialisable, f, indent=2, default=str)

    if args.dry_run:
        print("[dry-run] exiting before train().")
        return 0

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
