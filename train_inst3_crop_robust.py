"""Instruction 3 driver — crop robustness retrain.

Resumes from the Instruction 2 JPEG-robust checkpoint and adds the two
crop-robustness mechanisms required to pass the Crop Q80 gate:

  Part A  spread-spectrum carrier (each bit spread across the whole image
          via a fixed ±1 mask; translation-tolerant by construction)
  Part B  shift_pool extractor head with mean-pool (preserves per-bit sign
          under ±(K//2)-pixel spatial displacement) + crop-resize pre-attack
          applied from step 0 with ratio range (0.65, 0.95)

Target gate: Crop Q80 BER <= 0.20, Clean BER kept <= 0.01, JPEG Q75 real
BER kept <= 0.10.

Usage:
    python train_inst3_crop_robust.py
    python train_inst3_crop_robust.py --resume <inst2_ckpt_path>
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pprint

from train_v4 import CONFIG as V4_CONFIG, train


INST3_OVERRIDES = {
    "resume_path": "checkpoints_tifs/inst2_jpeg_robust/dscsp_v4_final.pt",
    "resume_load_optimizer": False,
    "resume_load_scheduler": False,
    "resume_load_disc_optimizer": False,
    "override_lrs_on_resume": True,

    # ── Architecture (Inst 3 changes) ────────────────────────────────────
    "num_bits": 256,
    "carrier_type": "spread_spectrum",     # Part A
    "spread_seed": 20260424,
    "extractor_mode": "shift_pool",        # Part B
    "shift_pool_mode": "mean",
    "shift_pool_kernel": 5,
    "base_channels": 48,

    # ── Perturbation / loss weights (kept moderate while retraining) ─────
    "max_perturbation": 0.10,
    "lambda_perc": 0.1,
    "lambda_spec": 0.004,
    "lambda_adv": 0.20,
    "use_adversarial": True,
    "adv_start_step": 500,
    "adv_ramp_steps": 2000,
    "disc_steps_per_gen_step": 3,

    # ── Keep JPEG mix from Inst 2 ────────────────────────────────────────
    "jpeg_weight": 0.55,
    "jpeg_quality_range": (50, 85),
    "real_jpeg_quality_range": (50, 85),
    "real_jpeg_prob_start": 0.50,
    "real_jpeg_prob_end": 0.50,
    "real_jpeg_ramp_step": 0,
    "real_jpeg_prob": 0.0,

    "robustness_start_step": 0,            # attacks from step 0 (resuming)
    "robustness_ramp_steps": 1000,
    "robustness_max_prob": 0.70,

    # ── Instruction 3 crop schedule (from step 0) ────────────────────────
    "crop_always_prob": 0.5,
    "crop_ratio_range": (0.65, 0.95),

    # ── Skip bootstraps on resume (already have trained weights) ─────────
    "extractor_bootstrap_steps": 0,
    "controller_bootstrap_steps": 0,
    "bit_curriculum_start_bits": 256,
    "bit_curriculum_end_bits": 256,
    "bit_curriculum_steps": 0,

    "use_combined_attacks": False,

    "lr_controller": 7e-5,
    "lr_extractor": 2.0e-4,
    "lr_min": 1e-6,

    "total_steps": 6000,
    "batch_size": 4,
    "checkpoint_dir": "checkpoints_tifs/inst3_crop_robust",
    "output_dir": "outputs_tifs/inst3_crop_robust",
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
    parser.add_argument("--crop-always-prob", type=float, default=None)
    parser.add_argument("--shift-pool-mode", choices=["max", "mean"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = build_config(INST3_OVERRIDES)
    if args.resume is not None:
        cfg["resume_path"] = args.resume
    if args.total_steps is not None:
        cfg["total_steps"] = args.total_steps
    if args.crop_always_prob is not None:
        cfg["crop_always_prob"] = args.crop_always_prob
    if args.shift_pool_mode is not None:
        cfg["shift_pool_mode"] = args.shift_pool_mode

    print("=" * 72)
    print("Instruction 3 — crop robustness retrain (spread-spectrum + shift_pool)")
    print("=" * 72)
    highlights = [
        "resume_path", "total_steps", "num_bits",
        "carrier_type", "spread_seed",
        "extractor_mode", "shift_pool_mode", "shift_pool_kernel",
        "max_perturbation",
        "jpeg_weight", "jpeg_quality_range",
        "real_jpeg_prob_start", "real_jpeg_prob_end", "real_jpeg_ramp_step",
        "robustness_start_step", "robustness_max_prob",
        "crop_always_prob", "crop_ratio_range",
        "lr_controller", "lr_extractor",
        "checkpoint_dir", "output_dir",
    ]
    pprint.pprint({k: cfg[k] for k in highlights if k in cfg}, width=100)
    print("=" * 72, flush=True)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "inst3_config.json"), "w") as f:
        serialisable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
        json.dump(serialisable, f, indent=2, default=str)

    if args.dry_run:
        print("[dry-run] exiting before train().")
        return 0

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
