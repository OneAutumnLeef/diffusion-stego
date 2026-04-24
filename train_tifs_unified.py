"""Unified TIFS training driver — stacks Inst 2 / 3 / 4 changes in one run.

Resumes from the Instruction-2-extend checkpoint (JPEG-robust) and enables,
simultaneously, the crop-robustness and detectability mechanisms:

  * Spread-spectrum carrier + shift_pool mean head    (Inst 3)
  * Crop-from-step-0 with ratio range (0.65, 0.95)    (Inst 3)
  * Mode-spread regularizer (ramped from step 1000)   (Inst 4A)
  * SRM feature-matching loss                         (Inst 4B)
  * Gradient penalty on discriminator, 6 D / G        (Inst 4C)
  * epsilon_max = 0.06                                (Inst 4D)
  * Batch 16 on the 5090 for stabler gradients
  * Real-libjpeg STE held at 0.50                     (Inst 2)

Target: PASS all 4 gates in a single training run rather than chaining
three separate runs with their own resume overhead.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pprint

from train_v4 import CONFIG as V4_CONFIG, train


UNIFIED_OVERRIDES = {
    # ── Resume from JPEG-robust base ─────────────────────────────────────
    "resume_path": "checkpoints_tifs/inst2_extend/dscsp_v4_final.pt",
    "resume_load_optimizer": False,
    "resume_load_scheduler": False,
    "resume_load_disc_optimizer": False,
    "override_lrs_on_resume": True,

    # ── Architecture (Inst 3 changes) ────────────────────────────────────
    "num_bits": 256,
    "carrier_type": "spread_spectrum",
    "spread_seed": 20260424,
    "extractor_mode": "shift_pool",
    "shift_pool_mode": "mean",
    "shift_pool_kernel": 5,
    "base_channels": 48,

    # ── Tight perturbation budget (Inst 4D) ─────────────────────────────
    # Starts at 0.08 — 0.06 is aggressive; can tighten via --max-perturbation
    "max_perturbation": 0.08,
    "lambda_perc": 0.15,
    "lambda_spec": 0.006,

    # ── Detectability losses (Inst 4) ────────────────────────────────────
    # Delayed: let BER/crop stabilise under the new carrier first, then add
    # detectability pressure. disc_steps=3 keeps wall-clock reasonable while
    # the gradient-penalty adds its own effective D-depth.
    # adv/spread start-steps are ABSOLUTE — set relative to the resumed
    # step (15000) so the first 2000 fresh steps train BER/crop only.
    "use_adversarial": True,
    "lambda_adv": 0.55,
    "adv_start_step": 17000,
    "adv_ramp_steps": 2000,
    "disc_steps_per_gen_step": 3,
    "lambda_gp": 10.0,
    "lambda_spread": 0.15,
    "lambda_spread_start_step": 18000,
    "lambda_spread_k": 8,
    "lambda_srm_match": 0.05,

    # ── JPEG mix carried over from Inst 2 ────────────────────────────────
    "jpeg_weight": 0.50,
    "jpeg_quality_range": (50, 85),
    "real_jpeg_quality_range": (50, 85),
    "real_jpeg_prob_start": 0.50,
    "real_jpeg_prob_end": 0.50,
    "real_jpeg_ramp_step": 0,
    "real_jpeg_prob": 0.0,

    "robustness_start_step": 0,
    "robustness_ramp_steps": 500,
    "robustness_max_prob": 0.80,

    # ── Crop-from-step-0 (Inst 3) ───────────────────────────────────────
    "crop_always_prob": 0.5,
    "crop_ratio_range": (0.65, 0.95),

    # Skip bootstraps — resuming (controller learned BER-recovery path;
    # extractor still needs to relearn for the new spread-spectrum +
    # shift_pool decoding, but main training handles it).
    "extractor_bootstrap_steps": 0,
    "controller_bootstrap_steps": 0,
    "bit_curriculum_start_bits": 256,
    "bit_curriculum_end_bits": 256,
    "bit_curriculum_steps": 0,

    "use_combined_attacks": False,

    # ── Batch 8: balanced for GP's second-order grads on 5090 ───────────
    "batch_size": 8,

    # Fresh LRs for a mid-regime fine-tune
    "lr_controller": 8e-5,
    "lr_extractor": 2.5e-4,
    "disc_lr": 1e-4,
    "lr_min": 1e-6,

    "total_steps": 12000,
    "checkpoint_dir": "checkpoints_tifs/unified",
    "output_dir": "outputs_tifs/unified",
    "checkpoint_interval": 2000,
    "log_interval": 50,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-perturbation", type=float, default=None)
    parser.add_argument("--lambda-spread", type=float, default=None)
    parser.add_argument("--lambda-gp", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = copy.deepcopy(V4_CONFIG)
    cfg.update(UNIFIED_OVERRIDES)

    if args.resume is not None:
        cfg["resume_path"] = args.resume
    if args.total_steps is not None:
        cfg["total_steps"] = args.total_steps
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.max_perturbation is not None:
        cfg["max_perturbation"] = args.max_perturbation
    if args.lambda_spread is not None:
        cfg["lambda_spread"] = args.lambda_spread
    if args.lambda_gp is not None:
        cfg["lambda_gp"] = args.lambda_gp

    print("=" * 72)
    print("TIFS unified training — Inst 2 + 3 + 4 stacked in one run")
    print("=" * 72)
    highlights = [
        "resume_path", "total_steps", "batch_size",
        "carrier_type", "extractor_mode", "shift_pool_mode",
        "max_perturbation",
        "lambda_adv", "disc_steps_per_gen_step", "lambda_gp",
        "lambda_spread", "lambda_spread_start_step", "lambda_srm_match",
        "jpeg_weight", "jpeg_quality_range", "real_jpeg_prob_start",
        "robustness_max_prob", "crop_always_prob", "crop_ratio_range",
        "lr_controller", "lr_extractor", "disc_lr",
        "checkpoint_dir", "output_dir",
    ]
    pprint.pprint({k: cfg[k] for k in highlights if k in cfg}, width=100)
    print("=" * 72, flush=True)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "unified_config.json"), "w") as f:
        serialisable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
        json.dump(serialisable, f, indent=2, default=str)

    if args.dry_run:
        print("[dry-run] exiting before train().")
        return 0

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
