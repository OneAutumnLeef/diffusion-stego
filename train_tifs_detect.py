"""Targeted detectability fine-tune.

Resumes from the Instruction-2-extend checkpoint (clean BER ~0.01, real
JPEG Q75 BER 0.08 — already passes JPEG gate) and fine-tunes ONLY for
detectability:

  * Mode-spread regularizer (ramped from step 300)
  * SRM feature-matching loss
  * Strong WGAN-GP adversarial (λ_gp = 10, 6 D / G)
  * Tightened ε = 0.06

No crop pressure during training — we accept the crop limitation per the
instruction set's explicit allowance ("whatever it is"). JPEG mix held at
p_real = 0.5 to preserve the JPEG gate.

Target: keep clean BER ≤ 0.01, keep JPEG ≤ 0.10, drive AUC ≤ 0.70.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pprint

from train_v4 import CONFIG as V4_CONFIG, train


DETECT_OVERRIDES = {
    "resume_path": "checkpoints_tifs/inst2_extend/dscsp_v4_final.pt",
    "resume_load_optimizer": False,
    "resume_load_scheduler": False,
    "resume_load_disc_optimizer": False,
    "override_lrs_on_resume": True,

    # Match inst2_extend architecture (no carrier/head change)
    "num_bits": 256,
    "carrier_type": "nearest",
    "extractor_mode": "spatial_map",
    "extractor_grid_size": 16,
    "base_channels": 48,

    # Tighter perturbation budget
    "max_perturbation": 0.06,
    "lambda_perc": 0.15,
    "lambda_spec": 0.008,

    # Strong detectability stack
    "use_adversarial": True,
    "lambda_adv": 0.6,
    "adv_start_step": 15000,            # absolute — resume begins at 15000
    "adv_ramp_steps": 1500,
    "disc_steps_per_gen_step": 6,
    "lambda_gp": 10.0,
    "lambda_spread": 0.15,
    "lambda_spread_start_step": 15300,  # absolute (300 fresh steps after resume)
    "lambda_spread_k": 8,
    "lambda_srm_match": 0.05,

    # Hold JPEG mix so the JPEG gate stays passed
    "jpeg_weight": 0.5,
    "jpeg_quality_range": (50, 85),
    "real_jpeg_quality_range": (50, 85),
    "real_jpeg_prob_start": 0.5,
    "real_jpeg_prob_end": 0.5,
    "real_jpeg_ramp_step": 0,
    "real_jpeg_prob": 0.0,

    # Attacks on from step 0 of resume
    "robustness_start_step": 0,
    "robustness_ramp_steps": 500,
    "robustness_max_prob": 0.55,

    # No crop pressure during training (paper reports crop honestly)
    "crop_always_prob": 0.0,

    # Skip bootstraps
    "extractor_bootstrap_steps": 0,
    "controller_bootstrap_steps": 0,
    "bit_curriculum_start_bits": 256,
    "bit_curriculum_end_bits": 256,
    "bit_curriculum_steps": 0,

    "use_combined_attacks": False,

    "batch_size": 8,
    "lr_controller": 6e-5,
    "lr_extractor": 2e-4,
    "disc_lr": 1e-4,
    "lr_min": 1e-6,

    # 7000 fresh steps (15000 resume + 7000)
    "total_steps": 22000,
    "checkpoint_dir": "checkpoints_tifs/detect_harden",
    "output_dir": "outputs_tifs/detect_harden",
    "checkpoint_interval": 2000,
    "log_interval": 50,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-perturbation", type=float, default=None)
    parser.add_argument("--lambda-spread", type=float, default=None)
    parser.add_argument("--lambda-gp", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = copy.deepcopy(V4_CONFIG)
    cfg.update(DETECT_OVERRIDES)

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
    if args.resume is not None:
        cfg["resume_path"] = args.resume

    print("=" * 72)
    print("Detectability-only fine-tune")
    print("=" * 72)
    highlights = [
        "resume_path", "total_steps", "batch_size", "max_perturbation",
        "lambda_adv", "adv_start_step", "disc_steps_per_gen_step", "lambda_gp",
        "lambda_spread", "lambda_spread_start_step", "lambda_srm_match",
        "jpeg_weight", "real_jpeg_prob_start", "real_jpeg_prob_end",
        "robustness_start_step", "robustness_max_prob", "crop_always_prob",
        "lr_controller", "lr_extractor", "disc_lr",
        "checkpoint_dir", "output_dir",
    ]
    pprint.pprint({k: cfg[k] for k in highlights if k in cfg}, width=100)
    print("=" * 72, flush=True)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "detect_config.json"), "w") as f:
        serialisable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
        json.dump(serialisable, f, indent=2, default=str)

    if args.dry_run:
        print("[dry-run] exiting before train().")
        return 0

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
