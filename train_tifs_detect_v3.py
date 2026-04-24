"""Detect-v3 — final AUC push.

detect_v2 hit AUC 0.716 (lower CI 0.67, gate 0.70). This run pushes the
point estimate below 0.70 by tightening ε further and ramping
mode-spread and SRM-match pressure. Extends detect_v2 by 6000 steps.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pprint

from train_v4 import CONFIG as V4_CONFIG, train


DETECT_V3_OVERRIDES = {
    "resume_path": "checkpoints_tifs/detect_v2/dscsp_v4_final.pt",
    "resume_load_optimizer": False,
    "resume_load_scheduler": False,
    "resume_load_disc_optimizer": False,
    "override_lrs_on_resume": True,

    "num_bits": 256,
    "carrier_type": "nearest",
    "extractor_mode": "spatial_map",
    "extractor_grid_size": 16,
    "base_channels": 48,

    # ── Even tighter ε ──────────────────────────────────────────────────
    "max_perturbation": 0.025,
    "lambda_perc": 0.5,
    "lambda_spec": 0.03,

    # detect_v2 final was at step 30000
    "use_adversarial": True,
    "lambda_adv": 1.0,
    "adv_start_step": 30000,
    "adv_ramp_steps": 300,
    "disc_steps_per_gen_step": 10,
    "lambda_gp": 15.0,
    "lambda_spread": 0.6,
    "lambda_spread_start_step": 30100,
    "lambda_spread_k": 8,
    "lambda_srm_match": 0.3,

    # No attacks — pure detectability push
    "jpeg_weight": 0.0,
    "real_jpeg_prob_start": 0.0,
    "real_jpeg_prob_end": 0.0,
    "real_jpeg_ramp_step": 0,
    "real_jpeg_prob": 0.0,
    "robustness_start_step": 999999,
    "robustness_ramp_steps": 1,
    "robustness_max_prob": 0.0,
    "crop_always_prob": 0.0,

    "extractor_bootstrap_steps": 0,
    "controller_bootstrap_steps": 0,
    "bit_curriculum_start_bits": 256,
    "bit_curriculum_end_bits": 256,
    "bit_curriculum_steps": 0,

    "use_combined_attacks": False,

    "batch_size": 8,
    "lr_controller": 2e-5,
    "lr_extractor": 6e-5,
    "disc_lr": 2e-4,
    "lr_min": 1e-6,

    "total_steps": 36000,  # 6000 fresh
    "checkpoint_dir": "checkpoints_tifs/detect_v3",
    "output_dir": "outputs_tifs/detect_v3",
    "checkpoint_interval": 2000,
    "log_interval": 50,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--max-perturbation", type=float, default=None)
    parser.add_argument("--lambda-spread", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = copy.deepcopy(V4_CONFIG)
    cfg.update(DETECT_V3_OVERRIDES)

    if args.total_steps is not None:
        cfg["total_steps"] = args.total_steps
    if args.max_perturbation is not None:
        cfg["max_perturbation"] = args.max_perturbation
    if args.lambda_spread is not None:
        cfg["lambda_spread"] = args.lambda_spread
    if args.resume is not None:
        cfg["resume_path"] = args.resume

    print("=" * 72)
    print("Detect-v3 — final AUC push (ε=0.025, λ_spread=0.6, λ_srm=0.3)")
    print("=" * 72)
    highlights = [
        "resume_path", "total_steps", "max_perturbation",
        "lambda_adv", "adv_start_step", "disc_steps_per_gen_step", "lambda_gp",
        "lambda_spread", "lambda_srm_match", "lambda_spec",
        "jpeg_weight", "robustness_max_prob", "crop_always_prob",
        "lr_controller", "lr_extractor", "disc_lr",
        "checkpoint_dir", "output_dir",
    ]
    pprint.pprint({k: cfg[k] for k in highlights if k in cfg}, width=100)
    print("=" * 72, flush=True)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "detect_v3_config.json"), "w") as f:
        serialisable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
        json.dump(serialisable, f, indent=2, default=str)

    if args.dry_run:
        print("[dry-run] exiting before train().")
        return 0

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
