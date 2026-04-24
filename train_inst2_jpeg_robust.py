"""Instruction 2 driver — JPEG robustness retrain.

From a clean initialization, runs the ``ber_recovery_curriculum`` variant
of train_v4 for 8000 steps with a mixed DiffJPEG + real-libjpeg-STE attack
path. ``p_real`` ramps 0.20 -> 0.50 at step 2000. Quality range [50, 85]
for both the DiffJPEG and real-libjpeg branches.

Target gate: real-libjpeg Q75 BER <= 0.10.

Usage:
    python train_inst2_jpeg_robust.py
    python train_inst2_jpeg_robust.py --total-steps 15000    # extended run
    python train_inst2_jpeg_robust.py --p-real-end 0.70      # fallback schedule
    python train_inst2_jpeg_robust.py --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pprint

from train_v4 import CONFIG as V4_CONFIG, train


INST2_OVERRIDES = {
    # ── Resume target: NONE (clean init, per Instruction 2) ──────────────
    "resume_path": "",

    # ── Architecture ─────────────────────────────────────────────────────
    "num_bits": 256,
    "base_channels": 48,
    "extractor_mode": "spatial_map",

    # ── ber_recovery_curriculum variant defaults ─────────────────────────
    "extractor_bootstrap_steps": 700,
    "extractor_bootstrap_perturbation": 0.16,
    "extractor_bootstrap_attack_prob": 0.1,
    "controller_bootstrap_steps": 700,
    "controller_bootstrap_align_weight": 16.0,
    "controller_bootstrap_attack_prob": 0.1,
    "controller_bootstrap_freeze_extractor": True,

    "max_perturbation": 0.10,
    "lambda_perc": 0.1,
    "lambda_spec": 0.004,

    # Adversarial is weak but on — the JPEG fix is the primary change
    "use_adversarial": True,
    "lambda_adv": 0.15,
    "adv_start_step": 1200,
    "adv_ramp_steps": 3000,
    "disc_steps_per_gen_step": 3,

    # ── Instruction 2: mixed JPEG attack path ────────────────────────────
    "jpeg_weight": 0.6,
    "jpeg_quality_range": (50, 85),          # both DiffJPEG and real branches
    "real_jpeg_quality_range": (50, 85),
    "real_jpeg_prob_start": 0.20,
    "real_jpeg_prob_end": 0.50,
    "real_jpeg_ramp_step": 2000,
    # Legacy single-value key stays 0 so the dynamic schedule always wins.
    "real_jpeg_prob": 0.0,

    "robustness_start_step": 1000,
    "robustness_ramp_steps": 3000,
    "robustness_max_prob": 0.45,

    # Crop held to the variant default (Instruction 3 will tighten it)
    "crop_ratio_range": (0.7, 0.95),

    "use_combined_attacks": False,
    "bit_curriculum_start_bits": 32,
    "bit_curriculum_end_bits": 256,
    "bit_curriculum_steps": 3000,
    "bit_curriculum_start_step": 1,

    # ── Rates / ops ──────────────────────────────────────────────────────
    "lr_controller": 8e-5,
    "lr_extractor": 2.5e-4,
    "lr_min": 1e-6,

    "total_steps": 8000,
    "batch_size": 4,
    "checkpoint_dir": "checkpoints_tifs/inst2_jpeg_robust",
    "output_dir": "outputs_tifs/inst2_jpeg_robust",
    "checkpoint_interval": 2000,
    "log_interval": 25,
}


def build_config(overrides: dict) -> dict:
    cfg = copy.deepcopy(V4_CONFIG)
    cfg.update(overrides)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--p-real-start", type=float, default=None)
    parser.add_argument("--p-real-end", type=float, default=None)
    parser.add_argument("--p-real-ramp-step", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = build_config(INST2_OVERRIDES)
    if args.total_steps is not None:
        cfg["total_steps"] = args.total_steps
    if args.p_real_start is not None:
        cfg["real_jpeg_prob_start"] = args.p_real_start
    if args.p_real_end is not None:
        cfg["real_jpeg_prob_end"] = args.p_real_end
    if args.p_real_ramp_step is not None:
        cfg["real_jpeg_ramp_step"] = args.p_real_ramp_step
    if args.resume is not None:
        cfg["resume_path"] = args.resume

    print("=" * 72)
    print("Instruction 2 — JPEG robustness retrain (mixed DiffJPEG + real STE)")
    print("=" * 72)
    highlights = [
        "resume_path", "total_steps", "num_bits", "max_perturbation",
        "jpeg_weight", "jpeg_quality_range", "real_jpeg_quality_range",
        "real_jpeg_prob_start", "real_jpeg_prob_end", "real_jpeg_ramp_step",
        "robustness_start_step", "robustness_max_prob",
        "lr_controller", "lr_extractor",
        "checkpoint_dir", "output_dir",
    ]
    pprint.pprint({k: cfg[k] for k in highlights if k in cfg}, width=100)
    print("=" * 72, flush=True)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "inst2_config.json"), "w") as f:
        serialisable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
        json.dump(serialisable, f, indent=2, default=str)

    if args.dry_run:
        print("[dry-run] exiting before train().")
        return 0

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
