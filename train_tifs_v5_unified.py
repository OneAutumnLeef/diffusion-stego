"""V5 unified TIFS driver — RedundantCoder + shift_pool + detectability stack.

Simpler, faster path than stacking spread-spectrum on top of a no-redundancy
v4 checkpoint. Uses:

  * RedundantCoder 128 msg -> 256 phys (R=2)   crop robustness via spatial
                                               replicas + majority vote
  * nearest-grid carrier                       proven to train, compatible
                                               with the existing extractor
  * shift_pool MEAN head, kernel=9             ±4 pixel shift tolerance;
                                               wider support than ±2
  * real-libjpeg STE mix at p_real=0.5         (retains Inst-2 JPEG gate)
  * always-on crop-resize pre-attack           with ratio (0.65, 0.95)
  * L_spread + L_srm_match + GP + adversary   (Inst 4 detectability)
  * epsilon_max = 0.08                         relaxed from 0.06 for stability

Trains from a clean init (extractor / controller bootstraps run). Target:
all 4 gates in ~12000 steps at batch 8 on the 5090 (~25 min).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pprint

from train_v5 import CONFIG as V5_CONFIG, train


V5_UNIFIED_OVERRIDES = {
    # Clean init (bootstraps train extractor/controller from scratch)
    "resume_path": "",

    # ── Payload: 128 message -> 256 physical (2x redundancy) ─────────────
    "num_bits": 256,
    "message_bits": 128,
    "use_redundant_coding": True,
    "redundancy_factor": 2,

    # ── Architecture ─────────────────────────────────────────────────────
    # Global FC head (ResNet-18 backbone + GAP + Linear to num_bits).
    # GAP is inherently translation-invariant — handles arbitrary
    # crop-resize shifts, unlike the spatial_map head which collapses at
    # shifts beyond the pool kernel radius.
    "carrier_type": "nearest",
    "extractor_mode": "global_fc",
    "base_channels": 48,

    # ── Perturbation budget ─────────────────────────────────────────────
    "max_perturbation": 0.08,
    "lambda_perc": 0.15,
    "lambda_spec": 0.006,

    # ── Detectability ───────────────────────────────────────────────────
    # Adversarial and mode-spread come online late, after BER is stable
    "use_adversarial": True,
    "lambda_adv": 0.4,
    "adv_start_step": 5000,
    "adv_ramp_steps": 3000,
    "disc_steps_per_gen_step": 3,
    "lambda_gp": 10.0,
    "lambda_spread": 0.10,
    "lambda_spread_start_step": 6000,
    "lambda_spread_k": 8,
    "lambda_srm_match": 0.05,

    # ── JPEG mix (Inst 2) ───────────────────────────────────────────────
    "jpeg_weight": 0.6,
    "jpeg_quality_range": (50, 85),
    "real_jpeg_quality_range": (50, 85),
    "real_jpeg_prob_start": 0.2,
    "real_jpeg_prob_end": 0.5,
    "real_jpeg_ramp_step": 2000,
    "real_jpeg_prob": 0.0,

    # Attacks ramp in gently after bootstraps so the extractor isn't
    # overwhelmed before it converges on clean decoding.
    "robustness_start_step": 1500,
    "robustness_ramp_steps": 3000,
    "robustness_max_prob": 0.65,

    # Crop-always pre-attack, starts gentler (0.80-0.95) during warmup
    "crop_always_prob": 0.3,
    "crop_ratio_range": (0.70, 0.95),

    # Bootstraps (from-scratch init) — longer for global_fc which must
    # memorize 256 position->bit associations globally.
    "extractor_bootstrap_steps": 800,
    "extractor_bootstrap_perturbation": 0.16,
    "extractor_bootstrap_attack_prob": 0.0,
    "controller_bootstrap_steps": 700,
    "controller_bootstrap_align_weight": 16.0,
    "controller_bootstrap_attack_prob": 0.0,
    "controller_bootstrap_freeze_extractor": True,

    "use_combined_attacks": False,
    "bit_curriculum_start_bits": 256,
    "bit_curriculum_end_bits": 256,
    "bit_curriculum_steps": 0,

    # ── Batch 8 on 5090 ─────────────────────────────────────────────────
    "batch_size": 8,

    "lr_controller": 1e-4,
    "lr_extractor": 3e-4,
    "disc_lr": 1e-4,
    "lr_min": 1e-6,

    "total_steps": 12000,
    "checkpoint_dir": "checkpoints_tifs/v5_unified",
    "output_dir": "outputs_tifs/v5_unified",
    "checkpoint_interval": 2000,
    "log_interval": 50,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-perturbation", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = copy.deepcopy(V5_CONFIG)
    cfg.update(V5_UNIFIED_OVERRIDES)

    if args.total_steps is not None:
        cfg["total_steps"] = args.total_steps
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.max_perturbation is not None:
        cfg["max_perturbation"] = args.max_perturbation
    if args.resume is not None:
        cfg["resume_path"] = args.resume

    print("=" * 72)
    print("V5 unified TIFS training (RedundantCoder + shift_pool + detectability)")
    print("=" * 72)
    highlights = [
        "resume_path", "total_steps", "batch_size",
        "num_bits", "message_bits", "redundancy_factor",
        "carrier_type", "extractor_mode", "shift_pool_mode", "shift_pool_kernel",
        "max_perturbation",
        "lambda_adv", "disc_steps_per_gen_step", "lambda_gp",
        "lambda_spread", "lambda_srm_match",
        "jpeg_weight", "real_jpeg_prob_start", "real_jpeg_prob_end",
        "robustness_start_step", "robustness_max_prob",
        "crop_always_prob", "crop_ratio_range",
        "lr_controller", "lr_extractor",
        "checkpoint_dir", "output_dir",
    ]
    pprint.pprint({k: cfg[k] for k in highlights if k in cfg}, width=100)
    print("=" * 72, flush=True)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "v5_unified_config.json"), "w") as f:
        serialisable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
        json.dump(serialisable, f, indent=2, default=str)

    if args.dry_run:
        print("[dry-run] exiting before train().")
        return 0

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
