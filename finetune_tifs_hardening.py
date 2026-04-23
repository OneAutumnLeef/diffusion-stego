"""TIFS hardening fine-tune driver.

Overlays ``HARDENING_OVERRIDES`` onto ``train_v5.CONFIG`` and resumes training
from the cycle-11 ``robust_balance`` checkpoint. Goal: trade a little clean BER
for substantially better PSNR / detectability under a tightened perturbation
budget and a stronger adversary.

Usage
-----
    python finetune_tifs_hardening.py                 # full run, 4k hardening steps
    python finetune_tifs_hardening.py --resume PATH   # override resume target
    python finetune_tifs_hardening.py --dry-run       # print effective config and exit

Config rationale
----------------
* ``extractor_mode="spatial_map"`` — ``shift_pool`` scrambles the 256-bit
  non-redundant head; see HANDOFF_2026-04-24.md §4.1.
* ``max_perturbation`` dropped to 0.065 — pushes PSNR above 30 dB.
* ``lambda_adv`` raised to 0.7 and ``disc_steps_per_gen_step`` to 4 for
  detectability pressure.
* ``real_jpeg_prob=0.35`` — real libjpeg round-trip via STE closes the
  DiffJPEG/libjpeg gap characterised in Section X of the paper.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pprint
import subprocess
import sys

from train_v5 import CONFIG as V5_CONFIG, train


HARDENING_OVERRIDES = {
    # ── Resume target ────────────────────────────────────────────────────
    "resume_path": "checkpoints_tifs/ber_recovery/major_cycle_011_robust_balance.pt",
    "resume_load_optimizer": False,
    "resume_load_scheduler": False,
    "resume_load_disc_optimizer": False,
    "override_lrs_on_resume": True,

    # ── Capacity / architecture ──────────────────────────────────────────
    "num_bits": 256,
    "message_bits": 256,
    "redundancy_factor": 1,
    "use_redundant_coding": False,
    "extractor_mode": "spatial_map",        # NOT shift_pool — see HANDOFF §4.1
    "extractor_grid_size": 16,
    "base_channels": 48,

    # ── Perturbation budget — tightened for detectability / PSNR ─────────
    "max_perturbation": 0.065,              # down from 0.09 in cycle 11
    "lambda_perc": 0.15,
    "lambda_spec": 0.006,

    # ── Stronger adversary ───────────────────────────────────────────────
    "lambda_adv": 0.7,                      # up from 0.35
    "disc_steps_per_gen_step": 4,           # up from 3
    "adv_start_step": 800,
    "adv_ramp_steps": 3000,

    # ── Real-libjpeg STE + aggressive JPEG ───────────────────────────────
    "jpeg_weight": 0.7,
    "jpeg_quality_range": (60, 92),
    "real_jpeg_prob": 0.35,                 # NEW — see HANDOFF §4.4

    # ── Combined + crop ──────────────────────────────────────────────────
    "combined_attack_ratio": 0.2,
    "combined_attack_start_step": 2500,
    "n_combined": 2,
    "crop_ratio_range": (0.75, 0.95),       # NOT (0.6, 0.9) from broken run 1

    # ── Rates ────────────────────────────────────────────────────────────
    "lr_controller": 7e-5,
    "lr_extractor": 2e-4,
    "lr_min": 1e-6,

    # ── Ops ──────────────────────────────────────────────────────────────
    "total_steps": 27600,                   # = 23600 resume + 4000 hardening
    "batch_size": 4,
    "checkpoint_dir": "checkpoints_tifs/hardening",
    "output_dir": "outputs_tifs/hardening",
    "checkpoint_interval": 1500,
    "log_interval": 25,
}


def build_config(overrides: dict) -> dict:
    cfg = copy.deepcopy(V5_CONFIG)
    cfg.update(overrides)
    return cfg


def _config_summary(cfg: dict) -> str:
    interesting = [
        "resume_path", "total_steps", "batch_size",
        "num_bits", "message_bits", "use_redundant_coding", "extractor_mode",
        "max_perturbation", "lambda_perc", "lambda_spec",
        "lambda_adv", "disc_steps_per_gen_step", "adv_start_step", "adv_ramp_steps",
        "jpeg_weight", "jpeg_quality_range", "real_jpeg_prob",
        "combined_attack_ratio", "combined_attack_start_step", "n_combined",
        "crop_ratio_range",
        "lr_controller", "lr_extractor", "lr_min",
        "checkpoint_dir", "output_dir",
    ]
    view = {k: cfg[k] for k in interesting if k in cfg}
    return pprint.pformat(view, width=100)


def main() -> int:
    parser = argparse.ArgumentParser(description="TIFS hardening fine-tune.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Override resume checkpoint path.")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Override total_steps (resume + hardening).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print effective config and exit without training.")
    args = parser.parse_args()

    cfg = build_config(HARDENING_OVERRIDES)
    if args.resume is not None:
        cfg["resume_path"] = args.resume
    if args.total_steps is not None:
        cfg["total_steps"] = args.total_steps

    print("=" * 72)
    print("DSCSP TIFS hardening fine-tune")
    print("=" * 72)
    print(_config_summary(cfg))
    print("=" * 72, flush=True)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    with open(os.path.join(cfg["output_dir"], "hardening_config.json"), "w") as f:
        serialisable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
        json.dump(serialisable, f, indent=2, default=str)

    if args.dry_run:
        print("[dry-run] exiting before train().")
        return 0

    train(cfg)

    # ── Remedy for the LFS-budget incident ─────────────────────────────────
    # After a successful run, mirror the final hardening checkpoint directly
    # to HF so future pods never need a GitHub LFS pull. Opt-in via HF_TOKEN.
    if os.environ.get("HF_TOKEN"):
        final_ckpt = os.path.join(cfg["checkpoint_dir"], "dscsp_v5_final.pt")
        if os.path.exists(final_ckpt) and os.path.getsize(final_ckpt) > 1_000_000:
            print(f"[hardening] HF_TOKEN detected; mirroring {final_ckpt} to HF.")
            cmd = [sys.executable, "push_checkpoints_to_hf.py",
                   "--include", f"{cfg['checkpoint_dir']}/*.pt"]
            try:
                subprocess.run(cmd, check=False)
            except Exception as exc:
                print(f"[hardening] HF mirror failed (non-fatal): {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
