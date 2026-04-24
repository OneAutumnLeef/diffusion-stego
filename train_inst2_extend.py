"""Instruction 2 fallback — extend JPEG-robust training to 15000 steps.

The 8000-step Instruction-2 run dropped real-libjpeg Q75 BER from ~0.34 to
~0.21 — progress, but still above the 0.10 gate. Per the instruction set,
extend to 15000 steps before considering the more aggressive p_real=0.70
retrain-from-scratch fallback.

Resumes from the 8000-step final checkpoint with fresh optimizer/scheduler
state and 7000 additional steps of training under the same mixed-JPEG
schedule (p_real held at 0.50). LR is configured for fine-tuning magnitudes.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pprint

from train_v4 import CONFIG as V4_CONFIG, train
from train_inst2_jpeg_robust import INST2_OVERRIDES


EXTEND_OVERRIDES = {
    # Resume from the 8000-step checkpoint, reset schedules
    "resume_path": "checkpoints_tifs/inst2_jpeg_robust/dscsp_v4_final.pt",
    "resume_load_optimizer": False,
    "resume_load_scheduler": False,
    "resume_load_disc_optimizer": False,
    "override_lrs_on_resume": True,

    # Skip bootstraps (already trained)
    "extractor_bootstrap_steps": 0,
    "controller_bootstrap_steps": 0,

    # Fresh fine-tune LR + reduced cosine amplitude
    "lr_controller": 5e-5,
    "lr_extractor": 1.5e-4,
    "lr_min": 1e-6,

    # 15000 total steps = 8000 already trained + 7000 new
    "total_steps": 15000,

    # Hold the JPEG mix at the mature p_real=0.5
    "real_jpeg_prob_start": 0.5,
    "real_jpeg_prob_end": 0.5,
    "real_jpeg_ramp_step": 0,

    # Keep attack prob at max from step 0 of resume, push attacks harder
    "robustness_start_step": 0,
    "robustness_ramp_steps": 500,
    "robustness_max_prob": 0.75,   # up from 0.45 — push JPEG pressure
    "jpeg_weight": 0.75,           # up from 0.60 — concentrate on JPEG

    # Turn off bit curriculum (use full 256 bits)
    "bit_curriculum_start_bits": 256,
    "bit_curriculum_end_bits": 256,
    "bit_curriculum_steps": 0,

    "checkpoint_dir": "checkpoints_tifs/inst2_extend",
    "output_dir": "outputs_tifs/inst2_extend",
    "checkpoint_interval": 2000,
    "log_interval": 50,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = copy.deepcopy(V4_CONFIG)
    cfg.update(INST2_OVERRIDES)   # base Instruction-2 config
    cfg.update(EXTEND_OVERRIDES)  # extension deltas win

    if args.resume is not None:
        cfg["resume_path"] = args.resume
    if args.total_steps is not None:
        cfg["total_steps"] = args.total_steps

    print("=" * 72)
    print("Instruction 2 extension — 8000 -> 15000 steps")
    print("=" * 72)
    highlights = [
        "resume_path", "resume_load_optimizer", "resume_load_scheduler",
        "total_steps", "lr_controller", "lr_extractor",
        "real_jpeg_prob_start", "real_jpeg_prob_end",
        "robustness_start_step", "robustness_max_prob",
        "checkpoint_dir", "output_dir",
    ]
    pprint.pprint({k: cfg[k] for k in highlights if k in cfg}, width=100)
    print("=" * 72, flush=True)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "inst2_extend_config.json"), "w") as f:
        serialisable = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
        json.dump(serialisable, f, indent=2, default=str)

    if args.dry_run:
        print("[dry-run] exiting before train().")
        return 0

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
