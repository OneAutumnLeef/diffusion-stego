"""
DSCSP v5 Clean-Domain Calibration.

Loads the quality fine-tuned checkpoint (max_perturbation=0.04) and runs
pure clean training (no attacks, no adversarial) so the extractor learns to
reliably decode 0.04-amplitude signals.

Problem addressed:
  After quality fine-tune, clean BER = 0.13 because the extractor was
  originally calibrated on 0.08-amplitude signals and 0.18-amplitude
  bootstrap. Removing attacks forces the controller/extractor pair to
  converge on the clean domain at the new scale.

After this phase, a short robustness top-up is run (light JPEG attacks) to
restore JPEG Q75 BER without blowing up PSNR.

Usage:
  python finetune_v5_clean.py \
      --checkpoint checkpoints_v5_quality/dscsp_v5_final.pt
"""

import argparse
import copy
import os
import sys

from train_v5 import CONFIG as BASE_CONFIG, train

# Phase 1: pure clean calibration — force extractor to decode at 0.04 amplitude
CLEAN_OVERRIDES = {
    "max_perturbation": 0.04,

    # No attacks at all — pure clean domain
    "robustness_max_prob": 0.0,
    "use_combined_attacks": False,

    # No adversarial — focus entirely on msg + perceptual
    "use_adversarial": False,
    "lambda_adv": 0.0,

    # Strong perceptual + delta penalty to hold PSNR gains
    "lambda_perc": 2.0,
    "lambda_spec": 0.05,
    "lambda_srm_match": 0.0,   # no discriminator, no SRM layer
    "lambda_ctrl": 0.5,

    # Very low LR — fine calibration, not a new training run
    "lr_controller": 1e-5,
    "lr_extractor": 3e-5,
    "lr_min": 1e-8,

    # Bootstrap disabled
    "extractor_bootstrap_steps": 0,
    "controller_bootstrap_steps": 0,

    "pool_refresh_interval": 99999,
    "output_dir": "outputs_v5_clean",
    "checkpoint_dir": "checkpoints_v5_clean",
    "resume_load_optimizer": False,
    "resume_load_scheduler": False,
    "resume_load_disc_optimizer": False,
    "override_lrs_on_resume": True,
    "log_interval": 10,
    "save_image_interval": 500,
    "checkpoint_interval": 1000,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True,
        help="Quality fine-tuned checkpoint (checkpoints_v5_quality/dscsp_v5_final.pt)",
    )
    parser.add_argument(
        "--steps", type=int, default=4000,
        help="Clean calibration steps (default: 4000)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    import torch as _torch
    ckpt_meta = _torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    start_step = int(ckpt_meta.get("step", 0))
    del ckpt_meta
    absolute_total = start_step + args.steps

    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.update(CLEAN_OVERRIDES)
    cfg["resume_path"] = args.checkpoint
    cfg["total_steps"] = absolute_total

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    print("=" * 60)
    print("DSCSP v5 Clean-Domain Calibration")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint} (step {start_step})")
    print(f"Steps: {args.steps} clean-only → total_steps={absolute_total}")
    print(f"max_perturbation: {cfg['max_perturbation']}  attacks: NONE  adv: OFF")
    print()

    train(cfg)


if __name__ == "__main__":
    main()
