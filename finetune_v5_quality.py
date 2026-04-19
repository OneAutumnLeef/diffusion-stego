"""
DSCSP v5 Quality Fine-Tuning.

Loads a trained v5 checkpoint and runs an additional phase that prioritises
image quality (PSNR) while maintaining message robustness.

Problem addressed:
  The main v5 training saturates max_perturbation=0.08 uniformly (PSNR≈28 dB).
  This phase raises lambda_perc to 2.0 and reduces attack_prob_max to 0.40 so
  the controller learns to concentrate perturbation in textured regions rather
  than flooding every pixel.

Target: PSNR >= 35 dB with BER(clean) ≈ 0.000 and JPEG Q75 BER < 0.10.

Usage:
  python finetune_v5_quality.py \
      --checkpoint checkpoints_v5/dscsp_v5_final.pt \
      --steps 8000
"""

import argparse
import copy
import os
import sys

# Import train_v5 components
from train_v5 import CONFIG as BASE_CONFIG, train

# Quality fine-tune overrides
QUALITY_OVERRIDES = {
    # ── PSNR fix: reduce max_perturbation 0.08 → 0.04 ──────────────────────
    # max_perturbation is a plain float attribute (not nn.Parameter), so it is
    # NOT saved/restored by state_dict. The controller is re-initialised here
    # with 0.04, loaded with old weights, and immediately outputs tanh(x)*0.04
    # instead of tanh(x)*0.08 → PSNR jumps from 28 dB to ~34 dB on step 1.
    # The extractor and controller then fine-tune to the new scale.
    "max_perturbation": 0.04,    # was 0.08 → halved amplitude for PSNR ≥ 35

    # Dramatically increase perceptual penalty to force smaller perturbations
    "lambda_perc": 2.0,          # was 0.3 — 6× higher
    "lambda_spec": 0.05,         # was 0.01 — penalise freq artifacts harder
    "lambda_srm_match": 0.2,     # was 0.05 — enforce SRM stealth harder
    "lambda_ctrl": 1.0,          # explicit L2 delta penalty (was 0.0)

    # Reduce attack pressure so model can trade some robustness for quality
    "robustness_max_prob": 0.40, # was 0.85 — fewer attacked batches
    "jpeg_weight": 0.70,         # was 0.80 — slightly less JPEG focus
    "use_combined_attacks": False,  # disable crop+JPEG compound attacks

    # Keep adversarial training but lighter — JPEG-reliance fix:
    # high lambda_adv caused controller to embed in JPEG artifacts (clean BER 0.18)
    "lambda_adv": 0.15,            # was 0.5 — reduce adversarial pressure 3×
    "disc_steps_per_gen_step": 2,  # was 4 — less discriminator pressure

    # Lower LR for fine-tuning stability
    "lr_controller": 3e-5,
    "lr_extractor": 1e-4,
    "lr_min": 1e-7,

    # Bootstrap disabled (already done)
    "extractor_bootstrap_steps": 0,
    "controller_bootstrap_steps": 0,

    # Pool refresh less often during short run
    "pool_refresh_interval": 99999,

    # Outputs
    "output_dir": "outputs_v5_quality",
    "checkpoint_dir": "checkpoints_v5_quality",

    # Resume settings
    "resume_load_optimizer": False,   # fresh optimizer — old momentum stale
    "resume_load_scheduler": False,
    "resume_load_disc_optimizer": False,
    "override_lrs_on_resume": True,

    # Logging
    "log_interval": 10,
    "save_image_interval": 500,
    "checkpoint_interval": 2000,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained v5 checkpoint (e.g. checkpoints_v5/dscsp_v5_final.pt)",
    )
    parser.add_argument(
        "--steps", type=int, default=12000,
        help="Number of quality fine-tuning steps (default: 12000)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Peek at checkpoint to get the saved step, then compute absolute total_steps.
    # train() uses total_steps as an absolute value, not "steps to add", so without
    # this correction the training loop would run 0 iterations when resuming from
    # step 50000 with total_steps=8000.
    import torch as _torch
    ckpt_meta = _torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    start_step = int(ckpt_meta.get("step", 0))
    del ckpt_meta
    absolute_total = start_step + args.steps

    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.update(QUALITY_OVERRIDES)
    cfg["resume_path"] = args.checkpoint
    cfg["total_steps"] = absolute_total

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    print("=" * 60)
    print("DSCSP v5 Quality Fine-Tuning")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint} (step {start_step})")
    print(f"Steps: {args.steps} → total_steps={absolute_total}")
    print(f"lambda_perc: {cfg['lambda_perc']} (was 0.3)")
    print(f"attack_prob_max: {cfg['robustness_max_prob']} (was 0.85)")
    print(f"combined_attacks: disabled")
    print()

    train(cfg)


if __name__ == "__main__":
    main()
