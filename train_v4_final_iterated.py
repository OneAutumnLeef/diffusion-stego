"""
Iterated final training orchestration for DSCSP v4.

This script runs multi-stage training with conservative-to-strong robustness
schedules and optional quick evaluations after each stage. It is designed for
final target-driven training on high-VRAM GPUs (e.g., RTX 5090).

Usage:
    python train_v4_final_iterated.py
    python train_v4_final_iterated.py --resume checkpoints_v4_5090/s1_stability/dscsp_v4_final.pt
    python train_v4_final_iterated.py --quick-eval-samples 256
"""

import argparse
import copy
import json
import os
from typing import Dict, List

import torch

import train_v4


TARGETS = {
    "clean_ber": 1e-4,
    "jpeg_q75_ber": 0.01,
    "jpeg_q50_ber": 0.05,
    "psnr": 35.0,
    "ssim": 0.97,
    "lpips": 0.05,
}


BASE_5090_OVERRIDES = {
    # Keep proven optimization dynamics from 3060 runs.
    "batch_size": 4,
    "base_channels": 48,
    "lr_controller": 1e-4,
    "lr_extractor": 3e-4,
    "disc_lr": 1e-4,
    "pool_size": 128,
    "pool_refresh_interval": 2000,
    "checkpoint_interval": 1000,
    "save_image_interval": 250,
    "lambda_adv": 0.5,
    "disc_steps_per_gen_step": 3,
    "jpeg_weight": 0.6,
    "robustness_max_prob": 0.85,
    "lambda_spec": 0.01,
}


FINAL_STAGES: List[Dict] = [
    {
        "name": "s1_msg_warmup",
        "end_step": 12000,
        "overrides": {
            "max_perturbation": 0.12,
            "lambda_perc": 0.0,
            "lambda_spec": 0.0,
            "lambda_adv": 0.0,
            "use_adversarial": False,
            "adv_start_step": 10000,
            "adv_ramp_steps": 4000,
            "robustness_start_step": 7000,
            "robustness_ramp_steps": 3000,
            "robustness_max_prob": 0.45,
            "jpeg_weight": 0.40,
            "jpeg_quality_range": (55, 95),
            "use_combined_attacks": False,
            "bit_curriculum_start_bits": 32,
            "bit_curriculum_end_bits": 256,
            "bit_curriculum_steps": 9000,
            "bit_curriculum_start_step": 1,
        },
    },
    {
        "name": "s2_robust_start",
        "end_step": 32000,
        "overrides": {
            "max_perturbation": 0.09,
            "lambda_perc": 0.15,
            "lambda_adv": 0.4,
            "use_adversarial": True,
            "adv_start_step": 3000,
            "adv_ramp_steps": 5000,
            "robustness_start_step": 1500,
            "robustness_ramp_steps": 3000,
            "robustness_max_prob": 0.80,
            "jpeg_weight": 0.65,
            "jpeg_quality_range": (40, 95),
            "use_combined_attacks": True,
            "combined_attack_start_step": 18000,
            "combined_attack_ratio": 0.15,
            "n_combined": 2,
            "lambda_spec": 0.006,
            "bit_curriculum_start_bits": 256,
            "bit_curriculum_end_bits": 256,
            "bit_curriculum_steps": 0,
            "bit_curriculum_start_step": 1,
        },
    },
    {
        "name": "s3_jpeg_harden",
        "end_step": 52000,
        "overrides": {
            "lambda_adv": 0.55,
            "disc_steps_per_gen_step": 3,
            "robustness_max_prob": 0.88,
            "jpeg_weight": 0.72,
            "jpeg_quality_range": (35, 90),
            "combined_attack_ratio": 0.25,
            "n_combined": 2,
            "lambda_spec": 0.010,
        },
    },
    {
        "name": "s4_finalize",
        "end_step": 72000,
        "overrides": {
            "lambda_adv": 0.60,
            "disc_steps_per_gen_step": 4,
            "robustness_max_prob": 0.9,
            "jpeg_weight": 0.78,
            "jpeg_quality_range": (30, 85),
            "combined_attack_ratio": 0.35,
            "n_combined": 3,
            "lambda_spec": 0.015,
            "lr_controller": 7e-5,
            "lr_extractor": 2e-4,
            "disc_lr": 8e-5,
        },
    },
]


def _is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except UnicodeDecodeError:
        return False
    except OSError:
        return False


def _quick_eval(checkpoint_path: str, num_samples: int) -> Dict:
    import evaluate_v4 as ev

    device = "cuda" if torch.cuda.is_available() else "cpu"
    controller, extractor, config = ev.load_v4_model(checkpoint_path, device)
    num_bits = config["num_bits"]

    diffusion = ev.DiffusionWrapper(
        model_id=config.get("model_id", "google/ddpm-celebahq-256"),
        device=device,
        num_inference_steps=config.get("ddim_steps", 20),
    )

    n_total = max(num_samples, 64)
    images = ev.generate_images(diffusion, n_total, device)

    clean = ev.eval_clean(controller, extractor, images[:num_samples], device, num_bits)
    jpeg = ev.eval_jpeg(
        controller,
        extractor,
        images[:num_samples],
        device,
        [50, 75],
        num_bits,
    )

    del controller, extractor, diffusion, images
    torch.cuda.empty_cache()

    return {
        "clean": clean,
        "jpeg": jpeg,
        "targets": TARGETS,
    }


def _meets_targets(results: Dict) -> bool:
    clean = results.get("clean", {})
    jpeg = results.get("jpeg", {})

    return (
        clean.get("ber", 1.0) <= TARGETS["clean_ber"]
        and jpeg.get("jpeg_q75", 1.0) < TARGETS["jpeg_q75_ber"]
        and jpeg.get("jpeg_q50", 1.0) < TARGETS["jpeg_q50_ber"]
        and clean.get("psnr", 0.0) >= TARGETS["psnr"]
        and clean.get("ssim", 0.0) >= TARGETS["ssim"]
        and clean.get("lpips", 1.0) <= TARGETS["lpips"]
    )


def _build_stage_config(
    stage: Dict,
    output_root: str,
    checkpoint_root: str,
    resume_path: str,
) -> Dict:
    cfg = copy.deepcopy(train_v4.CONFIG)
    cfg.update(BASE_5090_OVERRIDES)
    cfg.update(stage["overrides"])

    # Allow stages to specify head_type (shift_pool/spatial_map/global_fc) as an alias
    # for extractor_mode so stage configs stay readable without knowing the internal key.
    if "head_type" in cfg:
        cfg["extractor_mode"] = cfg.pop("head_type")

    cfg["total_steps"] = stage["end_step"]
    cfg["output_dir"] = os.path.join(output_root, stage["name"])
    cfg["checkpoint_dir"] = os.path.join(checkpoint_root, stage["name"])

    if resume_path:
        cfg["resume_path"] = resume_path
        # For stage transitions, keep model weights but reinitialize optimizer/scheduler.
        cfg["resume_load_optimizer"] = False
        cfg["resume_load_scheduler"] = False
        cfg["resume_load_disc_optimizer"] = False
        cfg["override_lrs_on_resume"] = True
    else:
        cfg["resume_path"] = ""

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterated final DSCSP v4 training")
    parser.add_argument("--resume", default="", help="Checkpoint to resume from")
    parser.add_argument(
        "--output-root",
        default="outputs_v4_5090_final",
        help="Root output folder for stage logs and images",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints_v4_5090_final",
        help="Root folder for stage checkpoints",
    )
    parser.add_argument(
        "--quick-eval-samples",
        type=int,
        default=128,
        help="Samples for quick post-stage eval",
    )
    parser.add_argument(
        "--skip-quick-eval",
        action="store_true",
        help="Skip quick eval after each stage",
    )
    parser.add_argument(
        "--stop-on-goal",
        action="store_true",
        help="Stop early when quick eval reaches targets",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        help="1-based stage index to start from",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_stage < 1 or args.start_stage > len(FINAL_STAGES):
        raise ValueError(f"start-stage must be in [1, {len(FINAL_STAGES)}]")

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.checkpoint_root, exist_ok=True)

    resume_path = args.resume.strip()
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        if _is_lfs_pointer(resume_path):
            raise RuntimeError(
                "Resume checkpoint looks like a Git LFS pointer file. "
                "Run 'git lfs pull' first."
            )

    summary = {
        "targets": TARGETS,
        "stages": [],
    }

    print("=" * 72)
    print("DSCSP v4 Iterated Final Training")
    print("=" * 72)
    print(f"Output root: {args.output_root}")
    print(f"Checkpoint root: {args.checkpoint_root}")
    print(f"Start stage: {args.start_stage}")
    if resume_path:
        print(f"Initial resume: {resume_path}")
    print()

    for idx, stage in enumerate(FINAL_STAGES, start=1):
        if idx < args.start_stage:
            continue

        print("-" * 72)
        print(f"Stage {idx}/{len(FINAL_STAGES)}: {stage['name']}")
        print(f"End step: {stage['end_step']}")
        print("-" * 72)

        cfg = _build_stage_config(
            stage=stage,
            output_root=args.output_root,
            checkpoint_root=args.checkpoint_root,
            resume_path=resume_path,
        )

        train_v4.train(cfg)

        final_ckpt = os.path.join(cfg["checkpoint_dir"], "dscsp_v4_final.pt")
        if not os.path.exists(final_ckpt):
            raise RuntimeError(f"Stage final checkpoint missing: {final_ckpt}")

        stage_record = {
            "stage": stage["name"],
            "checkpoint": final_ckpt,
            "config": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in cfg.items()
            },
        }

        if not args.skip_quick_eval:
            print("Running quick post-stage evaluation...")
            quick = _quick_eval(final_ckpt, args.quick_eval_samples)
            stage_record["quick_eval"] = quick
            print(
                "Quick eval: "
                f"clean BER={quick['clean']['ber']:.6f}, "
                f"PSNR={quick['clean']['psnr']:.2f}, "
                f"JPEG Q75 BER={quick['jpeg']['jpeg_q75']:.6f}, "
                f"JPEG Q50 BER={quick['jpeg']['jpeg_q50']:.6f}"
            )

            if args.stop_on_goal and _meets_targets(quick):
                print("Quick-eval targets met. Stopping iterative training early.")
                summary["stages"].append(stage_record)
                summary["goal_met"] = True
                summary["final_checkpoint"] = final_ckpt
                break

        summary["stages"].append(stage_record)
        summary["goal_met"] = False
        summary["final_checkpoint"] = final_ckpt

        # Resume from the final checkpoint for the next stage.
        resume_path = final_ckpt

    summary_path = os.path.join(args.output_root, "iterated_final_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 72)
    print("Iterated final training complete")
    print(f"Summary: {summary_path}")
    print(f"Final checkpoint: {summary.get('final_checkpoint', 'N/A')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
