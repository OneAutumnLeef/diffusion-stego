#!/usr/bin/env python3
"""
Autonomous TIFS-focused training loop for DSCSP.

What it does:
- Iterates training cycles with BER-first and robustness/detectability variants.
- Evaluates each cycle with evaluate_tifs.py logic.
- Writes ONLY major milestones (significant improvements or target hit).
- Resumes from previous state when rerun.

Usage:
  python training_ops/tifs_autopilot.py
  python training_ops/tifs_autopilot.py --max-cycles 20 --steps-per-cycle 3000
"""

import argparse
import copy
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evaluate_tifs
import train_v4


REQUIRED = {
    "detectability_auc_max": 0.70,
    "crop80_ber_max": 0.35,
    "clean_ber_max": 0.10,
    "jpeg_q75_ber_max": 0.20,
}


VARIANTS = [
    {
        "name": "ber_recovery_clean_first",
        "overrides": {
            "extractor_mode": "spatial_map",
            "extractor_bootstrap_steps": 900,
            "extractor_bootstrap_perturbation": 0.18,
            "extractor_bootstrap_attack_prob": 0.0,
            "controller_bootstrap_steps": 900,
            "controller_bootstrap_align_weight": 20.0,
            "controller_bootstrap_attack_prob": 0.0,
            "controller_bootstrap_freeze_extractor": True,
            "max_perturbation": 0.14,
            "lambda_perc": 0.0,
            "lambda_spec": 0.0,
            "lambda_adv": 0.0,
            "use_adversarial": False,
            "adv_start_step": 999999,
            "robustness_start_step": 999999,
            "robustness_max_prob": 0.0,
            "jpeg_weight": 0.0,
            "use_combined_attacks": False,
            "bit_curriculum_start_bits": 16,
            "bit_curriculum_end_bits": 256,
            "bit_curriculum_steps": 2500,
            "bit_curriculum_start_step": 1,
            "lr_controller": 8e-5,
            "lr_extractor": 2.5e-4,
        },
    },
    {
        "name": "ber_recovery_curriculum",
        "overrides": {
            "extractor_mode": "spatial_map",
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
            "lambda_adv": 0.15,
            "use_adversarial": True,
            "adv_start_step": 1200,
            "adv_ramp_steps": 3000,
            "robustness_start_step": 1000,
            "robustness_ramp_steps": 3000,
            "robustness_max_prob": 0.45,
            "jpeg_weight": 0.45,
            "jpeg_quality_range": (50, 95),
            "use_combined_attacks": False,
            "bit_curriculum_start_bits": 32,
            "bit_curriculum_end_bits": 256,
            "bit_curriculum_steps": 3000,
            "bit_curriculum_start_step": 1,
            "lr_controller": 8e-5,
            "lr_extractor": 2.5e-4,
        },
    },
    {
        "name": "robust_balance",
        "overrides": {
            "extractor_mode": "spatial_map",
            "extractor_bootstrap_steps": 400,
            "extractor_bootstrap_perturbation": 0.14,
            "extractor_bootstrap_attack_prob": 0.15,
            "controller_bootstrap_steps": 450,
            "controller_bootstrap_align_weight": 10.0,
            "controller_bootstrap_attack_prob": 0.2,
            "controller_bootstrap_freeze_extractor": False,
            "max_perturbation": 0.09,
            "lambda_perc": 0.15,
            "lambda_spec": 0.006,
            "lambda_adv": 0.35,
            "use_adversarial": True,
            "adv_start_step": 800,
            "adv_ramp_steps": 3000,
            "robustness_start_step": 500,
            "robustness_ramp_steps": 2500,
            "robustness_max_prob": 0.70,
            "jpeg_weight": 0.60,
            "jpeg_quality_range": (40, 95),
            "use_combined_attacks": True,
            "combined_attack_start_step": 2500,
            "combined_attack_ratio": 0.20,
            "n_combined": 2,
            "bit_curriculum_start_bits": 256,
            "bit_curriculum_end_bits": 256,
            "bit_curriculum_steps": 0,
            "bit_curriculum_start_step": 1,
            "lr_controller": 7e-5,
            "lr_extractor": 2.0e-4,
        },
    },
    {
        "name": "detectability_harden",
        "overrides": {
            "extractor_mode": "spatial_map",
            "extractor_bootstrap_steps": 250,
            "extractor_bootstrap_perturbation": 0.12,
            "extractor_bootstrap_attack_prob": 0.2,
            "controller_bootstrap_steps": 250,
            "controller_bootstrap_align_weight": 8.0,
            "controller_bootstrap_attack_prob": 0.25,
            "controller_bootstrap_freeze_extractor": False,
            "max_perturbation": 0.08,
            "lambda_perc": 0.2,
            "lambda_spec": 0.01,
            "lambda_adv": 0.55,
            "use_adversarial": True,
            "adv_start_step": 1,
            "adv_ramp_steps": 1500,
            "disc_steps_per_gen_step": 4,
            "robustness_start_step": 300,
            "robustness_ramp_steps": 2000,
            "robustness_max_prob": 0.85,
            "jpeg_weight": 0.72,
            "jpeg_quality_range": (35, 90),
            "use_combined_attacks": True,
            "combined_attack_start_step": 1500,
            "combined_attack_ratio": 0.30,
            "n_combined": 3,
            "bit_curriculum_start_bits": 256,
            "bit_curriculum_end_bits": 256,
            "bit_curriculum_steps": 0,
            "bit_curriculum_start_step": 1,
            "lr_controller": 6e-5,
            "lr_extractor": 1.8e-4,
            "disc_lr": 8e-5,
        },
    },
]


def _load_json(path: str, default: Dict) -> Dict:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _safe_get_step(checkpoint_path: Optional[str]) -> int:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return int(ckpt.get("step", 0))
    except Exception:
        return 0


def _meets_required(eval_result: Dict, tried_variants: List[str], min_variants: int) -> bool:
    clean_ber = float(eval_result["clean"]["ber"])
    jpeg_q75 = float(eval_result["jpeg"]["q75_ber"])
    crop80 = float(eval_result["crop"]["q80_ber"])
    auc_adapt = float(eval_result["adaptive_steganalysis"]["auc_adaptive"])

    cross_ok = bool(eval_result["areas"]["cross_dataset"]["pass"])
    ablation_ok = len(set(tried_variants)) >= min_variants

    return (
        clean_ber <= REQUIRED["clean_ber_max"]
        and jpeg_q75 <= REQUIRED["jpeg_q75_ber_max"]
        and crop80 <= REQUIRED["crop80_ber_max"]
        and auc_adapt <= REQUIRED["detectability_auc_max"]
        and cross_ok
        and ablation_ok
    )


def _is_significant(prev_best: Optional[Dict], cur_eval: Dict) -> bool:
    if prev_best is None:
        return True

    cur_clean = float(cur_eval["clean"]["ber"])
    cur_crop = float(cur_eval["crop"]["q80_ber"])
    cur_auc = float(cur_eval["adaptive_steganalysis"]["auc_adaptive"])

    best_clean = float(prev_best["clean_ber"])
    best_crop = float(prev_best["crop80_ber"])
    best_auc = float(prev_best["auc_adaptive"])

    clean_imp = cur_clean <= (best_clean - 0.01)
    crop_imp = cur_crop <= (best_crop - 0.02)
    auc_imp = cur_auc <= (best_auc - 0.02)

    pass_transition = (
        cur_auc <= REQUIRED["detectability_auc_max"] < best_auc
        or cur_crop <= REQUIRED["crop80_ber_max"] < best_crop
        or cur_clean <= REQUIRED["clean_ber_max"] < best_clean
    )

    return clean_imp or crop_imp or auc_imp or pass_transition


def _append_major_log(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def _write_ablation_report(history: List[Dict], path: str) -> None:
    by_variant = {}
    for item in history:
        v = item["variant"]
        by_variant.setdefault(v, {"clean": [], "crop": [], "auc": []})
        by_variant[v]["clean"].append(item["clean_ber"])
        by_variant[v]["crop"].append(item["crop80_ber"])
        by_variant[v]["auc"].append(item["auc_adaptive"])

    summary = {"variants": {}, "n_cycles": len(history)}
    for v, vals in by_variant.items():
        summary["variants"][v] = {
            "n": len(vals["clean"]),
            "clean_ber_mean": float(sum(vals["clean"]) / len(vals["clean"])),
            "crop80_ber_mean": float(sum(vals["crop"]) / len(vals["crop"])),
            "auc_adaptive_mean": float(sum(vals["auc"]) / len(vals["auc"])),
        }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _build_cycle_config(
    variant: Dict,
    total_steps: int,
    output_dir: str,
    checkpoint_dir: str,
    steps_per_cycle: int,
    resume_path: str,
) -> Dict:
    cfg = copy.deepcopy(train_v4.CONFIG)
    cfg.update(variant["overrides"])

    cfg["batch_size"] = 4
    cfg["base_channels"] = 48
    cfg["pool_size"] = 128
    cfg["pool_refresh_interval"] = 2000

    cfg["total_steps"] = total_steps
    cfg["output_dir"] = output_dir
    cfg["checkpoint_dir"] = checkpoint_dir

    cfg["checkpoint_interval"] = max(steps_per_cycle, 500)
    cfg["save_image_interval"] = max(steps_per_cycle * 10, 10000)

    cfg["resume_path"] = resume_path
    cfg["resume_load_optimizer"] = False
    cfg["resume_load_scheduler"] = False
    cfg["resume_load_disc_optimizer"] = False
    cfg["override_lrs_on_resume"] = True

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous TIFS-focused DSCSP runner")
    parser.add_argument("--max-cycles", type=int, default=16)
    parser.add_argument("--steps-per-cycle", type=int, default=3000)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--cross-dataset-samples", type=int, default=400)
    parser.add_argument("--adaptive-train", type=int, default=800)
    parser.add_argument("--adaptive-test", type=int, default=200)
    parser.add_argument("--cross-dataset-every", type=int, default=3)
    parser.add_argument("--min-ablation-variants", type=int, default=3)
    parser.add_argument("--output-root", default="outputs_tifs")
    parser.add_argument("--eval-root", default="outputs_tifs_eval")
    parser.add_argument("--checkpoint-root", default="checkpoints_tifs/ber_recovery")
    parser.add_argument("--state-path", default="outputs_tifs/autopilot_state.json")
    parser.add_argument("--major-log", default="outputs_tifs/MAJOR_LOG.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.eval_root, exist_ok=True)
    os.makedirs(args.checkpoint_root, exist_ok=True)

    if args.dry_run:
        print("Dry run mode enabled. No training executed.")
        print(f"Variants: {[v['name'] for v in VARIANTS]}")
        return

    state = _load_json(
        args.state_path,
        {
            "cycle": 0,
            "phase": "init",
            "active_variant": "",
            "cycle_total": int(args.max_cycles),
            "resume_checkpoint": "",
            "best": None,
            "history": [],
            "tried_variants": [],
            "finished": False,
        },
    )

    if state.get("finished", False):
        print("Autopilot already marked as finished. Delete state file to restart.")
        return

    _append_major_log(
        args.major_log,
        [
            "",
            f"## Autopilot Start {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            f"- Max cycles: {args.max_cycles}",
            f"- Steps per cycle: {args.steps_per_cycle}",
        ],
    )

    start_cycle = int(state.get("cycle", 0)) + 1

    for cycle in range(start_cycle, args.max_cycles + 1):
        variant = VARIANTS[(cycle - 1) % len(VARIANTS)]
        variant_name = variant["name"]

        resume_path = state.get("resume_checkpoint", "")
        current_step = _safe_get_step(resume_path)
        total_steps = current_step + args.steps_per_cycle

        cycle_output = os.path.join(args.output_root, f"cycle_{cycle:03d}_{variant_name}")
        cfg = _build_cycle_config(
            variant=variant,
            total_steps=total_steps,
            output_dir=cycle_output,
            checkpoint_dir=args.checkpoint_root,
            steps_per_cycle=args.steps_per_cycle,
            resume_path=resume_path,
        )

        print("=" * 72)
        print(f"Cycle {cycle}/{args.max_cycles} | variant={variant_name} | step {current_step} -> {total_steps}")
        print("=" * 72)

        state["phase"] = "training"
        state["active_variant"] = variant_name
        state["cycle"] = cycle
        state["cycle_total"] = int(args.max_cycles)
        state["step_start"] = int(current_step)
        state["step_target"] = int(total_steps)
        _save_json(args.state_path, state)

        train_v4.train(cfg)

        final_ckpt = os.path.join(args.checkpoint_root, "dscsp_v4_final.pt")
        if not os.path.exists(final_ckpt):
            raise RuntimeError(f"Missing final checkpoint after cycle {cycle}: {final_ckpt}")

        skip_cross = args.cross_dataset_every > 0 and (cycle % args.cross_dataset_every != 0)
        eval_out = os.path.join(args.eval_root, f"cycle_{cycle:03d}_{variant_name}.json")

        state["phase"] = "evaluating"
        state["eval_output"] = eval_out
        _save_json(args.state_path, state)

        eval_result = evaluate_tifs.evaluate_checkpoint(
            checkpoint=final_ckpt,
            num_samples=args.eval_samples,
            cross_dataset_samples=args.cross_dataset_samples,
            skip_cross_dataset=skip_cross,
            adaptive_train=args.adaptive_train,
            adaptive_test=args.adaptive_test,
        )
        _save_json(eval_out, eval_result)

        cycle_summary = {
            "cycle": cycle,
            "variant": variant_name,
            "checkpoint": final_ckpt,
            "clean_ber": float(eval_result["clean"]["ber"]),
            "jpeg_q75_ber": float(eval_result["jpeg"]["q75_ber"]),
            "crop80_ber": float(eval_result["crop"]["q80_ber"]),
            "auc_adaptive": float(eval_result["adaptive_steganalysis"]["auc_adaptive"]),
            "eval_path": eval_out,
        }

        state["history"].append(cycle_summary)
        state["tried_variants"].append(variant_name)
        state["cycle"] = cycle
        state["resume_checkpoint"] = final_ckpt
        state["phase"] = "cycle_complete"

        _write_ablation_report(state["history"], "outputs_tifs_ablation/latest_ablation.json")

        prev_best = state.get("best")
        significant = _is_significant(prev_best, eval_result)

        if prev_best is None or cycle_summary["clean_ber"] < prev_best["clean_ber"]:
            state["best"] = {
                "cycle": cycle,
                "variant": variant_name,
                "clean_ber": cycle_summary["clean_ber"],
                "crop80_ber": cycle_summary["crop80_ber"],
                "auc_adaptive": cycle_summary["auc_adaptive"],
                "checkpoint": final_ckpt,
            }

        if significant:
            major_ckpt = os.path.join(
                args.checkpoint_root,
                f"major_cycle_{cycle:03d}_{variant_name}.pt",
            )
            shutil.copy2(final_ckpt, major_ckpt)
            _append_major_log(
                args.major_log,
                [
                    "",
                    f"### Major Milestone | Cycle {cycle} | {variant_name}",
                    f"- Checkpoint: {major_ckpt}",
                    f"- Clean BER: {cycle_summary['clean_ber']:.6f}",
                    f"- JPEG Q75 BER: {cycle_summary['jpeg_q75_ber']:.6f}",
                    f"- Crop Q80 BER: {cycle_summary['crop80_ber']:.6f}",
                    f"- Adaptive AUC: {cycle_summary['auc_adaptive']:.4f}",
                    f"- Eval JSON: {eval_out}",
                ],
            )

        done = _meets_required(
            eval_result,
            state.get("tried_variants", []),
            args.min_ablation_variants,
        )

        if done:
            state["finished"] = True
            state["finished_cycle"] = cycle
            state["phase"] = "finished"
            _append_major_log(
                args.major_log,
                [
                    "",
                    "## TARGETS MET",
                    f"- Cycle: {cycle}",
                    f"- Best checkpoint: {state['best']['checkpoint']}",
                    f"- Clean BER: {cycle_summary['clean_ber']:.6f}",
                    f"- Crop Q80 BER: {cycle_summary['crop80_ber']:.6f}",
                    f"- Adaptive AUC: {cycle_summary['auc_adaptive']:.4f}",
                ],
            )
            _save_json(args.state_path, state)
            print("Targets met. Autopilot stopping.")
            return

        _save_json(args.state_path, state)

    state["finished"] = False
    state["stopped_reason"] = "max_cycles_reached"
    state["phase"] = "stopped"
    _save_json(args.state_path, state)
    _append_major_log(
        args.major_log,
        [
            "",
            "## Autopilot Stop",
            "- Reason: reached max cycles without meeting all required targets",
            f"- Last cycle: {state.get('cycle', 0)}",
            f"- Best checkpoint: {state.get('best', {}).get('checkpoint', 'N/A')}",
        ],
    )
    print("Reached max cycles without meeting all targets.")


if __name__ == "__main__":
    main()
