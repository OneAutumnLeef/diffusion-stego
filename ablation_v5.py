"""
DSCSP v5 Ablation Study.

Trains 6 lightweight variants (8 000 steps each) from a common 5 000-step
warm-started checkpoint and evaluates each with evaluate_tifs.py.

Variants:
  full           — full v5 model (baseline)
  no_redundant   — message_bits=256, no RedundantCoder
  no_srm_match   — lambda_srm_match=0.0
  no_spectral    — lambda_spec=0.0
  no_adv         — use_adversarial=False
  uniform_attacks— jpeg_weight=0.33 (uniform attack distribution)

Usage:
  # First, train the warm-start base:
  python ablation_v5.py --phase warmstart

  # Then, run all variants in sequence:
  python ablation_v5.py --phase variants

  # Or run a specific variant:
  python ablation_v5.py --phase variants --only full
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time


# Paths
WARMSTART_CKPT = "outputs_v5_ablation/warmstart/dscsp_v5_step_005000.pt"
ABLATION_DIR = "outputs_v5_ablation"
RESULTS_FILE = os.path.join(ABLATION_DIR, "ablation_results.json")
REPORT_FILE = os.path.join(ABLATION_DIR, "ablation_report.md")

WARMSTART_STEPS = 5000
VARIANT_STEPS = 8000


def base_config():
    """Return the base v5 config for all ablation variants."""
    from train_v5 import CONFIG as V5_CONFIG
    cfg = copy.deepcopy(V5_CONFIG)
    cfg["total_steps"] = VARIANT_STEPS
    cfg["checkpoint_interval"] = VARIANT_STEPS  # only final checkpoint
    cfg["save_image_interval"] = VARIANT_STEPS
    cfg["log_interval"] = 100
    cfg["pool_refresh_interval"] = 99999  # no refresh during short runs
    cfg["resume_load_optimizer"] = True
    cfg["resume_load_scheduler"] = False  # each variant starts fresh scheduler
    cfg["resume_load_disc_optimizer"] = True
    cfg["override_lrs_on_resume"] = True
    return cfg


def warmstart_config():
    """Config for the shared warm-start run (5 000 steps)."""
    from train_v5 import CONFIG as V5_CONFIG
    cfg = copy.deepcopy(V5_CONFIG)
    cfg["total_steps"] = WARMSTART_STEPS
    cfg["output_dir"] = os.path.join(ABLATION_DIR, "warmstart")
    cfg["checkpoint_dir"] = os.path.join(ABLATION_DIR, "warmstart")
    cfg["checkpoint_interval"] = WARMSTART_STEPS
    cfg["save_image_interval"] = WARMSTART_STEPS
    cfg["log_interval"] = 50
    cfg["pool_refresh_interval"] = 99999
    return cfg


VARIANTS = {
    "full": {
        # No changes — full v5 model
    },
    "no_redundant": {
        "use_redundant_coding": False,
        "message_bits": 256,    # physical == message (no redundancy)
        "redundancy_factor": 1,
    },
    "no_srm_match": {
        "lambda_srm_match": 0.0,
    },
    "no_spectral": {
        "lambda_spec": 0.0,
    },
    "no_adv": {
        "use_adversarial": False,
        "lambda_adv": 0.0,
    },
    "uniform_attacks": {
        "jpeg_weight": 0.33,    # uniform across jpeg / blur / noise / crop / bc
    },
}


def _run_training(config, label):
    """Launch train_v5.train() in-process with the given config override."""
    import train_v5
    import importlib
    importlib.reload(train_v5)

    print(f"\n{'='*60}")
    print(f"ABLATION VARIANT: {label}")
    print(f"{'='*60}")
    for k, v in config.items():
        if k not in train_v5.CONFIG or train_v5.CONFIG[k] != v:
            print(f"  OVERRIDE: {k} = {v}")
    print()

    train_v5.train(config)


def _run_evaluate(checkpoint_path, output_json):
    """Run evaluate_tifs.py on a checkpoint and collect results."""
    cmd = [
        sys.executable, "evaluate_tifs.py",
        "--checkpoint", checkpoint_path,
        "--num-samples", "200",
        "--skip-cross-dataset",
        "--output", output_json,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  WARNING: evaluate_tifs.py returned {result.returncode}")
        print(result.stderr[-2000:] if result.stderr else "")
        return None

    if os.path.exists(output_json):
        with open(output_json) as f:
            return json.load(f)
    return None


def _extract_metrics(eval_results):
    """Pull the key TIFS metrics from an evaluate_tifs result dict."""
    if eval_results is None:
        return {"clean_ber": None, "jpeg_q75_ber": None, "crop_q80_ber": None, "auc": None,
                "psnr": None, "lpips": None}
    clean = eval_results.get("clean", {})
    jpeg = eval_results.get("jpeg", {})
    crop = eval_results.get("crop", {})
    adap = eval_results.get("adaptive_steganalysis", {})
    lpips_val = eval_results.get("lpips") or clean.get("lpips")
    return {
        "clean_ber": clean.get("ber"),
        "jpeg_q75_ber": jpeg.get("q75_ber"),
        "crop_q80_ber": crop.get("q80_ber"),
        "auc": adap.get("auc_adaptive"),
        "psnr": clean.get("psnr"),
        "lpips": lpips_val,
    }


def _format_val(v, fmt=".4f"):
    if v is None:
        return "N/A"
    return f"{v:{fmt}}"


def _write_report(results: dict):
    """Write a markdown ablation table to REPORT_FILE."""
    os.makedirs(ABLATION_DIR, exist_ok=True)
    lines = [
        "# DSCSP v5 Ablation Results\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n",
        f"Warm-start: {WARMSTART_STEPS} steps | Variant training: {VARIANT_STEPS} steps\n",
        "",
        "| Variant | Clean BER | JPEG Q75 BER | Crop Q80 BER | AUC | PSNR (dB) | LPIPS |",
        "|---------|-----------|--------------|--------------|-----|-----------|-------|",
    ]
    for name, m in results.items():
        row = (
            f"| {name:<18} "
            f"| {_format_val(m.get('clean_ber'))} "
            f"| {_format_val(m.get('jpeg_q75_ber'))} "
            f"| {_format_val(m.get('crop_q80_ber'))} "
            f"| {_format_val(m.get('auc'))} "
            f"| {_format_val(m.get('psnr'), '.2f')} "
            f"| {_format_val(m.get('lpips'))} |"
        )
        lines.append(row)

    lines += [
        "",
        "## TIFS Targets",
        "- Clean BER: ≈ 0.000",
        "- JPEG Q75 BER: < 0.10",
        "- Crop Q80 BER: ≤ 0.30",
        "- AUC: ≤ 0.70",
        "- PSNR: ≥ 35 dB",
        "- LPIPS: ≤ 0.05",
    ]

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAblation report written to: {REPORT_FILE}")


def phase_warmstart():
    """Train the shared warm-start checkpoint."""
    os.makedirs(ABLATION_DIR, exist_ok=True)
    if os.path.exists(WARMSTART_CKPT):
        print(f"Warm-start checkpoint already exists: {WARMSTART_CKPT}")
        print("Delete it to re-run warm-start.")
        return

    print(f"Training warm-start for {WARMSTART_STEPS} steps...")
    cfg = warmstart_config()
    _run_training(cfg, "warmstart")

    # The checkpoint is saved as dscsp_v5_step_005000.pt
    expected = os.path.join(ABLATION_DIR, "warmstart", f"dscsp_v5_step_{WARMSTART_STEPS:06d}.pt")
    if os.path.exists(expected):
        print(f"\nWarm-start checkpoint ready: {expected}")
    else:
        print(f"\nWARNING: Expected checkpoint not found: {expected}")


def phase_variants(only=None):
    """Train and evaluate all (or one specific) variant."""
    if not os.path.exists(WARMSTART_CKPT):
        print(f"ERROR: Warm-start checkpoint not found: {WARMSTART_CKPT}")
        print("Run --phase warmstart first.")
        sys.exit(1)

    # Load existing results if any (enables resuming interrupted ablation runs)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results from {RESULTS_FILE}")
    else:
        all_results = {}

    for name, overrides in VARIANTS.items():
        if only and name != only:
            continue
        if name in all_results:
            print(f"  Skipping {name} (already in results)")
            continue

        print(f"\n{'='*60}")
        print(f"VARIANT: {name}")
        print(f"{'='*60}")

        variant_dir = os.path.join(ABLATION_DIR, name)
        os.makedirs(variant_dir, exist_ok=True)

        # Build config
        cfg = base_config()
        cfg["resume_path"] = WARMSTART_CKPT
        cfg["output_dir"] = variant_dir
        cfg["checkpoint_dir"] = variant_dir
        cfg.update(overrides)

        # Fix num_bits consistency for no_redundant variant
        if name == "no_redundant":
            cfg["num_bits"] = 256
            cfg["message_bits"] = 256
            cfg["use_redundant_coding"] = False
            cfg["redundancy_factor"] = 1

        # Train
        try:
            _run_training(cfg, name)
        except Exception as exc:
            print(f"  ERROR during training variant {name}: {exc}")
            all_results[name] = {"error": str(exc)}
            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            continue

        # Find final checkpoint
        final_ckpt = os.path.join(variant_dir, "dscsp_v5_final.pt")
        if not os.path.exists(final_ckpt):
            # Fall back to last step checkpoint
            step_ckpts = sorted([
                p for p in os.listdir(variant_dir) if p.startswith("dscsp_v5_step_")
            ])
            if step_ckpts:
                final_ckpt = os.path.join(variant_dir, step_ckpts[-1])
            else:
                print(f"  WARNING: No checkpoint found for variant {name}")
                all_results[name] = {"error": "no checkpoint"}
                continue

        # Evaluate
        eval_json = os.path.join(variant_dir, "eval_results.json")
        eval_results = _run_evaluate(final_ckpt, eval_json)
        metrics = _extract_metrics(eval_results)
        all_results[name] = metrics

        print(f"\n  Results for {name}:")
        for k, v in metrics.items():
            print(f"    {k}: {_format_val(v)}")

        # Save results after each variant
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved to {RESULTS_FILE}")

    _write_report(all_results)
    print("\nAblation study complete.")


def main():
    parser = argparse.ArgumentParser(description="DSCSP v5 Ablation Study")
    parser.add_argument(
        "--phase", choices=["warmstart", "variants", "report"], default="variants",
        help="warmstart: train shared checkpoint; variants: run all variants; report: regenerate report",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help=f"Run only one variant. Choices: {list(VARIANTS.keys())}",
    )
    args = parser.parse_args()

    if args.phase == "warmstart":
        phase_warmstart()
    elif args.phase == "variants":
        phase_variants(only=args.only)
    elif args.phase == "report":
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE) as f:
                all_results = json.load(f)
            _write_report(all_results)
        else:
            print(f"No results file found: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
