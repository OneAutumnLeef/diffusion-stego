#!/usr/bin/env python3
"""
Generate publication figures and LaTeX table rows for the DSCSP TIFS paper.

The script reads only local repo artifacts and code-derived metadata.
It does not rerun training or evaluation.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _latex_num(value, precision: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value.replace("_", "\\_")
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.{precision}f}"


def _display_num(value, precision: int = 4) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.{precision}f}"


def _pass_fail(flag: bool) -> str:
    return "Yes" if flag else "No"


def _short_variant(name: str) -> str:
    mapping = {
        "ber_recovery_clean_first": "BER clean-first",
        "ber_recovery_curriculum": "BER curriculum",
        "robust_balance": "Robust balance",
        "detectability_harden": "Detectability harden",
        "full": "Full",
        "no_srm_match": "No SRM match",
        "no_spectral": "No spectral",
        "no_adv": "No adversarial",
        "uniform_attacks": "Uniform attacks",
        "no_redundant": "No redundant coding",
    }
    return mapping.get(name, name.replace("_", " "))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _get_model_metadata() -> Dict:
    meta = {
        "controller_params": None,
        "extractor_params": None,
        "discriminator_params": None,
        "ddim_steps": 20,
        "num_bits": 256,
        "base_channels": 48,
        "extractor_grid_size": 16,
        "message_bits_v5": 128,
        "redundancy_factor_v5": 2,
    }
    try:
        sys.path.insert(0, str(ROOT))
        import train_v4
        from dscsp.models.controller_v4 import ContentAdaptiveController
        from train_v4 import GlobalExtractor, SNDiscriminator

        cfg = train_v4.CONFIG
        controller = ContentAdaptiveController(
            num_bits=cfg["num_bits"],
            max_perturbation=cfg["max_perturbation"],
            base_channels=cfg["base_channels"],
        )
        extractor = GlobalExtractor(
            num_bits=cfg["num_bits"],
            mode=cfg.get("extractor_mode", "global_fc"),
            grid_size=cfg.get("extractor_grid_size", 16),
        )
        discriminator = SNDiscriminator(use_srm=cfg["use_srm"])

        meta.update(
            {
                "controller_params": int(sum(p.numel() for p in controller.parameters())),
                "extractor_params": int(sum(p.numel() for p in extractor.parameters())),
                "discriminator_params": int(sum(p.numel() for p in discriminator.parameters())),
                "ddim_steps": int(cfg["ddim_steps"]),
                "num_bits": int(cfg["num_bits"]),
                "base_channels": int(cfg["base_channels"]),
                "extractor_grid_size": int(cfg.get("extractor_grid_size", 16)),
            }
        )
    except Exception as exc:  # noqa: BLE001
        meta["import_warning"] = str(exc)
    return meta


def load_data() -> Dict:
    state = _load_json(ROOT / "outputs_tifs" / "autopilot_state.json")
    evals = {}
    for item in state["history"]:
        evals[item["cycle"]] = _load_json(ROOT / item["eval_path"])

    data = {
        "state": state,
        "evals": evals,
        "tifs_ablation": _load_json(ROOT / "outputs_tifs_ablation" / "latest_ablation.json"),
        "v5_ablation": _load_json(ROOT / "outputs_v5_ablation" / "ablation_results.json"),
        "security_v5": _load_json(ROOT / "outputs_v5_analysis" / "security_analysis.json"),
        "model_meta": _get_model_metadata(),
    }
    return data


def generate_table_rows(data: Dict) -> None:
    state = data["state"]
    evals = data["evals"]
    selected_cycles = [9, 10, 11]

    main_rows = []
    for cycle in selected_cycles:
        hist = next(item for item in state["history"] if item["cycle"] == cycle)
        ev = evals[cycle]
        main_rows.append(
            "        "
            + " & ".join(
                [
                    str(cycle),
                    _short_variant(hist["variant"]),
                    _latex_num(ev["clean"]["ber"]),
                    _latex_num(ev["jpeg"]["q75_ber"]),
                    _latex_num(ev["crop"]["q80_ber"]),
                    _latex_num(ev["adaptive_steganalysis"]["auc_adaptive"]),
                    _latex_num(ev["clean"]["psnr"], precision=2),
                    _pass_fail(ev["areas"]["ber_focus"]["pass"]),
                    _pass_fail(ev["areas"]["detectability"]["pass"]),
                ]
            )
            + r" \\"
        )
    _write_text(FIG_DIR / "table_main_results_rows.tex", "\n".join(main_rows) + "\n")

    cross_rows = []
    for cycle in selected_cycles:
        hist = next(item for item in state["history"] if item["cycle"] == cycle)
        ev = evals[cycle]
        cifar = ev["cross_dataset"]["cifar10"]
        stl = ev["cross_dataset"]["stl10"]
        cross_rows.append(
            "        "
            + " & ".join(
                [
                    str(cycle),
                    _short_variant(hist["variant"]),
                    _latex_num(cifar["clean_ber"]),
                    _latex_num(cifar["crop80_ber"]),
                    _latex_num(stl["clean_ber"]),
                    _latex_num(stl["crop80_ber"]),
                ]
            )
            + r" \\"
        )
    _write_text(FIG_DIR / "table_cross_dataset_rows.tex", "\n".join(cross_rows) + "\n")

    v4_rows = []
    for name, vals in data["tifs_ablation"]["variants"].items():
        v4_rows.append(
            "        "
            + " & ".join(
                [
                    _short_variant(name),
                    str(vals["n"]),
                    _latex_num(vals["clean_ber_mean"]),
                    _latex_num(vals["crop80_ber_mean"]),
                    _latex_num(vals["auc_adaptive_mean"]),
                ]
            )
            + r" \\"
        )
    _write_text(FIG_DIR / "table_ablation_v4_rows.tex", "\n".join(v4_rows) + "\n")

    v5_rows = []
    for name, vals in data["v5_ablation"].items():
        if "error" in vals:
            v5_rows.append(
                "        "
                + " & ".join(
                    [
                        _short_variant(name),
                        "training failed",
                        "--",
                        "--",
                        "--",
                        vals["error"].replace("_", "\\_"),
                    ]
                )
                + r" \\"
            )
            continue

        v5_rows.append(
            "        "
            + " & ".join(
                [
                    _short_variant(name),
                    _latex_num(vals["clean_ber"]),
                    _latex_num(vals["jpeg_q75_ber"]),
                    _latex_num(vals["crop_q80_ber"]),
                    _latex_num(vals["auc"]),
                    _latex_num(vals["psnr"], precision=2),
                ]
            )
            + r" \\"
        )
    _write_text(FIG_DIR / "table_ablation_v5_rows.tex", "\n".join(v5_rows) + "\n")


def _setup_axis(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _box(ax, x, y, w, h, text, fc, ec, fontsize=9, lw=1.3, ls="-"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle=ls,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(ax, start: Tuple[float, float], end: Tuple[float, float], color="#444444", ls="-"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.4,
        color=color,
        linestyle=ls,
    )
    ax.add_patch(patch)
    return patch


def plot_system_overview(data: Dict) -> None:
    meta = data["model_meta"]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    _setup_axis(ax)

    ax.text(0.02, 0.93, "Primary TIFS v4 line", fontsize=11, weight="bold")
    ax.text(0.62, 0.93, "Supporting v5 analysis line", fontsize=11, weight="bold")

    _box(
        ax,
        0.03,
        0.62,
        0.16,
        0.18,
        f"Frozen DDIM sampler\n{meta['ddim_steps']} steps\ngoogle/ddpm-celebahq-256",
        "#e8eef7",
        "#5875a4",
    )
    _box(ax, 0.24, 0.62, 0.10, 0.18, "Clean image\n$x_c$", "#f4f4f4", "#666666")
    _box(
        ax,
        0.03,
        0.24,
        0.16,
        0.18,
        "Payload bits\n256 physical bits",
        "#eef7ea",
        "#5f8c4d",
    )
    _box(
        ax,
        0.24,
        0.24,
        0.16,
        0.18,
        "Controller\ncontent-adaptive U-Net\n+ carrier branches\n"
        f"{meta['controller_params'] / 1e6:.2f}M params",
        "#dce9f7",
        "#3f6ea7",
    )
    _box(
        ax,
        0.46,
        0.43,
        0.14,
        0.18,
        "Stego image\n$x_s=\\mathrm{clip}(x_c+\\delta)$",
        "#fdf4dc",
        "#ad8b32",
    )
    _box(
        ax,
        0.66,
        0.58,
        0.15,
        0.18,
        "Attack channel\nJPEG / crop / blur /\nnoise / brightness",
        "#f9e7dc",
        "#c36d32",
    )
    _box(
        ax,
        0.66,
        0.24,
        0.15,
        0.18,
        "Extractor\nResNet-18 spatial map\n"
        f"{meta['extractor_params'] / 1e6:.2f}M params",
        "#e3f4ef",
        "#2f8c73",
    )
    _box(
        ax,
        0.86,
        0.24,
        0.11,
        0.18,
        "Decoded bits\nBER metric",
        "#f4f4f4",
        "#666666",
    )
    _box(
        ax,
        0.86,
        0.58,
        0.11,
        0.18,
        "Adaptive\nsteganalysis\nAUC metric",
        "#f4e6ef",
        "#9a4d7b",
    )

    _box(
        ax,
        0.62,
        0.80,
        0.18,
        0.10,
        "Optional RedundantCoder\n128 message -> 256 physical",
        "#ffffff",
        "#5f8c4d",
        fontsize=8.5,
        ls="--",
    )
    _box(
        ax,
        0.82,
        0.80,
        0.15,
        0.10,
        "SRM-match / security\nanalysis branch",
        "#ffffff",
        "#c36d32",
        fontsize=8.5,
        ls="--",
    )

    _arrow(ax, (0.19, 0.71), (0.24, 0.71))
    _arrow(ax, (0.19, 0.33), (0.24, 0.33))
    _arrow(ax, (0.34, 0.71), (0.46, 0.52))
    _arrow(ax, (0.40, 0.33), (0.46, 0.43))
    _arrow(ax, (0.60, 0.52), (0.66, 0.67))
    _arrow(ax, (0.60, 0.45), (0.66, 0.33))
    _arrow(ax, (0.81, 0.33), (0.86, 0.33))
    _arrow(ax, (0.81, 0.67), (0.86, 0.67))
    _arrow(ax, (0.71, 0.80), (0.32, 0.42), ls="--", color="#5f8c4d")
    _arrow(ax, (0.89, 0.80), (0.73, 0.76), ls="--", color="#c36d32")

    ax.text(
        0.50,
        0.06,
        "Figure built from train_v4.py, controller_v4.py, diffusion_wrapper.py, discriminator.py, "
        "redundant_coding.py, and current artifact metadata.",
        fontsize=8,
        ha="center",
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "system_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_architecture_detail(data: Dict) -> None:
    meta = data["model_meta"]
    fig, ax = plt.subplots(figsize=(13, 6.4))
    _setup_axis(ax)

    ax.text(0.02, 0.95, "Detailed controller/extractor/training path", fontsize=12, weight="bold")

    # Controller internals.
    _box(ax, 0.03, 0.70, 0.11, 0.12, "Payload\n256 bits", "#eef7ea", "#5f8c4d")
    _box(ax, 0.17, 0.70, 0.13, 0.12, "Message MLP\n-> 4x4 map", "#eef7ea", "#5f8c4d")
    _box(ax, 0.33, 0.70, 0.14, 0.12, "Upsample to\n256x256", "#eef7ea", "#5f8c4d")
    _box(ax, 0.03, 0.46, 0.11, 0.12, "Clean image\n$x_c$", "#f4f4f4", "#666666")
    _box(ax, 0.17, 0.46, 0.16, 0.12, "Concat image +\nmessage map", "#dce9f7", "#3f6ea7")
    _box(
        ax,
        0.36,
        0.46,
        0.18,
        0.12,
        "3-level U-Net\nenc/dec + skips\nbase ch="
        f"{meta['base_channels']}",
        "#dce9f7",
        "#3f6ea7",
    )
    _box(ax, 0.03, 0.20, 0.15, 0.12, "Fixed carrier path\nnearest-neighbor grid", "#fff7e1", "#ad8b32")
    _box(ax, 0.21, 0.20, 0.15, 0.12, "Direct linear path\n16x16 payload map", "#fff7e1", "#ad8b32")
    _box(ax, 0.57, 0.46, 0.14, 0.12, "Add + tanh\nmax pert=0.08", "#dce9f7", "#3f6ea7")
    _box(ax, 0.74, 0.46, 0.12, 0.12, "Stego image\n$x_s$", "#fdf4dc", "#ad8b32")

    # Training/eval path.
    _box(ax, 0.74, 0.70, 0.15, 0.12, "JPEG-heavy attack layer\nor combined attacks", "#f9e7dc", "#c36d32")
    _box(
        ax,
        0.74,
        0.20,
        0.17,
        0.12,
        "Extractor\nResNet-18 no maxpool\n1x1 head -> 16x16 logits",
        "#e3f4ef",
        "#2f8c73",
    )
    _box(ax, 0.94, 0.20, 0.05, 0.12, "BER", "#e3f4ef", "#2f8c73")
    _box(
        ax,
        0.94,
        0.46,
        0.05,
        0.12,
        "SN D",
        "#f4e6ef",
        "#9a4d7b",
    )
    _box(ax, 0.94, 0.70, 0.05, 0.12, "AUC", "#f4e6ef", "#9a4d7b")

    # Phase boxes.
    _box(ax, 0.43, 0.84, 0.16, 0.08, "Phase 1\nextractor bootstrap", "#ffffff", "#666666", fontsize=8.5, ls="--")
    _box(ax, 0.61, 0.84, 0.16, 0.08, "Phase 2\ncontroller alignment", "#ffffff", "#666666", fontsize=8.5, ls="--")
    _box(ax, 0.79, 0.84, 0.16, 0.08, "Phase 3\nfull adversarial training", "#ffffff", "#666666", fontsize=8.5, ls="--")

    _arrow(ax, (0.14, 0.76), (0.17, 0.76))
    _arrow(ax, (0.30, 0.76), (0.33, 0.76))
    _arrow(ax, (0.14, 0.52), (0.17, 0.52))
    _arrow(ax, (0.33, 0.52), (0.36, 0.52))
    _arrow(ax, (0.47, 0.76), (0.47, 0.58))
    _arrow(ax, (0.18, 0.26), (0.57, 0.52), color="#ad8b32")
    _arrow(ax, (0.36, 0.26), (0.57, 0.49), color="#ad8b32")
    _arrow(ax, (0.54, 0.52), (0.57, 0.52))
    _arrow(ax, (0.71, 0.52), (0.74, 0.52))
    _arrow(ax, (0.80, 0.58), (0.80, 0.70), color="#c36d32")
    _arrow(ax, (0.80, 0.46), (0.80, 0.32), color="#2f8c73")
    _arrow(ax, (0.89, 0.76), (0.96, 0.76), color="#9a4d7b")
    _arrow(ax, (0.86, 0.52), (0.96, 0.52), color="#9a4d7b")
    _arrow(ax, (0.91, 0.26), (0.94, 0.26), color="#2f8c73")

    ax.text(
        0.50,
        0.08,
        "Controller: {:.2f}M params  |  Extractor: {:.2f}M params  |  Discriminator: {:.2f}M params".format(
            meta["controller_params"] / 1e6,
            meta["extractor_params"] / 1e6,
            meta["discriminator_params"] / 1e6,
        ),
        ha="center",
        fontsize=9,
        weight="bold",
    )
    ax.text(
        0.50,
        0.03,
        "Losses from code: BCE reconstruction, LPIPS perceptual loss, FFT spectral loss, and adversarial loss. "
        "Supporting v5 analysis adds redundant coding and SRM feature matching.",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "architecture_detail.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_autopilot_cycles(data: Dict) -> None:
    state = data["state"]
    cycles = [item["cycle"] for item in state["history"]]
    clean = [item["clean_ber"] for item in state["history"]]
    jpeg = [item["jpeg_q75_ber"] for item in state["history"]]
    crop = [item["crop80_ber"] for item in state["history"]]
    auc = [item["auc_adaptive"] for item in state["history"]]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)

    axes[0].plot(cycles, clean, marker="o", linewidth=2, label="Clean BER", color="#2f8c73")
    axes[0].plot(cycles, jpeg, marker="s", linewidth=2, label="JPEG Q75 BER", color="#3f6ea7")
    axes[0].axhline(0.10, color="#666666", linestyle="--", linewidth=1, label="BER focus gate")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("BER (log scale)")
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(cycles, crop, marker="o", linewidth=2, color="#c36d32")
    axes[1].axhline(0.35, color="#666666", linestyle="--", linewidth=1, label="Crop gate")
    axes[1].axhline(0.50, color="#999999", linestyle=":", linewidth=1, label="Random guess")
    axes[1].set_ylabel("Crop Q80 BER")
    axes[1].set_ylim(0.49, 0.5055)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].plot(cycles, auc, marker="o", linewidth=2, color="#9a4d7b")
    axes[2].axhline(0.70, color="#666666", linestyle="--", linewidth=1, label="AUC gate")
    axes[2].set_ylabel("Adaptive AUC")
    axes[2].set_xlabel("Autopilot cycle")
    axes[2].set_ylim(0.65, 1.02)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.25)

    for ax in axes:
        ax.axvspan(8.5, 11.5, color="#e9eef5", alpha=0.55)
        ax.set_xticks(cycles)

    axes[0].annotate(
        "Cycle 9: best clean/JPEG BER\nbut AUC remains near 1.0",
        xy=(9, clean[8]),
        xytext=(6.2, 0.002),
        arrowprops=dict(arrowstyle="->", color="#2f8c73"),
        fontsize=8,
    )
    axes[2].annotate(
        "Cycle 11: first AUC pass\n(0.6915)",
        xy=(11, auc[10]),
        xytext=(8.2, 0.76),
        arrowprops=dict(arrowstyle="->", color="#9a4d7b"),
        fontsize=8,
    )
    axes[1].annotate(
        "Crop never leaves chance regime",
        xy=(11, crop[10]),
        xytext=(6.0, 0.5036),
        arrowprops=dict(arrowstyle="->", color="#c36d32"),
        fontsize=8,
    )

    fig.suptitle(
        "Autopilot trajectory from outputs_tifs/autopilot_state.json\n"
        "Cycles 1-8 used 320 evaluation samples; cycles 9-11 used 200 after evaluator fixes.",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "autopilot_cycles.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ablation(data: Dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Left: v4 variant trade-off scatter.
    variants = data["tifs_ablation"]["variants"]
    x = [variants[name]["clean_ber_mean"] for name in variants]
    y = [variants[name]["auc_adaptive_mean"] for name in variants]
    c = [variants[name]["crop80_ber_mean"] for name in variants]
    labels = [_short_variant(name) for name in variants]
    axes[0].scatter(x, y, c=c, cmap="plasma", s=140, edgecolor="black")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Mean clean BER (log scale)")
    axes[0].set_ylabel("Mean adaptive AUC")
    axes[0].axhline(0.70, color="#666666", linestyle="--", linewidth=1)
    for xi, yi, label in zip(x, y, labels):
        axes[0].annotate(label, (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=8)
    axes[0].set_title("v4 autopilot variant means")
    axes[0].grid(alpha=0.25)

    # Right: v5 short-run supporting ablation bars.
    v5_order = ["full", "no_srm_match", "no_spectral", "no_adv", "uniform_attacks"]
    valid = [name for name in v5_order if name in data["v5_ablation"] and "error" not in data["v5_ablation"][name]]
    metric_names = ["clean_ber", "jpeg_q75_ber", "crop_q80_ber", "auc"]
    colors = ["#2f8c73", "#3f6ea7", "#c36d32", "#9a4d7b"]
    xpos = np.arange(len(valid))
    width = 0.18
    for idx, metric_name in enumerate(metric_names):
        vals = [data["v5_ablation"][name][metric_name] for name in valid]
        axes[1].bar(xpos + (idx - 1.5) * width, vals, width=width, color=colors[idx], label=metric_name)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([_short_variant(name) for name in valid], rotation=20, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Metric value")
    axes[1].set_title("v5 supporting component ablations")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Ablation summary from outputs_tifs_ablation/latest_ablation.json and outputs_v5_ablation/ablation_results.json\n"
        "The no_redundant v5 run failed to train and is listed only in the table.",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_security_analysis(data: Dict) -> None:
    security = data["security_v5"]
    power = security["power_spectrum"]["ring_diff_sample"]
    svd = security["svd_spectrum"]["top_32_singular_values_normalised"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    rings = [int(k) for k in power.keys()]
    diffs = [power[str(k)] for k in rings]
    axes[0].bar(rings, diffs, color="#3f6ea7")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Power spectrum ring differences")
    axes[0].set_xlabel("Radial ring")
    axes[0].set_ylabel("Relative diff")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(np.arange(1, len(svd) + 1), svd, color="#c36d32")
    axes[1].set_yscale("log")
    axes[1].set_title("Normalised singular values")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Value (log scale)")
    axes[1].grid(alpha=0.25, axis="y")

    axes[2].set_axis_off()
    summary_lines = [
        "Supporting v5 security summary",
        "",
        f"KL(stego||clean): {_display_num(security['kl_divergence']['kl_stego_vs_clean'])}",
        f"Power-spectrum p: {_display_num(security['power_spectrum']['p_value'])}",
        f"Detectable at p<0.05: {_pass_fail(security['power_spectrum']['detectable_p05'])}",
        f"SVD gini: {_display_num(security['svd_spectrum']['gini_coefficient'])}",
        f"Bootstrap BER: {_display_num(security['bootstrap_ci']['ber']['mean'])}",
        f"Bootstrap PSNR: {_display_num(security['bootstrap_ci']['psnr']['mean'], 2)} dB",
        f"Bootstrap LPIPS: {_display_num(security['bootstrap_ci']['lpips']['mean'])}",
        "",
        "Interpretation:",
        "Low KL does not guarantee spectral concealment.",
        "This v5 checkpoint remains statistically detectable",
        "in the power-spectrum test despite modest",
        "Inception-feature divergence.",
    ]
    axes[2].text(
        0.01,
        0.98,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
    )

    fig.suptitle(
        "Security/statistical analysis from outputs_v5_analysis/security_analysis.json",
        fontsize=11,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "security_analysis_v5.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_crop_failure(data: Dict) -> None:
    final_eval = data["evals"][11]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # Left: final cycle BERs with CI.
    names = ["Clean", "JPEG Q75", "Crop Q80"]
    means = [
        final_eval["clean"]["ber"],
        final_eval["jpeg"]["q75_ber"],
        final_eval["crop"]["q80_ber"],
    ]
    lows = [
        final_eval["clean"]["ber"] - final_eval["clean"]["ber_ci95"][0],
        final_eval["jpeg"]["q75_ber"] - final_eval["jpeg"]["q75_ber_ci95"][0],
        final_eval["crop"]["q80_ber"] - final_eval["crop"]["q80_ber_ci95"][0],
    ]
    highs = [
        final_eval["clean"]["ber_ci95"][1] - final_eval["clean"]["ber"],
        final_eval["jpeg"]["q75_ber_ci95"][1] - final_eval["jpeg"]["q75_ber"],
        final_eval["crop"]["q80_ber_ci95"][1] - final_eval["crop"]["q80_ber"],
    ]
    colors = ["#2f8c73", "#3f6ea7", "#c36d32"]
    x = np.arange(len(names))
    axes[0].bar(x, means, yerr=[lows, highs], capsize=4, color=colors)
    axes[0].axhline(0.35, color="#666666", linestyle="--", linewidth=1, label="Crop target")
    axes[0].axhline(0.50, color="#999999", linestyle=":", linewidth=1, label="Random guess")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylim(0, 0.55)
    axes[0].set_ylabel("BER")
    axes[0].set_title("Cycle 11 operating point")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, axis="y")

    # Right: post-fix cross-dataset crop BER.
    cycles = [9, 10, 11]
    id_crop = [data["evals"][cycle]["crop"]["q80_ber"] for cycle in cycles]
    cifar_crop = [data["evals"][cycle]["cross_dataset"]["cifar10"]["crop80_ber"] for cycle in cycles]
    stl_crop = [data["evals"][cycle]["cross_dataset"]["stl10"]["crop80_ber"] for cycle in cycles]
    xpos = np.arange(len(cycles))
    width = 0.24
    axes[1].bar(xpos - width, id_crop, width=width, label="CelebA-HQ", color="#c36d32")
    axes[1].bar(xpos, cifar_crop, width=width, label="CIFAR10", color="#3f6ea7")
    axes[1].bar(xpos + width, stl_crop, width=width, label="STL10", color="#9a4d7b")
    axes[1].axhline(0.35, color="#666666", linestyle="--", linewidth=1)
    axes[1].axhline(0.50, color="#999999", linestyle=":", linewidth=1)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([str(cycle) for cycle in cycles])
    axes[1].set_xlabel("Post-fix evaluation cycle")
    axes[1].set_ylabel("Crop Q80 BER")
    axes[1].set_ylim(0.49, 0.5055)
    axes[1].set_title("Crop BER stays near chance across datasets")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Crop robustness failure analysis from outputs_tifs_eval/cycle_011_robust_balance.json "
        "and post-fix cross-dataset checks",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "crop_failure_analysis.pdf", bbox_inches="tight")
    plt.close(fig)


def write_manifest(data: Dict) -> None:
    state = data["state"]
    final_eval = data["evals"][11]
    manifest = {
        "sources": {
            "autopilot_state": "outputs_tifs/autopilot_state.json",
            "eval_cycles": [item["eval_path"] for item in state["history"]],
            "tifs_ablation": "outputs_tifs_ablation/latest_ablation.json",
            "v5_ablation": "outputs_v5_ablation/ablation_results.json",
            "v5_security": "outputs_v5_analysis/security_analysis.json",
        },
        "headline_cycle_11": {
            "clean_ber": final_eval["clean"]["ber"],
            "jpeg_q75_ber": final_eval["jpeg"]["q75_ber"],
            "crop_q80_ber": final_eval["crop"]["q80_ber"],
            "adaptive_auc": final_eval["adaptive_steganalysis"]["auc_adaptive"],
            "clean_psnr": final_eval["clean"]["psnr"],
        },
        "model_metadata": data["model_meta"],
    }
    _write_text(FIG_DIR / "paper_metrics_manifest.json", json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    generate_table_rows(data)
    plot_system_overview(data)
    plot_architecture_detail(data)
    plot_autopilot_cycles(data)
    plot_ablation(data)
    plot_security_analysis(data)
    plot_crop_failure(data)
    write_manifest(data)
    print(f"Generated figures and table rows in {FIG_DIR}")


if __name__ == "__main__":
    main()
