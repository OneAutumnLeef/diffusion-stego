#!/usr/bin/env python3
"""
Regenerate publication figures and LaTeX table rows for the submission
manuscript. Writes assets into paper_submission/figures/ without any
mention of the internal TIFS sprint codename.

This script reads repository artifacts only; it does not rerun training
or evaluation.
"""

from __future__ import annotations

import json
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
OUT_DIR = ROOT / "paper_submission" / "figures"


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


def _display(value, precision: int = 4) -> str:
    return _latex_num(value, precision)


def _yesno(flag: bool) -> str:
    return "Yes" if flag else "No"


def _short_variant(name: str) -> str:
    mapping = {
        "ber_recovery_clean_first": "Clean-first",
        "ber_recovery_curriculum": "Curriculum",
        "robust_balance": "Balanced",
        "detectability_harden": "Detectability",
        "full": "Full",
        "no_srm_match": "w/o SRM match",
        "no_spectral": "w/o spectral",
        "no_adv": "w/o adversarial",
        "uniform_attacks": "Uniform attacks",
        "no_redundant": "w/o redundancy",
    }
    return mapping.get(name, name.replace("_", " "))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _model_metadata() -> Dict:
    meta = {
        "controller_params": 4392772,
        "extractor_params": 11308353,
        "discriminator_params": 2762945,
        "ddim_steps": 20,
        "num_bits": 256,
        "base_channels": 48,
        "extractor_grid_size": 16,
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
        meta.update({
            "controller_params": int(sum(p.numel() for p in controller.parameters())),
            "extractor_params": int(sum(p.numel() for p in extractor.parameters())),
            "discriminator_params": int(sum(p.numel() for p in discriminator.parameters())),
            "ddim_steps": int(cfg["ddim_steps"]),
            "num_bits": int(cfg["num_bits"]),
            "base_channels": int(cfg["base_channels"]),
            "extractor_grid_size": int(cfg.get("extractor_grid_size", 16)),
        })
    except Exception as exc:
        meta["import_warning"] = str(exc)
    return meta


def load_data() -> Dict:
    state = _load_json(ROOT / "outputs_tifs" / "autopilot_state.json")
    evals = {
        item["cycle"]: _load_json(ROOT / item["eval_path"])
        for item in state["history"]
    }
    return {
        "state": state,
        "evals": evals,
        "variant_stats": _load_json(ROOT / "outputs_tifs_ablation" / "latest_ablation.json"),
        "component_ablation": _load_json(ROOT / "outputs_v5_ablation" / "ablation_results.json"),
        "security": _load_json(ROOT / "outputs_v5_analysis" / "security_analysis.json"),
        "model_meta": _model_metadata(),
    }


# --------------------------------------------------------------------------- #
# Tables                                                                      #
# --------------------------------------------------------------------------- #
def write_tables(data: Dict) -> None:
    state = data["state"]
    evals = data["evals"]
    main_cycles = [9, 10, 11]

    rows = []
    for cycle in main_cycles:
        h = next(x for x in state["history"] if x["cycle"] == cycle)
        ev = evals[cycle]
        rows.append("        " + " & ".join([
            str(cycle),
            _short_variant(h["variant"]),
            _latex_num(ev["clean"]["ber"]),
            _latex_num(ev["jpeg"]["q75_ber"]),
            _latex_num(ev["jpeg"]["q50_ber"]),
            _latex_num(ev["crop"]["q80_ber"]),
            _latex_num(ev["adaptive_steganalysis"]["auc_adaptive"]),
            _latex_num(ev["clean"]["psnr"], 2),
        ]) + r" \\")
    _write(OUT_DIR / "table_main_results_rows.tex", "\n".join(rows) + "\n")

    cross_rows = []
    for cycle in main_cycles:
        h = next(x for x in state["history"] if x["cycle"] == cycle)
        ev = evals[cycle]
        c = ev["cross_dataset"]["cifar10"]
        s = ev["cross_dataset"]["stl10"]
        cross_rows.append("        " + " & ".join([
            str(cycle),
            _short_variant(h["variant"]),
            _latex_num(c["clean_ber"]),
            _latex_num(c["crop80_ber"]),
            _latex_num(s["clean_ber"]),
            _latex_num(s["crop80_ber"]),
        ]) + r" \\")
    _write(OUT_DIR / "table_cross_dataset_rows.tex", "\n".join(cross_rows) + "\n")

    var_rows = []
    for name, v in data["variant_stats"]["variants"].items():
        var_rows.append("        " + " & ".join([
            _short_variant(name),
            str(v["n"]),
            _latex_num(v["clean_ber_mean"]),
            _latex_num(v["crop80_ber_mean"]),
            _latex_num(v["auc_adaptive_mean"]),
        ]) + r" \\")
    _write(OUT_DIR / "table_variant_summary_rows.tex", "\n".join(var_rows) + "\n")

    comp_rows = []
    for name, v in data["component_ablation"].items():
        if "error" in v:
            comp_rows.append("        " + " & ".join([
                _short_variant(name),
                "--", "--", "--", "--",
                "training failed",
            ]) + r" \\")
            continue
        comp_rows.append("        " + " & ".join([
            _short_variant(name),
            _latex_num(v["clean_ber"]),
            _latex_num(v["jpeg_q75_ber"]),
            _latex_num(v["crop_q80_ber"]),
            _latex_num(v["auc"]),
            _latex_num(v["psnr"], 2),
        ]) + r" \\")
    _write(OUT_DIR / "table_component_ablation_rows.tex", "\n".join(comp_rows) + "\n")


# --------------------------------------------------------------------------- #
# Figure helpers                                                              #
# --------------------------------------------------------------------------- #
def _blank(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _box(ax, x, y, w, h, text, fc, ec, fontsize=9, ls="-", lw=1.3):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, a, b, color="#444", ls="-"):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=12, linewidth=1.4,
        color=color, linestyle=ls,
    ))


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
def fig_system_overview(data: Dict) -> None:
    m = data["model_meta"]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    _blank(ax)

    _box(ax, 0.02, 0.60, 0.17, 0.22,
         f"Frozen diffusion sampler\n(DDIM, {m['ddim_steps']} steps)\n"
         "google/ddpm-celebahq-256",
         "#e8eef7", "#5875a4")
    _box(ax, 0.24, 0.62, 0.10, 0.18, "Cover\n$x_c$", "#f4f4f4", "#666")
    _box(ax, 0.02, 0.22, 0.17, 0.22,
         f"Payload bits\n$b \\in \\{{0,1\\}}^{{{m['num_bits']}}}$",
         "#eef7ea", "#5f8c4d")
    _box(ax, 0.24, 0.22, 0.16, 0.22,
         "Content-adaptive\ncontroller $C_\\phi$\n"
         f"U-Net + carriers\n{m['controller_params']/1e6:.2f}M params",
         "#dce9f7", "#3f6ea7")
    _box(ax, 0.45, 0.45, 0.14, 0.18,
         "Stego\n$x_s = \\mathrm{clip}(x_c + \\delta)$",
         "#fdf4dc", "#ad8b32")
    _box(ax, 0.64, 0.60, 0.15, 0.22,
         "Attack channel\nJPEG / crop-resize /\nblur / noise / b-c",
         "#f9e7dc", "#c36d32")
    _box(ax, 0.64, 0.22, 0.15, 0.22,
         "Extractor $E_\\psi$\nResNet-18 spatial\n"
         f"{m['extractor_params']/1e6:.2f}M params",
         "#e3f4ef", "#2f8c73")
    _box(ax, 0.84, 0.22, 0.13, 0.22,
         "Decoded $\\hat{b}$\nBER + PSNR",
         "#f4f4f4", "#666")
    _box(ax, 0.84, 0.60, 0.13, 0.22,
         "SRM discriminator\n$D_\\theta$ / steganalyser\n"
         f"{m['discriminator_params']/1e6:.2f}M params",
         "#f4e6ef", "#9a4d7b")

    _arrow(ax, (0.19, 0.71), (0.24, 0.71))
    _arrow(ax, (0.19, 0.33), (0.24, 0.33))
    _arrow(ax, (0.34, 0.71), (0.45, 0.54))
    _arrow(ax, (0.40, 0.33), (0.45, 0.48))
    _arrow(ax, (0.59, 0.54), (0.64, 0.68))
    _arrow(ax, (0.59, 0.48), (0.64, 0.33))
    _arrow(ax, (0.79, 0.33), (0.84, 0.33))
    _arrow(ax, (0.79, 0.68), (0.84, 0.68))

    ax.text(0.50, 0.03,
            "Forward path from $(x_c,b)$ through the additive perturbation "
            "$\\delta=C_\\phi(x_c,b)$ and differentiable attack channel; "
            "$D_\\theta$ supervises undetectability during training.",
            ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "system_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_architecture_detail(data: Dict) -> None:
    m = data["model_meta"]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    _blank(ax)

    _box(ax, 0.03, 0.72, 0.11, 0.11, "Payload\n256 bits", "#eef7ea", "#5f8c4d")
    _box(ax, 0.17, 0.72, 0.14, 0.11, "Message MLP\n$\\to\\,4{\\times}4$ map", "#eef7ea", "#5f8c4d")
    _box(ax, 0.33, 0.72, 0.13, 0.11, "Bilinear up\n$\\to 256{\\times}256$", "#eef7ea", "#5f8c4d")
    _box(ax, 0.03, 0.48, 0.11, 0.11, "Cover\n$x_c$", "#f4f4f4", "#666")
    _box(ax, 0.17, 0.48, 0.16, 0.11, "Concat:\nimage+message", "#dce9f7", "#3f6ea7")
    _box(ax, 0.36, 0.48, 0.18, 0.11,
         f"3-level U-Net\nenc/dec + skips\nbase ch={m['base_channels']}",
         "#dce9f7", "#3f6ea7")
    _box(ax, 0.03, 0.22, 0.17, 0.11,
         "Fixed carrier path\n$16{\\times}16$ grid, channel-shifted",
         "#fff7e1", "#ad8b32", fontsize=8.5)
    _box(ax, 0.23, 0.22, 0.15, 0.11,
         "Direct linear path\n$\\text{Lin}\\to 3\\cdot16^2$",
         "#fff7e1", "#ad8b32", fontsize=8.5)
    _box(ax, 0.57, 0.48, 0.15, 0.11, "Sum + $\\tanh$\n$\\times\\, \\epsilon_{\\max}$",
         "#dce9f7", "#3f6ea7")
    _box(ax, 0.75, 0.48, 0.12, 0.11, "Stego $x_s$", "#fdf4dc", "#ad8b32")

    _box(ax, 0.75, 0.72, 0.15, 0.11,
         "Attack layer:\nDiffJPEG, crop,\nblur, noise, B/C",
         "#f9e7dc", "#c36d32", fontsize=8.5)
    _box(ax, 0.75, 0.22, 0.17, 0.11,
         "Extractor\nResNet-18 (no maxpool)\n$1{\\times}1\\,\\to 16^2$ logits",
         "#e3f4ef", "#2f8c73", fontsize=8.5)
    _box(ax, 0.94, 0.22, 0.05, 0.11, "BER", "#e3f4ef", "#2f8c73")
    _box(ax, 0.94, 0.48, 0.05, 0.11, "$D_\\theta$", "#f4e6ef", "#9a4d7b")
    _box(ax, 0.94, 0.72, 0.05, 0.11, "AUC", "#f4e6ef", "#9a4d7b")

    _box(ax, 0.03, 0.88, 0.20, 0.08, "Stage A: extractor warm-up",
         "#ffffff", "#666", fontsize=8.5, ls="--")
    _box(ax, 0.27, 0.88, 0.22, 0.08, "Stage B: controller alignment",
         "#ffffff", "#666", fontsize=8.5, ls="--")
    _box(ax, 0.53, 0.88, 0.44, 0.08,
         "Stage C: joint adversarial training with JPEG-heavy attack schedule",
         "#ffffff", "#666", fontsize=8.5, ls="--")

    _arrow(ax, (0.14, 0.77), (0.17, 0.77))
    _arrow(ax, (0.31, 0.77), (0.33, 0.77))
    _arrow(ax, (0.14, 0.53), (0.17, 0.53))
    _arrow(ax, (0.33, 0.53), (0.36, 0.53))
    _arrow(ax, (0.47, 0.72), (0.47, 0.59))
    _arrow(ax, (0.20, 0.27), (0.57, 0.52), color="#ad8b32")
    _arrow(ax, (0.38, 0.27), (0.57, 0.50), color="#ad8b32")
    _arrow(ax, (0.54, 0.53), (0.57, 0.53))
    _arrow(ax, (0.72, 0.53), (0.75, 0.53))
    _arrow(ax, (0.81, 0.59), (0.81, 0.72), color="#c36d32")
    _arrow(ax, (0.81, 0.48), (0.81, 0.33), color="#2f8c73")
    _arrow(ax, (0.90, 0.77), (0.94, 0.77), color="#9a4d7b")
    _arrow(ax, (0.87, 0.53), (0.94, 0.53), color="#9a4d7b")
    _arrow(ax, (0.92, 0.27), (0.94, 0.27), color="#2f8c73")

    ax.text(0.50, 0.06,
            "Controller {:.2f}M | Extractor {:.2f}M | Discriminator {:.2f}M parameters".format(
                m["controller_params"]/1e6,
                m["extractor_params"]/1e6,
                m["discriminator_params"]/1e6),
            ha="center", fontsize=9, weight="bold")
    ax.text(0.50, 0.02,
            "Objective: $\\mathcal{L}=\\mathcal{L}_{\\mathrm{msg}}"
            "+\\lambda_p\\mathcal{L}_{\\mathrm{LPIPS}}"
            "+\\lambda_s\\mathcal{L}_{\\mathrm{FFT}}"
            "+\\lambda_a\\mathcal{L}_{\\mathrm{adv}}$",
            ha="center", fontsize=9, color="#333")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "architecture_detail.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_training_trajectory(data: Dict) -> None:
    state = data["state"]
    cycles = [h["cycle"] for h in state["history"]]
    clean = [h["clean_ber"] for h in state["history"]]
    jpeg = [h["jpeg_q75_ber"] for h in state["history"]]
    crop = [h["crop80_ber"] for h in state["history"]]
    auc = [h["auc_adaptive"] for h in state["history"]]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True)

    axes[0].plot(cycles, clean, marker="o", lw=2, label="Clean BER", color="#2f8c73")
    axes[0].plot(cycles, jpeg, marker="s", lw=2, label="JPEG Q75 BER", color="#3f6ea7")
    axes[0].axhline(0.10, color="#666", ls="--", lw=1, label="Recovery threshold")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("BER (log scale)")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(cycles, crop, marker="o", lw=2, color="#c36d32")
    axes[1].axhline(0.35, color="#666", ls="--", lw=1, label="Robustness threshold")
    axes[1].axhline(0.50, color="#999", ls=":", lw=1, label="Random guess")
    axes[1].set_ylabel("Crop-resize Q80 BER")
    axes[1].set_ylim(0.49, 0.5055)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].plot(cycles, auc, marker="o", lw=2, color="#9a4d7b")
    axes[2].axhline(0.70, color="#666", ls="--", lw=1, label="Detectability threshold")
    axes[2].set_ylabel("Adaptive AUC")
    axes[2].set_xlabel("Refinement cycle")
    axes[2].set_ylim(0.65, 1.02)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.25)

    for ax in axes:
        ax.set_xticks(cycles)

    axes[0].annotate("cycle 9: minimal\nclean/JPEG BER",
                     xy=(9, clean[8]), xytext=(5.8, 0.0015),
                     arrowprops=dict(arrowstyle="->", color="#2f8c73"), fontsize=8)
    axes[2].annotate("cycle 11: AUC drops\nto 0.6915",
                     xy=(11, auc[10]), xytext=(7.8, 0.78),
                     arrowprops=dict(arrowstyle="->", color="#9a4d7b"), fontsize=8)

    fig.suptitle("Refinement trajectory across eleven training cycles", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "training_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_ablation(data: Dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    variants = data["variant_stats"]["variants"]
    x = [variants[n]["clean_ber_mean"] for n in variants]
    y = [variants[n]["auc_adaptive_mean"] for n in variants]
    c = [variants[n]["crop80_ber_mean"] for n in variants]
    labels = [_short_variant(n) for n in variants]
    sc = axes[0].scatter(x, y, c=c, cmap="plasma", s=180, edgecolor="black")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Mean clean BER (log scale)")
    axes[0].set_ylabel("Mean adaptive AUC")
    axes[0].axhline(0.70, color="#666", ls="--", lw=1)
    for xi, yi, label in zip(x, y, labels):
        axes[0].annotate(label, (xi, yi), textcoords="offset points",
                         xytext=(7, 7), fontsize=8.5)
    axes[0].set_title("Strategy frontier (mean over cycles)")
    axes[0].grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=axes[0], pad=0.02)
    cb.set_label("Mean crop Q80 BER", fontsize=8.5)

    order = ["full", "no_srm_match", "no_spectral", "no_adv", "uniform_attacks"]
    valid = [n for n in order if n in data["component_ablation"]
             and "error" not in data["component_ablation"][n]]
    metrics = ["clean_ber", "jpeg_q75_ber", "crop_q80_ber", "auc"]
    metric_labels = ["Clean BER", "JPEG Q75 BER", "Crop Q80 BER", "Adaptive AUC"]
    colors = ["#2f8c73", "#3f6ea7", "#c36d32", "#9a4d7b"]
    xpos = np.arange(len(valid))
    width = 0.19
    for i, (m, label) in enumerate(zip(metrics, metric_labels)):
        vals = [data["component_ablation"][n][m] for n in valid]
        axes[1].bar(xpos + (i - 1.5) * width, vals, width=width,
                    color=colors[i], label=label)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([_short_variant(n) for n in valid],
                            rotation=18, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Metric value")
    axes[1].set_title("Component ablation (short-run)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_spectral_security(data: Dict) -> None:
    s = data["security"]
    rings = s["power_spectrum"]["ring_diff_sample"]
    svd = s["svd_spectrum"]["top_32_singular_values_normalised"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))

    keys = sorted(int(k) for k in rings.keys())
    vals = [rings[str(k)] for k in keys]
    axes[0].bar(keys, vals, color="#3f6ea7", width=1.6)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Radial power-spectrum residual")
    axes[0].set_xlabel("Ring index")
    axes[0].set_ylabel("Stego $-$ clean (normalised)")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(np.arange(1, len(svd) + 1), svd, color="#c36d32")
    axes[1].set_yscale("log")
    axes[1].set_title("Singular-value spectrum of mean $\\delta$")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Normalised value (log)")
    axes[1].grid(alpha=0.25, axis="y")

    axes[2].set_axis_off()
    lines = [
        "Statistical probes",
        "",
        f"KL(stego $\\|$ clean) Inception = {_display(s['kl_divergence']['kl_stego_vs_clean'])}",
        f"Power-spectrum $\\chi^2$ p = {_display(s['power_spectrum']['p_value'])}",
        f"SVD Gini on mean $\\delta$ = {_display(s['svd_spectrum']['gini_coefficient'])}",
        "",
        f"Bootstrap BER mean = {_display(s['bootstrap_ci']['ber']['mean'])}",
        f"Bootstrap PSNR = {_display(s['bootstrap_ci']['psnr']['mean'], 2)} dB",
        f"Bootstrap LPIPS = {_display(s['bootstrap_ci']['lpips']['mean'])}",
        "",
        "Low Inception divergence does not",
        "imply spectral indistinguishability.",
    ]
    axes[2].text(0.02, 0.98, "\n".join(lines),
                 ha="left", va="top", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "spectral_security.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_robustness_analysis(data: Dict) -> None:
    ev = data["evals"][11]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    names = ["Clean", "JPEG Q75", "JPEG Q50", "Crop Q80", "Crop Q70"]
    means = [
        ev["clean"]["ber"],
        ev["jpeg"]["q75_ber"],
        ev["jpeg"]["q50_ber"],
        ev["crop"]["q80_ber"],
        ev["crop"]["q70_ber"],
    ]
    lows_highs = [
        (ev["clean"]["ber"] - ev["clean"]["ber_ci95"][0],
         ev["clean"]["ber_ci95"][1] - ev["clean"]["ber"]),
        (ev["jpeg"]["q75_ber"] - ev["jpeg"]["q75_ber_ci95"][0],
         ev["jpeg"]["q75_ber_ci95"][1] - ev["jpeg"]["q75_ber"]),
        (0.0, 0.0),
        (ev["crop"]["q80_ber"] - ev["crop"]["q80_ber_ci95"][0],
         ev["crop"]["q80_ber_ci95"][1] - ev["crop"]["q80_ber"]),
        (0.0, 0.0),
    ]
    lows = [a for a, _ in lows_highs]
    highs = [b for _, b in lows_highs]
    colors = ["#2f8c73", "#3f6ea7", "#3f6ea7", "#c36d32", "#c36d32"]
    x = np.arange(len(names))
    axes[0].bar(x, means, yerr=[lows, highs], capsize=4, color=colors)
    axes[0].axhline(0.35, color="#666", ls="--", lw=1, label="Robustness threshold")
    axes[0].axhline(0.50, color="#999", ls=":", lw=1, label="Random guess")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=10)
    axes[0].set_ylim(0, 0.55)
    axes[0].set_ylabel("Bit-error rate")
    axes[0].set_title("Distortion panel, final operating point")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, axis="y")

    cycles = [9, 10, 11]
    id_crop = [data["evals"][c]["crop"]["q80_ber"] for c in cycles]
    cifar_crop = [data["evals"][c]["cross_dataset"]["cifar10"]["crop80_ber"] for c in cycles]
    stl_crop = [data["evals"][c]["cross_dataset"]["stl10"]["crop80_ber"] for c in cycles]
    xpos = np.arange(len(cycles))
    w = 0.24
    axes[1].bar(xpos - w, id_crop, w, label="CelebA-HQ", color="#c36d32")
    axes[1].bar(xpos, cifar_crop, w, label="CIFAR-10", color="#3f6ea7")
    axes[1].bar(xpos + w, stl_crop, w, label="STL-10", color="#9a4d7b")
    axes[1].axhline(0.50, color="#999", ls=":", lw=1)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([str(c) for c in cycles])
    axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("Crop Q80 BER")
    axes[1].set_ylim(0.49, 0.5055)
    axes[1].set_title("Cross-dataset crop-resize behaviour")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "robustness_analysis.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_version_evolution() -> None:
    """Conceptual diagram comparing v1/v2 tiled, v3 refined, v4 adaptive."""
    fig, ax = plt.subplots(figsize=(12, 3.4))
    _blank(ax)

    _box(ax, 0.02, 0.30, 0.22, 0.55,
         "v1 / v2: tiled controller\n"
         "$8{\\times}8$ grid of learned\nper-cell perturbations\n"
         "AUC $\\approx 1.0$ from periodic\nboundary signatures",
         "#fff0e6", "#c36d32", fontsize=9)
    _box(ax, 0.27, 0.30, 0.22, 0.55,
         "v3: smaller budget\n"
         "same grid, $\\epsilon_{\\max}\\!\\downarrow$\n"
         "PSNR 44.3 dB, SSIM 0.98\nbut detectability unchanged",
         "#fff7e0", "#ad8b32", fontsize=9)
    _box(ax, 0.52, 0.30, 0.22, 0.55,
         "v4 (this work): content-\nadaptive U-Net + carriers\n"
         "trade-off frontier accessible\nlow BER or low AUC",
         "#dce9f7", "#3f6ea7", fontsize=9)
    _box(ax, 0.77, 0.30, 0.22, 0.55,
         "v5: exploratory extensions\n"
         "redundant coding + SRM\nfeature matching (short runs)",
         "#e3f4ef", "#2f8c73", fontsize=9)

    _arrow(ax, (0.24, 0.57), (0.27, 0.57))
    _arrow(ax, (0.49, 0.57), (0.52, 0.57))
    _arrow(ax, (0.74, 0.57), (0.77, 0.57))

    ax.text(0.50, 0.10,
            "Architectural evolution: tiled lookup $\\to$ budget-reduced tiled $\\to$ "
            "content-adaptive U-Net $\\to$ extended variants.",
            ha="center", fontsize=8.5, color="#444")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "version_evolution.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    write_tables(data)
    fig_system_overview(data)
    fig_architecture_detail(data)
    fig_training_trajectory(data)
    fig_ablation(data)
    fig_spectral_security(data)
    fig_robustness_analysis(data)
    fig_version_evolution()
    print(f"Wrote figures and tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
