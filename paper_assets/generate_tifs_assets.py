import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "paper_assets" / "generated_tables"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def draw_box(ax, x, y, w, h, label, fc="#E8F0FE", ec="#355C7D", fs=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)


def draw_system_overview(path: Path):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, 0.03, 0.62, 0.15, 0.2, "Message bits\n(128)", fc="#FDEBD0", ec="#B9770E")
    draw_box(ax, 0.22, 0.62, 0.17, 0.2, "RedundantCoder\n128 -> 256", fc="#FDEBD0", ec="#B9770E")
    draw_box(ax, 0.03, 0.18, 0.20, 0.22, "Frozen Diffusion\nDDPM + DDIM-20", fc="#E8ECEF", ec="#616A6B")
    draw_box(ax, 0.43, 0.55, 0.17, 0.27, "Content-Adaptive\nController\n(U-Net)", fc="#D6EAF8", ec="#1F618D")
    draw_box(ax, 0.64, 0.56, 0.14, 0.24, "Stego image\nclip(x+delta)", fc="#EAFAF1", ec="#117A65")
    draw_box(ax, 0.82, 0.56, 0.15, 0.24, "Extractor\nResNet-18\nspatial_map", fc="#D5F5E3", ec="#1D8348")
    draw_box(ax, 0.83, 0.16, 0.14, 0.20, "Discriminator\nSRM + SN", fc="#FADBD8", ec="#943126")

    ax.annotate("", xy=(0.22, 0.72), xytext=(0.18, 0.72), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.annotate("", xy=(0.43, 0.72), xytext=(0.39, 0.72), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.annotate("", xy=(0.64, 0.68), xytext=(0.60, 0.68), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.annotate("", xy=(0.82, 0.68), xytext=(0.78, 0.68), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.annotate("", xy=(0.43, 0.30), xytext=(0.23, 0.30), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.annotate("", xy=(0.83, 0.26), xytext=(0.71, 0.56), arrowprops=dict(arrowstyle="->", lw=1.2, linestyle="--"))

    ax.text(0.25, 0.33, "clean image x", fontsize=9)
    ax.text(0.47, 0.86, "delta = C(x, payload)", fontsize=9)
    ax.text(0.84, 0.84, "decoded bits", fontsize=9)

    ax.set_title("DSCSP TIFS Pipeline Overview (code-verified)", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def draw_architecture_detail(path: Path):
    fig, ax = plt.subplots(figsize=(11.5, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Controller detail
    draw_box(ax, 0.03, 0.65, 0.15, 0.15, "Payload\n(256 physical)", fc="#FCF3CF", ec="#B7950B")
    draw_box(ax, 0.21, 0.65, 0.12, 0.15, "MLP\n-> 4x4 map", fc="#FCF3CF", ec="#B7950B")
    draw_box(ax, 0.35, 0.65, 0.12, 0.15, "Upsample\n-> HxW", fc="#FCF3CF", ec="#B7950B")
    draw_box(ax, 0.03, 0.42, 0.16, 0.14, "Cover image\n(3x256x256)", fc="#E8ECEF", ec="#616A6B")
    draw_box(ax, 0.21, 0.42, 0.12, 0.14, "Concat", fc="#D6EAF8", ec="#1F618D")
    draw_box(ax, 0.35, 0.34, 0.20, 0.32, "U-Net\nenc: C->2C->4C\ndec: 4C->2C->C\nskip connections", fc="#D6EAF8", ec="#1F618D")
    draw_box(ax, 0.58, 0.45, 0.12, 0.14, "Direct payload\ncarrier", fc="#FDEDEC", ec="#943126")
    draw_box(ax, 0.72, 0.42, 0.16, 0.18, "delta\n=tanh(.)*max_pert", fc="#EAFAF1", ec="#117A65")

    draw_box(ax, 0.72, 0.72, 0.16, 0.18, "AttackLayer\n(JPEG/blur/noise\ncrop/BC)", fc="#EBDEF0", ec="#6C3483")
    draw_box(ax, 0.90, 0.72, 0.08, 0.18, "Extractor\n(ResNet-18)", fc="#D5F5E3", ec="#1D8348", fs=8)

    # Arrows
    for x1, y1, x2, y2 in [
        (0.18, 0.725, 0.21, 0.725),
        (0.33, 0.725, 0.35, 0.725),
        (0.19, 0.49, 0.21, 0.49),
        (0.33, 0.49, 0.35, 0.49),
        (0.55, 0.52, 0.58, 0.52),
        (0.70, 0.52, 0.72, 0.52),
        (0.80, 0.60, 0.80, 0.72),
        (0.88, 0.81, 0.90, 0.81),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3))

    ax.text(0.74, 0.36, "x_stego = clip(x + delta)", fontsize=9)
    ax.set_title("Detailed Controller + Training Path (from train_v5.py and module code)", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_cycle_performance(history, path: Path):
    cycles = [h["cycle"] for h in history]
    clean = [h["clean_ber"] for h in history]
    jpeg = [h["jpeg_q75_ber"] for h in history]
    crop = [h["crop80_ber"] for h in history]
    auc = [h["auc_adaptive"] for h in history]
    names = [h["variant"] for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True)

    axes[0].plot(cycles, clean, marker="o", label="Clean BER", color="#1f77b4")
    axes[0].plot(cycles, jpeg, marker="s", label="JPEG Q75 BER", color="#ff7f0e")
    axes[0].plot(cycles, crop, marker="^", label="Crop Q80 BER", color="#d62728")
    axes[0].axhline(0.10, color="#1f77b4", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].axhline(0.20, color="#ff7f0e", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].axhline(0.35, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("BER")
    axes[0].set_ylim(0, 0.55)
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=9, loc="upper center")

    axes[1].plot(cycles, auc, marker="d", color="#2ca02c", linewidth=2.0)
    axes[1].axhline(0.70, color="#2ca02c", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Adaptive AUC")
    axes[1].set_xlabel("Autopilot cycle")
    axes[1].set_ylim(0.65, 1.02)
    axes[1].grid(alpha=0.2)

    for x, y, n in zip(cycles, auc, names):
        short = n.replace("ber_recovery_", "ber_").replace("detectability_", "det_")
        axes[1].text(x, y + 0.012, short, fontsize=7, ha="center")

    fig.suptitle("Cycle-by-Cycle TIFS Autopilot Metrics", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_ablation(ablation_results, path: Path):
    ordered = ["full", "no_srm_match", "no_spectral", "no_adv", "uniform_attacks"]
    names = [n for n in ordered if n in ablation_results and "error" not in ablation_results[n]]

    clean = [ablation_results[n]["clean_ber"] for n in names]
    jpeg = [ablation_results[n]["jpeg_q75_ber"] for n in names]
    crop = [ablation_results[n]["crop_q80_ber"] for n in names]
    auc = [ablation_results[n]["auc"] for n in names]

    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True)
    panels = [
        (axes[0, 0], clean, "Clean BER", 0.10, "#1f77b4"),
        (axes[0, 1], jpeg, "JPEG Q75 BER", 0.20, "#ff7f0e"),
        (axes[1, 0], crop, "Crop Q80 BER", 0.35, "#d62728"),
        (axes[1, 1], auc, "Adaptive AUC", 0.70, "#2ca02c"),
    ]

    for ax, values, title, gate, color in panels:
        ax.bar(x, values, color=color, alpha=0.85)
        ax.axhline(gate, linestyle="--", color="black", linewidth=1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)

    for ax in axes[1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)

    fig.suptitle("v5 Ablation (outputs_v5_ablation/ablation_results.json)", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_security(security, path: Path):
    ring = security["power_spectrum"]["ring_diff_sample"]
    rings = np.array([int(k) for k in ring.keys()])
    diffs = np.array([float(v) for v in ring.values()])

    sv = np.array(security["svd_spectrum"]["top_32_singular_values_normalised"], dtype=float)
    ci = security["bootstrap_ci"]

    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.4))

    # Ring diff
    axes[0, 0].plot(rings, diffs, marker="o", color="#1f77b4")
    axes[0, 0].axhline(0.0, linestyle="--", color="black", linewidth=0.8)
    axes[0, 0].set_title("FFT Ring Relative Difference")
    axes[0, 0].set_xlabel("Ring index")
    axes[0, 0].set_ylabel("(stego-clean)/clean")
    axes[0, 0].grid(alpha=0.2)

    # Singular values
    axes[0, 1].plot(np.arange(1, len(sv) + 1), sv, marker="s", color="#d62728")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("SVD Spectrum of Mean Delta")
    axes[0, 1].set_xlabel("Rank")
    axes[0, 1].set_ylabel("Normalized singular value (log)")
    axes[0, 1].grid(alpha=0.2)

    # Bootstrap CI
    labels = ["BER", "PSNR", "LPIPS"]
    means = np.array([ci["ber"]["mean"], ci["psnr"]["mean"], ci["lpips"]["mean"]], dtype=float)
    lo = np.array([
        ci["ber"]["mean"] - ci["ber"]["ci95_lo"],
        ci["psnr"]["mean"] - ci["psnr"]["ci95_lo"],
        ci["lpips"]["mean"] - ci["lpips"]["ci95_lo"],
    ])
    hi = np.array([
        ci["ber"]["ci95_hi"] - ci["ber"]["mean"],
        ci["psnr"]["ci95_hi"] - ci["psnr"]["mean"],
        ci["lpips"]["ci95_hi"] - ci["lpips"]["mean"],
    ])
    xs = np.arange(len(labels))
    axes[1, 0].bar(xs, means, color=["#1f77b4", "#2ca02c", "#ff7f0e"], alpha=0.85)
    axes[1, 0].errorbar(xs, means, yerr=[lo, hi], fmt="none", ecolor="black", capsize=4)
    axes[1, 0].set_xticks(xs)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_title("Bootstrap 95% CIs")
    axes[1, 0].grid(axis="y", alpha=0.2)

    # Scalars
    kl = float(security["kl_divergence"]["kl_stego_vs_clean"])
    p = float(security["power_spectrum"]["p_value"])
    gini = float(security["svd_spectrum"]["gini_coefficient"])
    scalars = [kl, gini, -math.log10(max(p, 1e-300))]
    names = ["KL", "Gini", "-log10(p)"]
    axes[1, 1].bar(np.arange(3), scalars, color=["#9467bd", "#8c564b", "#e377c2"], alpha=0.85)
    axes[1, 1].set_xticks(np.arange(3))
    axes[1, 1].set_xticklabels(names)
    axes[1, 1].set_title("Security Scalars")
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.suptitle("v5 Security/Statistical Analysis", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_crop_failure(cycle11, history, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    # Left: cycle 11 with CIs
    labels = ["Clean", "JPEG Q75", "Crop Q80"]
    vals = [cycle11["clean"]["ber"], cycle11["jpeg"]["q75_ber"], cycle11["crop"]["q80_ber"]]
    cis = [
        cycle11["clean"]["ber_ci95"],
        cycle11["jpeg"]["q75_ber_ci95"],
        cycle11["crop"]["q80_ber_ci95"],
    ]

    lo = [v - c[0] for v, c in zip(vals, cis)]
    hi = [c[1] - v for v, c in zip(vals, cis)]

    x = np.arange(3)
    axes[0].bar(x, vals, color=["#1f77b4", "#ff7f0e", "#d62728"], alpha=0.9)
    axes[0].errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor="black", capsize=4)
    axes[0].axhline(0.35, linestyle="--", color="black", linewidth=1.0, label="crop gate")
    axes[0].axhline(0.5, linestyle=":", color="gray", linewidth=1.0, label="random BER")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("BER")
    axes[0].set_title("Cycle 11 BER by condition")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(axis="y", alpha=0.2)

    # Right: crop trend over all cycles
    cycles = [h["cycle"] for h in history]
    crop = [h["crop80_ber"] for h in history]
    axes[1].plot(cycles, crop, marker="o", color="#d62728", linewidth=2)
    axes[1].axhline(0.35, linestyle="--", color="black", linewidth=1.0, label="target <= 0.35")
    axes[1].axhline(0.50, linestyle=":", color="gray", linewidth=1.0, label="random BER")
    axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("Crop Q80 BER")
    axes[1].set_ylim(0.45, 0.53)
    axes[1].set_title("Crop BER remains unresolved")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8, loc="lower right")

    fig.suptitle("Failure Analysis: Crop Robustness", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_tables(cycle10, cycle11, ablation_results, history):
    # Main quantitative table
    metrics = [
        ("Clean BER", 0.10, cycle10["clean"]["ber"], cycle11["clean"]["ber"]),
        ("JPEG Q75 BER", 0.20, cycle10["jpeg"]["q75_ber"], cycle11["jpeg"]["q75_ber"]),
        ("Crop Q80 BER", 0.35, cycle10["crop"]["q80_ber"], cycle11["crop"]["q80_ber"]),
        ("Adaptive AUC", 0.70, cycle10["adaptive_steganalysis"]["auc_adaptive"], cycle11["adaptive_steganalysis"]["auc_adaptive"]),
        ("PSNR (dB)", 35.0, cycle10["clean"]["psnr"], cycle11["clean"]["psnr"]),
    ]

    lines = [
        r"\\begin{tabular}{lcccc}",
        r"\\toprule",
        r"Metric & Gate & Cycle 10 & Cycle 11 & Gate@11 \\",
        r"\\midrule",
    ]
    for name, gate, v10, v11 in metrics:
        if name == "PSNR (dB)":
            ok = v11 >= gate
        else:
            ok = v11 <= gate
        lines.append(f"{name} & {gate:.3f} & {v10:.6f} & {v11:.6f} & {'PASS' if ok else 'FAIL'} \\\\")
    lines += [r"\\bottomrule", r"\\end{tabular}"]
    (TABLE_DIR / "main_quant_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Cross-dataset table from cycle 11
    cd = cycle11.get("cross_dataset", {})
    lines = [
        r"\\begin{tabular}{lcc}",
        r"\\toprule",
        r"Dataset & Clean BER & Crop Q80 BER \\",
        r"\\midrule",
    ]
    for ds in ["cifar10", "stl10"]:
        row = cd.get(ds, {})
        lines.append(f"{ds.upper()} & {row.get('clean_ber', float('nan')):.6f} & {row.get('crop80_ber', float('nan')):.6f} \\\\")
    lines += [r"\\bottomrule", r"\\end{tabular}"]
    (TABLE_DIR / "cross_dataset_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # v5 ablation table
    order = ["full", "no_srm_match", "no_spectral", "no_adv", "uniform_attacks", "no_redundant"]
    lines = [
        r"\\begin{tabular}{lcccc}",
        r"\\toprule",
        r"Variant & Clean BER & JPEG Q75 BER & Crop Q80 BER & AUC \\",
        r"\\midrule",
    ]
    for name in order:
        row = ablation_results.get(name, {})
        if "error" in row:
            lines.append(f"{latex_escape(name)} & N/A & N/A & N/A & N/A \\\\")
        else:
            lines.append(
                f"{latex_escape(name)} & {row['clean_ber']:.6f} & {row['jpeg_q75_ber']:.6f} & {row['crop_q80_ber']:.6f} & {row['auc']:.6f} \\\\")
    lines += [r"\\bottomrule", r"\\end{tabular}"]
    (TABLE_DIR / "v5_ablation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Cycle summary table
    lines = [
        r"\\begin{tabular}{ccccc}",
        r"\\toprule",
        r"Cycle & Variant & Clean BER & JPEG Q75 BER & Adaptive AUC \\",
        r"\\midrule",
    ]
    for h in history:
        lines.append(
            f"{h['cycle']} & {latex_escape(h['variant'])} & {h['clean_ber']:.6f} & {h['jpeg_q75_ber']:.6f} & {h['auc_adaptive']:.6f} \\\\")
    lines += [r"\\bottomrule", r"\\end{tabular}"]
    (TABLE_DIR / "cycle_summary_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    autopilot = load_json(ROOT / "outputs_tifs" / "autopilot_state.json")
    history = autopilot["history"]

    cycle10 = load_json(ROOT / "outputs_tifs_eval" / "cycle_010_ber_recovery_curriculum.json")
    cycle11 = load_json(ROOT / "outputs_tifs_eval" / "cycle_011_robust_balance.json")
    v5_ablation = load_json(ROOT / "outputs_v5_ablation" / "ablation_results.json")
    security = load_json(ROOT / "outputs_v5_analysis" / "security_analysis.json")

    draw_system_overview(FIG_DIR / "system_overview.png")
    draw_architecture_detail(FIG_DIR / "architecture_detail.png")
    plot_cycle_performance(history, FIG_DIR / "cycle_performance.png")
    plot_ablation(v5_ablation, FIG_DIR / "ablation_v5.png")
    plot_security(security, FIG_DIR / "security_v5.png")
    plot_crop_failure(cycle11, history, FIG_DIR / "crop_failure.png")

    write_tables(cycle10, cycle11, v5_ablation, history)

    manifest = {
        "figures": [
            "figures/system_overview.png",
            "figures/architecture_detail.png",
            "figures/cycle_performance.png",
            "figures/ablation_v5.png",
            "figures/security_v5.png",
            "figures/crop_failure.png",
        ],
        "tables": [
            "paper_assets/generated_tables/main_quant_table.tex",
            "paper_assets/generated_tables/cross_dataset_table.tex",
            "paper_assets/generated_tables/v5_ablation_table.tex",
            "paper_assets/generated_tables/cycle_summary_table.tex",
        ],
    }
    (ROOT / "paper_assets" / "generated_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Generated figures and tables for paper assets.")


if __name__ == "__main__":
    main()
