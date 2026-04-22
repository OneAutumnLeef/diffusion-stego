"""
DSCSP v5 Security & Statistical Analysis.

Generates the statistical analysis required for TIFS reviewers:
  1. KL divergence — diagonal Gaussian fit on InceptionV3 pool features
  2. Power spectrum chi-square test — FFT magnitude histogram comparison
  3. Singular value spectrum — SVD of mean stego delta (absence of grid pattern)
  4. Bootstrap CI table — BER / PSNR / LPIPS with 95% CI

Outputs:
  outputs_v5_analysis/security_analysis.json   — all numeric results
  outputs_v5_analysis/power_spectrum.png        — frequency domain comparison
  outputs_v5_analysis/svd_spectrum.png          — singular value decay plot
  outputs_v5_analysis/feature_distribution.png — feature distribution violin plots

Usage:
  python analyze_security.py --checkpoint checkpoints_v5/dscsp_v5_final.pt
  python analyze_security.py --checkpoint checkpoints_v5/dscsp_v5_final.pt \\
      --n-images 1000 --output-dir outputs_v5_analysis
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from scipy.stats import chi2_contingency

from dscsp.models.diffusion_wrapper import DiffusionWrapper
from dscsp.models.controller_v4 import ContentAdaptiveController
from dscsp.utils.metrics import compute_ber, compute_psnr
from dscsp.utils.redundant_coding import RedundantCoder


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_v5_model(checkpoint_path: str, device: str):
    """Load controller + extractor from a v5 checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    phys_bits = int(config.get("num_bits", 256))
    msg_bits = int(ckpt.get("message_bits", config.get("message_bits", 128)))
    R = int(ckpt.get("redundancy_factor", config.get("redundancy_factor", 2)))
    use_rc = bool(config.get("use_redundant_coding", True)) and (phys_bits != msg_bits)

    max_pert = float(config.get("max_perturbation", 0.08))
    base_ch = int(config.get("base_channels", 48))
    ext_mode = str(config.get("extractor_mode", "spatial_map"))
    ext_grid = int(config.get("extractor_grid_size", 16))

    controller = ContentAdaptiveController(
        num_bits=phys_bits, max_perturbation=max_pert, base_channels=base_ch,
    ).to(device)

    # Import GlobalExtractor from the training script
    sys.path.insert(0, os.path.dirname(__file__))
    if os.path.exists("train_v5.py"):
        from train_v5 import GlobalExtractor
    else:
        from train_v4 import GlobalExtractor
    extractor = GlobalExtractor(num_bits=phys_bits, mode=ext_mode, grid_size=ext_grid).to(device)

    controller.load_state_dict(ckpt["controller"])
    extractor.load_state_dict(ckpt["extractor"])
    controller.eval()
    extractor.eval()

    coder = None
    if use_rc:
        coder = RedundantCoder(message_bits=msg_bits, physical_slots=phys_bits, redundancy=R)
        print(f"  RedundantCoder: {msg_bits} msg bits → {phys_bits} physical (R={R})")
    else:
        msg_bits = phys_bits

    print(f"  Controller: {sum(p.numel() for p in controller.parameters())/1e6:.2f}M params")
    print(f"  Extractor:  {sum(p.numel() for p in extractor.parameters())/1e6:.2f}M params")
    return controller, extractor, coder, msg_bits, phys_bits, config


# ─── Image Generation ────────────────────────────────────────────────────────

@torch.no_grad()
def generate_pairs(controller, extractor, coder, msg_bits, phys_bits, config, device, n):
    """Generate N (clean, stego) image pairs and compute messages."""
    model_id = config.get("model_id", "google/ddpm-celebahq-256")
    ddim_steps = int(config.get("ddim_steps", 20))

    print(f"Generating {n} image pairs via diffusion model...")
    diffusion = DiffusionWrapper(model_id=model_id, device=device, num_inference_steps=ddim_steps)
    diffusion.unet.to(device)

    cleans, stegos, messages, deltas = [], [], [], []
    batch = 4
    for i in range(0, n, batch):
        bs = min(batch, n - i)
        z = torch.randn(bs, *diffusion.image_shape, device=device)
        for si, t in enumerate(diffusion.timesteps):
            noise_pred = diffusion.predict_noise(z, t)
            a = diffusion.get_alpha(t)
            ap = diffusion.get_prev_alpha(si)
            x0 = (z - (1 - a).sqrt() * noise_pred) / a.sqrt()
            z = ap.sqrt() * x0 + (1 - ap).sqrt() * noise_pred
        x_c = z.clamp(-1, 1)

        msg = torch.randint(0, 2, (bs, msg_bits), device=device).float()
        physical = coder.encode(msg) if coder else msg
        delta = controller(x_c, physical)
        x_s = (x_c + delta).clamp(-1, 1)

        cleans.append(x_c.cpu())
        stegos.append(x_s.cpu())
        messages.append(msg.cpu())
        deltas.append(delta.cpu())
        print(f"  {min(i + batch, n)}/{n}", end="\r")

    print()
    diffusion.unet.cpu()
    torch.cuda.empty_cache()
    return (
        torch.cat(cleans, 0),
        torch.cat(stegos, 0),
        torch.cat(messages, 0),
        torch.cat(deltas, 0),
    )


# ─── Bootstrap CI ────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    arr = np.asarray(values, dtype=np.float64)
    means = [arr[np.random.randint(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


# ─── 1. KL Divergence ────────────────────────────────────────────────────────

def compute_kl_divergence(clean_imgs, stego_imgs, device, batch=32):
    """
    KL divergence KL(stego || clean) using diagonal Gaussian fits on
    InceptionV3 pool-2048 features.
    """
    print("Computing KL divergence (InceptionV3 pool features)...")
    inception = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.DEFAULT)
    inception.fc = nn.Identity()
    inception.aux_logits = False
    inception = inception.to(device).eval()

    def get_features(imgs):
        feats = []
        imgs_rgb = (imgs + 1.0) / 2.0  # [-1,1] → [0,1]
        imgs_rgb = F.interpolate(imgs_rgb, size=(299, 299), mode="bilinear", align_corners=False)
        for i in range(0, len(imgs), batch):
            x = imgs_rgb[i:i+batch].to(device)
            with torch.no_grad():
                f = inception(x)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    f_clean = get_features(clean_imgs)
    f_stego = get_features(stego_imgs)

    mu_c, sigma_c = f_clean.mean(0), f_clean.std(0) + 1e-8
    mu_s, sigma_s = f_stego.mean(0), f_stego.std(0) + 1e-8

    # KL(N(mu_s, sigma_s^2) || N(mu_c, sigma_c^2)) per dimension, summed
    kl_per_dim = (
        np.log(sigma_c / sigma_s)
        + (sigma_s**2 + (mu_s - mu_c)**2) / (2 * sigma_c**2)
        - 0.5
    )
    kl = float(kl_per_dim.sum())
    print(f"  KL(stego||clean) = {kl:.4f}")

    del inception
    torch.cuda.empty_cache()
    return kl, float(f_clean.mean()), float(f_stego.mean())


# ─── 2. Power Spectrum Chi-Square Test ───────────────────────────────────────

def compute_power_spectrum_test(clean_imgs, stego_imgs, n_bins=64):
    """
    Chi-square test comparing FFT log-magnitude histograms of clean vs stego.
    Returns chi2 statistic, p-value, and per-frequency-ring difference.
    """
    print("Computing power spectrum chi-square test...")

    def log_mag_hist(imgs):
        imgs_t = imgs.float()
        fft = torch.fft.fft2(imgs_t)
        mag = torch.log1p(fft.abs()).mean(1)  # mean over channels → (N, H, W)
        # Flatten spatial
        return mag.reshape(len(imgs), -1)

    clean_mag = log_mag_hist(clean_imgs).numpy()
    stego_mag = log_mag_hist(stego_imgs).numpy()

    # Compute global histogram
    all_vals = np.concatenate([clean_mag, stego_mag])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

    hist_clean = np.histogram(clean_mag, bins=bin_edges)[0].astype(float) + 1  # Laplace smoothing
    hist_stego = np.histogram(stego_mag, bins=bin_edges)[0].astype(float) + 1

    # Normalize to equal total counts for chi-square
    hist_clean_n = hist_clean / hist_clean.sum() * hist_stego.sum()
    chi2_stat, p_value = chi2_contingency(
        np.stack([hist_clean_n, hist_stego], axis=0)
    )[:2]

    # Per-radial-ring mean difference (shows where in frequency domain artifacts live)
    h, w = clean_imgs.shape[-2], clean_imgs.shape[-1]
    cy, cx = h // 2, w // 2
    ys = torch.arange(h) - cy
    xs = torch.arange(w) - cx
    radii = (ys[:, None]**2 + xs[None, :]**2).float().sqrt()  # (H, W)
    max_r = int(radii.max().item())

    fft_clean = torch.fft.fftshift(torch.fft.fft2(clean_imgs.float().mean(1))).abs()
    fft_stego = torch.fft.fftshift(torch.fft.fft2(stego_imgs.float().mean(1))).abs()
    mean_diff_by_ring = {}
    for r in range(0, min(max_r, 64), 2):
        mask = ((radii >= r) & (radii < r + 2)).unsqueeze(0)
        c_r = fft_clean[:, mask[0]].mean().item()
        s_r = fft_stego[:, mask[0]].mean().item()
        mean_diff_by_ring[r] = (s_r - c_r) / (c_r + 1e-8)

    print(f"  Chi-square stat: {chi2_stat:.2f}, p-value: {p_value:.4e}")
    print(f"  Interpretation: {'DETECTABLE (p<0.05)' if p_value < 0.05 else 'NOT detectable (p≥0.05)'}")
    return float(chi2_stat), float(p_value), mean_diff_by_ring


# ─── 3. Singular Value Spectrum ──────────────────────────────────────────────

def compute_svd_spectrum(deltas, n_components=32):
    """
    SVD of the mean stego delta matrix.
    Flat/decaying spectrum = no periodic grid pattern.
    Sharp spike = periodic artifact (bad for TIFS).
    """
    print("Computing SVD spectrum of stego deltas...")
    mean_delta = deltas.mean(0)  # (3, H, W)
    grey = mean_delta.mean(0)    # (H, W)
    U, S, Vh = torch.linalg.svd(grey.float())
    S_np = S[:n_components].numpy()
    S_norm = S_np / (S_np.sum() + 1e-8)

    # Gini coefficient as a concentration measure (higher = more periodic)
    sorted_s = np.sort(S_norm)
    n = len(sorted_s)
    gini = (2 * np.arange(1, n + 1) - n - 1).dot(sorted_s) / (n * sorted_s.sum() + 1e-8)

    print(f"  Top singular values (normalised): {S_norm[:8]}")
    print(f"  Gini coefficient: {gini:.4f} (0=flat/good, 1=spike/bad)")
    return S_norm.tolist(), float(gini)


# ─── 4. Bootstrap CI Table ───────────────────────────────────────────────────

@torch.no_grad()
def compute_bootstrap_ci_table(
    controller, extractor, coder, msg_bits, phys_bits,
    clean_imgs, stego_imgs, messages, device
):
    """Compute per-sample BER, PSNR, LPIPS and return bootstrap 95% CI."""
    import lpips
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    bers, psnrs, lpips_vals = [], [], []
    for i in range(len(clean_imgs)):
        x_c = clean_imgs[i:i+1].to(device)
        x_s = stego_imgs[i:i+1].to(device)
        msg = messages[i:i+1].to(device)

        logits = extractor(x_s)
        if coder:
            msg_hard = coder.decode_hard(logits)
            ber = compute_ber(msg_hard, msg)
        else:
            ber = compute_ber(logits, msg)

        psnr = compute_psnr(x_s, x_c)
        lp = lpips_fn(x_s, x_c).item()

        bers.append(ber)
        psnrs.append(psnr)
        lpips_vals.append(lp)

    ber_lo, ber_hi = bootstrap_ci(bers)
    psnr_lo, psnr_hi = bootstrap_ci(psnrs)
    lpips_lo, lpips_hi = bootstrap_ci(lpips_vals)

    table = {
        "ber":   {"mean": float(np.mean(bers)),   "ci95_lo": ber_lo,   "ci95_hi": ber_hi},
        "psnr":  {"mean": float(np.mean(psnrs)),  "ci95_lo": psnr_lo,  "ci95_hi": psnr_hi},
        "lpips": {"mean": float(np.mean(lpips_vals)), "ci95_lo": lpips_lo, "ci95_hi": lpips_hi},
    }
    print("\nBootstrap 95% CI table:")
    for k, v in table.items():
        print(f"  {k}: {v['mean']:.4f} [{v['ci95_lo']:.4f}, {v['ci95_hi']:.4f}]")
    return table


# ─── Plotting ────────────────────────────────────────────────────────────────

def _save_plots(power_diff, svd_spectrum, ci_table, output_dir):
    """Save matplotlib figures (skips gracefully if matplotlib is unavailable)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    # Power spectrum ring difference
    rings = sorted(power_diff.keys())
    diffs = [power_diff[r] for r in rings]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(rings, diffs, width=1.8, color="steelblue", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Radial frequency ring")
    ax.set_ylabel("Relative power diff (stego - clean) / clean")
    ax.set_title("Power spectrum: stego vs. clean (positive = stego boosted)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "power_spectrum.png"), dpi=150)
    plt.close()
    print(f"  Saved power_spectrum.png")

    # SVD spectrum
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(len(svd_spectrum)), svd_spectrum, color="darkorange", alpha=0.8)
    ax.set_xlabel("Singular value rank")
    ax.set_ylabel("Normalised singular value")
    ax.set_title("SVD spectrum of mean stego delta (flat = no grid artifact)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "svd_spectrum.png"), dpi=150)
    plt.close()
    print(f"  Saved svd_spectrum.png")

    # Bootstrap CI bar chart
    metrics = list(ci_table.keys())
    means = [ci_table[m]["mean"] for m in metrics]
    lo = [ci_table[m]["mean"] - ci_table[m]["ci95_lo"] for m in metrics]
    hi = [ci_table[m]["ci95_hi"] - ci_table[m]["mean"] for m in metrics]
    fig, ax = plt.subplots(figsize=(6, 3))
    x = range(len(metrics))
    ax.bar(x, means, yerr=[lo, hi], capsize=5, color=["royalblue", "seagreen", "tomato"])
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_title("Key metrics with 95% bootstrap CI")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bootstrap_ci.png"), dpi=150)
    plt.close()
    print(f"  Saved bootstrap_ci.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DSCSP v5 Security Analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to v5 checkpoint")
    parser.add_argument("--n-images", type=int, default=500, help="Number of image pairs to generate")
    parser.add_argument("--output-dir", default="outputs_v5_analysis", help="Output directory")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detect if unset)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"DSCSP v5 Security Analysis")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Images: {args.n_images}")
    print()

    # Load model
    controller, extractor, coder, msg_bits, phys_bits, config = load_v5_model(
        args.checkpoint, device
    )

    # Generate pairs
    clean_imgs, stego_imgs, messages, deltas = generate_pairs(
        controller, extractor, coder, msg_bits, phys_bits, config, device, args.n_images
    )

    results = {}

    # 1. KL divergence
    kl, feat_mean_clean, feat_mean_stego = compute_kl_divergence(clean_imgs, stego_imgs, device)
    results["kl_divergence"] = {
        "kl_stego_vs_clean": kl,
        "inception_feat_mean_clean": feat_mean_clean,
        "inception_feat_mean_stego": feat_mean_stego,
        "interpretation": (
            "low (<1.0): distributions indistinguishable; "
            "high (>5.0): significant distribution shift"
        ),
    }

    # 2. Power spectrum test
    chi2_stat, p_value, diff_by_ring = compute_power_spectrum_test(clean_imgs, stego_imgs)
    results["power_spectrum"] = {
        "chi2_statistic": chi2_stat,
        "p_value": p_value,
        "detectable_p05": p_value < 0.05,
        "ring_diff_sample": {str(k): round(v, 6) for k, v in list(diff_by_ring.items())[:16]},
    }

    # 3. SVD spectrum
    svd_spectrum, gini = compute_svd_spectrum(deltas)
    results["svd_spectrum"] = {
        "top_32_singular_values_normalised": svd_spectrum,
        "gini_coefficient": gini,
        "interpretation": "gini<0.3: no periodic artifact; gini>0.7: grid pattern present",
    }

    # 4. Bootstrap CI table
    ci_table = compute_bootstrap_ci_table(
        controller, extractor, coder, msg_bits, phys_bits,
        clean_imgs, stego_imgs, messages, device
    )
    results["bootstrap_ci"] = ci_table

    # Save JSON
    out_json = os.path.join(args.output_dir, "security_analysis.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_json}")

    # Save plots
    _save_plots(diff_by_ring, svd_spectrum, ci_table, args.output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SECURITY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  KL divergence:          {kl:.4f}")
    print(f"  Power spectrum p-value: {p_value:.4e} "
          f"({'FAIL' if p_value < 0.05 else 'PASS'})")
    print(f"  SVD Gini coefficient:   {gini:.4f} "
          f"({'bad' if gini > 0.5 else 'good'})")
    print(f"  Clean BER:              {ci_table['ber']['mean']:.4f} "
          f"[{ci_table['ber']['ci95_lo']:.4f}, {ci_table['ber']['ci95_hi']:.4f}]")
    print(f"  PSNR:                   {ci_table['psnr']['mean']:.2f} dB "
          f"[{ci_table['psnr']['ci95_lo']:.2f}, {ci_table['psnr']['ci95_hi']:.2f}]")
    print(f"  LPIPS:                  {ci_table['lpips']['mean']:.4f} "
          f"[{ci_table['lpips']['ci95_lo']:.4f}, {ci_table['lpips']['ci95_hi']:.4f}]")


if __name__ == "__main__":
    main()
