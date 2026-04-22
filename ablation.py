"""
DSCSP Ablation Study Framework

Systematically trains and evaluates ablation variants to isolate the
contribution of each component. Each variant modifies a specific config
parameter while keeping everything else constant.

Variants:
  1. baseline         - Full system (reference)
  2. no_ortho         - Without orthogonality loss
  3. no_curriculum    - Attacks from step 0 (no curriculum)
  4. no_lpips         - Without perceptual loss
  5. no_attack        - Without attack layer (clean only)
  6. no_spatial_align - Random bit ordering (breaks grid alignment)
  7. no_adversarial   - Without discriminator

Each variant trains for a configurable number of steps (default 15K),
then evaluates on 200 clean samples. Results are saved incrementally
to ablation_results.json.

Usage:
    python ablation.py
    python ablation.py --steps 15000 --variants baseline no_ortho no_lpips
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dscsp.models.diffusion_wrapper import DiffusionWrapper
from dscsp.models.controller import Controller
from dscsp.models.extractor import Extractor
from dscsp.models.discriminator import Discriminator
from dscsp.losses.composite_loss import DSCSPLoss
from dscsp.utils.metrics import compute_ber, compute_psnr
from dscsp.augmentations.differentiable import AttackLayer


# ─── Base Configuration ─────────────────────────────────────────────────────

BASE_CONFIG = {
    "model_id": "google/ddpm-celebahq-256",
    "num_bits": 256,
    "bits_per_cell": 4,
    "ddim_steps": 20,
    "max_perturbation": 0.3,
    "batch_size": 8,
    "total_steps": 15000,
    "lr_controller": 5e-4,
    "lr_extractor": 5e-4,
    "lr_min": 1e-5,
    "weight_decay_controller": 0,
    "weight_decay_extractor": 1e-4,
    "grad_clip": 5.0,
    "pool_size": 128,
    "pool_refresh_interval": 2000,
    "lambda_perc": 0.1,
    "lambda_ctrl": 0.0,
    "lambda_ortho": 0.001,
    "warmup_steps": 0,
    "robustness_start_step": 3000,
    "robustness_ramp_steps": 4000,   # Shorter ramp for 15K total
    "robustness_max_prob": 0.7,
    "jpeg_quality_range": (50, 95),
    "blur_kernel_size": 5,
    "blur_sigma_range": (0.5, 2.0),
    "noise_sigma_range": (0.01, 0.05),
    "crop_ratio_range": (0.7, 0.95),
    "brightness_range": (-0.1, 0.1),
    "contrast_range": (0.8, 1.2),
    # Adversarial (baseline includes it)
    "use_adversarial": True,
    "use_srm": True,
    "lambda_adv": 0.1,
    "adv_start_step": 5000,
    "adv_ramp_steps": 5000,
    "disc_lr": 2e-4,
    "disc_weight_decay": 1e-4,
    "disc_label_smoothing": 0.1,
    # Ablation-specific
    "shuffle_bit_order": False,
}


# ─── Ablation Variants ──────────────────────────────────────────────────────

ABLATION_VARIANTS = {
    "baseline": {},

    "no_ortho": {
        "lambda_ortho": 0.0,
    },

    "no_curriculum": {
        "robustness_start_step": 0,
        "robustness_ramp_steps": 0,
        "robustness_max_prob": 0.7,
    },

    "no_lpips": {
        "lambda_perc": 0.0,
    },

    "no_attack": {
        "robustness_max_prob": 0.0,
    },

    "no_spatial_align": {
        "shuffle_bit_order": True,
    },

    "no_adversarial": {
        "use_adversarial": False,
        "lambda_adv": 0.0,
    },
}


# ─── Image Pool ─────────────────────────────────────────────────────────────

def generate_image_pool(diffusion, pool_size, device):
    """Generate a pool of clean images using the diffusion model."""
    images = []
    batch = 4
    diffusion.unet.to(device)

    with torch.no_grad():
        for i in range(0, pool_size, batch):
            n = min(batch, pool_size - i)
            z = torch.randn(n, *diffusion.image_shape, device=device)
            for step_i, t in enumerate(diffusion.timesteps):
                noise_pred = diffusion.predict_noise(z, t)
                alpha_t = diffusion.get_alpha(t)
                alpha_prev = diffusion.get_prev_alpha(step_i)
                x0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                z = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred
            images.append(z.clamp(-1, 1))

    pool = torch.cat(images, dim=0)[:pool_size]
    diffusion.unet.cpu()
    torch.cuda.empty_cache()
    return pool


# ─── Helpers ────────────────────────────────────────────────────────────────

def get_attack_prob(step, config):
    start = config["robustness_start_step"]
    ramp = config["robustness_ramp_steps"]
    max_prob = config["robustness_max_prob"]
    if step < start:
        return 0.0
    elif ramp > 0 and step < start + ramp:
        return max_prob * (step - start) / ramp
    else:
        return max_prob


def compute_ortho_loss(controller):
    patterns = controller.bit_patterns
    K = patterns.shape[1]
    pf = patterns.view(patterns.shape[0], K, -1)
    gram = torch.bmm(pf, pf.transpose(1, 2))
    eye = torch.eye(K, device=gram.device).unsqueeze(0)
    return ((gram - eye) ** 2).mean()


# ─── Train One Variant ──────────────────────────────────────────────────────

def train_variant(variant_name, config, diffusion, image_pool, device):
    """Train a single ablation variant and return results."""
    print(f"\n  Initializing {variant_name}...")

    controller = Controller(
        num_bits=config["num_bits"],
        bits_per_cell=config["bits_per_cell"],
        target_res=256,
        max_perturbation=config["max_perturbation"],
    ).to(device)

    extractor = Extractor(
        num_bits=config["num_bits"],
        bits_per_cell=config["bits_per_cell"],
    ).to(device)

    discriminator = None
    disc_optimizer = None
    if config["use_adversarial"]:
        discriminator = Discriminator(use_srm=config["use_srm"]).to(device)
        disc_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=config["disc_lr"],
            weight_decay=config["disc_weight_decay"],
            betas=(0.5, 0.999),
        )

    criterion = DSCSPLoss(
        lambda_perc=config["lambda_perc"],
        lambda_ctrl=config["lambda_ctrl"],
        warmup_steps=config["warmup_steps"],
        lambda_adv=config.get("lambda_adv", 0.0) if config["use_adversarial"] else 0.0,
        adv_start_step=config.get("adv_start_step", 5000),
        adv_ramp_steps=config.get("adv_ramp_steps", 5000),
    ).to(device)

    attack_layer = AttackLayer(config).to(device)

    optimizer = torch.optim.Adam([
        {"params": controller.parameters(), "lr": config["lr_controller"],
         "weight_decay": config["weight_decay_controller"]},
        {"params": extractor.parameters(), "lr": config["lr_extractor"],
         "weight_decay": config["weight_decay_extractor"]},
    ], betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["total_steps"], eta_min=config["lr_min"],
    )

    # Bit permutation for no_spatial_align variant
    bit_perm = None
    inv_perm = None
    if config.get("shuffle_bit_order", False):
        bit_perm = torch.randperm(config["num_bits"],
                                  generator=torch.Generator().manual_seed(999)).to(device)
        inv_perm = torch.argsort(bit_perm)

    # Training loop
    total_steps = config["total_steps"]
    best_ber = 1.0

    for step in range(1, total_steps + 1):
        controller.train()
        extractor.train()
        attack_layer.train()
        if discriminator is not None:
            discriminator.train()

        # Refresh pool
        if step > 1 and step % config["pool_refresh_interval"] == 0:
            image_pool = generate_image_pool(diffusion, config["pool_size"], device)

        # Sample batch
        idx = torch.randint(0, config["pool_size"], (config["batch_size"],))
        x_clean = image_pool[idx]
        payload = torch.randint(
            0, 2, (config["batch_size"], config["num_bits"]), device=device
        ).float()

        # Spatial alignment ablation
        if bit_perm is not None:
            payload_ctrl = payload[:, bit_perm]
        else:
            payload_ctrl = payload

        delta = controller(payload_ctrl)
        x_stego = (x_clean + delta).clamp(-1, 1)
        delta_norms = delta.pow(2).mean()

        # Discriminator step
        if discriminator is not None and step >= config.get("adv_start_step", 5000):
            disc_optimizer.zero_grad()
            logits_c = discriminator(x_clean.detach())
            logits_s = discriminator(x_stego.detach())
            ls = config["disc_label_smoothing"]
            d_loss = (
                F.binary_cross_entropy_with_logits(logits_c, torch.full_like(logits_c, ls)) +
                F.binary_cross_entropy_with_logits(logits_s, torch.full_like(logits_s, 1 - ls))
            ) / 2
            d_loss.backward()
            disc_optimizer.step()

        # Generator step
        optimizer.zero_grad()
        delta = controller(payload_ctrl)
        x_stego = (x_clean + delta).clamp(-1, 1)
        delta_norms = delta.pow(2).mean()

        attack_prob = get_attack_prob(step, config)
        x_for_extraction = attack_layer(x_stego, attack_prob=attack_prob)

        bits_pred = extractor(x_for_extraction)

        # Undo permutation for loss
        if inv_perm is not None:
            bits_pred_aligned = bits_pred[:, inv_perm]
        else:
            bits_pred_aligned = bits_pred

        disc_logits = None
        if discriminator is not None and step >= config.get("adv_start_step", 5000):
            disc_logits = discriminator(x_stego)

        loss, loss_dict = criterion(
            bits_pred_aligned, payload, x_stego, x_clean, delta_norms,
            step=step, disc_logits_stego=disc_logits,
        )

        if config["lambda_ortho"] > 0:
            ortho_loss = compute_ortho_loss(controller)
            loss = loss + config["lambda_ortho"] * ortho_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(controller.parameters()) + list(extractor.parameters()),
            max_norm=config["grad_clip"],
        )
        optimizer.step()
        scheduler.step()

        # Progress logging
        if step % 500 == 0:
            ber = compute_ber(bits_pred_aligned.detach(), payload)
            psnr = compute_psnr(x_stego.detach(), x_clean.detach())
            if ber < best_ber:
                best_ber = ber
            print(f"    [{variant_name}] Step {step:5d}/{total_steps} | "
                  f"BER: {ber:.4f} | PSNR: {psnr:.1f}dB | loss: {loss_dict['total']:.4f}")

        del x_stego, x_for_extraction, delta, bits_pred, loss, payload
        if step % 100 == 0:
            torch.cuda.empty_cache()

    # ── Evaluation ──
    print(f"  Evaluating {variant_name}...")
    controller.eval()
    extractor.eval()
    n_eval = 200
    all_ber, all_psnr = [], []

    with torch.no_grad():
        for i in range(n_eval):
            idx_ = torch.randint(0, config["pool_size"], (1,))
            xc = image_pool[idx_]
            pay = torch.randint(0, 2, (1, config["num_bits"]), device=device).float()

            if bit_perm is not None:
                pay_ctrl = pay[:, bit_perm]
            else:
                pay_ctrl = pay

            d = controller(pay_ctrl)
            xs = (xc + d).clamp(-1, 1)
            bp = extractor(xs)

            if inv_perm is not None:
                bp = bp[:, inv_perm]

            all_ber.append(compute_ber(bp, pay))
            all_psnr.append(compute_psnr(xs, xc))

    result = {
        "variant": variant_name,
        "overrides": {k: str(v) if isinstance(v, tuple) else v
                      for k, v in ABLATION_VARIANTS.get(variant_name, {}).items()},
        "ber": float(np.mean(all_ber)),
        "psnr": float(np.mean(all_psnr)),
        "best_ber": best_ber,
        "total_steps": total_steps,
    }

    # Clean up
    del controller, extractor, criterion, attack_layer, optimizer, scheduler
    if discriminator is not None:
        del discriminator, disc_optimizer
    torch.cuda.empty_cache()

    return result


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DSCSP Ablation Study")
    parser.add_argument('--steps', type=int, default=15000,
                        help='Training steps per variant')
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Specific variants to run (default: all)')
    parser.add_argument('--output', default='ablation_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Select variants
    if args.variants:
        variants_to_run = {k: v for k, v in ABLATION_VARIANTS.items()
                           if k in args.variants}
    else:
        variants_to_run = ABLATION_VARIANTS

    print(f"\nRunning {len(variants_to_run)} ablation variants: {list(variants_to_run.keys())}")
    print(f"Steps per variant: {args.steps}")

    # Load diffusion model once (shared across all variants)
    print("\nLoading diffusion model (shared)...")
    diffusion = DiffusionWrapper(
        model_id=BASE_CONFIG["model_id"],
        device=device,
        num_inference_steps=BASE_CONFIG["ddim_steps"],
    )

    # Generate initial image pool
    print("Generating shared image pool...")
    image_pool = generate_image_pool(diffusion, BASE_CONFIG["pool_size"], device)

    # Load existing results if resuming
    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {args.output}")

    # Run variants
    for variant_name, overrides in variants_to_run.items():
        if variant_name in results:
            print(f"\n{'='*70}")
            print(f"SKIPPING {variant_name} (already completed)")
            print(f"{'='*70}")
            continue

        print(f"\n{'='*70}")
        print(f"ABLATION: {variant_name}")
        print(f"  Overrides: {overrides}")
        print(f"{'='*70}")

        config = {**BASE_CONFIG, **overrides}
        config["total_steps"] = args.steps

        t0 = time.time()
        result = train_variant(variant_name, config, diffusion, image_pool, device)
        elapsed = time.time() - t0
        result["training_time_sec"] = round(elapsed, 1)

        results[variant_name] = result

        # Save incrementally
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Result: BER={result['ber']:.4f}, PSNR={result['psnr']:.1f}dB, "
              f"time={elapsed/60:.1f}min")
        print(f"  Saved to {args.output}")

    # ── Summary Table ──
    print(f"\n{'='*70}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<20s} {'BER':>8s} {'PSNR':>8s} {'BestBER':>8s} {'Time':>8s}")
    print("-" * 56)

    baseline_ber = results.get("baseline", {}).get("ber", None)
    baseline_psnr = results.get("baseline", {}).get("psnr", None)

    for name, res in results.items():
        ber_str = f"{res['ber']:.4f}"
        psnr_str = f"{res['psnr']:.1f}"
        best_ber_str = f"{res['best_ber']:.4f}"
        time_str = f"{res.get('training_time_sec', 0)/60:.0f}m"

        # Relative delta from baseline
        delta_info = ""
        if baseline_ber is not None and name != "baseline":
            ber_delta = res["ber"] - baseline_ber
            psnr_delta = res["psnr"] - baseline_psnr
            delta_info = f"  (BER {'+' if ber_delta >= 0 else ''}{ber_delta:.4f}, " \
                         f"PSNR {'+' if psnr_delta >= 0 else ''}{psnr_delta:.1f})"

        print(f"{name:<20s} {ber_str:>8s} {psnr_str:>8s} {best_ber_str:>8s} {time_str:>8s}{delta_info}")

    # LaTeX table
    print(f"\n{'='*70}")
    print("LaTeX Table (copy-paste ready):")
    print(f"{'='*70}")
    print(r"\begin{tabular}{@{}lccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Variant} & \textbf{BER} & \textbf{PSNR (dB)} & \textbf{$\Delta$ BER} \\")
    print(r"\midrule")
    for name, res in results.items():
        ber_delta = ""
        if baseline_ber is not None and name != "baseline":
            d = res["ber"] - baseline_ber
            ber_delta = f"{'+' if d >= 0 else ''}{d:.4f}"
        elif name == "baseline":
            ber_delta = "---"
        display_name = name.replace("_", r"\_")
        print(f"{display_name} & {res['ber']:.4f} & {res['psnr']:.1f} & {ber_delta} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
