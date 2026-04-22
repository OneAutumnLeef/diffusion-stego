"""
DSCSP v2 Training Script — TIFS-level upgrade.

Extends the original train.py with:
  1. Adversarial discriminator training (AUC 1.0 -> target <= 0.70)
  2. Optional redundant encoding for crop robustness
  3. Combined attack support (sequential multi-attack composition)
  4. 5-phase curriculum schedule

The diffusion model remains frozen. New components (Discriminator,
RedundantCoder) are integrated non-destructively alongside the existing
Controller + Extractor pipeline.

Usage:
    python train_v2.py
"""

import os
import random
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from dscsp.models.diffusion_wrapper import DiffusionWrapper
from dscsp.models.controller import Controller
from dscsp.models.extractor import Extractor
from dscsp.models.discriminator import Discriminator
from dscsp.losses.composite_loss import DSCSPLoss
from dscsp.utils.metrics import compute_ber, compute_psnr
from dscsp.utils.visualization import save_comparison
from dscsp.utils.redundant_coding import RedundantCoder
from dscsp.augmentations.differentiable import AttackLayer


# ─── Configuration ──────────────────────────────────────────────────────────

CONFIG = {
    # Model
    "model_id": "google/ddpm-celebahq-256",
    "num_bits": 256,            # Physical encoding slots
    "bits_per_cell": 4,
    "ddim_steps": 20,
    "max_perturbation": 0.3,

    # Training
    "batch_size": 8,
    "total_steps": 30000,
    "lr_controller": 5e-4,
    "lr_extractor": 5e-4,
    "lr_min": 1e-5,
    "weight_decay_controller": 0,
    "weight_decay_extractor": 1e-4,
    "grad_clip": 5.0,

    # Image pool
    "pool_size": 128,
    "pool_refresh_interval": 2000,

    # Loss weights
    "lambda_perc": 0.1,
    "lambda_ctrl": 0.0,
    "lambda_ortho": 0.001,
    "warmup_steps": 0,

    # Robustness curriculum
    "robustness_start_step": 3000,
    "robustness_ramp_steps": 7000,
    "robustness_max_prob": 0.7,

    # Attack parameters
    "jpeg_quality_range": (50, 95),
    "blur_kernel_size": 5,
    "blur_sigma_range": (0.5, 2.0),
    "noise_sigma_range": (0.01, 0.05),
    "crop_ratio_range": (0.7, 0.95),
    "brightness_range": (-0.1, 0.1),
    "contrast_range": (0.8, 1.2),

    # ─── NEW: Adversarial training ──
    "use_adversarial": True,
    "use_srm": True,
    "lambda_adv": 0.1,
    "adv_start_step": 5000,
    "adv_ramp_steps": 5000,
    "disc_lr": 2e-4,
    "disc_weight_decay": 1e-4,
    "disc_steps_per_gen_step": 1,
    "disc_label_smoothing": 0.1,    # One-sided label smoothing

    # ─── NEW: Redundant encoding ──
    "use_redundant_encoding": False,
    "message_bits": 128,            # Effective payload when redundancy is on
    "redundancy_factor": 2,

    # ─── NEW: Combined attacks ──
    "use_combined_attacks": True,
    "combined_attack_start_step": 15000,
    "combined_attack_ratio": 0.3,   # Fraction of attacked batches using combos
    "n_combined": 2,

    # Logging
    "log_interval": 10,
    "save_image_interval": 200,
    "checkpoint_interval": 1000,
    "output_dir": "outputs_v2",
    "checkpoint_dir": "checkpoints_v2",

    # Resume
    "resume_path": "",
}


# ─── Image Pool ─────────────────────────────────────────────────────────────

def generate_image_pool(diffusion, pool_size, device):
    """Generate a pool of clean images using the diffusion model."""
    print(f"  Generating {pool_size} clean images...")
    images = []
    batch = 4

    diffusion.unet.to(device)

    with torch.no_grad():
        for i in range(0, pool_size, batch):
            n = min(batch, pool_size - i)
            z = torch.randn(n, *diffusion.image_shape, device=device)
            timesteps = diffusion.timesteps

            for step_i, t in enumerate(timesteps):
                noise_pred = diffusion.predict_noise(z, t)
                alpha_t = diffusion.get_alpha(t)
                alpha_prev = diffusion.get_prev_alpha(step_i)
                x0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                z = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred

            images.append(z.clamp(-1, 1))
            print(f"    {len(images) * batch}/{pool_size}", end="")

    pool = torch.cat(images, dim=0)[:pool_size]
    print(f"  Image pool: {pool.shape}, range: [{pool.min():.3f}, {pool.max():.3f}]")

    diffusion.unet.cpu()
    torch.cuda.empty_cache()

    return pool


# ─── Schedules ──────────────────────────────────────────────────────────────

def get_attack_prob(step, config):
    """Compute attack probability for the current step."""
    start = config["robustness_start_step"]
    ramp = config["robustness_ramp_steps"]
    max_prob = config["robustness_max_prob"]

    if step < start:
        return 0.0
    elif step < start + ramp:
        return max_prob * (step - start) / ramp
    else:
        return max_prob


def should_use_combined_attack(step, config):
    """Determine whether this step should use combined attacks."""
    if not config["use_combined_attacks"]:
        return False
    if step < config["combined_attack_start_step"]:
        return False
    return random.random() < config["combined_attack_ratio"]


# ─── Orthogonality Regularization ───────────────────────────────────────────

def compute_ortho_loss(controller):
    """Encourage orthogonality among the bits_per_cell patterns in each cell."""
    patterns = controller.bit_patterns
    K = patterns.shape[1]
    pf = patterns.view(patterns.shape[0], K, -1)
    gram = torch.bmm(pf, pf.transpose(1, 2))
    eye = torch.eye(K, device=gram.device).unsqueeze(0)
    return ((gram - eye) ** 2).mean()


# ─── Discriminator Training Step ────────────────────────────────────────────

def train_discriminator_step(
    discriminator, disc_optimizer, x_clean, x_stego, label_smoothing=0.1
):
    """
    One discriminator update step.

    Clean images -> label 0 (with smoothing: label_smoothing)
    Stego images -> label 1 (with smoothing: 1 - label_smoothing)

    Returns:
        disc_loss: scalar loss value (detached)
        disc_acc: accuracy of discriminator predictions
    """
    disc_optimizer.zero_grad()

    logits_clean = discriminator(x_clean.detach())
    logits_stego = discriminator(x_stego.detach())

    # One-sided label smoothing
    target_clean = torch.full_like(logits_clean, label_smoothing)
    target_stego = torch.full_like(logits_stego, 1.0 - label_smoothing)

    loss_clean = F.binary_cross_entropy_with_logits(logits_clean, target_clean)
    loss_stego = F.binary_cross_entropy_with_logits(logits_stego, target_stego)
    loss_disc = (loss_clean + loss_stego) / 2

    loss_disc.backward()
    disc_optimizer.step()

    # Compute accuracy
    with torch.no_grad():
        pred_clean = (torch.sigmoid(logits_clean) < 0.5).float()
        pred_stego = (torch.sigmoid(logits_stego) > 0.5).float()
        acc = (pred_clean.sum() + pred_stego.sum()) / (
            pred_clean.numel() + pred_stego.numel()
        )

    return loss_disc.item(), acc.item()


# ─── Training ──────────────────────────────────────────────────────────────

def train(config=None):
    config = config or CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # ── Initialize components ──
    print("Loading diffusion model...")
    diffusion = DiffusionWrapper(
        model_id=config["model_id"],
        device=device,
        num_inference_steps=config["ddim_steps"],
    )
    print(f"  UNet params: {sum(p.numel() for p in diffusion.unet.parameters()) / 1e6:.1f}M (frozen, FP16)")
    print()

    print("Initializing controller...")
    controller = Controller(
        num_bits=config["num_bits"],
        bits_per_cell=config["bits_per_cell"],
        target_res=256,
        max_perturbation=config["max_perturbation"],
    ).to(device)
    print(f"  Controller params: {sum(p.numel() for p in controller.parameters()) / 1e3:.1f}K")

    print("Initializing extractor...")
    extractor = Extractor(
        num_bits=config["num_bits"],
        bits_per_cell=config["bits_per_cell"],
    ).to(device)
    print(f"  Extractor params: {sum(p.numel() for p in extractor.parameters()) / 1e6:.1f}M")

    # Discriminator (optional)
    discriminator = None
    disc_optimizer = None
    if config["use_adversarial"]:
        print("Initializing discriminator...")
        discriminator = Discriminator(use_srm=config["use_srm"]).to(device)
        disc_params = sum(p.numel() for p in discriminator.parameters())
        print(f"  Discriminator params: {disc_params / 1e6:.2f}M")
        disc_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=config["disc_lr"],
            weight_decay=config["disc_weight_decay"],
            betas=(0.5, 0.999),  # Standard GAN optimizer betas
        )

    # Redundant coder (optional)
    redundant_coder = None
    if config["use_redundant_encoding"]:
        print(f"Initializing redundant coder (M={config['message_bits']}, "
              f"N={config['num_bits']}, R={config['redundancy_factor']})...")
        redundant_coder = RedundantCoder(
            message_bits=config["message_bits"],
            physical_slots=config["num_bits"],
            redundancy=config["redundancy_factor"],
            bits_per_cell=config["bits_per_cell"],
        )
        print(f"  Effective payload: {config['message_bits']} bits "
              f"({config['redundancy_factor']}x redundancy)")
    print()

    criterion = DSCSPLoss(
        lambda_perc=config["lambda_perc"],
        lambda_ctrl=config["lambda_ctrl"],
        warmup_steps=config["warmup_steps"],
        lambda_adv=config["lambda_adv"] if config["use_adversarial"] else 0.0,
        adv_start_step=config["adv_start_step"],
        adv_ramp_steps=config["adv_ramp_steps"],
    ).to(device)

    attack_layer = AttackLayer(config).to(device)
    print(f"  Attack layer initialized (5 differentiable attacks, combined={config['use_combined_attacks']})")

    optimizer = torch.optim.Adam([
        {"params": controller.parameters(), "lr": config["lr_controller"],
         "weight_decay": config["weight_decay_controller"]},
        {"params": extractor.parameters(), "lr": config["lr_extractor"],
         "weight_decay": config["weight_decay_extractor"]},
    ], betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["total_steps"], eta_min=config["lr_min"],
    )

    # ── Resume from checkpoint ──
    start_step = 0
    best_ber = 1.0
    if config["resume_path"] and os.path.exists(config["resume_path"]):
        print(f"Resuming from {config['resume_path']}...")
        ckpt = torch.load(config["resume_path"], map_location=device, weights_only=False)
        controller.load_state_dict(ckpt["controller"])
        extractor.load_state_dict(ckpt["extractor"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        best_ber = ckpt.get("best_ber", 1.0)
        # Load discriminator if available
        if discriminator is not None and "discriminator" in ckpt:
            discriminator.load_state_dict(ckpt["discriminator"])
        if disc_optimizer is not None and "disc_optimizer" in ckpt:
            disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        print(f"  Resumed at step {start_step}, best BER: {best_ber:.4f}")
        del ckpt

    # ── Generate image pool ──
    print("Generating image pool...")
    image_pool = generate_image_pool(diffusion, config["pool_size"], device)

    # ── Directories ──
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # ── Training summary ──
    print(f"\nStarting DSCSP v2 training for {config['total_steps']} steps (from step {start_step})...")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Payload: {config['num_bits']} bits ({config['bits_per_cell']} bits/cell)")
    if config["use_redundant_encoding"]:
        print(f"  Redundant encoding: {config['message_bits']} msg bits -> "
              f"{config['num_bits']} physical ({config['redundancy_factor']}x)")
    print(f"  Max perturbation: {config['max_perturbation']}")
    print(f"  Lambda perc: {config['lambda_perc']}, ctrl: {config['lambda_ctrl']}, "
          f"ortho: {config['lambda_ortho']}")
    if config["use_adversarial"]:
        print(f"  Adversarial: lambda={config['lambda_adv']}, starts step {config['adv_start_step']}, "
              f"ramps over {config['adv_ramp_steps']} steps")
    if config["use_combined_attacks"]:
        print(f"  Combined attacks: starts step {config['combined_attack_start_step']}, "
              f"ratio {config['combined_attack_ratio']}")
    print(f"  Robustness: starts step {config['robustness_start_step']}, "
          f"ramps {config['robustness_ramp_steps']} steps, max prob {config['robustness_max_prob']}")
    print(f"  LR schedule: cosine {config['lr_controller']} -> {config['lr_min']}")
    print()

    step_times = []

    for step in range(start_step + 1, config["total_steps"] + 1):
        t0 = time.time()
        controller.train()
        extractor.train()
        attack_layer.train()
        if discriminator is not None:
            discriminator.train()

        # Refresh image pool periodically
        if step > 1 and step % config["pool_refresh_interval"] == 0:
            print("  Refreshing image pool...")
            image_pool = generate_image_pool(diffusion, config["pool_size"], device)

        # Sample batch from image pool
        idx = torch.randint(0, config["pool_size"], (config["batch_size"],))
        x_clean = image_pool[idx]

        # Generate payload
        if redundant_coder is not None:
            message = torch.randint(
                0, 2, (config["batch_size"], config["message_bits"]), device=device
            ).float()
            payload = redundant_coder.encode(message)
        else:
            message = None
            payload = torch.randint(
                0, 2, (config["batch_size"], config["num_bits"]), device=device
            ).float()

        # Controller forward: produce perturbation
        delta = controller(payload)
        x_stego = (x_clean + delta).clamp(-1, 1)
        delta_norms = delta.pow(2).mean()

        # ── Discriminator update ──
        disc_loss_val = 0.0
        disc_acc_val = 0.0
        if (
            discriminator is not None
            and step >= config["adv_start_step"]
        ):
            for _ in range(config["disc_steps_per_gen_step"]):
                disc_loss_val, disc_acc_val = train_discriminator_step(
                    discriminator, disc_optimizer,
                    x_clean, x_stego,
                    label_smoothing=config["disc_label_smoothing"],
                )

        # ── Generator (Controller + Extractor) update ──
        optimizer.zero_grad()

        # Recompute stego (needed because discriminator step used detached copy)
        delta = controller(payload)
        x_stego = (x_clean + delta).clamp(-1, 1)
        delta_norms = delta.pow(2).mean()

        # Apply differentiable attack
        attack_prob = get_attack_prob(step, config)
        use_combined = should_use_combined_attack(step, config)
        x_for_extraction = attack_layer(
            x_stego, attack_prob=attack_prob,
            combined=use_combined, n_combined=config["n_combined"],
        )

        # Extract bits
        bits_pred = extractor(x_for_extraction)

        # Compute adversarial logits (gradients flow to controller)
        disc_logits_stego = None
        if (
            discriminator is not None
            and config["lambda_adv"] > 0
            and step >= config["adv_start_step"]
        ):
            disc_logits_stego = discriminator(x_stego)

        # Compute loss
        # When using redundant encoding, compute dual-level loss
        if redundant_coder is not None and message is not None:
            # Physical-level loss (trains all replicas)
            loss_physical, loss_dict = criterion(
                bits_pred, payload, x_stego, x_clean, delta_norms,
                step=step, disc_logits_stego=disc_logits_stego,
            )
            # Message-level loss (aggregated, drives consistency)
            message_logits = redundant_coder.decode_soft(bits_pred)
            loss_message = F.binary_cross_entropy_with_logits(message_logits, message)
            loss = 0.5 * loss_physical + 0.5 * loss_message
            loss_dict["msg_agg"] = loss_message.item()
        else:
            loss, loss_dict = criterion(
                bits_pred, payload, x_stego, x_clean, delta_norms,
                step=step, disc_logits_stego=disc_logits_stego,
            )

        # Orthogonality regularization
        if config["lambda_ortho"] > 0:
            ortho_loss = compute_ortho_loss(controller)
            loss = loss + config["lambda_ortho"] * ortho_loss
            loss_dict["ortho"] = ortho_loss.item()

        loss.backward()

        # Clip gradients for controller and extractor only
        torch.nn.utils.clip_grad_norm_(
            list(controller.parameters()) + list(extractor.parameters()),
            max_norm=config["grad_clip"],
        )
        optimizer.step()
        scheduler.step()

        step_time = time.time() - t0
        step_times.append(step_time)

        # ── Logging ──
        if step % config["log_interval"] == 0:
            # Compute BER at the message level if using redundancy
            if redundant_coder is not None and message is not None:
                hard_bits = redundant_coder.decode_hard(bits_pred.detach())
                ber = (hard_bits != message).float().mean().item()
            else:
                ber = compute_ber(bits_pred.detach(), payload)

            psnr = compute_psnr(x_stego.detach(), x_clean.detach())
            avg_time = sum(step_times[-10:]) / len(step_times[-10:])
            remaining = avg_time * (config["total_steps"] - step)
            cur_lr = scheduler.get_last_lr()[0]

            log_line = (
                f"Step {step:5d}/{config['total_steps']} | "
                f"BER: {ber:.4f} | PSNR: {psnr:.1f}dB | "
                f"msg: {loss_dict['msg']:.4f} | perc: {loss_dict['perc']:.4f} | "
                f"atk: {attack_prob:.2f} | "
                f"lr: {cur_lr:.1e} | "
                f"{avg_time:.2f}s/step | ETA: {remaining/60:.0f}min"
            )
            if config["use_adversarial"] and step >= config["adv_start_step"]:
                log_line += (
                    f" | D_loss: {disc_loss_val:.3f} D_acc: {disc_acc_val:.2f}"
                    f" adv: {loss_dict.get('adv', 0):.4f}"
                )
            if "ortho" in loss_dict:
                log_line += f" | orth: {loss_dict['ortho']:.4f}"
            if use_combined:
                log_line += " | COMBO"
            print(log_line)

            if ber < best_ber:
                best_ber = ber

        # ── Save images ──
        if step % config["save_image_interval"] == 0:
            save_comparison(
                x_stego.detach(), x_clean.detach(),
                step=step, output_dir=config["output_dir"],
            )

        # ── Checkpoint ──
        if step % config["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(
                config["checkpoint_dir"], f"dscsp_v2_step_{step:06d}.pt"
            )
            ckpt_data = {
                "step": step,
                "controller": controller.state_dict(),
                "extractor": extractor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config,
                "best_ber": best_ber,
            }
            if discriminator is not None:
                ckpt_data["discriminator"] = discriminator.state_dict()
            if disc_optimizer is not None:
                ckpt_data["disc_optimizer"] = disc_optimizer.state_dict()
            torch.save(ckpt_data, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Free memory
        del x_stego, x_for_extraction, delta, delta_norms, bits_pred, loss, payload
        if disc_logits_stego is not None:
            del disc_logits_stego
        if message is not None:
            del message
        if step % 100 == 0:
            torch.cuda.empty_cache()

    # ── Final evaluation ──
    print()
    print("=" * 60)
    print("FINAL EVALUATION (200 samples)")
    print("=" * 60)

    controller.eval()
    extractor.eval()
    total_ber = 0
    total_psnr = 0
    n_eval = 200

    with torch.no_grad():
        for i in range(n_eval):
            idx_ = torch.randint(0, config["pool_size"], (1,))
            x_clean_ = image_pool[idx_]

            if redundant_coder is not None:
                msg_ = torch.randint(0, 2, (1, config["message_bits"]), device=device).float()
                pay_ = redundant_coder.encode(msg_)
            else:
                msg_ = None
                pay_ = torch.randint(0, 2, (1, config["num_bits"]), device=device).float()

            delta_ = controller(pay_)
            x_stego_ = (x_clean_ + delta_).clamp(-1, 1)
            bits_pred_ = extractor(x_stego_)

            if redundant_coder is not None and msg_ is not None:
                hard_ = redundant_coder.decode_hard(bits_pred_)
                total_ber += (hard_ != msg_).float().mean().item()
            else:
                total_ber += compute_ber(bits_pred_, pay_)

            total_psnr += compute_psnr(x_stego_, x_clean_)

    avg_ber = total_ber / n_eval
    avg_psnr = total_psnr / n_eval

    print(f"  Average BER:  {avg_ber:.4f} (target: < 0.05)")
    print(f"  Average PSNR: {avg_psnr:.1f} dB (target: > 25)")
    print(f"  Best BER during training: {best_ber:.4f}")

    # Discriminator accuracy on eval set
    if discriminator is not None:
        discriminator.eval()
        disc_correct = 0
        disc_total = 0
        with torch.no_grad():
            for i in range(min(100, config["pool_size"])):
                xc = image_pool[i:i+1]
                pay_eval = torch.randint(0, 2, (1, config["num_bits"]), device=device).float()
                d_eval = controller(pay_eval)
                xs = (xc + d_eval).clamp(-1, 1)

                lc = discriminator(xc)
                ls = discriminator(xs)
                disc_correct += (torch.sigmoid(lc) < 0.5).sum().item()
                disc_correct += (torch.sigmoid(ls) > 0.5).sum().item()
                disc_total += 2

        disc_acc_final = disc_correct / disc_total
        print(f"  Discriminator accuracy: {disc_acc_final:.4f} (target: ~0.50 = indistinguishable)")
    else:
        disc_acc_final = None

    print()

    if avg_ber < 0.05:
        print("  SUCCESS: BER target met!")
    else:
        print("  BER target not met -- consider more training steps or tuning.")

    # Save final checkpoint
    final_path = os.path.join(config["checkpoint_dir"], "dscsp_v2_final.pt")
    final_data = {
        "step": config["total_steps"],
        "controller": controller.state_dict(),
        "extractor": extractor.state_dict(),
        "config": config,
        "final_ber": avg_ber,
        "final_psnr": avg_psnr,
    }
    if discriminator is not None:
        final_data["discriminator"] = discriminator.state_dict()
        final_data["disc_acc"] = disc_acc_final
    torch.save(final_data, final_path)
    print(f"  Saved final model: {final_path}")

    # Save results JSON
    results = {
        "final_ber": avg_ber,
        "final_psnr": avg_psnr,
        "best_ber": best_ber,
        "disc_acc": disc_acc_final,
        "config": {k: str(v) if isinstance(v, tuple) else v for k, v in config.items()},
    }
    results_path = os.path.join(config["output_dir"], "train_v2_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results: {results_path}")


if __name__ == "__main__":
    train()
