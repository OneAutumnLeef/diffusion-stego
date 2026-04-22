"""
DSCSP Training Script - Phase 2: 256-bit + Robustness

Trains the Controller and Extractor networks end-to-end while keeping the
pre-trained diffusion model frozen. Includes differentiable attack layers
for robustness training with a curriculum schedule.

Key design: Pre-generates a pool of clean images via DDIM and reuses them
across training steps. UNet is moved to CPU between pool refreshes to
save ~440 MB VRAM.

Usage:
    python train.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dscsp.models.diffusion_wrapper import DiffusionWrapper
from dscsp.models.controller import Controller
from dscsp.models.extractor import Extractor
from dscsp.losses.composite_loss import DSCSPLoss
from dscsp.utils.metrics import compute_ber, compute_psnr
from dscsp.utils.visualization import save_comparison
from dscsp.augmentations.differentiable import AttackLayer


# ─── Configuration ──────────────────────────────────────────────────────────

CONFIG = {
    # Model
    "model_id": "google/ddpm-celebahq-256",
    "num_bits": 256,
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
    # Logging
    "log_interval": 10,
    "save_image_interval": 200,
    "checkpoint_interval": 1000,
    "output_dir": "outputs",
    "checkpoint_dir": "checkpoints",
    # Resume
    "resume_path": "",
}


# ─── Image Pool ─────────────────────────────────────────────────────────────

def generate_image_pool(diffusion, pool_size, device):
    """Generate a pool of clean images using the diffusion model."""
    print(f"  Generating {pool_size} clean images...")
    images = []
    batch = 4  # Generate 4 at a time

    # Ensure UNet is on the right device
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

    # Move UNet to CPU to save VRAM during training
    diffusion.unet.cpu()
    torch.cuda.empty_cache()

    return pool


# ─── Attack Probability Schedule ────────────────────────────────────────────

def get_attack_prob(step):
    """Compute attack probability for the current step."""
    start = CONFIG["robustness_start_step"]
    ramp = CONFIG["robustness_ramp_steps"]
    max_prob = CONFIG["robustness_max_prob"]

    if step < start:
        return 0.0
    elif step < start + ramp:
        return max_prob * (step - start) / ramp
    else:
        return max_prob


# ─── Orthogonality Regularization ───────────────────────────────────────────

def compute_ortho_loss(controller):
    """Encourage orthogonality among the bits_per_cell patterns in each cell."""
    patterns = controller.bit_patterns  # (num_cells, bits_per_cell, 3, C, C)
    K = patterns.shape[1]
    # Flatten spatial dims: (num_cells, bits_per_cell, 3*C*C)
    pf = patterns.view(patterns.shape[0], K, -1)
    # Gram matrix: (num_cells, K, K)
    gram = torch.bmm(pf, pf.transpose(1, 2))
    # Target: identity (orthogonal patterns)
    eye = torch.eye(K, device=gram.device).unsqueeze(0)
    return ((gram - eye) ** 2).mean()


# ─── Smoke Test ─────────────────────────────────────────────────────────────

def smoke_test(image_pool, controller, extractor, criterion, device):
    """Quick validation with image pool approach."""
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    # Random payload
    payload = torch.randint(0, 2, (1, CONFIG["num_bits"]), device=device).float()
    print(f"  Payload shape: {payload.shape}, sum: {payload.sum().item():.0f}")

    # Get one clean image from pool
    x_clean = image_pool[:1]

    # Controller forward
    delta = controller(payload)
    x_stego = (x_clean + delta).clamp(-1, 1)
    delta_norms = delta.pow(2).mean()

    print(f"  Stego shape: {x_stego.shape}, range: [{x_stego.min():.3f}, {x_stego.max():.3f}]")
    print(f"  Clean shape: {x_clean.shape}, range: [{x_clean.min():.3f}, {x_clean.max():.3f}]")
    print(f"  Delta norms: {delta_norms.item():.8f}")

    # Extractor
    bits_pred = extractor(x_stego)
    print(f"  Bits pred shape: {bits_pred.shape}, range: [{bits_pred.min():.3f}, {bits_pred.max():.3f}]")

    # Metrics
    ber = compute_ber(bits_pred, payload)
    psnr = compute_psnr(x_stego.detach(), x_clean.detach())
    print(f"  BER: {ber:.4f} (expected ~0.50)")
    print(f"  PSNR: {psnr:.1f} dB")

    # Loss
    loss, loss_dict = criterion(bits_pred, payload, x_stego, x_clean, delta_norms, step=0)
    print(f"  Loss: {loss.item():.4f} (msg={loss_dict['msg']:.4f}, "
          f"perc={loss_dict['perc']:.4f}, ctrl={loss_dict['ctrl']:.6f})")

    # Backward pass
    loss.backward()

    ctrl_has_grad = all(p.grad is not None for p in controller.parameters())
    ext_has_grad = all(p.grad is not None for p in extractor.parameters())
    ctrl_grad_norm = sum(p.grad.norm().item() for p in controller.parameters() if p.grad is not None)
    ext_grad_norm = sum(p.grad.norm().item() for p in extractor.parameters() if p.grad is not None)

    print(f"  Controller gradients: {'OK' if ctrl_has_grad else 'MISSING!'} (norm: {ctrl_grad_norm:.6f})")
    print(f"  Extractor gradients:  {'OK' if ext_has_grad else 'MISSING!'} (norm: {ext_grad_norm:.6f})")

    # Zero gradients after smoke test
    controller.zero_grad()
    extractor.zero_grad()

    # Save initial comparison image
    save_comparison(x_stego.detach(), x_clean.detach(), step=0, output_dir=CONFIG["output_dir"])
    print(f"  Saved initial comparison to {CONFIG['output_dir']}/step_000000.png")

    assert ctrl_has_grad, "Controller has no gradients — check implementation!"
    assert ext_has_grad, "Extractor has no gradients — check implementation!"

    print("=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    print()


# ─── Training Loop ──────────────────────────────────────────────────────────

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # ── Initialize components ──
    print("Loading diffusion model...")
    diffusion = DiffusionWrapper(
        model_id=CONFIG["model_id"],
        device=device,
        num_inference_steps=CONFIG["ddim_steps"],
    )
    print(f"  UNet params: {sum(p.numel() for p in diffusion.unet.parameters()) / 1e6:.1f}M (frozen, FP16)")
    print()

    print("Initializing controller...")
    controller = Controller(
        num_bits=CONFIG["num_bits"],
        bits_per_cell=CONFIG["bits_per_cell"],
        target_res=256,
        max_perturbation=CONFIG["max_perturbation"],
    ).to(device)
    print(f"  Controller params: {sum(p.numel() for p in controller.parameters()) / 1e3:.1f}K")

    print("Initializing extractor...")
    extractor = Extractor(
        num_bits=CONFIG["num_bits"],
        bits_per_cell=CONFIG["bits_per_cell"],
    ).to(device)
    print(f"  Extractor params: {sum(p.numel() for p in extractor.parameters()) / 1e6:.1f}M")
    print()

    criterion = DSCSPLoss(
        lambda_perc=CONFIG["lambda_perc"],
        lambda_ctrl=CONFIG["lambda_ctrl"],
        warmup_steps=CONFIG["warmup_steps"],
    ).to(device)

    attack_layer = AttackLayer(CONFIG).to(device)
    print(f"  Attack layer initialized (5 differentiable attacks)")

    optimizer = torch.optim.Adam([
        {"params": controller.parameters(), "lr": CONFIG["lr_controller"],
         "weight_decay": CONFIG["weight_decay_controller"]},
        {"params": extractor.parameters(), "lr": CONFIG["lr_extractor"],
         "weight_decay": CONFIG["weight_decay_extractor"]},
    ], betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["total_steps"], eta_min=CONFIG["lr_min"],
    )

    # ── Resume from checkpoint ──
    start_step = 0
    best_ber = 1.0
    if CONFIG["resume_path"] and os.path.exists(CONFIG["resume_path"]):
        print(f"Resuming from {CONFIG['resume_path']}...")
        ckpt = torch.load(CONFIG["resume_path"], map_location=device, weights_only=False)
        controller.load_state_dict(ckpt["controller"])
        extractor.load_state_dict(ckpt["extractor"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        best_ber = ckpt.get("best_ber", 1.0)
        print(f"  Resumed at step {start_step}, best BER: {best_ber:.4f}")
        del ckpt

    # ── Generate image pool ──
    print("Generating image pool...")
    image_pool = generate_image_pool(diffusion, CONFIG["pool_size"], device)

    # ── Directories ──
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # ── Smoke test ──
    smoke_test(image_pool, controller, extractor, criterion, device)

    # ── Training ──
    print(f"Starting training for {CONFIG['total_steps']} steps (from step {start_step})...")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Payload: {CONFIG['num_bits']} bits ({CONFIG['bits_per_cell']} bits/cell)")
    print(f"  Max perturbation: {CONFIG['max_perturbation']}")
    print(f"  Image pool size: {CONFIG['pool_size']}")
    print(f"  Pool refresh interval: {CONFIG['pool_refresh_interval']}")
    print(f"  Lambda perc: {CONFIG['lambda_perc']}, Lambda ctrl: {CONFIG['lambda_ctrl']}, "
          f"Lambda ortho: {CONFIG['lambda_ortho']}")
    print(f"  Robustness: starts at step {CONFIG['robustness_start_step']}, "
          f"ramps over {CONFIG['robustness_ramp_steps']} steps, "
          f"max prob {CONFIG['robustness_max_prob']}")
    print(f"  LR schedule: cosine {CONFIG['lr_controller']} -> {CONFIG['lr_min']}")
    print()

    step_times = []

    for step in range(start_step + 1, CONFIG["total_steps"] + 1):
        t0 = time.time()
        controller.train()
        extractor.train()
        attack_layer.train()

        # Refresh image pool periodically
        if step > 1 and step % CONFIG["pool_refresh_interval"] == 0:
            print("  Refreshing image pool...")
            image_pool = generate_image_pool(diffusion, CONFIG["pool_size"], device)

        optimizer.zero_grad()

        # Sample batch from image pool
        idx = torch.randint(0, CONFIG["pool_size"], (CONFIG["batch_size"],))
        x_clean = image_pool[idx]  # (B, 3, 256, 256), detached

        # Random payload
        payload = torch.randint(
            0, 2, (CONFIG["batch_size"], CONFIG["num_bits"]), device=device
        ).float()

        # Controller forward: produce perturbation
        delta = controller(payload)  # (B, 3, 256, 256)
        x_stego = (x_clean + delta).clamp(-1, 1)
        delta_norms = delta.pow(2).mean()

        # Apply differentiable attack (gradient flows through)
        attack_prob = get_attack_prob(step)
        x_for_extraction = attack_layer(x_stego, attack_prob=attack_prob)

        # Extract bits (from possibly-attacked image)
        bits_pred = extractor(x_for_extraction)

        # Compute loss (LPIPS uses x_stego vs x_clean, NOT attacked version)
        loss, loss_dict = criterion(
            bits_pred, payload, x_stego, x_clean, delta_norms, step=step
        )

        # Orthogonality regularization
        if CONFIG["lambda_ortho"] > 0:
            ortho_loss = compute_ortho_loss(controller)
            loss = loss + CONFIG["lambda_ortho"] * ortho_loss
            loss_dict["ortho"] = ortho_loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(controller.parameters()) + list(extractor.parameters()),
            max_norm=CONFIG["grad_clip"],
        )
        optimizer.step()
        scheduler.step()

        step_time = time.time() - t0
        step_times.append(step_time)

        # ── Logging ──
        if step % CONFIG["log_interval"] == 0:
            ber = compute_ber(bits_pred.detach(), payload)
            psnr = compute_psnr(x_stego.detach(), x_clean.detach())
            avg_time = sum(step_times[-10:]) / len(step_times[-10:])
            remaining = avg_time * (CONFIG["total_steps"] - step)
            cur_lr = scheduler.get_last_lr()[0]

            log_line = (
                f"Step {step:5d}/{CONFIG['total_steps']} | "
                f"BER: {ber:.4f} | PSNR: {psnr:.1f}dB | "
                f"msg: {loss_dict['msg']:.4f} | perc: {loss_dict['perc']:.4f} | "
                f"ctrl: {loss_dict['ctrl']:.6f} | "
                f"atk: {attack_prob:.2f} | "
                f"lr: {cur_lr:.1e} | "
                f"{avg_time:.2f}s/step | ETA: {remaining/60:.0f}min"
            )
            if "ortho" in loss_dict:
                log_line += f" | orth: {loss_dict['ortho']:.4f}"
            print(log_line)

            if ber < best_ber:
                best_ber = ber

        # ── Save images ──
        if step % CONFIG["save_image_interval"] == 0:
            save_comparison(
                x_stego.detach(), x_clean.detach(),
                step=step, output_dir=CONFIG["output_dir"],
            )

        # ── Checkpoint ──
        if step % CONFIG["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(
                CONFIG["checkpoint_dir"], f"dscsp_step_{step:06d}.pt"
            )
            torch.save({
                "step": step,
                "controller": controller.state_dict(),
                "extractor": extractor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": CONFIG,
                "best_ber": best_ber,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Free memory
        del x_stego, x_clean, x_for_extraction, delta, delta_norms, bits_pred, loss, payload
        if step % 100 == 0:
            torch.cuda.empty_cache()

    # ── Final evaluation ──
    print()
    print("=" * 60)
    print("FINAL EVALUATION (100 samples)")
    print("=" * 60)

    controller.eval()
    extractor.eval()
    total_ber = 0
    total_psnr = 0
    n_eval = 100

    with torch.no_grad():
        for i in range(n_eval):
            idx = torch.randint(0, CONFIG["pool_size"], (1,))
            x_clean = image_pool[idx]
            payload = torch.randint(0, 2, (1, CONFIG["num_bits"]), device=device).float()
            delta = controller(payload)
            x_stego = (x_clean + delta).clamp(-1, 1)
            bits_pred = extractor(x_stego)
            total_ber += compute_ber(bits_pred, payload)
            total_psnr += compute_psnr(x_stego, x_clean)

    avg_ber = total_ber / n_eval
    avg_psnr = total_psnr / n_eval

    print(f"  Average BER:  {avg_ber:.4f} (target: < 0.05)")
    print(f"  Average PSNR: {avg_psnr:.1f} dB (target: > 25)")
    print(f"  Best BER during training: {best_ber:.4f}")
    print()

    if avg_ber < 0.05:
        print("  SUCCESS: BER target met!")
    else:
        print("  BER target not met — consider more training steps or tuning hyperparameters.")

    # Save final checkpoint
    final_path = os.path.join(CONFIG["checkpoint_dir"], "dscsp_final.pt")
    torch.save({
        "step": CONFIG["total_steps"],
        "controller": controller.state_dict(),
        "extractor": extractor.state_dict(),
        "config": CONFIG,
        "final_ber": avg_ber,
        "final_psnr": avg_psnr,
    }, final_path)
    print(f"  Saved final model: {final_path}")


if __name__ == "__main__":
    train()
