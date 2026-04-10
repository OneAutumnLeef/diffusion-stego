"""
DSCSP v4 Training Script — Content-Adaptive Architecture.

Key changes from v2:
  1. ContentAdaptiveController (U-Net) replaces tiled-grid Controller
  2. Global-pool Extractor (ResNet-18 + GAP + FC) replaces grid-aligned head
  3. JPEG-heavy attack schedule (80% probability, from step 0)
  4. Adversarial training from step 0 with spectral normalization
  5. No orthogonality loss (no grid patterns to regularize)

Usage:
    python train_v4.py
"""

import os
import random
import signal
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.nn.utils import spectral_norm

from dscsp.models.diffusion_wrapper import DiffusionWrapper
from dscsp.models.controller_v4 import ContentAdaptiveController
from dscsp.models.discriminator import SRMFilterLayer
from dscsp.losses.composite_loss import DSCSPLoss
from dscsp.utils.metrics import compute_ber, compute_psnr
from dscsp.utils.visualization import save_comparison
from dscsp.augmentations.differentiable import AttackLayer


# ─── Configuration ──────────────────────────────────────────────────────────

CONFIG = {
    # Model
    "model_id": "google/ddpm-celebahq-256",
    "num_bits": 256,
    "ddim_steps": 20,
    "max_perturbation": 0.08,
    "base_channels": 48,

    # Training
    "batch_size": 4,
    "total_steps": 50000,
    "lr_controller": 1e-4,
    "lr_extractor": 3e-4,
    "lr_min": 1e-6,
    "weight_decay_controller": 1e-5,
    "weight_decay_extractor": 1e-4,
    "grad_clip": 5.0,

    # Image pool
    "pool_size": 128,
    "pool_refresh_interval": 2000,

    # Loss weights
    "lambda_perc": 0.3,
    "lambda_ctrl": 0.0,
    "warmup_steps": 0,

    # Attack schedule — JPEG-heavy
    "robustness_start_step": 500,
    "robustness_ramp_steps": 2000,
    "robustness_max_prob": 0.85,
    "jpeg_weight": 0.6,            # 60% of attacks are JPEG

    # Attack parameters
    "jpeg_quality_range": (40, 95),
    "blur_kernel_size": 5,
    "blur_sigma_range": (0.5, 2.0),
    "noise_sigma_range": (0.01, 0.05),
    "crop_ratio_range": (0.7, 0.95),
    "brightness_range": (-0.1, 0.1),
    "contrast_range": (0.8, 1.2),

    # Adversarial training — from step 0, strong
    "use_adversarial": True,
    "use_srm": True,
    "lambda_adv": 0.5,
    "adv_start_step": 1000,
    "adv_ramp_steps": 4000,
    "disc_lr": 1e-4,
    "disc_weight_decay": 1e-4,
    "disc_steps_per_gen_step": 3,
    "disc_label_smoothing": 0.1,

    # Combined attacks
    "use_combined_attacks": True,
    "combined_attack_start_step": 15000,
    "combined_attack_ratio": 0.3,
    "n_combined": 2,

    # Logging
    "log_interval": 10,
    "save_image_interval": 500,
    "checkpoint_interval": 2000,
    "output_dir": "outputs_v4",
    "checkpoint_dir": "checkpoints_v4",

    # Resume
    "resume_path": "",
}


# ─── Global-Pool Extractor ─────────────────────────────────────────────────

class GlobalExtractor(nn.Module):
    """
    Bit extractor using ResNet-18 with global average pooling.

    Unlike the grid-aligned v2 extractor (1x1 conv on 8x8 features),
    this uses GAP + FC to predict all bits from global features.
    No spatial alignment assumptions — works with content-adaptive controller.
    """

    def __init__(self, num_bits: int = 256):
        super().__init__()
        self.num_bits = num_bits
        backbone = tv_models.resnet18(weights=None)
        # Replace final FC layer
        backbone.fc = nn.Linear(512, num_bits)
        self.backbone = backbone

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, 256, 256) stego image in [-1, 1]
        Returns:
            (B, num_bits) logits
        """
        return self.backbone(image)


# ─── Spectral-Norm Discriminator ───────────────────────────────────────────

class SNDiscriminator(nn.Module):
    """
    Steganalysis discriminator with spectral normalization for training stability.

    Classifies images as clean (0) or stego (1).
    Uses SRM preprocessing + 5 conv blocks with spectral norm (no batch norm).
    """

    def __init__(self, use_srm: bool = True):
        super().__init__()
        self.use_srm = use_srm

        if use_srm:
            self.srm = SRMFilterLayer()
            in_channels = 9
        else:
            in_channels = 3

        # 5 conv blocks with spectral normalization (no BN — incompatible with SN)
        self.blocks = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 256, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 256, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = spectral_norm(nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.srm is not None:
            x = self.srm(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


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


# ─── JPEG-Heavy Attack Wrapper ──────────────────────────────────────────────

class JPEGHeavyAttackLayer(nn.Module):
    """
    Attack layer that applies JPEG compression with higher probability
    than other attacks. Wraps the base AttackLayer but overrides attack
    selection to weight JPEG more heavily.
    """

    def __init__(self, config):
        super().__init__()
        self.base_attack = AttackLayer(config)
        self.jpeg_weight = config.get("jpeg_weight", 0.6)

    def forward(self, x, attack_prob=0.0, combined=False, n_combined=2):
        if attack_prob <= 0 or random.random() > attack_prob:
            return x

        if combined and n_combined > 1:
            return self.base_attack(x, attack_prob=1.0, combined=True, n_combined=n_combined)

        # Weighted attack selection: JPEG gets higher weight
        if random.random() < self.jpeg_weight:
            return self.base_attack._apply_single(x, 'jpeg')
        else:
            # Apply a random non-JPEG attack
            non_jpeg = random.choice(['blur', 'noise', 'crop_resize', 'brightness_contrast'])
            return self.base_attack._apply_single(x, non_jpeg)


# ─── Schedules ──────────────────────────────────────────────────────────────

def get_attack_prob(step, config):
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
    if not config["use_combined_attacks"]:
        return False
    if step < config["combined_attack_start_step"]:
        return False
    return random.random() < config["combined_attack_ratio"]


# ─── Discriminator Training Step ────────────────────────────────────────────

def train_discriminator_step(discriminator, disc_optimizer, x_clean, x_stego, label_smoothing=0.1):
    disc_optimizer.zero_grad()
    logits_clean = discriminator(x_clean.detach())
    logits_stego = discriminator(x_stego.detach())

    target_clean = torch.full_like(logits_clean, label_smoothing)
    target_stego = torch.full_like(logits_stego, 1.0 - label_smoothing)

    loss_clean = F.binary_cross_entropy_with_logits(logits_clean, target_clean)
    loss_stego = F.binary_cross_entropy_with_logits(logits_stego, target_stego)
    loss_disc = (loss_clean + loss_stego) / 2

    loss_disc.backward()
    disc_optimizer.step()

    with torch.no_grad():
        pred_clean = (torch.sigmoid(logits_clean) < 0.5).float()
        pred_stego = (torch.sigmoid(logits_stego) > 0.5).float()
        acc = (pred_clean.sum() + pred_stego.sum()) / (pred_clean.numel() + pred_stego.numel())

    return loss_disc.item(), acc.item()


# ─── Spectral Loss ──────────────────────────────────────────────────────────

def compute_spectral_loss(x_stego, x_clean):
    """
    Penalize differences in the frequency domain (FFT magnitude spectrum).
    This discourages the controller from creating spectral artifacts that
    steganalysis detectors can easily identify.
    """
    # Compute 2D FFT magnitude
    fft_stego = torch.fft.fft2(x_stego)
    fft_clean = torch.fft.fft2(x_clean)

    mag_stego = torch.abs(fft_stego)
    mag_clean = torch.abs(fft_clean)

    # Log-magnitude difference (log to handle dynamic range)
    loss = F.mse_loss(
        torch.log1p(mag_stego),
        torch.log1p(mag_clean),
    )
    return loss


# ─── Training ──────────────────────────────────────────────────────────────

def train(config=None):
    config = config or CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # ── Signal handler — save checkpoint on SIGINT / SIGTERM ──
    # Allows `kill -TERM <pid>` or Ctrl-C to gracefully save progress (Lesson 14).
    _interrupted = [False]

    def _handle_signal(signum, frame):
        print(f"\n  [Signal {signum}] Interrupt received — will checkpoint and exit after this step.")
        _interrupted[0] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Initialize diffusion model ──
    print("Loading diffusion model...")
    diffusion = DiffusionWrapper(
        model_id=config["model_id"],
        device=device,
        num_inference_steps=config["ddim_steps"],
    )
    print(f"  UNet params: {sum(p.numel() for p in diffusion.unet.parameters()) / 1e6:.1f}M (frozen)")
    print()

    # ── Initialize v4 components ──
    print("Initializing content-adaptive controller...")
    controller = ContentAdaptiveController(
        num_bits=config["num_bits"],
        max_perturbation=config["max_perturbation"],
        base_channels=config["base_channels"],
    ).to(device)
    ctrl_params = sum(p.numel() for p in controller.parameters())
    print(f"  Controller params: {ctrl_params / 1e6:.2f}M")

    print("Initializing global-pool extractor...")
    extractor = GlobalExtractor(num_bits=config["num_bits"]).to(device)
    ext_params = sum(p.numel() for p in extractor.parameters())
    print(f"  Extractor params: {ext_params / 1e6:.2f}M")

    # Discriminator with spectral normalization
    discriminator = None
    disc_optimizer = None
    if config["use_adversarial"]:
        print("Initializing SN-discriminator...")
        discriminator = SNDiscriminator(use_srm=config["use_srm"]).to(device)
        disc_params = sum(p.numel() for p in discriminator.parameters())
        print(f"  Discriminator params: {disc_params / 1e6:.2f}M")
        disc_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=config["disc_lr"],
            weight_decay=config["disc_weight_decay"],
            betas=(0.0, 0.9),  # TTUR betas for spectral-norm discriminator
        )

    criterion = DSCSPLoss(
        lambda_perc=config["lambda_perc"],
        lambda_ctrl=config["lambda_ctrl"],
        warmup_steps=config["warmup_steps"],
        lambda_adv=config["lambda_adv"] if config["use_adversarial"] else 0.0,
        adv_start_step=config["adv_start_step"],
        adv_ramp_steps=config["adv_ramp_steps"],
    ).to(device)

    attack_layer = JPEGHeavyAttackLayer(config).to(device)
    print(f"  JPEG-heavy attack layer (JPEG weight: {config['jpeg_weight']})")

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
    total_params = ctrl_params + ext_params
    if discriminator is not None:
        total_params += disc_params
    print(f"\nStarting DSCSP v4 training for {config['total_steps']} steps (from step {start_step})...")
    print(f"  Total trainable params: {total_params / 1e6:.2f}M")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Payload: {config['num_bits']} bits")
    print(f"  Max perturbation: {config['max_perturbation']}")
    print(f"  Lambda perc: {config['lambda_perc']}, adv: {config['lambda_adv']}")
    print(f"  Adversarial: starts step {config['adv_start_step']}, "
          f"ramps {config['adv_ramp_steps']} steps, {config['disc_steps_per_gen_step']} D steps/G step")
    print(f"  Attacks: starts step {config['robustness_start_step']}, "
          f"max prob {config['robustness_max_prob']}, JPEG weight {config['jpeg_weight']}")
    print(f"  LR: controller={config['lr_controller']}, extractor={config['lr_extractor']}")
    print()

    step_times = []

    for step in range(start_step + 1, config["total_steps"] + 1):
        t0 = time.time()
        controller.train()
        extractor.train()
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
        payload = torch.randint(
            0, 2, (config["batch_size"], config["num_bits"]), device=device
        ).float()

        # Controller forward: content-adaptive perturbation
        delta = controller(x_clean, payload)
        x_stego = (x_clean + delta).clamp(-1, 1)
        delta_norms = delta.pow(2).mean()

        # ── Discriminator update ──
        disc_loss_val = 0.0
        disc_acc_val = 0.0
        if discriminator is not None and step >= config["adv_start_step"]:
            for _ in range(config["disc_steps_per_gen_step"]):
                disc_loss_val, disc_acc_val = train_discriminator_step(
                    discriminator, disc_optimizer,
                    x_clean, x_stego,
                    label_smoothing=config["disc_label_smoothing"],
                )

        # ── Generator (Controller + Extractor) update ──
        optimizer.zero_grad()

        # Recompute stego (discriminator step used detached copy)
        delta = controller(x_clean, payload)
        x_stego = (x_clean + delta).clamp(-1, 1)
        delta_norms = delta.pow(2).mean()

        # Apply JPEG-heavy differentiable attack
        attack_prob = get_attack_prob(step, config)
        use_combined = should_use_combined_attack(step, config)
        x_for_extraction = attack_layer(
            x_stego, attack_prob=attack_prob,
            combined=use_combined, n_combined=config["n_combined"],
        )

        # Extract bits
        bits_pred = extractor(x_for_extraction)

        # Compute adversarial logits
        disc_logits_stego = None
        if (
            discriminator is not None
            and config["lambda_adv"] > 0
            and step >= config["adv_start_step"]
        ):
            disc_logits_stego = discriminator(x_stego)

        # Compute composite loss
        loss, loss_dict = criterion(
            bits_pred, payload, x_stego, x_clean, delta_norms,
            step=step, disc_logits_stego=disc_logits_stego,
        )

        # Spectral loss (penalize frequency-domain differences)
        spec_loss = compute_spectral_loss(x_stego, x_clean)
        loss = loss + 0.01 * spec_loss
        loss_dict["spec"] = spec_loss.item()

        loss.backward()

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
            ber = compute_ber(bits_pred.detach(), payload)
            psnr = compute_psnr(x_stego.detach(), x_clean.detach())
            avg_time = sum(step_times[-10:]) / len(step_times[-10:])
            remaining = avg_time * (config["total_steps"] - step)
            cur_lr = scheduler.get_last_lr()[0]

            log_line = (
                f"Step {step:5d}/{config['total_steps']} | "
                f"BER: {ber:.4f} | PSNR: {psnr:.1f}dB | "
                f"msg: {loss_dict['msg']:.4f} | perc: {loss_dict['perc']:.4f} | "
                f"spec: {loss_dict['spec']:.4f} | "
                f"atk: {attack_prob:.2f} | "
                f"lr: {cur_lr:.1e} | "
                f"{avg_time:.2f}s/step | ETA: {remaining/60:.0f}min"
            )
            if config["use_adversarial"] and step >= config["adv_start_step"]:
                log_line += (
                    f" | D_loss: {disc_loss_val:.3f} D_acc: {disc_acc_val:.2f}"
                    f" adv: {loss_dict.get('adv', 0):.4f}"
                )
            print(log_line)

            if ber < best_ber:
                best_ber = ber

            # Write a small JSON progress file — allows monitoring without
            # stdout buffering (Lesson 5).
            progress = {
                "step": step,
                "total_steps": config["total_steps"],
                "ber": round(ber, 6),
                "psnr": round(psnr, 2),
                "best_ber": round(best_ber, 6),
                "msg_loss": round(loss_dict["msg"], 6),
                "perc_loss": round(loss_dict["perc"], 6),
                "attack_prob": round(attack_prob, 3),
                "lr": cur_lr,
                "avg_step_time_s": round(avg_time, 3),
                "eta_min": round(remaining / 60, 1),
            }
            progress_path = os.path.join(config["output_dir"], "progress.json")
            try:
                with open(progress_path, "w") as _f:
                    json.dump(progress, _f, indent=2)
            except OSError:
                pass

        # ── Save images ──
        if step % config["save_image_interval"] == 0:
            save_comparison(
                x_stego.detach(), x_clean.detach(),
                step=step, output_dir=config["output_dir"],
            )

        # ── Checkpoint ──
        if step % config["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(
                config["checkpoint_dir"], f"dscsp_v4_step_{step:06d}.pt"
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

        # ── Signal-triggered emergency checkpoint ──
        if _interrupted[0]:
            interrupt_path = os.path.join(
                config["checkpoint_dir"], f"dscsp_v4_interrupt_step_{step:06d}.pt"
            )
            ckpt_interrupt = {
                "step": step,
                "controller": controller.state_dict(),
                "extractor": extractor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config,
                "best_ber": best_ber,
            }
            if discriminator is not None:
                ckpt_interrupt["discriminator"] = discriminator.state_dict()
            if disc_optimizer is not None:
                ckpt_interrupt["disc_optimizer"] = disc_optimizer.state_dict()
            torch.save(ckpt_interrupt, interrupt_path)
            print(f"  Interrupt checkpoint saved: {interrupt_path}")
            break

        # Free memory
        del x_stego, x_for_extraction, delta, delta_norms, bits_pred, loss, payload
        if disc_logits_stego is not None:
            del disc_logits_stego
        if step % 50 == 0:
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
            pay_ = torch.randint(0, 2, (1, config["num_bits"]), device=device).float()

            delta_ = controller(x_clean_, pay_)
            x_stego_ = (x_clean_ + delta_).clamp(-1, 1)
            bits_pred_ = extractor(x_stego_)

            total_ber += compute_ber(bits_pred_, pay_)
            total_psnr += compute_psnr(x_stego_, x_clean_)

    avg_ber = total_ber / n_eval
    avg_psnr = total_psnr / n_eval

    print(f"  Average BER:  {avg_ber:.6f} (target: 0.0000)")
    print(f"  Average PSNR: {avg_psnr:.1f} dB (target: >= 35)")
    print(f"  Best BER during training: {best_ber:.6f}")

    # Discriminator accuracy on eval set
    disc_acc_final = None
    if discriminator is not None:
        discriminator.eval()
        disc_correct = 0
        disc_total = 0
        with torch.no_grad():
            for i in range(min(100, config["pool_size"])):
                xc = image_pool[i:i+1]
                pay_eval = torch.randint(0, 2, (1, config["num_bits"]), device=device).float()
                d_eval = controller(xc, pay_eval)
                xs = (xc + d_eval).clamp(-1, 1)

                lc = discriminator(xc)
                ls = discriminator(xs)
                disc_correct += (torch.sigmoid(lc) < 0.5).sum().item()
                disc_correct += (torch.sigmoid(ls) > 0.5).sum().item()
                disc_total += 2

        disc_acc_final = disc_correct / disc_total
        print(f"  Discriminator accuracy: {disc_acc_final:.4f} (target: ~0.50)")

    print()

    if avg_ber < 0.001:
        print("  SUCCESS: Clean BER target met!")
    else:
        print("  WARNING: Clean BER not yet at target")

    # Save final checkpoint
    final_path = os.path.join(config["checkpoint_dir"], "dscsp_v4_final.pt")
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
    results_path = os.path.join(config["output_dir"], "train_v4_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results: {results_path}")


if __name__ == "__main__":
    train()
