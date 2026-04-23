"""
DSCSP v5 Training Script — TIFS-Ready.

Key changes from v4:
  1. RedundantCoder: 128 message bits → 256 physical slots (2× spatial redundancy)
     This is the primary fix for crop robustness (was ~50% BER, target ≤ 0.30).
  2. SRM feature-matching loss (λ=0.05): directly minimises the mean SRM residual
     gap between clean and stego, giving the controller a frequency-domain stealth
     signal beyond the GAN loss alone.
  3. JPEG-heavy attack refined: weight=0.80, quality range [70, 92] (realistic
     social-media threat model; training on Q40 is unrealistic and hurts Q75 BER).
  4. Compound attacks (crop+JPEG) start at step 5 000 (was 15 000) to build crop
     robustness earlier.
  5. Stronger adversarial: discriminator starts at step 500 (was 1 000), 4 D steps
     per G step (was 3).

Usage:
    python train_v5.py
    python train_v5.py --resume checkpoints_v5/dscsp_v5_step_020000.pt
"""

import argparse
import os
import random
import signal
import time
import json
import math
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
from dscsp.utils.redundant_coding import RedundantCoder
from dscsp.augmentations.differentiable import AttackLayer


# ─── Configuration ──────────────────────────────────────────────────────────

CONFIG = {
    # Model
    "model_id": "google/ddpm-celebahq-256",
    "num_bits": 256,              # physical encoding slots (unchanged from v4)
    "message_bits": 128,          # actual payload bits (2× redundancy)
    "use_redundant_coding": True,
    "redundancy_factor": 2,
    "ddim_steps": 20,
    "max_perturbation": 0.08,
    "base_channels": 48,
    "extractor_mode": "spatial_map",
    "extractor_grid_size": 16,

    # Training
    "batch_size": 4,
    "total_steps": 50000,
    "lr_controller": 1e-4,
    "lr_extractor": 3e-4,
    "lr_min": 1e-6,
    "weight_decay_controller": 1e-5,
    "weight_decay_extractor": 1e-4,
    "grad_clip": 5.0,

    # Spectral consistency penalty (FFT magnitude MSE)
    "lambda_spec": 0.01,

    # SRM feature-matching loss (NEW in v5)
    "lambda_srm_match": 0.05,

    # Image pool
    "pool_size": 128,
    "pool_refresh_interval": 2000,

    # Loss weights
    "lambda_perc": 0.3,
    "lambda_ctrl": 0.0,
    "warmup_steps": 0,

    # Bit curriculum — DISABLED in v5 (redundant coding replaces it)
    "bit_curriculum_start_bits": 256,
    "bit_curriculum_end_bits": 256,
    "bit_curriculum_steps": 0,
    "bit_curriculum_start_step": 1,

    # Extractor bootstrap (same as v4)
    "extractor_bootstrap_steps": 800,
    "extractor_bootstrap_start_step": 1,
    "extractor_bootstrap_perturbation": 0.18,
    "extractor_bootstrap_attack_prob": 0.0,

    # Controller bootstrap (same as v4)
    "controller_bootstrap_steps": 1000,
    "controller_bootstrap_start_step": 0,
    "controller_bootstrap_align_weight": 20.0,
    "controller_bootstrap_attack_prob": 0.0,
    "controller_bootstrap_freeze_extractor": True,

    # Attack schedule — JPEG-heavy, realistic quality range
    "robustness_start_step": 500,
    "robustness_ramp_steps": 2000,
    "robustness_max_prob": 0.85,
    "jpeg_weight": 0.80,              # was 0.60 in v4
    "jpeg_quality_range": (70, 92),   # was (40, 95); realistic social-media range

    # Attack parameters
    "blur_kernel_size": 5,
    "blur_sigma_range": (0.5, 2.0),
    "noise_sigma_range": (0.01, 0.05),
    "crop_ratio_range": (0.75, 0.95),
    "brightness_range": (-0.1, 0.1),
    "contrast_range": (0.8, 1.2),

    # Adversarial training — stronger in v5
    "use_adversarial": True,
    "use_srm": True,
    "lambda_adv": 0.5,
    "adv_start_step": 500,            # was 1000 in v4
    "adv_ramp_steps": 3000,           # was 4000 in v4
    "disc_lr": 1e-4,
    "disc_weight_decay": 1e-4,
    "disc_steps_per_gen_step": 4,     # was 3 in v4
    "disc_label_smoothing": 0.1,

    # Compound attacks — start earlier for crop robustness
    "use_combined_attacks": True,
    "combined_attack_start_step": 5000,  # was 15000 in v4
    "combined_attack_ratio": 0.35,
    "n_combined": 2,

    # Logging
    "log_interval": 10,
    "save_image_interval": 500,
    "checkpoint_interval": 2000,
    "output_dir": "outputs_v5",
    "checkpoint_dir": "checkpoints_v5",

    # Resume
    "resume_path": "",
    "resume_load_optimizer": True,
    "resume_load_scheduler": True,
    "resume_load_disc_optimizer": True,
    "override_lrs_on_resume": True,
}


# ─── Global-Pool Extractor (same as v4) ────────────────────────────────────

class GlobalExtractor(nn.Module):
    """ResNet-18 extractor with selectable head (global_fc, spatial_map, or shift_pool)."""

    def __init__(self, num_bits: int = 256, mode: str = "global_fc", grid_size: int = 16):
        super().__init__()
        self.num_bits = num_bits
        self.mode = mode
        self.grid_size = max(1, int(grid_size))

        backbone = tv_models.resnet18(weights=None)
        backbone.fc = nn.Linear(512, num_bits)

        if self.mode in ("spatial_map", "shift_pool"):
            backbone.maxpool = nn.Identity()
            self.spatial_head = nn.Conv2d(512, 1, kernel_size=1)
            self.grid_h, self.grid_w = self._infer_grid_shape(num_bits)

        self.backbone = backbone

    @staticmethod
    def _infer_grid_shape(num_bits: int):
        side = int(math.sqrt(num_bits))
        for h in range(side, 0, -1):
            if num_bits % h == 0:
                return h, num_bits // h
        return 1, num_bits

    def _forward_features(self, image: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(image)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.mode == "global_fc":
            return self.backbone(image)

        feat = self._forward_features(image)
        logits_map = self.spatial_head(feat)

        if logits_map.shape[2] != self.grid_h or logits_map.shape[3] != self.grid_w:
            logits_map = F.interpolate(
                logits_map, size=(self.grid_h, self.grid_w),
                mode="bilinear", align_corners=False,
            )

        if self.mode == "shift_pool":
            logits_map = F.max_pool2d(logits_map, kernel_size=5, stride=1, padding=2)

        logits = logits_map.flatten(1)
        if logits.shape[1] > self.num_bits:
            logits = logits[:, :self.num_bits]
        elif logits.shape[1] < self.num_bits:
            logits = F.pad(logits, (0, self.num_bits - logits.shape[1]))
        return logits


# ─── Spectral-Norm Discriminator (same as v4) ──────────────────────────────

class SNDiscriminator(nn.Module):
    """SRM + 5-block SN discriminator for steganalysis."""

    def __init__(self, use_srm: bool = True):
        super().__init__()
        self.use_srm = use_srm
        self.srm = SRMFilterLayer() if use_srm else None
        in_channels = 9 if use_srm else 3

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
            print(f"    {min((i // batch + 1) * batch, pool_size)}/{pool_size}", end="\r")
    pool = torch.cat(images, dim=0)[:pool_size]
    print(f"\n  Image pool: {pool.shape}, range: [{pool.min():.3f}, {pool.max():.3f}]")
    diffusion.unet.cpu()
    torch.cuda.empty_cache()
    return pool


# ─── JPEG-Heavy Attack Wrapper ──────────────────────────────────────────────

class JPEGHeavyAttackLayer(nn.Module):
    """JPEG-weighted attack layer with configurable quality range."""

    def __init__(self, config):
        super().__init__()
        self.base_attack = AttackLayer(config)
        self.jpeg_weight = config.get("jpeg_weight", 0.80)

    def forward(self, x, attack_prob=0.0, combined=False, n_combined=2):
        if attack_prob <= 0 or random.random() > attack_prob:
            return x
        if combined and n_combined > 1:
            return self.base_attack(x, attack_prob=1.0, combined=True, n_combined=n_combined)
        if random.random() < self.jpeg_weight:
            return self.base_attack._apply_single(x, 'jpeg')
        else:
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


def get_active_bits(step, config):
    num_bits = int(config["num_bits"])
    start_bits = int(config.get("bit_curriculum_start_bits", num_bits))
    end_bits = int(config.get("bit_curriculum_end_bits", num_bits))
    steps = int(config.get("bit_curriculum_steps", 0))
    start_step = int(config.get("bit_curriculum_start_step", 1))
    start_bits = max(1, min(num_bits, start_bits))
    end_bits = max(1, min(num_bits, end_bits))
    if steps <= 0 or start_bits == end_bits:
        return end_bits
    if step <= start_step:
        return start_bits
    progress = min(1.0, (step - start_step) / max(1, steps))
    active = int(round(start_bits + progress * (end_bits - start_bits)))
    if active < num_bits:
        active = max(1, (active // 8) * 8)
    return max(1, min(num_bits, active))


# ─── Discriminator Step ─────────────────────────────────────────────────────

def train_discriminator_step(discriminator, disc_optimizer, x_clean, x_stego, label_smoothing=0.1):
    disc_optimizer.zero_grad()
    logits_clean = discriminator(x_clean.detach())
    logits_stego = discriminator(x_stego.detach())
    target_clean = torch.full_like(logits_clean, label_smoothing)
    target_stego = torch.full_like(logits_stego, 1.0 - label_smoothing)
    loss_disc = (
        F.binary_cross_entropy_with_logits(logits_clean, target_clean) +
        F.binary_cross_entropy_with_logits(logits_stego, target_stego)
    ) / 2
    loss_disc.backward()
    disc_optimizer.step()
    with torch.no_grad():
        acc = (
            (torch.sigmoid(logits_clean) < 0.5).float().sum() +
            (torch.sigmoid(logits_stego) > 0.5).float().sum()
        ) / (logits_clean.numel() + logits_stego.numel())
    return loss_disc.item(), acc.item()


# ─── Auxiliary Losses ───────────────────────────────────────────────────────

def compute_spectral_loss(x_stego, x_clean):
    """Log-magnitude FFT MSE — penalises frequency-domain artifacts."""
    fft_s = torch.fft.fft2(x_stego)
    fft_c = torch.fft.fft2(x_clean)
    return F.mse_loss(torch.log1p(fft_s.abs()), torch.log1p(fft_c.abs()))


def compute_srm_match_loss(x_stego, x_clean, srm_layer):
    """
    SRM feature-matching loss: penalise the mean SRM residual difference.
    Gives the controller a direct stealth signal in the high-frequency domain.
    Gradient flows through x_stego only (x_clean is detached).
    """
    feats_stego = srm_layer(x_stego)          # (B, 9, H, W)
    feats_clean = srm_layer(x_clean).detach() # no gradient through clean reference
    return F.mse_loss(feats_stego.mean(0), feats_clean.mean(0))


# ─── Utilities ──────────────────────────────────────────────────────────────

def _load_state_dict_flexible(model: nn.Module, state_dict: dict, name: str):
    """Load checkpoint weights with shape/key filtering for architecture transitions."""
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"  Loaded {name} weights (strict)")
        return
    except RuntimeError as exc:
        print(f"  Strict load failed for {name}: {exc}")
        model_state = model.state_dict()
        compatible = {
            k: v for k, v in state_dict.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        model.load_state_dict(compatible, strict=False)
        print(
            f"  Loaded {name} weights (flex): "
            f"{len(compatible)}/{len(model_state)} tensors compatible"
        )


def _build_bootstrap_delta(payload: torch.Tensor, height: int, width: int, amplitude: float) -> torch.Tensor:
    """Deterministic synthetic channel for extractor bootstrap (same as v4)."""
    bsz, num_bits = payload.shape
    grid_h, grid_w = GlobalExtractor._infer_grid_shape(num_bits)
    signed = payload[:, :grid_h * grid_w] * 2.0 - 1.0
    signed = signed.view(bsz, 1, grid_h, grid_w)
    signed = F.interpolate(signed, size=(height, width), mode="nearest")
    sh = max(1, height // max(1, 2 * grid_h))
    sw = max(1, width // max(1, 2 * grid_w))
    delta = torch.cat([signed, torch.roll(signed, sh, 2), torch.roll(signed, sw, 3)], dim=1)
    return delta.clamp(-amplitude, amplitude)


# ─── Quick Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def quick_eval(controller, extractor, coder, image_pool, config, device, n=200, eval_msg_bits=None):
    """
    Evaluate BER and PSNR.

    eval_msg_bits: override message bits for multi-payload capacity evaluation.
        - None or 128: use full model (native)
        - 64: decode only first 64 message bits (first 128 physical slots via perm subset)
        - must be <= config["message_bits"]
    """
    controller.eval()
    extractor.eval()
    pool_size = image_pool.shape[0]
    total_ber = 0.0
    total_psnr = 0.0
    msg_bits = config.get("message_bits", 128)
    if eval_msg_bits is not None:
        eval_bits = min(eval_msg_bits, msg_bits)
    else:
        eval_bits = msg_bits

    for _ in range(n):
        idx = torch.randint(0, pool_size, (1,))
        x_c = image_pool[idx].to(device)
        msg = torch.randint(0, 2, (1, msg_bits), device=device).float()
        if coder is not None:
            physical = coder.encode(msg)
        else:
            physical = msg  # use_redundant_coding=False: message == physical payload

        delta = controller(x_c, physical)
        x_s = (x_c + delta).clamp(-1, 1)
        logits = extractor(x_s)

        if coder is not None and eval_bits < msg_bits:
            # Subset evaluation: take first eval_bits message positions
            perm_sub = coder.perm[:eval_bits].to(device)  # (eval_bits, R)
            gathered = torch.stack(
                [logits.gather(1, perm_sub[:, r].unsqueeze(0).expand(1, -1))
                 for r in range(coder.R)],
                dim=-1,
            )  # (1, eval_bits, R)
            votes = (gathered.mean(-1) > 0).float()
            total_ber += (votes - msg[:, :eval_bits]).abs().mean().item()
        elif coder is not None:
            msg_hard = coder.decode_hard(logits)
            total_ber += compute_ber(msg_hard, msg)
        else:
            msg_hard = (logits > 0).float()
            total_ber += compute_ber(msg_hard, msg)

        total_psnr += compute_psnr(x_s, x_c)

    return total_ber / n, total_psnr / n


# ─── Training ──────────────────────────────────────────────────────────────

def train(config=None):
    config = config or CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DSCSP v5 Training")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    _interrupted = [False]

    def _handle_signal(signum, frame):
        print(f"\n  [Signal {signum}] Will checkpoint and exit after this step.")
        _interrupted[0] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Redundant coder ──
    msg_bits = config.get("message_bits", 128)
    phys_bits = config["num_bits"]
    use_rc = config.get("use_redundant_coding", True)
    if use_rc:
        R = config.get("redundancy_factor", 2)
        assert phys_bits == msg_bits * R, (
            f"num_bits ({phys_bits}) must equal message_bits ({msg_bits}) * redundancy ({R})"
        )
        coder = RedundantCoder(message_bits=msg_bits, physical_slots=phys_bits, redundancy=R)
        print(f"RedundantCoder: {msg_bits} message bits → {phys_bits} physical slots (R={R})")
    else:
        coder = None
        msg_bits = phys_bits
        print(f"RedundantCoder: disabled ({phys_bits} message bits = physical slots)")
    print()

    # ── Diffusion model ──
    print("Loading diffusion model...")
    diffusion = DiffusionWrapper(
        model_id=config["model_id"],
        device=device,
        num_inference_steps=config["ddim_steps"],
    )
    print(f"  UNet params: {sum(p.numel() for p in diffusion.unet.parameters()) / 1e6:.1f}M (frozen)")
    print()

    # ── Models ──
    print("Initializing content-adaptive controller...")
    controller = ContentAdaptiveController(
        num_bits=phys_bits,
        max_perturbation=config["max_perturbation"],
        base_channels=config["base_channels"],
    ).to(device)
    print(f"  Controller params: {sum(p.numel() for p in controller.parameters()) / 1e6:.2f}M")

    print("Initializing global-pool extractor...")
    extractor = GlobalExtractor(
        num_bits=phys_bits,
        mode=config.get("extractor_mode", "global_fc"),
        grid_size=config.get("extractor_grid_size", 16),
    ).to(device)
    print(f"  Extractor params: {sum(p.numel() for p in extractor.parameters()) / 1e6:.2f}M")

    discriminator = None
    disc_optimizer = None
    srm_layer = None
    if config["use_adversarial"]:
        print("Initializing SN-discriminator...")
        discriminator = SNDiscriminator(use_srm=config["use_srm"]).to(device)
        print(f"  Discriminator params: {sum(p.numel() for p in discriminator.parameters()) / 1e6:.2f}M")
        disc_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=config["disc_lr"],
            weight_decay=config["disc_weight_decay"],
            betas=(0.0, 0.9),
        )
        if config["use_srm"] and config.get("lambda_srm_match", 0.0) > 0:
            srm_layer = SRMFilterLayer().to(device)
            srm_layer.eval()
            for p in srm_layer.parameters():
                p.requires_grad_(False)
            print(f"  SRM feature-matching loss enabled (λ={config['lambda_srm_match']})")

    criterion = DSCSPLoss(
        lambda_perc=config["lambda_perc"],
        lambda_ctrl=config["lambda_ctrl"],
        warmup_steps=config["warmup_steps"],
        lambda_adv=config["lambda_adv"] if config["use_adversarial"] else 0.0,
        adv_start_step=config["adv_start_step"],
        adv_ramp_steps=config["adv_ramp_steps"],
    ).to(device)

    attack_layer = JPEGHeavyAttackLayer(config).to(device)
    print(f"  Attack layer: JPEG weight {config['jpeg_weight']}, "
          f"quality {config['jpeg_quality_range']}")

    optimizer = torch.optim.Adam([
        {"params": controller.parameters(), "lr": config["lr_controller"],
         "weight_decay": config["weight_decay_controller"]},
        {"params": extractor.parameters(), "lr": config["lr_extractor"],
         "weight_decay": config["weight_decay_extractor"]},
    ], betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["total_steps"], eta_min=config["lr_min"],
    )

    # ── Resume ──
    start_step = 0
    best_ber = 1.0
    if config["resume_path"] and os.path.exists(config["resume_path"]):
        print(f"Resuming from {config['resume_path']}...")
        ckpt = torch.load(config["resume_path"], map_location=device, weights_only=False)
        _load_state_dict_flexible(controller, ckpt["controller"], "controller")
        _load_state_dict_flexible(extractor, ckpt["extractor"], "extractor")
        if config.get("resume_load_optimizer", True) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if config.get("resume_load_scheduler", True) and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        best_ber = ckpt.get("best_ber", 1.0)
        if discriminator is not None and "discriminator" in ckpt:
            _load_state_dict_flexible(discriminator, ckpt["discriminator"], "discriminator")
        if (disc_optimizer is not None and config.get("resume_load_disc_optimizer", True)
                and "disc_optimizer" in ckpt):
            disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        if config.get("override_lrs_on_resume", True):
            optimizer.param_groups[0]["lr"] = config["lr_controller"]
            optimizer.param_groups[1]["lr"] = config["lr_extractor"]
            if disc_optimizer is not None:
                disc_optimizer.param_groups[0]["lr"] = config["disc_lr"]
        print(f"  Resumed at step {start_step}, best BER: {best_ber:.4f}")
        del ckpt

    # ── Image pool ──
    print("Generating image pool...")
    image_pool = generate_image_pool(diffusion, config["pool_size"], device)

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    print(f"\nStarting DSCSP v5 training for {config['total_steps']} steps (from step {start_step})")
    print(f"  Payload: {msg_bits} message bits → {phys_bits} physical bits (R={config.get('redundancy_factor',1)})")
    print(f"  JPEG: weight={config['jpeg_weight']}, quality={config['jpeg_quality_range']}")
    print(f"  Adv: start={config['adv_start_step']}, D_steps={config['disc_steps_per_gen_step']}")
    print(f"  Compound attacks: start step {config['combined_attack_start_step']}")
    print()

    # Bootstrap window computation
    step_times = []
    ext_boot_start = start_step + int(config.get("extractor_bootstrap_start_step", 1))
    ext_boot_steps = int(config.get("extractor_bootstrap_steps", 0))
    ext_boot_end = ext_boot_start + ext_boot_steps - 1

    ctrl_boot_steps = int(config.get("controller_bootstrap_steps", 0))
    ctrl_boot_start_cfg = int(config.get("controller_bootstrap_start_step", 0))
    if ctrl_boot_start_cfg > 0:
        ctrl_boot_start = start_step + ctrl_boot_start_cfg
    else:
        ctrl_boot_start = ext_boot_end + 1
    ctrl_boot_end = ctrl_boot_start + ctrl_boot_steps - 1

    extractor_frozen = False

    for step in range(start_step + 1, config["total_steps"] + 1):
        t0 = time.time()
        controller.train()
        extractor.train()
        if discriminator is not None:
            discriminator.train()

        # Refresh image pool
        if step > 1 and step % config["pool_refresh_interval"] == 0:
            print("  Refreshing image pool...")
            image_pool = generate_image_pool(diffusion, config["pool_size"], device)

        # Sample batch
        idx = torch.randint(0, config["pool_size"], (config["batch_size"],))
        x_clean = image_pool[idx].to(device)

        # Generate message bits + encode to physical bits
        if use_rc:
            msg = torch.randint(0, 2, (config["batch_size"], msg_bits), device=device).float()
            payload = coder.encode(msg)  # (B, phys_bits)
        else:
            payload = torch.randint(0, 2, (config["batch_size"], phys_bits), device=device).float()
            msg = payload

        in_ext_boot = ext_boot_steps > 0 and ext_boot_start <= step <= ext_boot_end
        in_ctrl_boot = ctrl_boot_steps > 0 and ctrl_boot_start <= step <= ctrl_boot_end

        freeze_now = in_ctrl_boot and config.get("controller_bootstrap_freeze_extractor", True)
        if freeze_now != extractor_frozen:
            for p in extractor.parameters():
                p.requires_grad_(not freeze_now)
            extractor_frozen = freeze_now

        disc_loss_val = 0.0
        disc_acc_val = 0.0

        if in_ext_boot:
            # ── Extractor bootstrap: learn to decode synthetic channel ──
            optimizer.zero_grad()

            delta = _build_bootstrap_delta(
                payload, x_clean.shape[2], x_clean.shape[3],
                float(config.get("extractor_bootstrap_perturbation", 0.18)),
            )
            x_stego = (x_clean + delta).clamp(-1, 1)
            delta_norms = delta.pow(2).mean()

            boot_ap = float(config.get("extractor_bootstrap_attack_prob", 0.0))
            x_for_ext = (
                attack_layer(x_stego, attack_prob=boot_ap, combined=False, n_combined=1)
                if boot_ap > 0 else x_stego
            )

            logits = extractor(x_for_ext)

            # Bootstrap loss is always on physical bits (teaches raw bit positions)
            loss = F.binary_cross_entropy_with_logits(logits, payload)
            loss_dict = {"msg": loss.item(), "perc": 0.0, "ctrl": delta_norms.item(),
                         "adv": 0.0, "spec": 0.0, "srm_match": 0.0}
            attack_prob = boot_ap
            disc_logits_stego = None

            loss.backward()
            torch.nn.utils.clip_grad_norm_(extractor.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()

        elif in_ctrl_boot:
            # ── Controller bootstrap: align to synthetic target ──
            optimizer.zero_grad()

            target_amp = min(
                float(config.get("extractor_bootstrap_perturbation", 0.18)),
                float(config["max_perturbation"]),
            )
            target_delta = _build_bootstrap_delta(
                payload, x_clean.shape[2], x_clean.shape[3], target_amp,
            )

            delta = controller(x_clean, payload)
            x_stego = (x_clean + delta).clamp(-1, 1)
            delta_norms = delta.pow(2).mean()

            ctrl_ap = float(config.get("controller_bootstrap_attack_prob", 0.0))
            x_for_ext = (
                attack_layer(x_stego, attack_prob=ctrl_ap, combined=False, n_combined=1)
                if ctrl_ap > 0 else x_stego
            )

            logits = extractor(x_for_ext)
            loss_msg = F.binary_cross_entropy_with_logits(logits, payload)
            loss_align = F.mse_loss(delta, target_delta)
            align_w = float(config.get("controller_bootstrap_align_weight", 20.0))
            loss = (align_w * loss_align) if extractor_frozen else (loss_msg + align_w * loss_align)

            loss_dict = {"msg": loss_msg.item(), "perc": 0.0, "ctrl": delta_norms.item(),
                         "adv": 0.0, "spec": 0.0, "align": loss_align.item(), "srm_match": 0.0}
            attack_prob = ctrl_ap
            disc_logits_stego = None

            loss.backward()
            params = controller.parameters() if extractor_frozen else (
                list(controller.parameters()) + list(extractor.parameters())
            )
            torch.nn.utils.clip_grad_norm_(params, config["grad_clip"])
            optimizer.step()
            scheduler.step()

        else:
            # ── Main training ──

            # Controller forward
            delta = controller(x_clean, payload)
            x_stego = (x_clean + delta).clamp(-1, 1)
            delta_norms = delta.pow(2).mean()

            # Discriminator update
            if discriminator is not None and step >= config["adv_start_step"]:
                for _ in range(config["disc_steps_per_gen_step"]):
                    disc_loss_val, disc_acc_val = train_discriminator_step(
                        discriminator, disc_optimizer, x_clean, x_stego,
                        label_smoothing=config["disc_label_smoothing"],
                    )

            # Generator update
            optimizer.zero_grad()

            # Recompute stego with fresh gradients
            delta = controller(x_clean, payload)
            x_stego = (x_clean + delta).clamp(-1, 1)
            delta_norms = delta.pow(2).mean()

            # Attack
            attack_prob = get_attack_prob(step, config)
            use_combined = should_use_combined_attack(step, config)
            x_for_ext = attack_layer(
                x_stego, attack_prob=attack_prob,
                combined=use_combined, n_combined=config["n_combined"],
            )

            # Extract physical bits
            logits = extractor(x_for_ext)

            # Decode to message bits for loss
            if use_rc:
                msg_logits = coder.decode_soft(logits)  # (B, msg_bits)
                loss_target = msg
            else:
                msg_logits = logits
                loss_target = payload

            # Adversarial signal
            disc_logits_stego = None
            if (discriminator is not None and config["lambda_adv"] > 0
                    and step >= config["adv_start_step"]):
                disc_logits_stego = discriminator(x_stego)

            # Composite loss (msg + perc + ctrl + adv)
            loss, loss_dict = criterion(
                msg_logits, loss_target, x_stego, x_clean, delta_norms,
                step=step, disc_logits_stego=disc_logits_stego,
            )

            # Spectral loss
            spec_loss = compute_spectral_loss(x_stego, x_clean)
            loss = loss + config["lambda_spec"] * spec_loss
            loss_dict["spec"] = spec_loss.item()

            # SRM feature-matching loss (v5 new)
            srm_match_val = 0.0
            if srm_layer is not None and config.get("lambda_srm_match", 0.0) > 0:
                srm_loss = compute_srm_match_loss(x_stego, x_clean, srm_layer)
                loss = loss + config["lambda_srm_match"] * srm_loss
                srm_match_val = srm_loss.item()
            loss_dict["srm_match"] = srm_match_val

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(controller.parameters()) + list(extractor.parameters()),
                config["grad_clip"],
            )
            optimizer.step()
            scheduler.step()

        step_time = time.time() - t0
        step_times.append(step_time)

        # ── Logging ──
        if step % config["log_interval"] == 0:
            with torch.no_grad():
                if use_rc:
                    msg_hard = coder.decode_hard(logits.detach())
                    ber = compute_ber(msg_hard, msg)
                    ber_phys = compute_ber(logits.detach(), payload)
                else:
                    ber = compute_ber(logits.detach(), payload)
                    ber_phys = ber
            psnr = compute_psnr(x_stego.detach(), x_clean.detach())
            delta_rms = delta.pow(2).mean().sqrt().item()
            avg_time = sum(step_times[-10:]) / len(step_times[-10:])
            remaining = avg_time * (config["total_steps"] - step)
            cur_lr = scheduler.get_last_lr()[0]

            if in_ext_boot:
                phase = "BOOT-EXT"
            elif in_ctrl_boot:
                phase = "BOOT-CTRL"
            else:
                phase = "TRAIN"

            log_line = (
                f"{phase} {step:5d}/{config['total_steps']} | "
                f"BER(msg): {ber:.4f} | BER(phys): {ber_phys:.4f} | "
                f"PSNR: {psnr:.1f}dB | "
                f"msg: {loss_dict['msg']:.4f} | perc: {loss_dict['perc']:.4f} | "
                f"spec: {loss_dict['spec']:.4f} | srm_m: {loss_dict['srm_match']:.4f} | "
                f"atk: {attack_prob:.2f} | lr: {cur_lr:.1e} | "
                f"{avg_time:.2f}s/step | ETA: {remaining/60:.0f}min"
            )
            if phase == "TRAIN" and config["use_adversarial"] and step >= config["adv_start_step"]:
                log_line += (
                    f" | D: {disc_loss_val:.3f}/{disc_acc_val:.2f}"
                    f" adv: {loss_dict.get('adv', 0):.4f}"
                )
            print(log_line)

            if ber < best_ber:
                best_ber = ber

            progress = {
                "step": step,
                "total_steps": config["total_steps"],
                "ber": round(ber, 6),
                "ber_phys": round(ber_phys, 6),
                "psnr": round(psnr, 2),
                "best_ber": round(best_ber, 6),
                "phase": phase,
                "msg_loss": round(loss_dict["msg"], 6),
                "perc_loss": round(loss_dict["perc"], 6),
                "spec_loss": round(loss_dict["spec"], 6),
                "srm_match_loss": round(loss_dict["srm_match"], 6),
                "attack_prob": round(attack_prob, 3),
                "lr": cur_lr,
                "avg_step_time_s": round(avg_time, 3),
                "eta_min": round(remaining / 60, 1),
            }
            with open(os.path.join(config["output_dir"], "progress.json"), "w") as f:
                json.dump(progress, f, indent=2)

        # ── Save images ──
        if step % config["save_image_interval"] == 0:
            save_comparison(x_stego.detach(), x_clean.detach(),
                            step=step, output_dir=config["output_dir"])

        # ── Checkpoint ──
        def _ckpt_now(s):
            d = {
                "step": s,
                "controller": controller.state_dict(),
                "extractor": extractor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": {k: str(v) if isinstance(v, tuple) else v for k, v in config.items()},
                "best_ber": best_ber,
                "message_bits": msg_bits,
                "redundancy_factor": config.get("redundancy_factor", 1),
            }
            if discriminator is not None:
                d["discriminator"] = discriminator.state_dict()
            if disc_optimizer is not None:
                d["disc_optimizer"] = disc_optimizer.state_dict()
            return d

        if step % config["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(config["checkpoint_dir"], f"dscsp_v5_step_{step:06d}.pt")
            torch.save(_ckpt_now(step), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        if _interrupted[0]:
            ipath = os.path.join(config["checkpoint_dir"], f"dscsp_v5_interrupt_{step:06d}.pt")
            torch.save(_ckpt_now(step), ipath)
            print(f"  Interrupt checkpoint saved: {ipath}")
            break

        del x_stego, x_for_ext, delta, delta_norms, logits, loss, payload, msg
        if disc_logits_stego is not None:
            del disc_logits_stego
        if step % 50 == 0:
            torch.cuda.empty_cache()

    # ── Final evaluation ──
    # _make_ckpt defined here (not inside loop) so it's reachable even when
    # the training loop runs 0 iterations (e.g. resume already at total_steps).
    final_step = step if 'step' in dir() else start_step  # noqa: F821

    def _make_ckpt():
        d = {
            "step": final_step,
            "controller": controller.state_dict(),
            "extractor": extractor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": {k: str(v) if isinstance(v, tuple) else v for k, v in config.items()},
            "best_ber": best_ber,
            "message_bits": msg_bits,
            "redundancy_factor": config.get("redundancy_factor", 1),
        }
        if discriminator is not None:
            d["discriminator"] = discriminator.state_dict()
        if disc_optimizer is not None:
            d["disc_optimizer"] = disc_optimizer.state_dict()
        return d

    print()
    print("=" * 60)
    print("FINAL EVALUATION (200 samples) — v5")
    print("=" * 60)

    avg_ber, avg_psnr = quick_eval(
        controller, extractor, coder if use_rc else None,
        image_pool, config, device, n=200,
    )

    print(f"  Clean BER (message): {avg_ber:.6f} (target: 0.0000)")
    print(f"  PSNR:                {avg_psnr:.1f} dB (target: >= 35)")
    print(f"  Best BER during training: {best_ber:.6f}")

    if avg_ber < 0.001:
        print("  SUCCESS: Clean BER target met!")
    else:
        print("  WARNING: Clean BER not yet at target")

    final_path = os.path.join(config["checkpoint_dir"], "dscsp_v5_final.pt")
    final_data = _make_ckpt()
    final_data["final_ber"] = avg_ber
    final_data["final_psnr"] = avg_psnr
    torch.save(final_data, final_path)
    print(f"  Saved final model: {final_path}")

    results = {
        "final_ber": avg_ber,
        "final_psnr": avg_psnr,
        "best_ber": best_ber,
        "config": {k: str(v) if isinstance(v, tuple) else v for k, v in config.items()},
    }
    results_path = os.path.join(config["output_dir"], "train_v5_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    args = parser.parse_args()
    if args.resume:
        CONFIG["resume_path"] = args.resume
    train(CONFIG)
