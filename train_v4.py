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
from dscsp.augmentations.differentiable import AttackLayer


# ─── Configuration ──────────────────────────────────────────────────────────

CONFIG = {
    # Model
    "model_id": "google/ddpm-celebahq-256",
    "num_bits": 256,
    "ddim_steps": 20,
    "max_perturbation": 0.08,
    "base_channels": 48,
    "extractor_mode": "spatial_map",   # "global_fc" (legacy) or "spatial_map"
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

    # Spectral consistency penalty
    "lambda_spec": 0.01,

    # Image pool
    "pool_size": 128,
    "pool_refresh_interval": 2000,

    # Loss weights
    "lambda_perc": 0.3,
    "lambda_ctrl": 0.0,
    "warmup_steps": 0,

    # Payload-bit curriculum (helps escape random-guess regime)
    # If start/end are equal or steps <= 0, curriculum is disabled.
    "bit_curriculum_start_bits": 256,
    "bit_curriculum_end_bits": 256,
    "bit_curriculum_steps": 0,
    "bit_curriculum_start_step": 1,

    # Extractor bootstrap curriculum (major BER recovery mechanism)
    # Trains extractor on a synthetic bit-carrying channel for early steps,
    # then switches to normal controller-driven training.
    "extractor_bootstrap_steps": 800,
    "extractor_bootstrap_start_step": 1,
    "extractor_bootstrap_perturbation": 0.18,
    "extractor_bootstrap_attack_prob": 0.0,

    # Controller bootstrap phase (after extractor bootstrap):
    # align controller outputs to a synthetic target channel while keeping
    # message decoding objective active.
    "controller_bootstrap_steps": 1000,
    "controller_bootstrap_start_step": 0,   # 0 means "right after extractor bootstrap"
    "controller_bootstrap_align_weight": 20.0,
    "controller_bootstrap_attack_prob": 0.0,
    "controller_bootstrap_freeze_extractor": True,

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
    "resume_load_optimizer": True,
    "resume_load_scheduler": True,
    "resume_load_disc_optimizer": True,
    "override_lrs_on_resume": True,
}


# ─── Global-Pool Extractor ─────────────────────────────────────────────────

class GlobalExtractor(nn.Module):
    """
        Bit extractor with selectable head type:
            - global_fc  : legacy GAP + FC head
            - spatial_map: high-resolution spatial head (one bit per spatial cell)
            - shift_pool : spatial_map + ±2-pixel max-pool before flatten for
                           crop-shift robustness

        The spatial_map/shift_pool modes preserve local embedding evidence and
        avoid the severe compression bottleneck of forcing all bits through a
        single globally pooled vector.  shift_pool applies a 5×5 max-pool over
        the logit map so a random crop that shifts the grid by ≤2 pixels does
        not change which maximum is selected.
    """

    def __init__(self, num_bits: int = 256, mode: str = "global_fc", grid_size: int = 16):
        super().__init__()
        self.num_bits = num_bits
        self.mode = mode
        self.grid_size = max(1, int(grid_size))

        backbone = tv_models.resnet18(weights=None)

        # Keep FC shape aligned with num_bits for robust checkpoint compatibility.
        backbone.fc = nn.Linear(512, num_bits)

        if self.mode in ("spatial_map", "shift_pool"):
            # Preserve higher-frequency spatial details (256 -> 16 feature grid).
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
        """
        Args:
            image: (B, 3, 256, 256) stego image in [-1, 1]
        Returns:
            (B, num_bits) logits
        """
        if self.mode == "global_fc":
            return self.backbone(image)

        feat = self._forward_features(image)
        logits_map = self.spatial_head(feat)

        if logits_map.shape[2] != self.grid_h or logits_map.shape[3] != self.grid_w:
            logits_map = F.interpolate(
                logits_map,
                size=(self.grid_h, self.grid_w),
                mode="bilinear",
                align_corners=False,
            )

        if self.mode == "shift_pool":
            # Max over ±2-pixel neighbourhood — a random crop shifting the grid
            # by ≤2 pixels will not change the maximum, giving crop-shift tolerance.
            logits_map = F.max_pool2d(logits_map, kernel_size=5, stride=1, padding=2)

        logits = logits_map.flatten(1)
        if logits.shape[1] > self.num_bits:
            logits = logits[:, :self.num_bits]
        elif logits.shape[1] < self.num_bits:
            logits = F.pad(logits, (0, self.num_bits - logits.shape[1]))
        return logits


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


def get_active_bits(step, config):
    """
    Number of payload bits supervised at this step.

    Curriculum linearly increases supervised bits from
    bit_curriculum_start_bits to bit_curriculum_end_bits over
    bit_curriculum_steps, beginning at bit_curriculum_start_step.
    """
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
        active = start_bits
    else:
        progress = min(1.0, (step - start_step) / max(1, steps))
        active = int(round(start_bits + progress * (end_bits - start_bits)))

    # Keep changes smooth and aligned to byte boundaries where possible.
    if active < num_bits:
        active = max(1, (active // 8) * 8)

    return max(1, min(num_bits, active))


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
    """
    Deterministic synthetic channel for extractor bootstrap.

    Encodes payload bits as block-wise +/- patterns so extractor can learn a
    usable decoding prior before controller/extractor co-adaptation.
    """
    bsz, num_bits = payload.shape
    grid_h, grid_w = GlobalExtractor._infer_grid_shape(num_bits)

    signed = payload[:, :grid_h * grid_w] * 2.0 - 1.0
    signed = signed.view(bsz, 1, grid_h, grid_w)
    signed = F.interpolate(signed, size=(height, width), mode="nearest")

    # Channel-shifted carriers make the bootstrap less trivially luminance-only.
    sh = max(1, height // max(1, 2 * grid_h))
    sw = max(1, width // max(1, 2 * grid_w))
    d0 = signed
    d1 = torch.roll(signed, shifts=sh, dims=2)
    d2 = torch.roll(signed, shifts=sw, dims=3)

    delta = torch.cat([d0, d1, d2], dim=1)
    return delta.clamp(-amplitude, amplitude)


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
    extractor = GlobalExtractor(
        num_bits=config["num_bits"],
        mode=config.get("extractor_mode", "global_fc"),
        grid_size=config.get("extractor_grid_size", 16),
    ).to(device)
    ext_params = sum(p.numel() for p in extractor.parameters())
    print(f"  Extractor params: {ext_params / 1e6:.2f}M ({config.get('extractor_mode', 'global_fc')})")

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
    best_ber_active = 1.0
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
        best_ber_active = ckpt.get("best_ber_active", best_ber)
        if discriminator is not None and "discriminator" in ckpt:
            _load_state_dict_flexible(discriminator, ckpt["discriminator"], "discriminator")
        if (
            disc_optimizer is not None
            and config.get("resume_load_disc_optimizer", True)
            and "disc_optimizer" in ckpt
        ):
            disc_optimizer.load_state_dict(ckpt["disc_optimizer"])

        # Apply configured learning rates after resume (optimizer state may contain old LRs).
        if config.get("override_lrs_on_resume", True):
            if len(optimizer.param_groups) >= 2:
                optimizer.param_groups[0]["lr"] = config["lr_controller"]
                optimizer.param_groups[1]["lr"] = config["lr_extractor"]
            if disc_optimizer is not None and len(disc_optimizer.param_groups) >= 1:
                disc_optimizer.param_groups[0]["lr"] = config["disc_lr"]

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
    print(
        f"  Active bits curriculum: {config['bit_curriculum_start_bits']} -> "
        f"{config['bit_curriculum_end_bits']} over {config['bit_curriculum_steps']} steps"
    )
    print(f"  Max perturbation: {config['max_perturbation']}")
    print(
        f"  Extractor bootstrap: {config.get('extractor_bootstrap_steps', 0)} steps, "
        f"amp={config.get('extractor_bootstrap_perturbation', 0.0)}"
    )
    print(
        f"  Controller bootstrap: {config.get('controller_bootstrap_steps', 0)} steps, "
        f"align_w={config.get('controller_bootstrap_align_weight', 0.0)}"
    )
    print(
        f"  Lambda perc: {config['lambda_perc']}, "
        f"spec: {config['lambda_spec']}, adv: {config['lambda_adv']}"
    )
    print(f"  Adversarial: starts step {config['adv_start_step']}, "
          f"ramps {config['adv_ramp_steps']} steps, {config['disc_steps_per_gen_step']} D steps/G step")
    print(f"  Attacks: starts step {config['robustness_start_step']}, "
          f"max prob {config['robustness_max_prob']}, JPEG weight {config['jpeg_weight']}")
    print(f"  LR: controller={config['lr_controller']}, extractor={config['lr_extractor']}")
    print()

    step_times = []
    ext_bootstrap_start = start_step + int(config.get("extractor_bootstrap_start_step", 1))
    ext_bootstrap_steps = int(config.get("extractor_bootstrap_steps", 0))
    ext_bootstrap_end = ext_bootstrap_start + ext_bootstrap_steps - 1

    ctrl_bootstrap_steps = int(config.get("controller_bootstrap_steps", 0))
    ctrl_bootstrap_start_cfg = int(config.get("controller_bootstrap_start_step", 0))
    if ctrl_bootstrap_start_cfg > 0:
        ctrl_bootstrap_start = start_step + ctrl_bootstrap_start_cfg
    else:
        ctrl_bootstrap_start = ext_bootstrap_end + 1
    ctrl_bootstrap_end = ctrl_bootstrap_start + ctrl_bootstrap_steps - 1

    extractor_frozen = False

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

        in_extractor_bootstrap = (
            ext_bootstrap_steps > 0
            and step >= ext_bootstrap_start
            and step <= ext_bootstrap_end
        )
        in_controller_bootstrap = (
            ctrl_bootstrap_steps > 0
            and step >= ctrl_bootstrap_start
            and step <= ctrl_bootstrap_end
        )

        freeze_extractor_now = bool(
            in_controller_bootstrap
            and config.get("controller_bootstrap_freeze_extractor", True)
        )
        if freeze_extractor_now != extractor_frozen:
            for p in extractor.parameters():
                p.requires_grad_(not freeze_extractor_now)
            extractor_frozen = freeze_extractor_now

        if in_extractor_bootstrap:
            # Bootstrap phase: train extractor on a deterministic synthetic channel
            # to break the random-guess deadlock before joint training.
            optimizer.zero_grad()

            active_bits = get_active_bits(step, config)
            delta = _build_bootstrap_delta(
                payload,
                x_clean.shape[2],
                x_clean.shape[3],
                float(config.get("extractor_bootstrap_perturbation", 0.18)),
            )
            x_stego = (x_clean + delta).clamp(-1, 1)
            delta_norms = delta.pow(2).mean()

            boot_attack_prob = float(config.get("extractor_bootstrap_attack_prob", 0.0))
            if boot_attack_prob > 0:
                x_for_extraction = attack_layer(
                    x_stego, attack_prob=boot_attack_prob, combined=False, n_combined=1
                )
            else:
                x_for_extraction = x_stego

            bits_pred = extractor(x_for_extraction)
            bits_pred_loss = bits_pred[:, :active_bits]
            payload_loss = payload[:, :active_bits]

            loss = F.binary_cross_entropy_with_logits(bits_pred_loss, payload_loss)
            loss_dict = {
                "msg": loss.item(),
                "perc": 0.0,
                "ctrl": delta_norms.item(),
                "adv": 0.0,
                "spec": 0.0,
            }
            disc_loss_val = 0.0
            disc_acc_val = 0.0
            attack_prob = boot_attack_prob
            disc_logits_stego = None

            loss.backward()
            torch.nn.utils.clip_grad_norm_(extractor.parameters(), max_norm=config["grad_clip"])
            optimizer.step()
            scheduler.step()

        elif in_controller_bootstrap:
            # Bridge phase: teach controller to emit a decodable structure while
            # maintaining payload supervision through the real extractor path.
            optimizer.zero_grad()

            active_bits = get_active_bits(step, config)
            target_amp = min(
                float(config.get("extractor_bootstrap_perturbation", 0.18)),
                float(config["max_perturbation"]),
            )
            target_delta = _build_bootstrap_delta(
                payload,
                x_clean.shape[2],
                x_clean.shape[3],
                target_amp,
            )

            delta = controller(x_clean, payload)
            x_stego = (x_clean + delta).clamp(-1, 1)
            delta_norms = delta.pow(2).mean()

            ctrl_boot_attack = float(config.get("controller_bootstrap_attack_prob", 0.0))
            if ctrl_boot_attack > 0:
                x_for_extraction = attack_layer(
                    x_stego, attack_prob=ctrl_boot_attack, combined=False, n_combined=1
                )
            else:
                x_for_extraction = x_stego

            bits_pred = extractor(x_for_extraction)
            bits_pred_loss = bits_pred[:, :active_bits]
            payload_loss = payload[:, :active_bits]

            loss_msg = F.binary_cross_entropy_with_logits(bits_pred_loss, payload_loss)
            loss_align = F.mse_loss(delta, target_delta)
            align_w = float(config.get("controller_bootstrap_align_weight", 6.0))
            if extractor_frozen:
                # First make controller emit a decodable signal shape.
                # Message gradients are unreliable while extractor is frozen.
                loss = align_w * loss_align
            else:
                loss = loss_msg + align_w * loss_align

            loss_dict = {
                "msg": loss_msg.item(),
                "perc": 0.0,
                "ctrl": delta_norms.item(),
                "adv": 0.0,
                "spec": 0.0,
                "align": loss_align.item(),
            }
            disc_loss_val = 0.0
            disc_acc_val = 0.0
            attack_prob = ctrl_boot_attack
            disc_logits_stego = None

            loss.backward()
            if extractor_frozen:
                torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=config["grad_clip"])
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(controller.parameters()) + list(extractor.parameters()),
                    max_norm=config["grad_clip"],
                )
            optimizer.step()
            scheduler.step()

        else:
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
            active_bits = get_active_bits(step, config)
            bits_pred_loss = bits_pred[:, :active_bits]
            payload_loss = payload[:, :active_bits]

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
                bits_pred_loss, payload_loss, x_stego, x_clean, delta_norms,
                step=step, disc_logits_stego=disc_logits_stego,
            )

            # Spectral loss (penalize frequency-domain differences)
            spec_loss = compute_spectral_loss(x_stego, x_clean)
            loss = loss + config["lambda_spec"] * spec_loss
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
            ber_active = compute_ber(bits_pred_loss.detach(), payload_loss)
            psnr = compute_psnr(x_stego.detach(), x_clean.detach())
            delta_rms = delta.pow(2).mean().sqrt().item()
            avg_time = sum(step_times[-10:]) / len(step_times[-10:])
            remaining = avg_time * (config["total_steps"] - step)
            cur_lr = scheduler.get_last_lr()[0]

            if in_extractor_bootstrap:
                phase = "BOOT-EXT"
            elif in_controller_bootstrap:
                phase = "BOOT-CTRL"
            else:
                phase = "TRAIN"
            log_line = (
                f"{phase} Step {step:5d}/{config['total_steps']} | "
                f"BER: {ber:.4f} | "
                f"A-BER({active_bits}): {ber_active:.4f} | "
                f"PSNR: {psnr:.1f}dB | "
                f"msg: {loss_dict['msg']:.4f} | perc: {loss_dict['perc']:.4f} | "
                f"spec: {loss_dict['spec']:.4f} | "
                f"d_rms: {delta_rms:.4f} | "
                f"atk: {attack_prob:.2f} | "
                f"lr: {cur_lr:.1e} | "
                f"{avg_time:.2f}s/step | ETA: {remaining/60:.0f}min"
            )
            if (phase == "TRAIN") and config["use_adversarial"] and step >= config["adv_start_step"]:
                log_line += (
                    f" | D_loss: {disc_loss_val:.3f} D_acc: {disc_acc_val:.2f}"
                    f" adv: {loss_dict.get('adv', 0):.4f}"
                )
            print(log_line)

            if ber < best_ber:
                best_ber = ber
            if ber_active < best_ber_active:
                best_ber_active = ber_active

            # Write a small JSON progress file — allows monitoring without
            # stdout buffering (Lesson 5).
            progress = {
                "step": step,
                "total_steps": config["total_steps"],
                "ber": round(ber, 6),
                "ber_active": round(ber_active, 6),
                "active_bits": active_bits,
                "phase": phase,
                "psnr": round(psnr, 2),
                "best_ber": round(best_ber, 6),
                "best_ber_active": round(best_ber_active, 6),
                "msg_loss": round(loss_dict["msg"], 6),
                "perc_loss": round(loss_dict["perc"], 6),
                "delta_rms": round(delta_rms, 6),
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
            os.makedirs(config["output_dir"], exist_ok=True)
            save_comparison(
                x_stego.detach(), x_clean.detach(),
                step=step, output_dir=config["output_dir"],
            )

        # ── Checkpoint ──
        if step % config["checkpoint_interval"] == 0:
            os.makedirs(config["checkpoint_dir"], exist_ok=True)
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
                "best_ber_active": best_ber_active,
            }
            if discriminator is not None:
                ckpt_data["discriminator"] = discriminator.state_dict()
            if disc_optimizer is not None:
                ckpt_data["disc_optimizer"] = disc_optimizer.state_dict()
            torch.save(ckpt_data, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # ── Signal-triggered emergency checkpoint ──
        if _interrupted[0]:
            os.makedirs(config["checkpoint_dir"], exist_ok=True)
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
                "best_ber_active": best_ber_active,
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
    print(f"  Best active-bit BER during training: {best_ber_active:.6f}")

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
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    final_path = os.path.join(config["checkpoint_dir"], "dscsp_v4_final.pt")
    final_data = {
        "step": config["total_steps"],
        "controller": controller.state_dict(),
        "extractor": extractor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "final_ber": avg_ber,
        "final_psnr": avg_psnr,
        "best_ber": best_ber,
        "best_ber_active": best_ber_active,
    }
    if discriminator is not None:
        final_data["discriminator"] = discriminator.state_dict()
        final_data["disc_acc"] = disc_acc_final
    if disc_optimizer is not None:
        final_data["disc_optimizer"] = disc_optimizer.state_dict()
    torch.save(final_data, final_path)
    print(f"  Saved final model: {final_path}")

    # Save results JSON
    results = {
        "final_ber": avg_ber,
        "final_psnr": avg_psnr,
        "best_ber": best_ber,
        "best_ber_active": best_ber_active,
        "disc_acc": disc_acc_final,
        "config": {k: str(v) if isinstance(v, tuple) else v for k, v in config.items()},
    }
    results_path = os.path.join(config["output_dir"], "train_v4_results.json")
    os.makedirs(config["output_dir"], exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results: {results_path}")


if __name__ == "__main__":
    train()
