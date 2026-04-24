"""
Differentiable attack layers for robustness training.

All attacks are differentiable w.r.t. the input tensor so gradients
flow back through the attack to the controller during training.

Attacks:
  - Gaussian noise
  - Gaussian blur (depthwise conv)
  - Brightness/contrast adjustment
  - Crop + resize (bilinear interpolation)
  - JPEG compression (STE-based differentiable proxy)
  - Real libjpeg JPEG round-trip with straight-through gradient
"""

import io
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def real_jpeg_ste(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Real libjpeg JPEG round-trip with straight-through gradient.

    Forward: PIL.Image encode at ``quality``, decode back. Range-in/range-out [-1, 1].
    Backward: identity (STE) — gradient flows through ``x`` unchanged.
    """
    with torch.no_grad():
        imgs = ((x.detach().clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)
        out = np.empty_like(imgs)
        for i, arr in enumerate(imgs):
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
            buf.seek(0)
            out[i] = np.array(Image.open(buf).convert("RGB"))
        y = torch.from_numpy(out).to(x.device).permute(0, 3, 1, 2).float()
        y = (y / 127.5) - 1.0
    return x + (y - x).detach()


# ─── Individual Attacks ──────────────────────────────────────────────────────


def gaussian_noise(x, sigma_range=(0.01, 0.05)):
    """Add zero-mean Gaussian noise. Gradient: identity."""
    sigma = random.uniform(*sigma_range)
    noise = torch.randn_like(x) * sigma
    return (x + noise).clamp(-1, 1)


def gaussian_blur(x, kernel_size=5, sigma_range=(0.5, 2.0)):
    """Gaussian blur via depthwise convolution. Fully differentiable."""
    sigma = random.uniform(*sigma_range)
    # Build 1D Gaussian kernel
    half = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - half
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    # 2D kernel via outer product
    kernel_2d = g[:, None] * g[None, :]  # (K, K)
    # Shape for depthwise conv: (3, 1, K, K)
    kernel = kernel_2d.expand(3, 1, -1, -1).contiguous()
    padding = half
    return F.conv2d(x, kernel, padding=padding, groups=3)


def brightness_contrast(x, brightness_range=(-0.1, 0.1), contrast_range=(0.8, 1.2)):
    """Linear brightness/contrast adjustment. Gradient: contrast factor."""
    b = random.uniform(*brightness_range)
    c = random.uniform(*contrast_range)
    return (x * c + b).clamp(-1, 1)


def crop_resize(x, crop_ratio_range=(0.7, 0.95)):
    """Random crop followed by bilinear resize to original size."""
    ratio = random.uniform(*crop_ratio_range)
    H, W = x.shape[2], x.shape[3]
    crop_h = int(H * ratio)
    crop_w = int(W * ratio)
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)
    cropped = x[:, :, top:top + crop_h, left:left + crop_w]
    return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)


# ─── Differentiable JPEG ─────────────────────────────────────────────────────


class DiffJPEG(nn.Module):
    """
    Differentiable JPEG compression using Straight-Through Estimator (STE).

    Forward pass applies real JPEG quantization (RGB → YCbCr → block DCT →
    quantize → IDCT → RGB). Backward pass uses identity gradient through
    the quantization step (STE).
    """

    def __init__(self):
        super().__init__()
        # Standard JPEG luminance quantization table
        lum_q = torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ], dtype=torch.float32)

        # Standard JPEG chrominance quantization table
        chrom_q = torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ], dtype=torch.float32)

        self.register_buffer('lum_q', lum_q)
        self.register_buffer('chrom_q', chrom_q)

        # Precompute DCT basis matrix (8x8)
        dct_mat = self._build_dct_matrix(8)
        self.register_buffer('dct_mat', dct_mat)
        self.register_buffer('dct_mat_t', dct_mat.t())

    @staticmethod
    def _build_dct_matrix(n):
        """Build the NxN DCT-II transform matrix."""
        mat = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    mat[k, i] = 1.0 / math.sqrt(n)
                else:
                    mat[k, i] = math.sqrt(2.0 / n) * math.cos(
                        math.pi * (2 * i + 1) * k / (2 * n)
                    )
        return mat

    def _get_scaled_q_tables(self, quality):
        """Scale quantization tables by JPEG quality factor."""
        if quality < 50:
            scale = 5000.0 / quality
        else:
            scale = 200.0 - 2.0 * quality
        scale = max(scale, 1.0)
        q_lum = ((self.lum_q * scale + 50) / 100).clamp(min=1)
        q_chrom = ((self.chrom_q * scale + 50) / 100).clamp(min=1)
        return q_lum, q_chrom

    @staticmethod
    def _rgb_to_ycbcr(x):
        """Convert RGB [0, 255] to YCbCr."""
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
        return y, cb, cr

    @staticmethod
    def _ycbcr_to_rgb(y, cb, cr):
        """Convert YCbCr to RGB [0, 255]."""
        r = y + 1.402 * (cr - 128)
        g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
        b = y + 1.772 * (cb - 128)
        return torch.cat([r, g, b], dim=1)

    def _block_dct(self, x):
        """Apply 8x8 block DCT to a single-channel image."""
        B, _, H, W = x.shape
        bh, bw = H // 8, W // 8
        # Reshape into 8x8 blocks: (B, bh, bw, 8, 8)
        x = x.view(B, 1, bh, 8, bw, 8)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()  # (B, 1, bh, bw, 8, 8)
        x = x.view(-1, 8, 8)  # (B*bh*bw, 8, 8)
        # DCT: D @ block @ D^T
        x = self.dct_mat @ x @ self.dct_mat_t
        x = x.view(B, 1, bh, bw, 8, 8)
        return x

    def _block_idct(self, x):
        """Apply 8x8 block inverse DCT."""
        B_bh_bw = x.shape[0] * x.shape[2] * x.shape[3]
        x = x.view(-1, 8, 8)
        # IDCT: D^T @ block @ D
        x = self.dct_mat_t @ x @ self.dct_mat
        B = x.shape[0] // (x.shape[0] // B_bh_bw) if B_bh_bw > 0 else 1
        return x

    def forward(self, x, quality=75):
        """
        Args:
            x: (B, 3, H, W) in [-1, 1]
            quality: JPEG quality factor (1-100)
        Returns:
            JPEG-compressed image in [-1, 1]
        """
        B, C, H, W = x.shape
        bh, bw = H // 8, W // 8

        # Convert to [0, 255]
        x_255 = (x + 1) * 127.5

        # RGB -> YCbCr
        y, cb, cr = self._rgb_to_ycbcr(x_255)

        # Get scaled quantization tables
        q_lum, q_chrom = self._get_scaled_q_tables(quality)

        result_channels = []
        for ch, q_table in [(y, q_lum), (cb, q_chrom), (cr, q_chrom)]:
            # Center around 0 (JPEG subtracts 128 from Y channel)
            ch = ch - 128

            # Block DCT
            blocks = ch.view(B, 1, bh, 8, bw, 8)
            blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
            blocks = blocks.view(-1, 8, 8)
            dct_blocks = self.dct_mat @ blocks @ self.dct_mat_t

            # STE quantization: forward = round, backward = identity
            q = q_table.view(1, 8, 8)
            quantized = torch.round(dct_blocks / q) * q
            dct_blocks = dct_blocks + (quantized - dct_blocks).detach()

            # Block IDCT
            spatial = self.dct_mat_t @ dct_blocks @ self.dct_mat

            # Reconstruct channel
            spatial = spatial.view(B, 1, bh, bw, 8, 8)
            spatial = spatial.permute(0, 1, 2, 4, 3, 5).contiguous()
            spatial = spatial.view(B, 1, H, W)

            # Undo centering
            spatial = spatial + 128
            result_channels.append(spatial)

        # YCbCr -> RGB
        rgb = self._ycbcr_to_rgb(result_channels[0], result_channels[1], result_channels[2])

        # Back to [-1, 1]
        return (rgb / 127.5 - 1).clamp(-1, 1)


# ─── Attack Layer Wrapper ────────────────────────────────────────────────────


class AttackLayer(nn.Module):
    """
    Applies differentiable attacks during training for robustness.

    Supports two modes:
      - Single attack (default): one random attack per batch.
      - Combined attacks: sequentially compose multiple random attacks,
        simulating realistic post-processing pipelines (e.g., JPEG + noise).

    When not training or when attack_prob threshold is not met, returns
    input unchanged (identity).
    """

    ATTACK_NAMES = ['jpeg', 'blur', 'noise', 'crop_resize', 'brightness_contrast']

    def __init__(self, config=None):
        super().__init__()
        self.diff_jpeg = DiffJPEG()

        # Attack parameter ranges (can be overridden via config)
        cfg = config or {}
        self.config = cfg
        self.jpeg_quality_range = cfg.get('jpeg_quality_range', (50, 95))
        self.blur_kernel_size = cfg.get('blur_kernel_size', 5)
        self.blur_sigma_range = cfg.get('blur_sigma_range', (0.5, 2.0))
        self.noise_sigma_range = cfg.get('noise_sigma_range', (0.01, 0.05))
        self.crop_ratio_range = cfg.get('crop_ratio_range', (0.7, 0.95))
        self.brightness_range = cfg.get('brightness_range', (-0.1, 0.1))
        self.contrast_range = cfg.get('contrast_range', (0.8, 1.2))
        self.real_jpeg_prob = float(cfg.get('real_jpeg_prob', 0.0) or 0.0)
        # Optional second quality range for the real-libjpeg branch.
        self.real_jpeg_quality_range = cfg.get(
            'real_jpeg_quality_range', self.jpeg_quality_range
        )

    def set_real_jpeg_prob(self, p: float):
        """Update the real-libjpeg probability at runtime (used by train schedules)."""
        self.real_jpeg_prob = float(max(0.0, min(1.0, p)))

    def _apply_single(self, x, attack_name):
        """Apply a single named attack. Returns attacked tensor."""
        if attack_name == 'jpeg':
            if self.real_jpeg_prob > 0.0 and self.training and torch.rand(1).item() < self.real_jpeg_prob:
                qr = self.real_jpeg_quality_range
                q = int(torch.randint(int(qr[0]), int(qr[1]) + 1, (1,)).item())
                return real_jpeg_ste(x, q)
            quality = random.randint(*self.jpeg_quality_range)
            return self.diff_jpeg(x, quality)
        elif attack_name == 'blur':
            return gaussian_blur(x, self.blur_kernel_size, self.blur_sigma_range)
        elif attack_name == 'noise':
            return gaussian_noise(x, self.noise_sigma_range)
        elif attack_name == 'crop_resize':
            return crop_resize(x, self.crop_ratio_range)
        elif attack_name == 'brightness_contrast':
            return brightness_contrast(x, self.brightness_range, self.contrast_range)
        return x

    def forward(self, x, attack_prob=0.5, combined=False, n_combined=2):
        """
        Args:
            x: (B, 3, H, W) stego image
            attack_prob: probability of applying an attack this batch
            combined: if True, apply multiple attacks sequentially
            n_combined: number of attacks to compose when combined=True
        Returns:
            Attacked or unchanged image (same shape)
        """
        if not self.training or random.random() > attack_prob:
            return x

        if combined:
            n = min(n_combined, len(self.ATTACK_NAMES))
            attacks = random.sample(self.ATTACK_NAMES, n)
        else:
            attacks = [random.choice(self.ATTACK_NAMES)]

        for attack_name in attacks:
            x = self._apply_single(x, attack_name)

        return x
