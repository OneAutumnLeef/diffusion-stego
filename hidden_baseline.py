"""HiDDeN-compatible baseline (Zhu et al., 2018) — in-repo reimplementation.

Architecture follows the original paper:
  Encoder: 4 ConvBNReLU(3->64) -> concat(replicated message map) -> ConvBNReLU
           -> 1x1 conv to 3 channels -> delta added to cover image.
  Decoder: 7 ConvBNReLU(3->64) -> 1x1 conv to num_bits -> GAP -> FC(num_bits).
  Optional adversary (SN-disc — same as DSCSP's).

Input resolution 256x256, payload 256 bits (matched to DSCSP's physical payload
size). Cover images come from the same ``google/ddpm-celebahq-256`` DDIM pool
used by DSCSP, so JPEG/crop/AUC comparisons are on the exact same distribution.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class HiDDeNEncoder(nn.Module):
    """Encoder E(C, M) -> delta. Matches the original layer sequence."""

    def __init__(self, num_bits: int = 256, n_blocks: int = 4, ch: int = 64,
                 max_perturbation: float = 0.08):
        super().__init__()
        self.num_bits = num_bits
        self.max_perturbation = max_perturbation
        self.n_blocks = n_blocks
        self.ch = ch

        blocks = [_conv_bn_relu(3, ch)]
        for _ in range(n_blocks - 1):
            blocks.append(_conv_bn_relu(ch, ch))
        self.stem = nn.Sequential(*blocks)

        # After concatenation with replicated message + original cover:
        # ch (feat) + num_bits (message) + 3 (cover) channels
        self.fuse = _conv_bn_relu(ch + num_bits + 3, ch)
        self.out = nn.Conv2d(ch, 3, kernel_size=1)

    def forward(self, cover: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        B, _, H, W = cover.shape
        feat = self.stem(cover)
        # Replicate (B, num_bits) -> (B, num_bits, H, W)
        msg_map = message.view(B, self.num_bits, 1, 1).expand(B, self.num_bits, H, W)
        fused = self.fuse(torch.cat([feat, msg_map, cover], dim=1))
        delta_raw = self.out(fused)
        # Bound the residual — keeps stego within [-1, 1] after addition.
        return torch.tanh(delta_raw) * self.max_perturbation


class HiDDeNDecoder(nn.Module):
    """Decoder D(stego) -> logits of length num_bits."""

    def __init__(self, num_bits: int = 256, n_blocks: int = 7, ch: int = 64):
        super().__init__()
        self.num_bits = num_bits
        blocks = [_conv_bn_relu(3, ch)]
        for _ in range(n_blocks - 1):
            blocks.append(_conv_bn_relu(ch, ch))
        self.trunk = nn.Sequential(*blocks)
        self.head = nn.Conv2d(ch, num_bits, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_bits, num_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        h = self.head(h)
        h = self.gap(h).flatten(1)
        return self.fc(h)


class HiDDeNAdversary(nn.Module):
    """3-block CNN adversary — cf. original paper's discriminator."""

    def __init__(self, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, stride=1, padding=1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, stride=1, padding=1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, stride=1, padding=1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = self.pool(h).flatten(1)
        return self.fc(h)


def encode(encoder: HiDDeNEncoder, cover: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    """Full stego generation. Returns clamped stego image."""
    delta = encoder(cover, message)
    return (cover + delta).clamp(-1, 1)
