"""
Content-adaptive steganographic controller for DSCSP v4.

Replaces the tiled-grid controller (v1/v2) with a U-Net encoder-decoder
that produces image-dependent perturbations. Key differences:

  1. Takes BOTH cover image and payload as input -> content-adaptive delta
  2. No periodic grid structure -> smooth perturbation, no tile boundary artifacts
  3. Skip connections preserve spatial detail for targeted embedding

Architecture:
  Payload (B,256) -> MLP -> (B, ch, 4, 4) -> upsample to (B, ch, 256, 256)
  Concat with cover image -> (B, ch+3, 256, 256)
  3-level U-Net encoder-decoder with skip connections
  Output: tanh(delta) * max_perturbation -> (B, 3, 256, 256)

~3.5M parameters with base_channels=48.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv + BatchNorm + ReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Downsample by 2x with two conv layers."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample by 2x with ConvTranspose + conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.bn_up = nn.BatchNorm2d(out_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.relu(self.bn_up(self.up(x)), inplace=True)
        return self.conv(x)


class ContentAdaptiveController(nn.Module):
    """
    Content-adaptive steganographic encoder.

    Takes cover image + payload bits as input. Outputs a perturbation
    delta that is spatially adapted to the image content, concentrating
    changes in textured/edge regions and minimizing changes in smooth areas.

    Args:
        num_bits: Number of payload bits to embed (default 256)
        max_perturbation: Maximum absolute perturbation magnitude
        base_channels: Base channel count for U-Net (affects model size)
    """

    def __init__(
        self,
        num_bits: int = 256,
        max_perturbation: float = 0.1,
        base_channels: int = 48,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.max_perturbation = max_perturbation
        ch = base_channels

        # Message preparation: bits -> spatial feature map
        self.msg_encoder = nn.Sequential(
            nn.Linear(num_bits, ch * 16),
            nn.ReLU(inplace=True),
        )
        # Output: (B, ch, 4, 4) -> upsampled to (B, ch, 256, 256)

        # Initial convolution: image (3ch) + message (ch) -> ch
        self.initial = ConvBlock(3 + ch, ch)

        # Encoder path (3 levels of downsampling)
        self.enc1 = DownBlock(ch, ch * 2)       # 256 -> 128
        self.enc2 = DownBlock(ch * 2, ch * 4)   # 128 -> 64
        self.enc3 = DownBlock(ch * 4, ch * 4)   # 64 -> 32

        # Decoder path with skip connections
        self.dec3 = UpBlock(ch * 4, ch * 4)     # 32 -> 64
        # After concat with enc2 skip: ch*4 + ch*4 = ch*8
        self.dec2_reduce = ConvBlock(ch * 8, ch * 4)
        self.dec2 = UpBlock(ch * 4, ch * 2)     # 64 -> 128
        # After concat with enc1 skip: ch*2 + ch*2 = ch*4
        self.dec1_reduce = ConvBlock(ch * 4, ch * 2)
        self.dec1 = UpBlock(ch * 2, ch)         # 128 -> 256

        # Final output: concat with initial skip (ch + ch = ch*2) -> 3
        self.final = nn.Sequential(
            ConvBlock(ch * 2, ch),
            nn.Conv2d(ch, 3, kernel_size=1),
        )

    def forward(self, image: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, 256, 256) cover image in [-1, 1]
            payload: (B, num_bits) payload bits in {0, 1}

        Returns:
            delta: (B, 3, 256, 256) perturbation in [-max_pert, +max_pert]
        """
        B = image.shape[0]
        H, W = image.shape[2], image.shape[3]

        # Prepare message as spatial feature map
        msg = self.msg_encoder(payload)             # (B, ch*16)
        ch = msg.shape[1] // 16
        msg = msg.view(B, ch, 4, 4)                 # (B, ch, 4, 4)
        msg = F.interpolate(msg, size=(H, W), mode='bilinear', align_corners=False)

        # Concatenate image + message and initial conv
        d0 = self.initial(torch.cat([image, msg], dim=1))   # (B, ch, 256, 256)

        # Encoder
        d1 = self.enc1(d0)      # (B, ch*2, 128, 128)
        d2 = self.enc2(d1)      # (B, ch*4, 64, 64)
        d3 = self.enc3(d2)      # (B, ch*4, 32, 32)

        # Decoder with skip connections
        u3 = self.dec3(d3)                                      # (B, ch*4, 64, 64)
        u3 = self.dec2_reduce(torch.cat([u3, d2], dim=1))       # (B, ch*4, 64, 64)
        u2 = self.dec2(u3)                                      # (B, ch*2, 128, 128)
        u2 = self.dec1_reduce(torch.cat([u2, d1], dim=1))       # (B, ch*2, 128, 128)
        u1 = self.dec1(u2)                                      # (B, ch, 256, 256)

        # Final with skip from initial
        delta = self.final(torch.cat([u1, d0], dim=1))  # (B, 3, 256, 256)

        return torch.tanh(delta) * self.max_perturbation
