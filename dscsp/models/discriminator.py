"""
Adversarial discriminator for DSCSP steganalysis training.

A lightweight 5-block strided CNN that classifies images as clean (0) or
stego (1). Includes optional SRM (Spatial Rich Model) high-pass filter
preprocessing to amplify steganographic artifacts, following best practices
from the steganalysis literature (SRNet, XuNet).

The discriminator is trained adversarially: it learns to detect stego images,
while the Controller learns to fool it --- driving perturbation patterns
toward statistical undetectability.

Architecture:
    [Optional SRM filters] -> 5x (Conv4x4-s2 + BN + LeakyReLU) -> GAP -> FC(1)
    ~2.76M parameters, fits within ~33 MB VRAM (params + Adam states).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRMFilterLayer(nn.Module):
    """
    Fixed Spatial Rich Model high-pass filters applied per input channel.

    Uses 3 classic SRM kernels (1st-order, 2nd-order, 3rd-order residuals)
    applied independently to each RGB channel, producing 9 output channels.
    Weights are frozen (not learned).
    """

    def __init__(self):
        super().__init__()
        # 3 SRM filter kernels (5x5)
        # Filter 1: 1st-order horizontal edge
        f1 = torch.tensor([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  1, -1,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=torch.float32)

        # Filter 2: 2nd-order Laplacian
        f2 = torch.tensor([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  0],
            [ 0,  1, -4,  1,  0],
            [ 0,  0,  1,  0,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=torch.float32)

        # Filter 3: 3rd-order edge
        f3 = torch.tensor([
            [ 0,  0,  0,  0,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  2, -4,  2,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=torch.float32)

        # Stack: (3, 1, 5, 5) -- 3 filters, each applied per-channel
        filters = torch.stack([f1, f2, f3], dim=0).unsqueeze(1)
        # Repeat for 3 RGB channels: (9, 1, 5, 5) for groups=3 depthwise
        # Actually we need (9, 1, 5, 5) with groups=3, input 3 channels
        # Each group of 3 filters applies to one input channel
        filters = filters.repeat(3, 1, 1, 1)  # (9, 1, 5, 5)

        self.register_buffer('filters', filters)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image

        Returns:
            (B, 9, H, W) high-pass filtered output
        """
        return F.conv2d(x, self.filters, padding=2, groups=3)


class Discriminator(nn.Module):
    """
    Steganalysis discriminator for adversarial training.

    Classifies images as clean (label 0) or stego (label 1).
    Uses 5 strided conv blocks with batch normalization and LeakyReLU,
    followed by global average pooling and a single FC output.

    Args:
        use_srm: If True, prepend SRM high-pass filter preprocessing.
                 Increases discriminator strength for detecting steganographic
                 artifacts at the cost of 6 input channels -> 9 channels.
    """

    def __init__(self, use_srm: bool = True):
        super().__init__()
        self.use_srm = use_srm

        if use_srm:
            self.srm = SRMFilterLayer()
            in_channels = 9
        else:
            self.srm = None
            in_channels = 3

        # 5 conv blocks with stride-2 downsampling
        # Input 256x256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.blocks = nn.Sequential(
            # Block 1: no BN (standard GAN practice)
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 5
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image in [-1, 1]

        Returns:
            (B, 1) logits. Positive = predicted stego, negative = predicted clean.
        """
        if self.srm is not None:
            x = self.srm(x)

        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
