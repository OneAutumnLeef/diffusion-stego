"""
Extractor network for DSCSP.

Recovers payload bits from a stego image using a ResNet-18 backbone
with a spatially-aware multi-channel 1x1 conv head.

Architecture (v3 - Multi-bit Spatial Head):
  ResNet-18 backbone (minus avgpool and fc) produces (B, 512, 8, 8)
  feature maps when given 256x256 input (stride 32).

  A 1x1 conv reduces 512 channels to `bits_per_cell` channels,
  giving (B, bits_per_cell, 8, 8). After permuting to (B, 8, 8, bits_per_cell)
  and flattening, this yields (B, num_bits) logits.

  The permute ensures bit ordering matches the Controller's view(B, 64, 4):
  bit i -> cell i // bits_per_cell, within-cell position i % bits_per_cell.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class Extractor(nn.Module):
    """
    Extracts hidden payload bits from a stego image.

    Uses a ResNet-18 backbone with a multi-channel 1x1 conv spatial head.
    Each 8x8 spatial position reads `bits_per_cell` bits from its
    corresponding 32x32 cell, aligned with the Controller's tiling grid.
    """

    def __init__(self, num_bits: int = 256, bits_per_cell: int = 4):
        super().__init__()
        self.num_bits = num_bits
        self.bits_per_cell = bits_per_cell

        # ResNet-18 backbone without pretrained weights
        backbone = models.resnet18(weights=None)

        # Remove avgpool and fc — keep everything up to layer4
        # children: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # Multi-channel 1x1 conv spatial head: (B, 512, 8, 8) -> (B, bits_per_cell, 8, 8)
        self.head = nn.Conv2d(512, bits_per_cell, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, 256, 256) in [-1, 1] range

        Returns:
            (B, num_bits) logits (apply sigmoid for bit probabilities)
        """
        feat = self.features(image)        # (B, 512, 8, 8)
        out = self.head(feat)               # (B, bits_per_cell, 8, 8)
        # Permute so flatten order matches controller's view(B, 64, 4):
        # bit i -> cell i // bits_per_cell, channel i % bits_per_cell
        out = out.permute(0, 2, 3, 1)      # (B, 8, 8, bits_per_cell)
        return out.reshape(out.size(0), -1)  # (B, 256)
