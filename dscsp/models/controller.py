"""
Controller network for DSCSP.

Maps payload bits to a spatial perturbation using multi-bit spatial tiling.

Architecture (v5 - Multi-bit per Cell):
  The image is divided into an 8x8 grid of 32x32-pixel cells.
  Each cell encodes `bits_per_cell` bits (default 4), giving 256 total bits.
  Each bit has its own learned 3x32x32 pattern within its cell.
  Bits are mapped from {0,1} to {-1,+1} signs, patterns are sign-modulated
  and summed within each cell, then tiled into the full image.

  Within-cell SNR = 1/(bits_per_cell-1) = 1/3 for 4 bits/cell.
  The extractor's 8x8 feature map (ResNet stride-32) aligns with the grid,
  and a 4-channel 1x1 conv head reads each cell's 4 bits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(nn.Module):
    """
    Maps payload bits to a spatial perturbation delta using multi-bit
    spatial tiling. Each 32x32 cell encodes `bits_per_cell` bits.
    """

    def __init__(
        self,
        num_bits: int = 256,
        bits_per_cell: int = 4,
        target_res: int = 256,
        max_perturbation: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.bits_per_cell = bits_per_cell
        self.target_res = target_res
        self.max_perturbation = max_perturbation

        # Grid layout: 8x8 for 64 cells, 4 bits each = 256 bits
        self.num_cells = num_bits // bits_per_cell
        self.grid_size = 8
        self.cell_size = target_res // self.grid_size  # 32

        assert num_bits % bits_per_cell == 0, \
            f"num_bits {num_bits} not divisible by bits_per_cell {bits_per_cell}"
        assert self.num_cells == self.grid_size ** 2, \
            f"num_cells {self.num_cells} != grid_size^2 {self.grid_size**2}"
        assert target_res % self.grid_size == 0, \
            f"target_res {target_res} not divisible by grid_size {self.grid_size}"

        # Learned spatial patterns: (num_cells, bits_per_cell, 3, cell_size, cell_size)
        # Each cell has `bits_per_cell` independent patterns
        self.bit_patterns = nn.Parameter(
            torch.randn(self.num_cells, bits_per_cell, 3, self.cell_size, self.cell_size) * 0.25
        )

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bits: (B, num_bits) float tensor with values in {0, 1}

        Returns:
            delta: (B, 3, target_res, target_res) perturbation,
                   constrained to [-max_perturbation, max_perturbation]
        """
        B = bits.shape[0]
        G = self.grid_size
        C = self.cell_size
        K = self.bits_per_cell

        # Group bits by cell: (B, num_cells, bits_per_cell)
        bits_grouped = bits.view(B, self.num_cells, K)

        # Map {0, 1} -> {-1, +1}
        signs = bits_grouped * 2 - 1  # (B, num_cells, K)

        # Apply signs to patterns: (B, num_cells, K, 3, C, C)
        signed = signs[:, :, :, None, None, None] * self.bit_patterns[None]

        # Sum over bits_per_cell dimension within each cell: (B, num_cells, 3, C, C)
        cell_patterns = signed.sum(dim=2)

        # Reshape to grid: (B, G, G, 3, C, C)
        cell_patterns = cell_patterns.view(B, G, G, 3, C, C)

        # Rearrange to image: (B, 3, G*C, G*C)
        delta = cell_patterns.permute(0, 3, 1, 4, 2, 5).contiguous()
        delta = delta.view(B, 3, G * C, G * C)

        # Constrain perturbation magnitude
        delta = torch.tanh(delta) * self.max_perturbation

        return delta
