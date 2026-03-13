"""
Visualization utilities for DSCSP.
"""

import os
import torch
import torchvision.utils as vutils


def save_comparison(
    x_stego: torch.Tensor,
    x_clean: torch.Tensor,
    step: int,
    output_dir: str,
):
    """
    Save side-by-side comparison of clean vs stego images, plus an amplified
    difference map for debugging.

    Args:
        x_stego: (B, 3, H, W) stego images in [-1, 1]
        x_clean: (B, 3, H, W) clean images in [-1, 1]
        step: training step number (for filename)
        output_dir: directory to save images
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert from [-1, 1] to [0, 1] for saving
    stego_vis = (x_stego[:4].detach().cpu() + 1) / 2
    clean_vis = (x_clean[:4].detach().cpu() + 1) / 2

    # Amplified difference map (10x)
    diff = (x_stego[:4] - x_clean[:4]).abs().detach().cpu() * 10
    diff = diff.clamp(0, 1)

    # Side-by-side: clean | stego | difference
    grid = vutils.make_grid(
        torch.cat([clean_vis, stego_vis, diff], dim=0),
        nrow=min(4, x_stego.size(0)),
        padding=2,
        normalize=False,
    )

    save_path = os.path.join(output_dir, f"step_{step:06d}.png")
    vutils.save_image(grid, save_path)
