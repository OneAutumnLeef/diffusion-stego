"""
Evaluation metrics for DSCSP.
"""

import torch
import torch.nn.functional as F


def compute_ber(bits_pred_logits: torch.Tensor, bits_target: torch.Tensor) -> float:
    """
    Compute Bit Error Rate: fraction of incorrectly decoded bits.

    Args:
        bits_pred_logits: (B, num_bits) logits from extractor
        bits_target: (B, num_bits) ground truth bits in {0, 1}

    Returns:
        BER as a float in [0, 1]. 0.5 = random chance, 0.0 = perfect.
    """
    predicted = (torch.sigmoid(bits_pred_logits) > 0.5).float()
    errors = (predicted != bits_target).float().mean()
    return errors.item()


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute PSNR between two image tensors in [-1, 1] range.

    Args:
        img1, img2: (B, C, H, W) tensors in [-1, 1]

    Returns:
        PSNR in dB (higher is better). Infinity if images are identical.
    """
    mse = F.mse_loss(img1, img2)
    if mse.item() == 0:
        return float("inf")
    # Range is [-1, 1], so max pixel value difference is 2, max squared is 4
    return (10 * torch.log10(torch.tensor(4.0) / mse)).item()
