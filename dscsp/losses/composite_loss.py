"""
Composite loss for DSCSP training.

Combines:
  1. Message reconstruction loss (BCE with logits)
  2. Perceptual fidelity loss (LPIPS between stego and clean reference)
  3. Controller regularization (L2 on perturbation magnitudes)
  4. Adversarial loss (optional, for reducing steganalysis detectability)

Supports curriculum training via warmup_steps: perceptual and regularization
losses are linearly ramped from 0 to their full weight over the warmup period.
Adversarial loss has its own independent curriculum controlled by
adv_start_step and adv_ramp_steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class DSCSPLoss(nn.Module):
    """Combined training loss for the DSCSP steganography system."""

    def __init__(
        self,
        lambda_perc: float = 1.0,
        lambda_ctrl: float = 0.5,
        warmup_steps: int = 500,
        lambda_adv: float = 0.0,
        adv_start_step: int = 5000,
        adv_ramp_steps: int = 5000,
    ):
        super().__init__()
        self.lambda_perc = lambda_perc
        self.lambda_ctrl = lambda_ctrl
        self.warmup_steps = warmup_steps
        self.lambda_adv = lambda_adv
        self.adv_start_step = adv_start_step
        self.adv_ramp_steps = adv_ramp_steps

        # LPIPS perceptual loss (frozen VGG backbone)
        self.lpips_fn = lpips.LPIPS(net="vgg")
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)

    def _get_adv_ramp(self, step: int) -> float:
        """Compute adversarial loss ramp factor based on current step."""
        if self.lambda_adv <= 0 or step < self.adv_start_step:
            return 0.0
        if self.adv_ramp_steps <= 0:
            return 1.0
        progress = (step - self.adv_start_step) / self.adv_ramp_steps
        return min(progress, 1.0)

    def forward(
        self,
        bits_pred: torch.Tensor,
        bits_target: torch.Tensor,
        x_stego: torch.Tensor,
        x_clean: torch.Tensor,
        delta_norms: torch.Tensor,
        step: int = 0,
        disc_logits_stego: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute the composite loss with curriculum warmup.

        Args:
            bits_pred: (B, num_bits) logits from extractor
            bits_target: (B, num_bits) ground truth bits in {0, 1}
            x_stego: (B, 3, H, W) stego images in [-1, 1]
            x_clean: (B, 3, H, W) clean reference images in [-1, 1]
            delta_norms: scalar, mean L2 of controller perturbations
            step: current training step (for curriculum warmup)
            disc_logits_stego: (B, 1) discriminator logits for stego images,
                               or None to skip adversarial loss.

        Returns:
            total_loss: scalar tensor for backpropagation
            loss_dict: dictionary of individual loss components (detached floats)
        """
        # Curriculum: linearly ramp perceptual/ctrl weights from 0 to full
        if self.warmup_steps > 0 and step < self.warmup_steps:
            ramp = step / self.warmup_steps
        else:
            ramp = 1.0

        # 1. Message reconstruction loss (always full weight)
        loss_msg = F.binary_cross_entropy_with_logits(bits_pred, bits_target)

        # 2. Perceptual loss (ramped)
        loss_perc = self.lpips_fn(x_stego, x_clean).mean()

        # 3. Controller regularization (ramped)
        loss_ctrl = delta_norms

        # Total with curriculum ramping
        total = (
            loss_msg
            + ramp * self.lambda_perc * loss_perc
            + ramp * self.lambda_ctrl * loss_ctrl
        )

        loss_dict = {
            "msg": loss_msg.item(),
            "perc": loss_perc.item(),
            "ctrl": loss_ctrl.item(),
            "ramp": ramp,
        }

        # 4. Adversarial loss (independent curriculum)
        if disc_logits_stego is not None and self.lambda_adv > 0:
            ramp_adv = self._get_adv_ramp(step)
            if ramp_adv > 0:
                # Controller wants discriminator to classify stego as clean (0)
                loss_adv = F.binary_cross_entropy_with_logits(
                    disc_logits_stego,
                    torch.zeros_like(disc_logits_stego),
                )
                total = total + ramp_adv * self.lambda_adv * loss_adv
                loss_dict["adv"] = loss_adv.item()
                loss_dict["ramp_adv"] = ramp_adv
            else:
                loss_dict["adv"] = 0.0
                loss_dict["ramp_adv"] = 0.0
        else:
            loss_dict["adv"] = 0.0
            loss_dict["ramp_adv"] = 0.0

        loss_dict["total"] = total.item()

        return total, loss_dict
