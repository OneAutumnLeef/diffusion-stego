"""
Modified DDIM sampler for DSCSP.

Simplified architecture: the entire DDIM denoising chain runs under
torch.no_grad() to produce a clean base image. The Controller's perturbation
delta is then added directly to this final image (IN the gradient graph).

This gives the cleanest possible gradient path:
  loss → extractor → (x_clean + delta).clamp → delta → controller

The DDIM chain itself doesn't need gradients since we're perturbing the
output image directly. This approach converges much faster because:
  1. The extractor sees the full perturbation signal (no DDIM scaling)
  2. The controller gets clean gradients (no noisy pass through untrained extractor
     at intermediate DDIM steps)
  3. Both networks can learn their roles simultaneously

Later, once this base system converges, perturbation injection can be moved
earlier in the DDIM chain for more subtle encoding.
"""

import torch

from dscsp.models.diffusion_wrapper import DiffusionWrapper
from dscsp.models.controller import Controller


class StegoDDIM:
    """DDIM sampler that adds a learned perturbation to the final denoised image."""

    def __init__(self, diffusion: DiffusionWrapper, controller: Controller):
        self.diffusion = diffusion
        self.controller = controller

    def sample(
        self,
        payload_bits: torch.Tensor,
        z_T: torch.Tensor = None,
        return_clean: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Generate stego images encoding the payload.

        Args:
            payload_bits: (B, num_bits) float tensor with values in {0, 1}
            z_T: optional initial noise (B, 3, 256, 256). If None, sampled randomly.
            return_clean: if True, also return the clean (unperturbed) image

        Returns:
            x_stego: (B, 3, 256, 256) in [-1,1], IN gradient graph
            x_clean: (B, 3, 256, 256) in [-1,1], detached (None if return_clean=False)
            delta_norms: scalar tensor, mean L2 of the perturbation delta
        """
        B = payload_bits.size(0)
        device = payload_bits.device
        image_shape = self.diffusion.image_shape  # (3, 256, 256)

        # 1. Controller forward: produce perturbation delta (IN grad graph)
        delta = self.controller(payload_bits)  # (B, 3, 256, 256)

        # 2. Generate clean image via standard DDIM (entirely no_grad)
        if z_T is None:
            z_T = torch.randn(B, *image_shape, device=device)

        with torch.no_grad():
            z = z_T.clone()
            timesteps = self.diffusion.timesteps

            for i, t in enumerate(timesteps):
                noise_pred = self.diffusion.predict_noise(z, t)
                alpha_t = self.diffusion.get_alpha(t)
                alpha_prev = self.diffusion.get_prev_alpha(i)

                x0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                z = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred

            x_clean = z.clamp(-1, 1)

        # 3. Add perturbation to final image (IN gradient graph)
        x_stego = (x_clean + delta).clamp(-1, 1)

        # 4. Compute delta norm for regularization
        delta_norms = delta.pow(2).mean()

        # Return clean as detached if requested
        x_clean_out = x_clean if return_clean else None

        return x_stego, x_clean_out, delta_norms
