"""
Frozen diffusion model wrapper for DSCSP.

Loads a pre-trained HuggingFace DDPM UNet in FP16 and exposes
a clean interface for noise prediction and DDIM alpha schedules.
"""

import torch
from diffusers import DDPMPipeline, DDIMScheduler


class DiffusionWrapper:
    """Wraps a frozen HuggingFace UNet for use in the DSCSP sampling loop."""

    def __init__(
        self,
        model_id: str = "google/ddpm-celebahq-256",
        device: str = "cuda",
        num_inference_steps: int = 20,
    ):
        # Load pre-trained pipeline
        pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.unet = pipe.unet.to(device).eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)

        # Setup DDIM scheduler from the DDPM config
        self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.scheduler.set_timesteps(num_inference_steps)

        # Timesteps in descending order (e.g. [981, 961, ..., 1])
        self.timesteps = self.scheduler.timesteps.to(device)
        self.num_steps = num_inference_steps

        # Alpha cumulative products for the full schedule (indexed by timestep value)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(device).float()
        self.final_alpha_cumprod = torch.tensor(
            self.scheduler.final_alpha_cumprod, device=device, dtype=torch.float32
        )

        self.device = device
        self.image_shape = (3, 256, 256)

    @torch.no_grad()
    def predict_noise(self, z_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Run the frozen UNet to predict noise. Output is FP32, fully detached.

        Args:
            z_t: noisy image tensor, any dtype (will be cast to FP16)
            timestep: scalar or 1-element tensor with the diffusion timestep

        Returns:
            noise prediction as float32 tensor, detached from any grad graph
        """
        z_in = z_t.to(dtype=torch.float16)
        noise_pred = self.unet(z_in, timestep).sample
        return noise_pred.float()

    def get_alpha(self, timestep: int) -> torch.Tensor:
        """Get alpha_cumprod for a given timestep value."""
        return self.alphas_cumprod[timestep]

    def get_prev_alpha(self, step_idx: int) -> torch.Tensor:
        """
        Get alpha_cumprod for the previous timestep in the DDIM schedule.

        Args:
            step_idx: index into self.timesteps (0-based)

        Returns:
            alpha_cumprod for the next-lower timestep, or final_alpha_cumprod
            if we're at the last step
        """
        if step_idx + 1 < len(self.timesteps):
            prev_t = self.timesteps[step_idx + 1]
            return self.alphas_cumprod[prev_t]
        return self.final_alpha_cumprod

    @torch.no_grad()
    def sample_clean(self, batch_size: int = 1, z_T: torch.Tensor = None) -> torch.Tensor:
        """
        Generate a clean (unperturbed) image using standard DDIM sampling.
        Useful for testing that the model works.

        Returns:
            images in [-1, 1] range, shape (B, 3, 256, 256)
        """
        if z_T is None:
            z_T = torch.randn(batch_size, *self.image_shape, device=self.device)

        z = z_T
        for i, t in enumerate(self.timesteps):
            noise_pred = self.predict_noise(z, t)
            alpha_t = self.get_alpha(t)
            alpha_prev = self.get_prev_alpha(i)

            # Predict x_0
            x0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

            # DDIM deterministic step
            z = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred

        return z.clamp(-1, 1)
