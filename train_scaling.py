"""
DSCSP Capacity Scaling Experiments

Trains DSCSP at different payload capacities (128, 256, 512 bits)
by varying bits_per_cell while keeping the 8x8 grid fixed.

Usage:
    python train_scaling.py --bits_per_cell 2   # 128 bits
    python train_scaling.py --bits_per_cell 4   # 256 bits (baseline)
    python train_scaling.py --bits_per_cell 8   # 512 bits
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn
import lpips

from dscsp.models.diffusion_wrapper import DiffusionWrapper
from dscsp.models.controller import Controller
from dscsp.models.extractor import Extractor
from dscsp.losses.composite_loss import DSCSPLoss
from dscsp.utils.metrics import compute_ber, compute_psnr
from dscsp.utils.visualization import save_comparison


def generate_image_pool(diffusion, pool_size, device):
    """Generate a pool of clean images using the diffusion model."""
    print(f"  Generating {pool_size} clean images...")
    images = []
    batch = 4

    diffusion.unet.to(device)

    with torch.no_grad():
        for i in range(0, pool_size, batch):
            n = min(batch, pool_size - i)
            z = torch.randn(n, *diffusion.image_shape, device=device)
            for step_i, t in enumerate(diffusion.timesteps):
                noise_pred = diffusion.predict_noise(z, t)
                alpha_t = diffusion.get_alpha(t)
                alpha_prev = diffusion.get_prev_alpha(step_i)
                x0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                z = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred
            images.append(z.clamp(-1, 1))

    pool = torch.cat(images, dim=0)[:pool_size]
    diffusion.unet.cpu()
    torch.cuda.empty_cache()
    return pool


def compute_ssim_batch(img1, img2):
    """Compute mean SSIM for a batch."""
    a = ((img1.clamp(-1, 1) + 1) / 2).cpu().numpy()
    b = ((img2.clamp(-1, 1) + 1) / 2).cpu().numpy()
    vals = []
    for i in range(a.shape[0]):
        val = ssim_fn(a[i], b[i], channel_axis=0, data_range=1.0)
        vals.append(val)
    return np.mean(vals)


def compute_ortho_loss(controller):
    """Encourage orthogonality among patterns."""
    patterns = controller.bit_patterns
    K = patterns.shape[1]
    pf = patterns.view(patterns.shape[0], K, -1)
    gram = torch.bmm(pf, pf.transpose(1, 2))
    eye = torch.eye(K, device=gram.device).unsqueeze(0)
    return ((gram - eye) ** 2).mean()


def train_and_evaluate(bits_per_cell, total_steps=15000):
    """Train and evaluate a model at the given capacity."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_bits = 64 * bits_per_cell  # 8x8 grid * bits_per_cell

    print("=" * 70)
    print(f"CAPACITY SCALING: {num_bits} bits ({bits_per_cell} bits/cell)")
    print("=" * 70)

    # ── Initialize ──
    diffusion = DiffusionWrapper(
        model_id='google/ddpm-celebahq-256',
        device=device,
        num_inference_steps=20,
    )

    controller = Controller(
        num_bits=num_bits,
        bits_per_cell=bits_per_cell,
        max_perturbation=0.3,
    ).to(device)
    print(f"  Controller params: {sum(p.numel() for p in controller.parameters()) / 1e3:.1f}K")

    extractor = Extractor(
        num_bits=num_bits,
        bits_per_cell=bits_per_cell,
    ).to(device)
    print(f"  Extractor params: {sum(p.numel() for p in extractor.parameters()) / 1e6:.1f}M")

    criterion = DSCSPLoss(lambda_perc=0.1, lambda_ctrl=0.0, warmup_steps=0).to(device)

    optimizer = torch.optim.Adam([
        {'params': controller.parameters(), 'lr': 5e-4, 'weight_decay': 0},
        {'params': extractor.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4},
    ], betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5,
    )

    # ── Image pool ──
    image_pool = generate_image_pool(diffusion, 128, device)

    # ── Train (clean only, no attacks) ──
    print(f"\n  Training for {total_steps} steps (clean, no attacks)...")
    batch_size = 8
    best_ber = 1.0

    for step in range(1, total_steps + 1):
        controller.train()
        extractor.train()
        optimizer.zero_grad()

        # Refresh pool
        if step > 1 and step % 2000 == 0:
            image_pool = generate_image_pool(diffusion, 128, device)

        idx = torch.randint(0, 128, (batch_size,))
        x_clean = image_pool[idx]
        payload = torch.randint(0, 2, (batch_size, num_bits), device=device).float()

        delta = controller(payload)
        x_stego = (x_clean + delta).clamp(-1, 1)
        delta_norms = delta.pow(2).mean()

        bits_pred = extractor(x_stego)
        loss, loss_dict = criterion(bits_pred, payload, x_stego, x_clean, delta_norms, step=step)

        # Orthogonality regularization
        ortho_loss = compute_ortho_loss(controller)
        loss = loss + 0.001 * ortho_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(controller.parameters()) + list(extractor.parameters()), max_norm=5.0
        )
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            ber = compute_ber(bits_pred.detach(), payload)
            psnr = compute_psnr(x_stego.detach(), x_clean.detach())
            if ber < best_ber:
                best_ber = ber
            print(f"    Step {step:5d}/{total_steps} | BER: {ber:.4f} | PSNR: {psnr:.1f}dB")

        del x_stego, x_clean, delta, bits_pred, loss, payload
        if step % 100 == 0:
            torch.cuda.empty_cache()

    # ── Final evaluation ──
    print(f"\n  Final evaluation (200 samples)...")
    controller.eval()
    extractor.eval()

    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()

    diffusion.unet.to(device)
    all_ber, all_psnr, all_ssim, all_lpips = [], [], [], []

    with torch.no_grad():
        for bi in range(0, 200, 4):
            n = min(4, 200 - bi)
            z_T = torch.randn(n, 3, 256, 256, device=device)
            z = z_T
            for i, t in enumerate(diffusion.timesteps):
                noise_pred = diffusion.predict_noise(z, t)
                alpha_t = diffusion.get_alpha(t)
                alpha_prev = diffusion.get_prev_alpha(i)
                x0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                z = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred
            x_clean = z.clamp(-1, 1)

            payload = torch.randint(0, 2, (n, num_bits), device=device).float()
            delta = controller(payload)
            x_stego = (x_clean + delta).clamp(-1, 1)
            bits_pred = extractor(x_stego)

            all_ber.append(compute_ber(bits_pred, payload))
            all_psnr.append(compute_psnr(x_stego, x_clean))
            all_ssim.append(compute_ssim_batch(x_stego, x_clean))
            all_lpips.append(lpips_fn(x_stego, x_clean).mean().item())

    result = {
        'num_bits': num_bits,
        'bits_per_cell': bits_per_cell,
        'snr': 1.0 / max(bits_per_cell - 1, 1),
        'ber': float(np.mean(all_ber)),
        'psnr': float(np.mean(all_psnr)),
        'ssim': float(np.mean(all_ssim)),
        'lpips': float(np.mean(all_lpips)),
        'training_steps': total_steps,
    }

    print(f"\n  Results for {num_bits} bits ({bits_per_cell} bits/cell):")
    print(f"    BER:   {result['ber']:.4f}")
    print(f"    PSNR:  {result['psnr']:.2f} dB")
    print(f"    SSIM:  {result['ssim']:.4f}")
    print(f"    LPIPS: {result['lpips']:.4f}")

    # Save checkpoint
    ckpt_dir = 'checkpoints_scaling'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'dscsp_{num_bits}bit_final.pt')
    torch.save({
        'controller': controller.state_dict(),
        'extractor': extractor.state_dict(),
        'config': {
            'num_bits': num_bits,
            'bits_per_cell': bits_per_cell,
            'max_perturbation': 0.3,
            'model_id': 'google/ddpm-celebahq-256',
            'ddim_steps': 20,
        },
        'results': result,
    }, ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    # Save results
    result_path = f'scaling_results_{num_bits}bit.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved results: {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits_per_cell', type=int, required=True,
                        choices=[2, 4, 8], help='Bits per cell (2=128bit, 4=256bit, 8=512bit)')
    parser.add_argument('--total_steps', type=int, default=15000)
    args = parser.parse_args()

    result = train_and_evaluate(args.bits_per_cell, args.total_steps)
    print("\nDone!")


if __name__ == '__main__':
    main()
