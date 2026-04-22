"""
DSCSP Comprehensive Evaluation Script

Evaluates the trained DSCSP model with:
  - Quality metrics: PSNR, SSIM, LPIPS, FID
  - Clean BER
  - Real (non-differentiable) attack robustness:
    * Real JPEG compression (Pillow) at Q={50,60,70,80,90,95}
    * Real Gaussian noise at sigma={0.01,0.02,0.03,0.05}
    * Real Gaussian blur (Pillow) at radius={0.5,1.0,1.5,2.0}
    * Real brightness/contrast (Pillow)
    * Real crop+resize (Pillow) at ratios={0.7,0.8,0.9,0.95}
  - Combined attack robustness (sequential multi-attack pipelines)
  - Optional redundant encoding support for crop-robust models

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/dscsp_final.pt --num_samples 500
    python evaluate.py --checkpoint checkpoints_v2/dscsp_v2_final.pt --num_samples 500
"""

import os
import io
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from skimage.metrics import structural_similarity as ssim_fn
import lpips

from dscsp.models.diffusion_wrapper import DiffusionWrapper
from dscsp.models.controller import Controller
from dscsp.models.extractor import Extractor
from dscsp.utils.metrics import compute_ber, compute_psnr
from dscsp.utils.redundant_coding import RedundantCoder


# ─── Real Attacks (non-differentiable, using Pillow/numpy) ────────────────

def tensor_to_pil(t):
    """Convert (3, H, W) tensor in [-1,1] to PIL Image."""
    img = ((t.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img)


def pil_to_tensor(img, device):
    """Convert PIL Image to (1, 3, H, W) tensor in [-1,1]."""
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def real_jpeg(t, quality, device):
    """Apply real JPEG compression via Pillow."""
    results = []
    for i in range(t.shape[0]):
        pil_img = tensor_to_pil(t[i])
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        results.append(pil_to_tensor(pil_img, device))
    return torch.cat(results, dim=0)


def real_gaussian_noise(t, sigma):
    """Add real Gaussian noise using numpy."""
    noise = torch.randn_like(t) * sigma
    return (t + noise).clamp(-1, 1)


def real_gaussian_blur(t, radius, device):
    """Apply real Gaussian blur via Pillow."""
    results = []
    for i in range(t.shape[0]):
        pil_img = tensor_to_pil(t[i])
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
        results.append(pil_to_tensor(pil_img, device))
    return torch.cat(results, dim=0)


def real_brightness_contrast(t, brightness_factor, contrast_factor, device):
    """Apply real brightness/contrast via Pillow."""
    results = []
    for i in range(t.shape[0]):
        pil_img = tensor_to_pil(t[i])
        pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness_factor)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast_factor)
        results.append(pil_to_tensor(pil_img, device))
    return torch.cat(results, dim=0)


def real_crop_resize(t, ratio, device):
    """Apply real crop+resize via Pillow."""
    results = []
    for i in range(t.shape[0]):
        pil_img = tensor_to_pil(t[i])
        W, H = pil_img.size
        crop_w, crop_h = int(W * ratio), int(H * ratio)
        left = (W - crop_w) // 2
        top = (H - crop_h) // 2
        pil_img = pil_img.crop((left, top, left + crop_w, top + crop_h))
        pil_img = pil_img.resize((W, H), Image.BILINEAR)
        results.append(pil_to_tensor(pil_img, device))
    return torch.cat(results, dim=0)


# ─── Metrics ──────────────────────────────────────────────────────────────

def compute_ssim_batch(img1, img2):
    """Compute mean SSIM for a batch. Inputs: (B,3,H,W) in [-1,1]."""
    # Convert to [0,1] numpy
    a = ((img1.clamp(-1, 1) + 1) / 2).cpu().numpy()
    b = ((img2.clamp(-1, 1) + 1) / 2).cpu().numpy()
    vals = []
    for i in range(a.shape[0]):
        # channel_axis=0 for (C,H,W)
        val = ssim_fn(a[i], b[i], channel_axis=0, data_range=1.0)
        vals.append(val)
    return np.mean(vals)


# ─── Main Evaluation ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/dscsp_final.pt')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_images', action='store_true',
                        help='Save clean/stego images for FID computation')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load model ──
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']

    controller = Controller(
        num_bits=config['num_bits'],
        bits_per_cell=config['bits_per_cell'],
        max_perturbation=config['max_perturbation'],
    ).to(device)
    controller.load_state_dict(ckpt['controller'])
    controller.eval()

    extractor = Extractor(
        num_bits=config['num_bits'],
        bits_per_cell=config['bits_per_cell'],
    ).to(device)
    extractor.load_state_dict(ckpt['extractor'])
    extractor.eval()

    print(f"  Config: {config['num_bits']} bits, {config['bits_per_cell']} bits/cell")

    # ── Check for redundant encoding ──
    redundant_coder = None
    use_redundant = config.get('use_redundant_encoding', False)
    if use_redundant:
        msg_bits = config.get('message_bits', 128)
        redundancy = config.get('redundancy_factor', 2)
        redundant_coder = RedundantCoder(
            message_bits=msg_bits,
            physical_slots=config['num_bits'],
            redundancy=redundancy,
            bits_per_cell=config['bits_per_cell'],
        )
        print(f"  Redundant encoding: {msg_bits} msg bits, {redundancy}x redundancy")

    # ── Load diffusion model for clean image generation ──
    print("Loading diffusion model...")
    diffusion = DiffusionWrapper(
        model_id=config['model_id'],
        device=device,
        num_inference_steps=config.get('ddim_steps', 20),
    )

    # ── LPIPS model ──
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()

    # ── Generate samples and evaluate ──
    print(f"\nEvaluating {args.num_samples} samples...")

    all_psnr, all_ssim, all_lpips_val, all_ber = [], [], [], []

    # FID directories
    if args.save_images:
        os.makedirs('eval_clean', exist_ok=True)
        os.makedirs('eval_stego', exist_ok=True)

    img_idx = 0
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    for batch_i in range(num_batches):
        n = min(args.batch_size, args.num_samples - batch_i * args.batch_size)

        # Generate clean images
        with torch.no_grad():
            z_T = torch.randn(n, 3, 256, 256, device=device)
            z = z_T
            for i, t in enumerate(diffusion.timesteps):
                noise_pred = diffusion.predict_noise(z, t)
                alpha_t = diffusion.get_alpha(t)
                alpha_prev = diffusion.get_prev_alpha(i)
                x0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                z = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred
            x_clean = z.clamp(-1, 1)

        # Generate stego
        if redundant_coder is not None:
            message = torch.randint(0, 2, (n, config.get('message_bits', 128)), device=device).float()
            payload = redundant_coder.encode(message)
        else:
            message = None
            payload = torch.randint(0, 2, (n, config['num_bits']), device=device).float()

        with torch.no_grad():
            delta = controller(payload)
            x_stego = (x_clean + delta).clamp(-1, 1)
            bits_pred = extractor(x_stego)

        # Quality metrics
        if redundant_coder is not None and message is not None:
            hard_bits = redundant_coder.decode_hard(bits_pred)
            ber = (hard_bits != message).float().mean().item()
        else:
            ber = compute_ber(bits_pred, payload)
        psnr = compute_psnr(x_stego, x_clean)
        ssim_val = compute_ssim_batch(x_stego.detach(), x_clean.detach())
        with torch.no_grad():
            lpips_val = lpips_fn(x_stego, x_clean).mean().item()

        all_ber.append(ber)
        all_psnr.append(psnr)
        all_ssim.append(ssim_val)
        all_lpips_val.append(lpips_val)

        # Save images for FID
        if args.save_images:
            for j in range(n):
                clean_pil = tensor_to_pil(x_clean[j])
                stego_pil = tensor_to_pil(x_stego[j])
                clean_pil.save(f'eval_clean/{img_idx:05d}.png')
                stego_pil.save(f'eval_stego/{img_idx:05d}.png')
                img_idx += 1

        if (batch_i + 1) % 10 == 0 or batch_i == num_batches - 1:
            print(f"  Batch {batch_i+1}/{num_batches} | "
                  f"BER: {np.mean(all_ber):.4f} | PSNR: {np.mean(all_psnr):.2f} | "
                  f"SSIM: {np.mean(all_ssim):.4f} | LPIPS: {np.mean(all_lpips_val):.4f}")

    # Move UNet to CPU for remaining evaluation
    diffusion.unet.cpu()
    torch.cuda.empty_cache()

    results = {
        'clean': {
            'ber': float(np.mean(all_ber)),
            'psnr': float(np.mean(all_psnr)),
            'ssim': float(np.mean(all_ssim)),
            'lpips': float(np.mean(all_lpips_val)),
        }
    }

    print("\n" + "=" * 70)
    print("CLEAN IMAGE QUALITY METRICS")
    print("=" * 70)
    print(f"  BER:   {results['clean']['ber']:.4f}")
    print(f"  PSNR:  {results['clean']['psnr']:.2f} dB")
    print(f"  SSIM:  {results['clean']['ssim']:.4f}")
    print(f"  LPIPS: {results['clean']['lpips']:.4f}")

    # ── Real Attack Robustness Evaluation ──
    # Use a subset of stego images from a fresh pool for attack tests
    print("\n" + "=" * 70)
    print("REAL ATTACK ROBUSTNESS EVALUATION")
    print("=" * 70)

    # Pre-generate a pool of clean/stego pairs for attack testing
    print("Generating 200 test pairs for attack evaluation...")
    diffusion.unet.to(device)
    test_clean = []
    test_stego = []
    test_payload = []
    test_message = []  # For redundant encoding
    n_attack_test = 200

    with torch.no_grad():
        for bi in range(0, n_attack_test, 4):
            n = min(4, n_attack_test - bi)
            z_T = torch.randn(n, 3, 256, 256, device=device)
            z = z_T
            for i, t in enumerate(diffusion.timesteps):
                noise_pred = diffusion.predict_noise(z, t)
                alpha_t = diffusion.get_alpha(t)
                alpha_prev = diffusion.get_prev_alpha(i)
                x0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                z = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred
            xc = z.clamp(-1, 1)

            if redundant_coder is not None:
                m = torch.randint(0, 2, (n, config.get('message_bits', 128)), device=device).float()
                p = redundant_coder.encode(m)
                test_message.append(m)
            else:
                p = torch.randint(0, 2, (n, config['num_bits']), device=device).float()

            delta = controller(p)
            xs = (xc + delta).clamp(-1, 1)

            test_clean.append(xc)
            test_stego.append(xs)
            test_payload.append(p)

    diffusion.unet.cpu()
    torch.cuda.empty_cache()

    test_stego = torch.cat(test_stego, dim=0)
    test_payload = torch.cat(test_payload, dim=0)
    if test_message:
        test_message = torch.cat(test_message, dim=0)
    else:
        test_message = None
    print(f"  Generated {test_stego.shape[0]} test pairs")

    attack_results = {}

    # Helper: evaluate BER on attacked images
    def eval_attack(name, attacked_images, payloads, messages=None):
        with torch.no_grad():
            total_ber = 0
            n_batches = (attacked_images.shape[0] + 7) // 8
            for bi in range(n_batches):
                start = bi * 8
                end = min(start + 8, attacked_images.shape[0])
                bp = extractor(attacked_images[start:end])
                if redundant_coder is not None and messages is not None:
                    hard = redundant_coder.decode_hard(bp)
                    batch_ber = (hard != messages[start:end]).float().mean().item()
                else:
                    batch_ber = compute_ber(bp, payloads[start:end])
                total_ber += batch_ber * (end - start)
        return total_ber / attacked_images.shape[0]

    # JPEG compression
    print("\n  [JPEG Compression]")
    for q in [50, 60, 70, 80, 90, 95]:
        attacked = real_jpeg(test_stego, q, device)
        ber = eval_attack(f'jpeg_q{q}', attacked, test_payload, test_message)
        attack_results[f'jpeg_q{q}'] = ber
        print(f"    Q={q:3d}: BER = {ber:.4f}")
        del attacked

    # Gaussian noise
    print("\n  [Gaussian Noise]")
    for sigma in [0.01, 0.02, 0.03, 0.05]:
        attacked = real_gaussian_noise(test_stego, sigma)
        ber = eval_attack(f'noise_s{sigma}', attacked, test_payload, test_message)
        attack_results[f'noise_sigma{sigma}'] = ber
        print(f"    sigma={sigma:.2f}: BER = {ber:.4f}")
        del attacked

    # Gaussian blur
    print("\n  [Gaussian Blur]")
    for radius in [0.5, 1.0, 1.5, 2.0]:
        attacked = real_gaussian_blur(test_stego, radius, device)
        ber = eval_attack(f'blur_r{radius}', attacked, test_payload, test_message)
        attack_results[f'blur_radius{radius}'] = ber
        print(f"    radius={radius:.1f}: BER = {ber:.4f}")
        del attacked

    # Brightness/contrast
    print("\n  [Brightness/Contrast]")
    for bf, cf in [(1.1, 1.0), (0.9, 1.0), (1.0, 1.2), (1.0, 0.8), (1.1, 1.2)]:
        attacked = real_brightness_contrast(test_stego, bf, cf, device)
        label = f'bc_b{bf}_c{cf}'
        ber = eval_attack(label, attacked, test_payload, test_message)
        attack_results[label] = ber
        print(f"    brightness={bf}, contrast={cf}: BER = {ber:.4f}")
        del attacked

    # Crop + resize
    print("\n  [Crop + Resize]")
    for ratio in [0.95, 0.9, 0.8, 0.7]:
        attacked = real_crop_resize(test_stego, ratio, device)
        ber = eval_attack(f'crop_r{ratio}', attacked, test_payload, test_message)
        attack_results[f'crop_ratio{ratio}'] = ber
        print(f"    ratio={ratio:.2f}: BER = {ber:.4f}")
        del attacked

    results['attacks'] = attack_results

    # ── Combined Attack Robustness ──
    print("\n" + "=" * 70)
    print("COMBINED ATTACK ROBUSTNESS EVALUATION")
    print("=" * 70)

    combined_results = {}

    combined_attacks = [
        ("jpeg_q80 + noise_0.02", [
            ("jpeg", {"quality": 80}),
            ("noise", {"sigma": 0.02}),
        ]),
        ("jpeg_q70 + blur_1.0", [
            ("jpeg", {"quality": 70}),
            ("blur", {"radius": 1.0}),
        ]),
        ("crop_0.9 + jpeg_q80", [
            ("crop", {"ratio": 0.9}),
            ("jpeg", {"quality": 80}),
        ]),
        ("noise_0.03 + bc_1.1_1.1", [
            ("noise", {"sigma": 0.03}),
            ("bc", {"bf": 1.1, "cf": 1.1}),
        ]),
        ("jpeg_q80 + noise_0.02 + blur_0.5", [
            ("jpeg", {"quality": 80}),
            ("noise", {"sigma": 0.02}),
            ("blur", {"radius": 0.5}),
        ]),
    ]

    for combo_name, attack_list in combined_attacks:
        attacked = test_stego.clone()
        for atk_type, atk_params in attack_list:
            if atk_type == "jpeg":
                attacked = real_jpeg(attacked, atk_params["quality"], device)
            elif atk_type == "noise":
                attacked = real_gaussian_noise(attacked, atk_params["sigma"])
            elif atk_type == "blur":
                attacked = real_gaussian_blur(attacked, atk_params["radius"], device)
            elif atk_type == "crop":
                attacked = real_crop_resize(attacked, atk_params["ratio"], device)
            elif atk_type == "bc":
                attacked = real_brightness_contrast(
                    attacked, atk_params["bf"], atk_params["cf"], device
                )
        ber = eval_attack(combo_name, attacked, test_payload, test_message)
        combined_results[combo_name] = ber
        print(f"  {combo_name}: BER = {ber:.4f}")
        del attacked

    results['combined_attacks'] = combined_results

    # ── Save results ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Clean: BER={results['clean']['ber']:.4f}, "
          f"PSNR={results['clean']['psnr']:.2f}dB, "
          f"SSIM={results['clean']['ssim']:.4f}, "
          f"LPIPS={results['clean']['lpips']:.4f}")

    # Save to JSON
    with open('eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to eval_results.json")


if __name__ == '__main__':
    main()
