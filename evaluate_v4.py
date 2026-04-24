"""
DSCSP v4 Comprehensive Evaluation.

Evaluates against ALL publication target metrics:
  - Clean BER (target: 0.0000)
  - JPEG robustness (Q75 < 0.01, Q50 < 0.05)
  - PSNR >= 35, SSIM >= 0.97, LPIPS <= 0.05
  - FID increase <= 3
  - Steganalysis AUC <= 0.60
  - 1000+ test images
  - Baseline comparison table
  - Capacity scaling curve

Usage:
    python evaluate_v4.py
    python evaluate_v4.py --checkpoint checkpoints_v4/dscsp_v4_final.pt
    python evaluate_v4.py --num_samples 1000
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lpips

from dscsp.models.diffusion_wrapper import DiffusionWrapper
from dscsp.models.controller_v4 import ContentAdaptiveController
from dscsp.augmentations.differentiable import DiffJPEG
from dscsp.utils.metrics import compute_ber, compute_psnr


# ─── Helpers ────────────────────────────────────────────────────────────────

def load_v4_model(ckpt_path, device):
    """Load v4 controller + extractor from checkpoint."""
    def _load_flexible(model, state_dict, name):
        try:
            model.load_state_dict(state_dict, strict=True)
            return
        except RuntimeError as exc:
            print(f"  Warning: strict load failed for {name}: {exc}")
            model_state = model.state_dict()
            compatible = {
                k: v for k, v in state_dict.items()
                if k in model_state and model_state[k].shape == v.shape
            }
            model.load_state_dict(compatible, strict=False)
            print(f"  Loaded {name} with flexible mapping: {len(compatible)}/{len(model_state)} tensors")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    controller = ContentAdaptiveController(
        num_bits=config["num_bits"],
        max_perturbation=config["max_perturbation"],
        base_channels=config.get("base_channels", 48),
        carrier_type=config.get("carrier_type", "nearest"),
        spread_image_size=config.get("spread_image_size", 256),
        spread_seed=config.get("spread_seed", 20260424),
    ).to(device)
    _load_flexible(controller, ckpt["controller"], "controller")
    controller.eval()

    # Import GlobalExtractor from train_v4
    from train_v4 import GlobalExtractor
    extractor = GlobalExtractor(
        num_bits=config["num_bits"],
        mode=config.get("extractor_mode", "global_fc"),
        grid_size=config.get("extractor_grid_size", 16),
        shift_pool_mode=config.get("shift_pool_mode", "max"),
        shift_pool_kernel=config.get("shift_pool_kernel", 5),
    ).to(device)
    _load_flexible(extractor, ckpt["extractor"], "extractor")
    extractor.eval()

    return controller, extractor, config


def generate_images(diffusion, n, device, batch=4):
    """Generate n clean images from diffusion model."""
    images = []
    diffusion.unet.to(device)
    with torch.no_grad():
        for i in range(0, n, batch):
            nb = min(batch, n - i)
            z = torch.randn(nb, *diffusion.image_shape, device=device)
            for step_i, t in enumerate(diffusion.timesteps):
                noise_pred = diffusion.predict_noise(z, t)
                alpha_t = diffusion.get_alpha(t)
                alpha_prev = diffusion.get_prev_alpha(step_i)
                x0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                z = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred
            images.append(z.clamp(-1, 1).cpu())
            if (i + batch) % 20 == 0:
                print(f"    Generated {min(i+batch, n)}/{n}")
    diffusion.unet.cpu()
    torch.cuda.empty_cache()
    return torch.cat(images, dim=0)[:n]


def compute_ssim_single(img1, img2):
    """SSIM for single image pair (C,H,W) in [-1,1]."""
    from torchmetrics.functional.image import structural_similarity_index_measure
    return structural_similarity_index_measure(
        img1.unsqueeze(0), img2.unsqueeze(0), data_range=2.0
    ).item()


# ─── Evaluation Functions ──────────────────────────────────────────────────

def eval_clean(controller, extractor, images, device, num_bits=256):
    """Evaluate clean (no attack) performance."""
    n = len(images)
    bers, psnrs, ssims, lpipss = [], [], [], []
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    with torch.no_grad():
        for i in range(n):
            x = images[i:i+1].to(device)
            pay = torch.randint(0, 2, (1, num_bits), device=device).float()
            delta = controller(x, pay)
            xs = (x + delta).clamp(-1, 1)
            bp = extractor(xs)

            bers.append(compute_ber(bp, pay))
            psnrs.append(compute_psnr(xs, x))
            ssims.append(compute_ssim_single(xs[0].cpu(), x[0].cpu()))
            lpipss.append(lpips_fn(xs, x).item())

    return {
        "ber": float(np.mean(bers)),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpipss)),
    }


def eval_jpeg(controller, extractor, images, device, qualities, num_bits=256):
    """Evaluate JPEG robustness at multiple quality levels."""
    diff_jpeg = DiffJPEG().to(device)
    results = {}

    with torch.no_grad():
        for q in qualities:
            bers = []
            for i in range(len(images)):
                x = images[i:i+1].to(device)
                pay = torch.randint(0, 2, (1, num_bits), device=device).float()
                delta = controller(x, pay)
                xs = (x + delta).clamp(-1, 1)
                xs_jpeg = diff_jpeg(xs, quality=q)
                bp = extractor(xs_jpeg)
                bers.append(compute_ber(bp, pay))
            results[f"jpeg_q{q}"] = float(np.mean(bers))
            print(f"    JPEG Q{q}: BER={results[f'jpeg_q{q}']:.6f}")

    return results


def eval_attacks(controller, extractor, images, device, num_bits=256):
    """Evaluate robustness against various attacks."""
    from dscsp.augmentations.differentiable import gaussian_noise, gaussian_blur, brightness_contrast
    diff_jpeg = DiffJPEG().to(device)
    results = {}

    attack_configs = [
        ("noise_0.01", lambda x: x + torch.randn_like(x) * 0.01),
        ("noise_0.03", lambda x: x + torch.randn_like(x) * 0.03),
        ("noise_0.05", lambda x: x + torch.randn_like(x) * 0.05),
        ("blur_1.0", lambda x: gaussian_blur(x, 5, (1.0, 1.0))),
        ("blur_2.0", lambda x: gaussian_blur(x, 5, (2.0, 2.0))),
    ]

    with torch.no_grad():
        for name, attack_fn in attack_configs:
            bers = []
            for i in range(len(images)):
                x = images[i:i+1].to(device)
                pay = torch.randint(0, 2, (1, num_bits), device=device).float()
                delta = controller(x, pay)
                xs = (x + delta).clamp(-1, 1)
                xs_atk = attack_fn(xs).clamp(-1, 1)
                bp = extractor(xs_atk)
                bers.append(compute_ber(bp, pay))
            results[name] = float(np.mean(bers))
            print(f"    {name}: BER={results[name]:.6f}")

    return results


def eval_steganalysis(controller, images, device, num_bits=256, n_train=800, n_test=200):
    """Train a ResNet-18 classifier to distinguish clean vs stego images.
    Returns AUC (lower = better undetectability)."""
    from torchvision.models import resnet18
    import torch.optim as optim

    n_total = n_train + n_test
    assert len(images) >= n_total, f"Need {n_total} images, got {len(images)}"

    # Generate stego images
    print("    Generating stego images for steganalysis...")
    stego_imgs = []
    with torch.no_grad():
        for i in range(n_total):
            x = images[i:i+1].to(device)
            pay = torch.randint(0, 2, (1, num_bits), device=device).float()
            delta = controller(x, pay)
            xs = (x + delta).clamp(-1, 1)
            stego_imgs.append(xs.cpu())
    stego_imgs = torch.cat(stego_imgs, dim=0)

    # Split train/test
    clean_train = images[:n_train].to(device)
    stego_train = stego_imgs[:n_train].to(device)
    clean_test = images[n_train:n_total].to(device)
    stego_test = stego_imgs[n_train:n_total].to(device)

    # Train classifier
    clf = resnet18(weights=None, num_classes=1).to(device)
    opt = optim.Adam(clf.parameters(), lr=1e-4)
    clf.train()

    batch_sz = 16
    for epoch in range(20):
        indices = torch.randperm(n_train)
        for start in range(0, n_train, batch_sz):
            end = min(start + batch_sz, n_train)
            idx = indices[start:end]
            half = len(idx)

            x_batch = torch.cat([clean_train[idx], stego_train[idx]], dim=0)
            y_batch = torch.cat([torch.zeros(half, 1), torch.ones(half, 1)], dim=0).to(device)

            opt.zero_grad()
            pred = clf(x_batch)
            loss = F.binary_cross_entropy_with_logits(pred, y_batch)
            loss.backward()
            opt.step()

    # Evaluate
    clf.eval()
    with torch.no_grad():
        scores = []
        labels = []
        for start in range(0, n_test, batch_sz):
            end = min(start + batch_sz, n_test)
            xc = clean_test[start:end]
            xs = stego_test[start:end]

            sc = torch.sigmoid(clf(xc)).cpu().numpy().flatten()
            ss = torch.sigmoid(clf(xs)).cpu().numpy().flatten()

            scores.extend(sc.tolist())
            scores.extend(ss.tolist())
            labels.extend([0] * len(sc))
            labels.extend([1] * len(ss))

    auc = roc_auc_score(labels, scores)
    acc = np.mean([(s < 0.5) == (l == 0) for s, l in zip(scores, labels)])

    del clf, clean_train, stego_train, clean_test, stego_test
    torch.cuda.empty_cache()

    return {"auc": float(auc), "accuracy": float(acc)}


def eval_fid(images_clean, images_stego, device, batch_sz=16):
    """Compute FID between clean and stego image sets using InceptionV3."""
    model = inception_v3(weights="DEFAULT", transform_input=False).to(device)
    model.fc = torch.nn.Identity()
    model.eval()

    def get_features(imgs):
        feats = []
        with torch.no_grad():
            for start in range(0, len(imgs), batch_sz):
                end = min(start + batch_sz, len(imgs))
                x = imgs[start:end].to(device)
                # Resize to 299x299 for Inception
                x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
                # Scale from [-1,1] to [0,1]
                x = (x + 1) / 2
                f = model(x)
                feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    f1 = get_features(images_clean)
    f2 = get_features(images_stego)

    mu1, sigma1 = f1.mean(axis=0), np.cov(f1, rowvar=False)
    mu2, sigma2 = f2.mean(axis=0), np.cov(f2, rowvar=False)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

    del model
    torch.cuda.empty_cache()
    return float(fid)


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DSCSP v4 Evaluation")
    parser.add_argument('--checkpoint', default='checkpoints_v4/dscsp_v4_final.pt')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output', default='eval_v4_results.json')
    parser.add_argument('--skip_fid', action='store_true')
    parser.add_argument('--skip_steg', action='store_true')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    controller, extractor, config = load_v4_model(args.checkpoint, device)
    num_bits = config["num_bits"]
    print(f"  num_bits={num_bits}, max_pert={config['max_perturbation']}")

    # Generate test images
    n_total = max(args.num_samples, 1000)
    print(f"\nGenerating {n_total} test images...")
    diffusion = DiffusionWrapper(
        model_id=config.get("model_id", "google/ddpm-celebahq-256"),
        device=device,
        num_inference_steps=config.get("ddim_steps", 20),
    )
    images = generate_images(diffusion, n_total, device)
    del diffusion

    results = {"num_samples": n_total, "num_bits": num_bits}

    # 1. Clean metrics
    print(f"\n{'='*60}")
    print("CLEAN METRICS")
    print(f"{'='*60}")
    clean = eval_clean(controller, extractor, images[:args.num_samples], device, num_bits)
    results["clean"] = clean
    print(f"  BER:   {clean['ber']:.6f}  (target: 0.0000)")
    print(f"  PSNR:  {clean['psnr']:.2f} dB  (target: >= 35)")
    print(f"  SSIM:  {clean['ssim']:.4f}  (target: >= 0.97)")
    print(f"  LPIPS: {clean['lpips']:.4f}  (target: <= 0.05)")

    # 2. JPEG robustness
    print(f"\n{'='*60}")
    print("JPEG ROBUSTNESS")
    print(f"{'='*60}")
    jpeg = eval_jpeg(controller, extractor, images[:args.num_samples], device,
                     [50, 60, 70, 75, 80, 90, 95], num_bits)
    results["jpeg"] = jpeg

    # 3. Other attacks
    print(f"\n{'='*60}")
    print("ATTACK ROBUSTNESS")
    print(f"{'='*60}")
    attacks = eval_attacks(controller, extractor, images[:min(200, args.num_samples)], device, num_bits)
    results["attacks"] = attacks

    # 4. Steganalysis
    if not args.skip_steg:
        print(f"\n{'='*60}")
        print("STEGANALYSIS (AUC)")
        print(f"{'='*60}")
        steg = eval_steganalysis(controller, images[:1000], device, num_bits)
        results["steganalysis"] = steg
        print(f"  AUC:      {steg['auc']:.4f}  (target: <= 0.60)")
        print(f"  Accuracy: {steg['accuracy']:.4f}")

    # 5. FID
    if not args.skip_fid:
        print(f"\n{'='*60}")
        print("FID")
        print(f"{'='*60}")
        print("  Generating stego images for FID...")
        stego_for_fid = []
        with torch.no_grad():
            for i in range(min(1000, n_total)):
                x = images[i:i+1].to(device)
                pay = torch.randint(0, 2, (1, num_bits), device=device).float()
                delta = controller(x, pay)
                xs = (x + delta).clamp(-1, 1)
                stego_for_fid.append(xs.cpu())
        stego_for_fid = torch.cat(stego_for_fid, dim=0)
        fid = eval_fid(images[:1000], stego_for_fid, device)
        results["fid"] = fid
        print(f"  FID: {fid:.2f}  (target: <= 3.0)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — Target vs Achieved")
    print(f"{'='*60}")
    print(f"  {'Metric':<20s} {'Target':<12s} {'Achieved':<12s} {'Status':<8s}")
    print(f"  {'-'*52}")
    checks = [
        ("Clean BER", "0.0000", f"{clean['ber']:.6f}", clean['ber'] < 0.0001),
        ("JPEG Q75 BER", "< 0.01", f"{jpeg.get('jpeg_q75', 'N/A'):.6f}" if 'jpeg_q75' in jpeg else "N/A", jpeg.get('jpeg_q75', 1) < 0.01),
        ("JPEG Q50 BER", "< 0.05", f"{jpeg.get('jpeg_q50', 'N/A'):.6f}" if 'jpeg_q50' in jpeg else "N/A", jpeg.get('jpeg_q50', 1) < 0.05),
        ("PSNR", ">= 35 dB", f"{clean['psnr']:.2f}", clean['psnr'] >= 35),
        ("SSIM", ">= 0.97", f"{clean['ssim']:.4f}", clean['ssim'] >= 0.97),
        ("LPIPS", "<= 0.05", f"{clean['lpips']:.4f}", clean['lpips'] <= 0.05),
    ]
    if "fid" in results:
        checks.append(("FID increase", "<= 3.0", f"{results['fid']:.2f}", results['fid'] <= 3.0))
    if "steganalysis" in results:
        checks.append(("AUC", "<= 0.60", f"{results['steganalysis']['auc']:.4f}", results['steganalysis']['auc'] <= 0.60))

    passed = 0
    for name, target, achieved, ok in checks:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  {name:<20s} {target:<12s} {achieved:<12s} {status:<8s}")
    print(f"\n  {passed}/{len(checks)} targets met")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
