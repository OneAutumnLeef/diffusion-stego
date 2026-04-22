"""
DSCSP Steganalysis Detector Evaluation

Trains a ResNet-18 binary classifier to distinguish clean vs stego images,
then reports ROC AUC, accuracy, precision, recall.

This simulates an SRNet-style steganalysis evaluation: if a simple
learned detector can reliably distinguish stego from clean, the
steganographic system is detectable.

Usage:
    python steganalysis.py
    python steganalysis.py --num_pairs 500 --epochs 30
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from dscsp.models.diffusion_wrapper import DiffusionWrapper
from dscsp.models.controller import Controller


def generate_pairs(controller, diffusion, num_pairs, config, device):
    """Generate clean/stego image pairs."""
    clean_images = []
    stego_images = []

    print(f"  Generating {num_pairs} clean/stego pairs...")
    batch = 4

    with torch.no_grad():
        for start in range(0, num_pairs, batch):
            n = min(batch, num_pairs - start)

            z_T = torch.randn(n, 3, 256, 256, device=device)
            z = z_T
            for i, t in enumerate(diffusion.timesteps):
                noise_pred = diffusion.predict_noise(z, t)
                alpha_t = diffusion.get_alpha(t)
                alpha_prev = diffusion.get_prev_alpha(i)
                x0_pred = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                z = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred
            x_clean = z.clamp(-1, 1)

            payload = torch.randint(0, 2, (n, config['num_bits']), device=device).float()
            delta = controller(payload)
            x_stego = (x_clean + delta).clamp(-1, 1)

            clean_images.append(x_clean.cpu())
            stego_images.append(x_stego.cpu())

            if (start + n) % 100 == 0 or start + n >= num_pairs:
                print(f"    {start + n}/{num_pairs}")

    return torch.cat(clean_images), torch.cat(stego_images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/dscsp_final.pt')
    parser.add_argument('--num_pairs', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load DSCSP model ──
    print(f"Loading DSCSP model: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']

    controller = Controller(
        num_bits=config['num_bits'],
        bits_per_cell=config['bits_per_cell'],
        max_perturbation=config['max_perturbation'],
    ).to(device)
    controller.load_state_dict(ckpt['controller'])
    controller.eval()

    # ── Load diffusion model ──
    print("Loading diffusion model...")
    diffusion = DiffusionWrapper(
        model_id=config['model_id'],
        device=device,
        num_inference_steps=config.get('ddim_steps', 20),
    )

    # ── Generate data ──
    clean_images, stego_images = generate_pairs(
        controller, diffusion, args.num_pairs, config, device
    )

    # Free diffusion model memory
    diffusion.unet.cpu()
    del diffusion
    torch.cuda.empty_cache()

    # ── Build dataset ──
    # Labels: 0 = clean, 1 = stego
    images = torch.cat([clean_images, stego_images], dim=0)
    labels = torch.cat([
        torch.zeros(len(clean_images)),
        torch.ones(len(stego_images)),
    ])

    # Shuffle and split 80/20
    n_total = len(images)
    perm = torch.randperm(n_total)
    images = images[perm]
    labels = labels[perm]

    n_train = int(0.8 * n_total)
    train_images, test_images = images[:n_train], images[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    print(f"\n  Train: {n_train} samples | Test: {n_total - n_train} samples")

    train_ds = TensorDataset(train_images, train_labels)
    test_ds = TensorDataset(test_images, test_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ── Build detector ──
    print("\nTraining steganalysis detector (ResNet-18)...")
    detector = models.resnet18(weights=None, num_classes=1).to(device)
    optimizer = torch.optim.Adam(detector.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # ── Training ──
    for epoch in range(1, args.epochs + 1):
        detector.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            logits = detector(batch_imgs).squeeze(-1)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_imgs)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == batch_labels).sum().item()
            total += len(batch_imgs)

        train_acc = correct / total
        avg_loss = total_loss / total

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

    # ── Evaluation ──
    print("\n" + "=" * 70)
    print("STEGANALYSIS DETECTION RESULTS")
    print("=" * 70)

    detector.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_imgs, batch_labels in test_loader:
            batch_imgs = batch_imgs.to(device)
            logits = detector(batch_imgs).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch_labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(float)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)

    print(f"  ROC AUC:    {auc:.4f}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print()

    if auc > 0.9:
        print("  WARNING: Detector achieves high AUC — stego images are detectable.")
    elif auc > 0.7:
        print("  MODERATE: Detector has some ability to distinguish stego from clean.")
    else:
        print("  GOOD: Detector struggles to distinguish stego from clean (low detectability).")

    # Save results
    det_results = {
        'auc': float(auc),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'num_train': n_train,
        'num_test': n_total - n_train,
        'epochs': args.epochs,
    }
    with open('steganalysis_results.json', 'w') as f:
        json.dump(det_results, f, indent=2)
    print(f"  Results saved to steganalysis_results.json")


import json

if __name__ == '__main__':
    main()
