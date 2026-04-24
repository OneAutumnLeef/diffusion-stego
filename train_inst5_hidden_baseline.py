"""Instruction 5 driver — HiDDeN matched-conditions baseline.

Trains the in-repo HiDDeN reimplementation (``hidden_baseline.py``) on the
exact same DDIM CelebA-HQ 256x256 cover distribution used by DSCSP, with the
same attack schedule. The resulting checkpoint can then be evaluated with
``evaluate_gates.py`` for a like-for-like comparison.

Falls back to 128 message bits -> 256 physical bits via RedundantCoder if
``--redundant`` is passed (recommended — 256-bit raw HiDDeN is often near
its capacity cliff).

Usage:
    python train_inst5_hidden_baseline.py
    python train_inst5_hidden_baseline.py --redundant
    python train_inst5_hidden_baseline.py --total-steps 12000 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from dscsp.augmentations.differentiable import AttackLayer, crop_resize
from dscsp.models.diffusion_wrapper import DiffusionWrapper
from dscsp.utils.metrics import compute_ber, compute_psnr
from dscsp.utils.redundant_coding import RedundantCoder
from hidden_baseline import HiDDeNAdversary, HiDDeNDecoder, HiDDeNEncoder


def generate_image_pool(diffusion, pool_size, device, batch=4):
    diffusion.unet.to(device)
    imgs = []
    with torch.no_grad():
        for i in range(0, pool_size, batch):
            n = min(batch, pool_size - i)
            z = torch.randn(n, *diffusion.image_shape, device=device)
            for step_i, t in enumerate(diffusion.timesteps):
                noise_pred = diffusion.predict_noise(z, t)
                at = diffusion.get_alpha(t)
                ap = diffusion.get_prev_alpha(step_i)
                x0 = (z - (1 - at).sqrt() * noise_pred) / at.sqrt()
                z = ap.sqrt() * x0 + (1 - ap).sqrt() * noise_pred
            imgs.append(z.clamp(-1, 1))
    pool = torch.cat(imgs, 0)[:pool_size]
    diffusion.unet.cpu()
    torch.cuda.empty_cache()
    return pool


def get_attack_prob(step, start, ramp, max_prob):
    if step < start:
        return 0.0
    if step < start + ramp:
        return max_prob * (step - start) / ramp
    return max_prob


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-bits", type=int, default=256)
    parser.add_argument("--redundant", action="store_true",
                        help="Use 128 msg bits -> 256 physical via RedundantCoder")
    parser.add_argument("--message-bits", type=int, default=128,
                        help="Only applies with --redundant")
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-perturbation", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-msg", type=float, default=1.0)
    parser.add_argument("--lambda-img", type=float, default=0.7)
    parser.add_argument("--lambda-adv", type=float, default=0.001)
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--pool-refresh", type=int, default=2000)
    parser.add_argument("--ckpt-dir", default="checkpoints_tifs/hidden_baseline")
    parser.add_argument("--out-dir", default="outputs_tifs/hidden_baseline")
    parser.add_argument("--ckpt-interval", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--real-jpeg-prob", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    interrupted = [False]

    def _handle(signum, frame):
        interrupted[0] = True
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    # Redundant coding path
    coder = None
    phys_bits = args.num_bits
    msg_bits = args.num_bits
    if args.redundant:
        msg_bits = args.message_bits
        assert phys_bits == msg_bits * 2, "--redundant expects phys == 2 * message-bits"
        coder = RedundantCoder(message_bits=msg_bits, physical_slots=phys_bits, redundancy=2)

    # Diffusion pool
    diffusion = DiffusionWrapper(
        model_id="google/ddpm-celebahq-256", device=device, num_inference_steps=20,
    )
    pool = generate_image_pool(diffusion, args.pool_size, device)

    # Models
    encoder = HiDDeNEncoder(num_bits=phys_bits, max_perturbation=args.max_perturbation).to(device)
    decoder = HiDDeNDecoder(num_bits=phys_bits).to(device)
    adversary = HiDDeNAdversary().to(device)

    opt_ed = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr, betas=(0.9, 0.999),
    )
    opt_adv = torch.optim.Adam(adversary.parameters(), lr=args.lr, betas=(0.9, 0.999))

    attack_cfg = {
        "jpeg_quality_range": (50, 85),
        "real_jpeg_quality_range": (50, 85),
        "real_jpeg_prob": args.real_jpeg_prob,
        "crop_ratio_range": (0.65, 0.95),
    }
    attack_layer = AttackLayer(attack_cfg).to(device)

    print(f"HiDDeN baseline — phys={phys_bits}, msg={msg_bits}, "
          f"{'redundant 2x' if coder else 'no coding'}, steps={args.total_steps}")

    step_times = []
    best_ber = 1.0
    for step in range(1, args.total_steps + 1):
        t0 = time.time()

        if step > 1 and step % args.pool_refresh == 0:
            pool = generate_image_pool(diffusion, args.pool_size, device)

        idx = torch.randint(0, args.pool_size, (args.batch_size,))
        x_clean = pool[idx].to(device)

        if coder is not None:
            msg = torch.randint(0, 2, (args.batch_size, msg_bits), device=device).float()
            payload = coder.encode(msg)
        else:
            payload = torch.randint(0, 2, (args.batch_size, phys_bits), device=device).float()
            msg = payload

        encoder.train(); decoder.train(); adversary.train()

        # ── Encoder / Decoder update ──
        opt_ed.zero_grad()
        delta = encoder(x_clean, payload)
        x_stego = (x_clean + delta).clamp(-1, 1)

        # Attack (crop pre-pass + main mix)
        x_pre = x_stego
        if random.random() < 0.5:
            x_pre = crop_resize(x_pre, attack_cfg["crop_ratio_range"])
        atk_p = get_attack_prob(step, start=500, ramp=2000, max_prob=0.7)
        x_att = attack_layer(x_pre, attack_prob=atk_p, combined=False)

        logits = decoder(x_att)
        loss_msg = F.binary_cross_entropy_with_logits(logits, payload)
        loss_img = F.mse_loss(x_stego, x_clean)

        # Adversarial pressure (cover=real, stego=fake)
        adv_logits = adversary(x_stego)
        loss_adv = F.binary_cross_entropy_with_logits(adv_logits, torch.zeros_like(adv_logits))

        loss = args.lambda_msg * loss_msg + args.lambda_img * loss_img + args.lambda_adv * loss_adv
        loss.backward()
        opt_ed.step()

        # ── Adversary update ──
        opt_adv.zero_grad()
        lc = adversary(x_clean.detach())
        ls = adversary(x_stego.detach())
        loss_a = (
            F.binary_cross_entropy_with_logits(lc, torch.zeros_like(lc)) +
            F.binary_cross_entropy_with_logits(ls, torch.ones_like(ls))
        ) / 2
        loss_a.backward()
        opt_adv.step()

        step_times.append(time.time() - t0)

        if step % args.log_interval == 0:
            with torch.no_grad():
                if coder is not None:
                    msg_hard = coder.decode_hard(logits.detach())
                    ber = compute_ber(msg_hard, msg)
                else:
                    ber = compute_ber(logits.detach(), payload)
            psnr = compute_psnr(x_stego.detach(), x_clean.detach())
            if ber < best_ber:
                best_ber = ber
            avg = sum(step_times[-10:]) / max(1, len(step_times[-10:]))
            eta = avg * (args.total_steps - step) / 60
            print(f"HIDDEN {step:5d}/{args.total_steps} | BER {ber:.4f} | "
                  f"PSNR {psnr:.1f} | msg {loss_msg.item():.3f} | img {loss_img.item():.4f} | "
                  f"adv {loss_adv.item():.3f} | atk {atk_p:.2f} | {avg:.2f}s/step | ETA {eta:.0f}min",
                  flush=True)
            with open(os.path.join(args.out_dir, "progress.json"), "w") as f:
                json.dump({
                    "step": step, "total_steps": args.total_steps,
                    "ber": ber, "psnr": psnr, "best_ber": best_ber,
                }, f, indent=2)

        if step % args.ckpt_interval == 0:
            save = {
                "step": step, "config": {
                    "num_bits": phys_bits, "max_perturbation": args.max_perturbation,
                    "redundant": bool(coder is not None),
                },
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "adversary": adversary.state_dict(),
                "message_bits": msg_bits,
                "redundancy_factor": 2 if coder is not None else 1,
                "best_ber": best_ber,
            }
            torch.save(save, os.path.join(args.ckpt_dir, f"hidden_step_{step:06d}.pt"))

        if interrupted[0]:
            print(f"[hidden] interrupt at step {step}")
            break

    # Final checkpoint
    final_path = os.path.join(args.ckpt_dir, "hidden_final.pt")
    torch.save({
        "step": step,
        "config": {
            "num_bits": phys_bits, "max_perturbation": args.max_perturbation,
            "redundant": bool(coder is not None),
        },
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "adversary": adversary.state_dict(),
        "message_bits": msg_bits,
        "redundancy_factor": 2 if coder is not None else 1,
        "best_ber": best_ber,
    }, final_path)
    print(f"  Saved final HiDDeN: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
