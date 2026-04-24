"""Locked 4-gate evaluation for DSCSP (Instruction 1).

Every experiment in the hardening sprint is evaluated against these four
numbers and only these four numbers:

  Clean BER                  <= 0.01
  JPEG Q75 BER (real libjpeg)<= 0.10
  Crop Q80 BER               <= 0.20
  Adaptive AUC (RN-18/34 ens)<= 0.70

JPEG metric uses real Pillow libjpeg round-trip — the DiffJPEG proxy is
explicitly NOT used at evaluation time. Outputs a JSON with the four
numbers plus bootstrap 95% CIs.

Usage:
    python evaluate_gates.py --checkpoint PATH --output outputs_tifs_eval/gates_X.json
    python evaluate_gates.py --checkpoint PATH --num-samples 200   # quick smoke test
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import models as tv_models

import evaluate_v4 as ev
from dscsp.models.diffusion_wrapper import DiffusionWrapper
from dscsp.utils.metrics import compute_ber


def load_hidden_model(ckpt_path: str, device: str):
    """Load a HiDDeN baseline checkpoint; returns (encoder, decoder, config)."""
    from hidden_baseline import HiDDeNEncoder, HiDDeNDecoder
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    encoder = HiDDeNEncoder(
        num_bits=cfg["num_bits"], max_perturbation=cfg["max_perturbation"],
    ).to(device)
    decoder = HiDDeNDecoder(num_bits=cfg["num_bits"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval(); decoder.eval()
    return encoder, decoder, cfg


GATES = {
    "clean_ber_max": 0.01,
    "jpeg_q75_real_ber_max": 0.10,
    "crop_q80_ber_max": 0.20,
    "adaptive_auc_ens_max": 0.70,
}


# ─── Utilities ──────────────────────────────────────────────────────────────

def _bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05,
                  seed: int = 0) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = arr.size
    means = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def _auc_bootstrap_ci(labels, scores, n_boot: int = 1000, alpha: float = 0.05,
                      seed: int = 0) -> Tuple[float, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    n = labels.size
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        l_b = labels[idx]
        if len(np.unique(l_b)) < 2:
            continue
        aucs.append(roc_auc_score(l_b, scores[idx]))
    if not aucs:
        return float("nan"), float("nan")
    lo = float(np.percentile(aucs, 100 * (alpha / 2)))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi


def _real_jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Real Pillow libjpeg round-trip. Input/output in [-1, 1]."""
    imgs = ((x.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    out = np.empty_like(imgs)
    for i, arr in enumerate(imgs):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        out[i] = np.array(Image.open(buf).convert("RGB"))
    y = torch.from_numpy(out).to(x.device).permute(0, 3, 1, 2).float()
    return y / 127.5 - 1.0


def _random_crop_resize(x: torch.Tensor, ratio: float, seed: int = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    b, c, h, w = x.shape
    ch, cw = max(1, int(h * ratio)), max(1, int(w * ratio))
    top = random.randint(0, h - ch)
    left = random.randint(0, w - cw)
    cropped = x[:, :, top:top + ch, left:left + cw]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


# ─── Payload helpers (handle v5 redundant coding) ──────────────────────────

def _prep_payload_model(ckpt_path: str, num_bits: int, device: str):
    meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    msg_bits = int(meta.get("message_bits", num_bits))
    R = int(meta.get("redundancy_factor", 1))
    del meta
    if R > 1 and msg_bits < num_bits:
        from dscsp.utils.redundant_coding import RedundantCoder
        coder = RedundantCoder(message_bits=msg_bits, physical_slots=num_bits, redundancy=R)
        return coder, msg_bits
    return None, num_bits


def _sample_payload(coder, msg_bits, num_bits, device):
    """Returns (physical_payload, msg). msg == physical when coder is None."""
    if coder is not None:
        msg = torch.randint(0, 2, (1, msg_bits), device=device).float()
        return coder.encode(msg), msg
    payload = torch.randint(0, 2, (1, num_bits), device=device).float()
    return payload, payload


def _decode_ber(logits, msg, coder):
    if coder is not None:
        hard = coder.decode_hard(logits)
        return float(compute_ber(hard, msg))
    return float(compute_ber(logits, msg))


# ─── Per-sample stego generation (kept on CPU pool, moved to GPU per batch) ─

@torch.no_grad()
def _build_stego_pool(controller, images, coder, msg_bits, num_bits, device):
    """Return (stego_tensor, list_of_msg). Runs the controller once per image."""
    stego = []
    msgs = []
    for i in range(len(images)):
        x = images[i:i + 1].to(device)
        payload, msg = _sample_payload(coder, msg_bits, num_bits, device)
        delta = controller(x, payload)
        xs = (x + delta).clamp(-1, 1)
        stego.append(xs.cpu())
        msgs.append(msg.cpu())
    return torch.cat(stego, 0), msgs


# ─── Core gates ────────────────────────────────────────────────────────────

@torch.no_grad()
def gate_clean(extractor, stego_pool, msgs, coder, device) -> Dict:
    bers = []
    for i in range(stego_pool.shape[0]):
        xs = stego_pool[i:i + 1].to(device)
        logits = extractor(xs)
        bers.append(_decode_ber(logits, msgs[i].to(device), coder))
    lo, hi = _bootstrap_ci(bers)
    return {"mean": float(np.mean(bers)), "ci95": [lo, hi], "n": len(bers)}


@torch.no_grad()
def gate_jpeg_q75(extractor, stego_pool, msgs, coder, device, quality: int = 75) -> Dict:
    bers = []
    for i in range(stego_pool.shape[0]):
        xs = stego_pool[i:i + 1].to(device)
        xa = _real_jpeg(xs, quality)
        logits = extractor(xa)
        bers.append(_decode_ber(logits, msgs[i].to(device), coder))
    lo, hi = _bootstrap_ci(bers)
    return {"mean": float(np.mean(bers)), "ci95": [lo, hi], "n": len(bers), "quality": quality}


@torch.no_grad()
def gate_crop_q80(extractor, stego_pool, msgs, coder, device, ratio: float = 0.80,
                  seed: int = 12345) -> Dict:
    random.seed(seed)
    bers = []
    for i in range(stego_pool.shape[0]):
        xs = stego_pool[i:i + 1].to(device)
        xa = _random_crop_resize(xs, ratio)
        logits = extractor(xa)
        bers.append(_decode_ber(logits, msgs[i].to(device), coder))
    lo, hi = _bootstrap_ci(bers)
    return {"mean": float(np.mean(bers)), "ci95": [lo, hi], "n": len(bers), "ratio": ratio}


# ─── Adaptive AUC (ResNet-18 + ResNet-34 ensemble) ─────────────────────────

def _adaptive_attack_eval(x: torch.Tensor) -> torch.Tensor:
    """Adaptive attack used during both detector training and evaluation."""
    r = random.random()
    if r < 0.4:
        q = random.randint(40, 90)
        return _real_jpeg(x, q)
    if r < 0.7:
        ratio = random.uniform(0.70, 0.95)
        return _random_crop_resize(x, ratio)
    sigma = random.uniform(0.01, 0.04)
    return (x + torch.randn_like(x) * sigma).clamp(-1, 1)


def gate_adaptive_auc(stego_pool, clean_pool, device,
                      n_train: int = 800, n_test: int = 200,
                      epochs: int = 12, batch: int = 16, seed: int = 0) -> Dict:
    assert stego_pool.shape[0] >= n_train + n_test
    assert clean_pool.shape[0] >= n_train + n_test

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    clean = clean_pool[:n_train + n_test].to(device)
    stego = stego_pool[:n_train + n_test].to(device)
    clean_train, clean_test = clean[:n_train], clean[n_train:]
    stego_train, stego_test = stego[:n_train], stego[n_train:]

    def _train_clf(arch):
        if arch == "resnet18":
            clf = tv_models.resnet18(weights=None)
        else:
            clf = tv_models.resnet34(weights=None)
        clf.fc = torch.nn.Linear(clf.fc.in_features, 1)
        clf = clf.to(device)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-4)
        clf.train()
        for _ in range(epochs):
            idx = torch.randperm(n_train)
            for st in range(0, n_train, batch):
                ed = min(st + batch, n_train)
                ids = idx[st:ed]
                if len(ids) == 0:
                    continue
                xc = _adaptive_attack_eval(clean_train[ids])
                xs = _adaptive_attack_eval(stego_train[ids])
                x_in = torch.cat([xc, xs], 0)
                y = torch.cat([
                    torch.zeros(len(ids), 1, device=device),
                    torch.ones(len(ids), 1, device=device),
                ], 0)
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(clf(x_in), y)
                loss.backward()
                opt.step()
        clf.eval()
        return clf

    clf18 = _train_clf("resnet18")
    clf34 = _train_clf("resnet34")

    scores18, scores34, labels = [], [], []
    with torch.no_grad():
        for st in range(0, n_test, batch):
            ed = min(st + batch, n_test)
            xc = clean_test[st:ed]
            xs = stego_test[st:ed]
            for x_batch, lbl in [(xc, 0), (xs, 1)]:
                xa = _adaptive_attack_eval(x_batch)
                scores18.extend(torch.sigmoid(clf18(xa)).cpu().numpy().flatten().tolist())
                scores34.extend(torch.sigmoid(clf34(xa)).cpu().numpy().flatten().tolist())
                labels.extend([lbl] * x_batch.shape[0])

    scores_ens = [(a + b) / 2 for a, b in zip(scores18, scores34)]
    auc18 = float(roc_auc_score(labels, scores18))
    auc34 = float(roc_auc_score(labels, scores34))
    auc_ens = float(roc_auc_score(labels, scores_ens))
    lo, hi = _auc_bootstrap_ci(labels, scores_ens, seed=seed)

    del clf18, clf34
    torch.cuda.empty_cache()
    return {
        "auc_resnet18": auc18,
        "auc_resnet34": auc34,
        "auc_ensemble": auc_ens,
        "auc_ensemble_ci95": [lo, hi],
        "n_train": n_train,
        "n_test": n_test,
    }


# ─── Main evaluation entry point ───────────────────────────────────────────

def evaluate_checkpoint(
    checkpoint: str,
    num_samples: int = 1000,
    adaptive_train: int = 800,
    adaptive_test: int = 200,
    seed: int = 0,
    model_type: str = "dscsp",
) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[gates] checkpoint={checkpoint}  model_type={model_type}", flush=True)
    if model_type == "hidden":
        controller, extractor, config = load_hidden_model(checkpoint, device)
    else:
        controller, extractor, config = ev.load_v4_model(checkpoint, device)
    num_bits = int(config["num_bits"])

    diffusion = DiffusionWrapper(
        model_id=config.get("model_id", "google/ddpm-celebahq-256"),
        device=device,
        num_inference_steps=config.get("ddim_steps", 20),
    )

    n_base = max(num_samples, adaptive_train + adaptive_test)
    print(f"[gates] generating {n_base} cover images via DDIM", flush=True)
    images = ev.generate_images(diffusion, n_base, device)

    # v5 redundant coding support
    coder, msg_bits = _prep_payload_model(checkpoint, num_bits, device)
    if coder is not None:
        print(f"[gates] RedundantCoder active: {msg_bits} msg -> {num_bits} phys (R={coder.R})")

    # Build the stego pool once and reuse it across every gate — deterministic
    # per-image payloads via the random seed above.
    print(f"[gates] building stego pool ({n_base} images)", flush=True)
    stego_pool, msgs = _build_stego_pool(
        controller, images[:n_base], coder, msg_bits, num_bits, device
    )

    # Gate 1: Clean BER
    print("[gates] Clean BER ...", flush=True)
    clean_res = gate_clean(extractor, stego_pool[:num_samples], msgs[:num_samples], coder, device)

    # Gate 2: JPEG Q75 real-libjpeg BER
    print("[gates] JPEG Q75 (real libjpeg) BER ...", flush=True)
    jpeg_res = gate_jpeg_q75(extractor, stego_pool[:num_samples], msgs[:num_samples], coder, device)

    # Gate 3: Crop Q80 BER
    print("[gates] Crop Q80 BER ...", flush=True)
    crop_res = gate_crop_q80(extractor, stego_pool[:num_samples], msgs[:num_samples], coder, device)

    # Gate 4: Adaptive ensemble AUC (uses clean + stego halves from full pool)
    print(f"[gates] Adaptive AUC (ResNet-18 + ResNet-34 ensemble, "
          f"train={adaptive_train} test={adaptive_test}) ...", flush=True)
    auc_res = gate_adaptive_auc(
        stego_pool=stego_pool, clean_pool=images[:stego_pool.shape[0]],
        device=device, n_train=adaptive_train, n_test=adaptive_test, seed=seed,
    )

    gates = {
        "clean_ber": {
            "value": clean_res["mean"], "ci95": clean_res["ci95"],
            "max": GATES["clean_ber_max"],
            "pass": clean_res["mean"] <= GATES["clean_ber_max"],
        },
        "jpeg_q75_real_ber": {
            "value": jpeg_res["mean"], "ci95": jpeg_res["ci95"],
            "max": GATES["jpeg_q75_real_ber_max"],
            "pass": jpeg_res["mean"] <= GATES["jpeg_q75_real_ber_max"],
        },
        "crop_q80_ber": {
            "value": crop_res["mean"], "ci95": crop_res["ci95"],
            "max": GATES["crop_q80_ber_max"],
            "pass": crop_res["mean"] <= GATES["crop_q80_ber_max"],
        },
        "adaptive_auc_ens": {
            "value": auc_res["auc_ensemble"], "ci95": auc_res["auc_ensemble_ci95"],
            "max": GATES["adaptive_auc_ens_max"],
            "pass": auc_res["auc_ensemble"] <= GATES["adaptive_auc_ens_max"],
        },
    }

    all_pass = all(g["pass"] for g in gates.values())
    summary = {
        "checkpoint": checkpoint,
        "num_samples": num_samples,
        "seed": seed,
        "gates": gates,
        "all_gates_pass": all_pass,
        "detail": {
            "clean": clean_res,
            "jpeg_q75_real": jpeg_res,
            "crop_q80": crop_res,
            "adaptive_auc": auc_res,
        },
    }

    del controller, extractor, diffusion, images, stego_pool
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Locked 4-gate DSCSP evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--adaptive-train", type=int, default=800)
    parser.add_argument("--adaptive-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-type", choices=["dscsp", "hidden"], default="dscsp",
                        help="Which checkpoint schema to load.")
    parser.add_argument("--output", default="outputs_tifs_eval/gates_latest.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        num_samples=args.num_samples,
        adaptive_train=args.adaptive_train,
        adaptive_test=args.adaptive_test,
        seed=args.seed,
        model_type=args.model_type,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 72)
    print("4-Gate Evaluation Summary")
    print("=" * 72)
    print(f"  Checkpoint: {args.checkpoint}")
    for name, g in result["gates"].items():
        v = g["value"]
        ci = g["ci95"]
        mark = "PASS" if g["pass"] else "FAIL"
        print(f"  {name:25s}  {v:.4f}  (CI95 {ci[0]:.4f}, {ci[1]:.4f})  "
              f"[max {g['max']}]  {mark}")
    print("-" * 72)
    print(f"  ALL GATES: {'PASS' if result['all_gates_pass'] else 'FAIL'}")
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
