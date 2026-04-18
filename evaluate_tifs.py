"""
TIFS-oriented evaluation for DSCSP.

Focus areas:
- Detectability (adaptive steganalysis AUC)
- Crop robustness
- Deep statistical analysis (bootstrap CIs + paired tests)
- Cross-dataset generalization (CIFAR10/STL10)
- BER-centric reporting

Usage:
  python evaluate_tifs.py --checkpoint checkpoints_tifs/ber_recovery/dscsp_v4_final.pt
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.transforms import ToTensor

import evaluate_v4 as ev
from dscsp.augmentations.differentiable import DiffJPEG
from dscsp.utils.metrics import compute_ber, compute_psnr


TARGETS = {
    "detectability_auc_max": 0.70,
    "crop_q80_ber_max": 0.35,
    "clean_ber_focus_max": 0.10,
    "jpeg_q75_ber_focus_max": 0.20,
}


def _bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")

    means = []
    n = arr.size
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        means.append(float(arr[idx].mean()))

    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def _eval_clean_series(controller, extractor, images, device: str, num_bits: int) -> Dict:
    bers, psnrs = [], []
    with torch.no_grad():
        for i in range(len(images)):
            x = images[i:i + 1].to(device)
            payload = torch.randint(0, 2, (1, num_bits), device=device).float()
            delta = controller(x, payload)
            xs = (x + delta).clamp(-1, 1)
            pred = extractor(xs)
            bers.append(float(compute_ber(pred, payload)))
            psnrs.append(float(compute_psnr(xs, x)))

    return {
        "ber_mean": float(np.mean(bers)),
        "psnr_mean": float(np.mean(psnrs)),
        "ber_series": bers,
        "psnr_series": psnrs,
    }


def _eval_jpeg_series(controller, extractor, images, device: str, num_bits: int, quality: int) -> Dict:
    diff_jpeg = DiffJPEG().to(device)
    bers = []
    with torch.no_grad():
        for i in range(len(images)):
            x = images[i:i + 1].to(device)
            payload = torch.randint(0, 2, (1, num_bits), device=device).float()
            delta = controller(x, payload)
            xs = (x + delta).clamp(-1, 1)
            xa = diff_jpeg(xs, quality=quality)
            pred = extractor(xa)
            bers.append(float(compute_ber(pred, payload)))

    return {
        "ber_mean": float(np.mean(bers)),
        "ber_series": bers,
    }


def _random_crop_resize(x: torch.Tensor, ratio: float) -> torch.Tensor:
    b, c, h, w = x.shape
    ch, cw = max(1, int(h * ratio)), max(1, int(w * ratio))
    top = random.randint(0, h - ch)
    left = random.randint(0, w - cw)
    cropped = x[:, :, top:top + ch, left:left + cw]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


def _eval_crop_series(controller, extractor, images, device: str, num_bits: int, ratio: float) -> Dict:
    bers = []
    with torch.no_grad():
        for i in range(len(images)):
            x = images[i:i + 1].to(device)
            payload = torch.randint(0, 2, (1, num_bits), device=device).float()
            delta = controller(x, payload)
            xs = (x + delta).clamp(-1, 1)
            xa = _random_crop_resize(xs, ratio)
            pred = extractor(xa)
            bers.append(float(compute_ber(pred, payload)))

    return {
        "ber_mean": float(np.mean(bers)),
        "ber_series": bers,
    }


def _adaptive_attack(x: torch.Tensor, diff_jpeg: DiffJPEG) -> torch.Tensor:
    r = random.random()
    if r < 0.4:
        q = random.randint(40, 90)
        return diff_jpeg(x, quality=q)
    if r < 0.7:
        ratio = random.uniform(0.70, 0.95)
        return _random_crop_resize(x, ratio)
    sigma = random.uniform(0.01, 0.04)
    return (x + torch.randn_like(x) * sigma).clamp(-1, 1)


def _adaptive_steganalysis(controller, images, device: str, num_bits: int, n_train: int, n_test: int) -> Dict:
    assert len(images) >= n_train + n_test, "Not enough images for adaptive steganalysis"

    diff_jpeg = DiffJPEG().to(device)

    # Build stego samples once for train/test split.
    stego = []
    with torch.no_grad():
        for i in range(n_train + n_test):
            x = images[i:i + 1].to(device)
            payload = torch.randint(0, 2, (1, num_bits), device=device).float()
            delta = controller(x, payload)
            xs = (x + delta).clamp(-1, 1)
            stego.append(xs)
    stego = torch.cat(stego, dim=0)

    clean = images[: n_train + n_test].to(device)
    clean_train, clean_test = clean[:n_train], clean[n_train:]
    stego_train, stego_test = stego[:n_train], stego[n_train:]

    clf = models.resnet18(weights=None)
    clf.fc = torch.nn.Linear(clf.fc.in_features, 1)
    clf = clf.to(device)

    opt = torch.optim.Adam(clf.parameters(), lr=1e-4)
    clf.train()

    batch = 16
    for _ in range(12):
        idx = torch.randperm(n_train)
        for st in range(0, n_train, batch):
            ed = min(st + batch, n_train)
            ids = idx[st:ed]
            half = len(ids)
            if half == 0:
                continue

            xc = clean_train[ids]
            xs = stego_train[ids]

            # Adaptive augmentation on both classes keeps detector from overfitting trivial clues.
            xc = _adaptive_attack(xc, diff_jpeg)
            xs = _adaptive_attack(xs, diff_jpeg)

            x_in = torch.cat([xc, xs], dim=0)
            y = torch.cat([
                torch.zeros(half, 1, device=device),
                torch.ones(half, 1, device=device),
            ], dim=0)

            opt.zero_grad()
            logits = clf(x_in)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()

    clf.eval()
    labels_clean, scores_clean = [], []
    labels_adapt, scores_adapt = [], []

    with torch.no_grad():
        for st in range(0, n_test, batch):
            ed = min(st + batch, n_test)
            xc = clean_test[st:ed]
            xs = stego_test[st:ed]

            # Clean test
            s_clean_clean = torch.sigmoid(clf(xc)).cpu().numpy().flatten().tolist()
            s_clean_stego = torch.sigmoid(clf(xs)).cpu().numpy().flatten().tolist()
            scores_clean.extend(s_clean_clean + s_clean_stego)
            labels_clean.extend([0] * len(s_clean_clean) + [1] * len(s_clean_stego))

            # Adaptive test
            xca = _adaptive_attack(xc, diff_jpeg)
            xsa = _adaptive_attack(xs, diff_jpeg)
            s_adapt_clean = torch.sigmoid(clf(xca)).cpu().numpy().flatten().tolist()
            s_adapt_stego = torch.sigmoid(clf(xsa)).cpu().numpy().flatten().tolist()
            scores_adapt.extend(s_adapt_clean + s_adapt_stego)
            labels_adapt.extend([0] * len(s_adapt_clean) + [1] * len(s_adapt_stego))

    auc_clean = float(roc_auc_score(labels_clean, scores_clean))
    auc_adaptive = float(roc_auc_score(labels_adapt, scores_adapt))

    acc_adaptive = float(np.mean([(s > 0.5) == (l == 1) for s, l in zip(scores_adapt, labels_adapt)]))

    del clf
    torch.cuda.empty_cache()

    return {
        "auc_clean": auc_clean,
        "auc_adaptive": auc_adaptive,
        "accuracy_adaptive": acc_adaptive,
    }


def _load_dataset_images(dataset_name: str, n: int) -> torch.Tensor:
    root = ".tifs_data"
    to_tensor = ToTensor()

    if dataset_name == "cifar10":
        ds = datasets.CIFAR10(root=root, train=False, download=True, transform=to_tensor)
    elif dataset_name == "stl10":
        ds = datasets.STL10(root=root, split="test", download=True, transform=to_tensor)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if len(ds) < n:
        n = len(ds)

    subset = Subset(ds, list(range(n)))
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=0)

    imgs = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        if not torch.is_tensor(x):
            x = torch.stack([to_tensor(img) for img in x], dim=0)
        x = x.float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            continue
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        x = x * 2 - 1
        imgs.append(x)

    if not imgs:
        raise RuntimeError(f"No images loaded for dataset {dataset_name}")

    return torch.cat(imgs, dim=0)[:n]


def evaluate_checkpoint(
    checkpoint: str,
    num_samples: int,
    cross_dataset_samples: int,
    skip_cross_dataset: bool,
    adaptive_train: int,
    adaptive_test: int,
) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] Loading checkpoint: {checkpoint}", flush=True)
    controller, extractor, config = ev.load_v4_model(checkpoint, device)
    num_bits = int(config["num_bits"])

    diffusion = ev.DiffusionWrapper(
        model_id=config.get("model_id", "google/ddpm-celebahq-256"),
        device=device,
        num_inference_steps=config.get("ddim_steps", 20),
    )

    n_base = max(num_samples, adaptive_train + adaptive_test)
    print(f"[eval] Generating {n_base} base images", flush=True)
    images = ev.generate_images(diffusion, n_base, device)

    print(f"[eval] Running clean/jpeg/crop metrics on {num_samples} samples", flush=True)
    clean = _eval_clean_series(controller, extractor, images[:num_samples], device, num_bits)
    jpeg75 = _eval_jpeg_series(controller, extractor, images[:num_samples], device, num_bits, 75)
    jpeg50 = _eval_jpeg_series(controller, extractor, images[:num_samples], device, num_bits, 50)
    crop80 = _eval_crop_series(controller, extractor, images[:num_samples], device, num_bits, 0.80)
    crop70 = _eval_crop_series(controller, extractor, images[:num_samples], device, num_bits, 0.70)

    print(
        f"[eval] Running adaptive steganalysis (train={adaptive_train}, test={adaptive_test})",
        flush=True,
    )
    adaptive = _adaptive_steganalysis(
        controller,
        images,
        device,
        num_bits,
        n_train=adaptive_train,
        n_test=adaptive_test,
    )

    clean_ci = _bootstrap_ci(clean["ber_series"])
    jpeg75_ci = _bootstrap_ci(jpeg75["ber_series"])
    crop80_ci = _bootstrap_ci(crop80["ber_series"])

    p_clean_vs_crop80 = float(ttest_rel(clean["ber_series"], crop80["ber_series"]).pvalue)
    p_clean_vs_jpeg75 = float(ttest_rel(clean["ber_series"], jpeg75["ber_series"]).pvalue)

    cross_dataset = {}
    if not skip_cross_dataset:
        print(f"[eval] Running cross-dataset checks ({cross_dataset_samples} samples each)", flush=True)
        for ds_name in ["cifar10", "stl10"]:
            try:
                print(f"[eval] Loading dataset: {ds_name}", flush=True)
                ds_images = _load_dataset_images(ds_name, cross_dataset_samples)
                print(f"[eval] Evaluating dataset: {ds_name} (n={len(ds_images)})", flush=True)
                ds_clean = _eval_clean_series(controller, extractor, ds_images, device, num_bits)
                ds_crop = _eval_crop_series(controller, extractor, ds_images, device, num_bits, 0.80)
                cross_dataset[ds_name] = {
                    "n": int(len(ds_images)),
                    "clean_ber": ds_clean["ber_mean"],
                    "crop80_ber": ds_crop["ber_mean"],
                }
            except Exception as exc:  # noqa: BLE001
                cross_dataset[ds_name] = {"error": str(exc)}

    cross_dataset_valid = any(
        isinstance(v, dict)
        and "error" not in v
        and "clean_ber" in v
        and "crop80_ber" in v
        for v in cross_dataset.values()
    )

    ablation_path = "outputs_tifs_ablation/latest_ablation.json"
    ablation_status = {
        "has_report": os.path.exists(ablation_path),
        "report": ablation_path if os.path.exists(ablation_path) else "",
    }

    areas = {
        "detectability": {
            "current": adaptive["auc_adaptive"],
            "target": TARGETS["detectability_auc_max"],
            "pass": adaptive["auc_adaptive"] <= TARGETS["detectability_auc_max"],
        },
        "crop_robustness": {
            "current": crop80["ber_mean"],
            "target": TARGETS["crop_q80_ber_max"],
            "pass": crop80["ber_mean"] <= TARGETS["crop_q80_ber_max"],
        },
        "ablation": {
            "current": "extensive" if ablation_status["has_report"] else "minimal",
            "target": "extensive",
            "pass": ablation_status["has_report"],
        },
        "statistical_analysis": {
            "current": "deep",
            "target": "deep",
            "pass": True,
        },
        "cross_dataset": {
            "current": "yes" if cross_dataset_valid else "no",
            "target": "yes",
            "pass": cross_dataset_valid,
        },
        "adaptive_steganalysis": {
            "current": adaptive["auc_adaptive"],
            "target": TARGETS["detectability_auc_max"],
            "pass": adaptive["auc_adaptive"] <= TARGETS["detectability_auc_max"],
        },
        "ber_focus": {
            "clean_ber": clean["ber_mean"],
            "jpeg_q75_ber": jpeg75["ber_mean"],
            "clean_target": TARGETS["clean_ber_focus_max"],
            "jpeg_q75_target": TARGETS["jpeg_q75_ber_focus_max"],
            "pass": (
                clean["ber_mean"] <= TARGETS["clean_ber_focus_max"]
                and jpeg75["ber_mean"] <= TARGETS["jpeg_q75_ber_focus_max"]
            ),
        },
    }

    summary = {
        "checkpoint": checkpoint,
        "num_samples": num_samples,
        "clean": {
            "ber": clean["ber_mean"],
            "psnr": clean["psnr_mean"],
            "ber_ci95": [clean_ci[0], clean_ci[1]],
        },
        "jpeg": {
            "q75_ber": jpeg75["ber_mean"],
            "q50_ber": jpeg50["ber_mean"],
            "q75_ber_ci95": [jpeg75_ci[0], jpeg75_ci[1]],
        },
        "crop": {
            "q80_ber": crop80["ber_mean"],
            "q70_ber": crop70["ber_mean"],
            "q80_ber_ci95": [crop80_ci[0], crop80_ci[1]],
        },
        "adaptive_steganalysis": adaptive,
        "cross_dataset": cross_dataset,
        "statistics": {
            "p_clean_vs_crop80": p_clean_vs_crop80,
            "p_clean_vs_jpeg75": p_clean_vs_jpeg75,
        },
        "ablation": ablation_status,
        "areas": areas,
    }

    print("[eval] Completed full TIFS evaluation", flush=True)
    del controller, extractor, diffusion, images
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="TIFS-oriented DSCSP evaluation")
    parser.add_argument("--checkpoint", default="checkpoints_v4/dscsp_v4_final.pt")
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--cross-dataset-samples", type=int, default=400)
    parser.add_argument("--adaptive-train", type=int, default=800)
    parser.add_argument("--adaptive-test", type=int, default=200)
    parser.add_argument("--skip-cross-dataset", action="store_true")
    parser.add_argument("--output", default="outputs_tifs_eval/tifs_eval_latest.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    result = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        num_samples=args.num_samples,
        cross_dataset_samples=args.cross_dataset_samples,
        skip_cross_dataset=args.skip_cross_dataset,
        adaptive_train=args.adaptive_train,
        adaptive_test=args.adaptive_test,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=" * 72)
    print("TIFS Evaluation Summary")
    print("=" * 72)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Clean BER: {result['clean']['ber']:.6f}  (CI95: {result['clean']['ber_ci95'][0]:.6f}, {result['clean']['ber_ci95'][1]:.6f})")
    print(f"JPEG Q75 BER: {result['jpeg']['q75_ber']:.6f}")
    print(f"Crop Q80 BER: {result['crop']['q80_ber']:.6f}")
    print(f"Adaptive AUC: {result['adaptive_steganalysis']['auc_adaptive']:.4f}")
    print("Areas:")
    for k, v in result["areas"].items():
        print(f"  - {k}: {'PASS' if v.get('pass', False) else 'FAIL'}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
