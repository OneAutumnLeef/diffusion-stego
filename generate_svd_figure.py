"""Generate the SVD-of-mean-perturbation figure for Section VIII (item B6b).

Produces:
  figures/svd_singular_vectors.pdf    top-3 left singular vectors (3-panel)
  figures/svd_singular_vectors.png    same, for preview
  figures/svd_singular_values.json    top-32 singular values + Gini coefficient

Usage
-----
    python generate_svd_figure.py                                    # defaults below
    python generate_svd_figure.py --checkpoint CKPT --n-samples 100  # override
    python generate_svd_figure.py --images covers.pt                 # skip diffusion

``--images`` expects a ``.pt`` tensor of shape ``(N, 3, H, W)`` in ``[-1, 1]``.
Omitting it generates fresh covers from the frozen diffusion model used during
training (``google/ddpm-celebahq-256`` at 20 DDIM steps).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluate_v4 import load_v4_model  # noqa: E402


DEFAULT_CKPT = "checkpoints_tifs/hardening/dscsp_v5_final.pt"


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative 1-D array. Returns 0 for uniform,
    approaches 1 for fully concentrated."""
    v = np.asarray(values, dtype=np.float64).flatten()
    if v.size == 0:
        return 0.0
    if np.amin(v) < 0:
        v = v - np.amin(v)
    v = v + 1e-12
    v = np.sort(v)
    n = v.size
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) / (n * np.sum(v))) - (n + 1) / n)


def _load_or_generate_images(n: int, device: torch.device,
                              images_path: str | None,
                              image_size: int) -> torch.Tensor:
    if images_path and os.path.exists(images_path):
        print(f"[svd] loading covers from {images_path}")
        imgs = torch.load(images_path, map_location="cpu")
        if imgs.dim() != 4 or imgs.shape[1] != 3:
            raise ValueError(f"expected (N,3,H,W); got {tuple(imgs.shape)}")
        return imgs[:n]

    print(f"[svd] generating {n} covers from ddpm-celebahq-256 (DDIM 20)")
    from dscsp.models.diffusion_wrapper import DiffusionWrapper
    from evaluate_v4 import generate_images
    diffusion = DiffusionWrapper(
        model_id="google/ddpm-celebahq-256",
        device=str(device),
        num_inference_steps=20,
    )
    covers = generate_images(diffusion, n, str(device), batch=4)
    if covers.shape[-1] != image_size:
        covers = torch.nn.functional.interpolate(
            covers, size=(image_size, image_size),
            mode="bilinear", align_corners=False,
        )
    return covers


def compute_mean_perturbation_svd(controller: torch.nn.Module,
                                    images: torch.Tensor,
                                    device: torch.device,
                                    num_bits: int,
                                    batch_size: int = 4,
                                    k_keep: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Stack per-image perturbations, run SVD, return (U_top3, S_topK).

    U_top3: (3, 3, H, W) — top-3 left singular vectors, reshaped back to image.
    S_topK: (k_keep,) — top-k singular values (descending).
    """
    controller.eval()
    deltas = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            x = images[i:i + batch_size].to(device)
            payload = torch.randint(0, 2, (x.shape[0], num_bits),
                                     device=device, dtype=torch.float32)
            delta = controller(x, payload).cpu()
            deltas.append(delta)
    D = torch.cat(deltas, dim=0)                      # (N, 3, H, W)
    N, C, H, W = D.shape
    flat = D.view(N, C * H * W).to(torch.float64).numpy()

    # Centre so SVD captures structured residual around the mean perturbation
    mean_row = flat.mean(axis=0, keepdims=True)
    Xc = flat - mean_row

    U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(3, S.shape[0])
    # Each column of U, scaled by S, mapped back through the feature basis
    # gives the contribution of that mode — we visualise the left singular
    # *mode image* by taking the top rows of Vt reshaped to (C,H,W).
    # numpy SVD returns Vt; we want the right singular vectors since images
    # live in the column space. Recompute accordingly.
    _U2, S2, Vt2 = np.linalg.svd(Xc, full_matrices=False)
    top_modes = Vt2[:k].reshape(k, C, H, W)
    top_sv = S2[:min(k_keep, S2.shape[0])]
    return top_modes.astype(np.float32), top_sv.astype(np.float64)


def _panel_normalize(mode: np.ndarray) -> np.ndarray:
    """Normalise a (3, H, W) mode to a (H, W, 3) uint-ish float panel in [0, 1]."""
    v = mode.transpose(1, 2, 0).astype(np.float32)
    lo, hi = np.percentile(v, 1.0), np.percentile(v, 99.0)
    if hi - lo < 1e-12:
        hi = lo + 1.0
    v = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return v


def save_figure(top_modes: np.ndarray, top_sv: np.ndarray,
                 out_pdf: Path, out_png: Path, out_json: Path,
                 title_suffix: str = "") -> None:
    gini = _gini(top_sv)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, mode, sv in zip(axes, top_modes, top_sv[:3]):
        panel = _panel_normalize(mode)
        ax.imshow(panel, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"σ = {sv:.3g}", fontsize=10)
    fig.suptitle(f"Top-3 singular vectors of the mean perturbation{title_suffix} "
                 f"— Gini(top-{len(top_sv)}) = {gini:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)

    with open(out_json, "w") as f:
        json.dump({
            "top_singular_values": top_sv.tolist(),
            "gini_coefficient": gini,
            "interpretation": ("Gini > 0.70 indicates the perturbation is "
                                "concentrated in a small number of spatial "
                                "modes — the subspace the ResNet steganalyser "
                                "keys on. Values near 0 indicate a diffuse, "
                                "isotropic perturbation."),
        }, f, indent=2)

    print(f"[svd] wrote {out_pdf}")
    print(f"[svd] wrote {out_png}")
    print(f"[svd] wrote {out_json}")
    print(f"[svd] top-3 singular values: {top_sv[:3].tolist()}")
    print(f"[svd] top-32 Gini: {gini:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SVD of mean perturbation figure.")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--images", default=None,
                        help="Optional .pt tensor of covers in [-1,1].")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--out-pdf", default="figures/svd_singular_vectors.pdf")
    parser.add_argument("--out-png", default="figures/svd_singular_vectors.png")
    parser.add_argument("--out-json", default="figures/svd_singular_values.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[svd] device={device} checkpoint={args.checkpoint}")

    controller, _extractor, config = load_v4_model(args.checkpoint, device)
    num_bits = int(config.get("num_bits", 256))
    print(f"[svd] loaded controller with num_bits={num_bits}")

    covers = _load_or_generate_images(
        n=args.n_samples,
        device=device,
        images_path=args.images,
        image_size=args.image_size,
    )

    top_modes, top_sv = compute_mean_perturbation_svd(
        controller, covers, device, num_bits=num_bits, batch_size=args.batch_size,
    )

    out_pdf = Path(args.out_pdf); out_png = Path(args.out_png); out_json = Path(args.out_json)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    tag = "" if args.checkpoint == DEFAULT_CKPT else f" ({os.path.basename(args.checkpoint)})"
    save_figure(top_modes, top_sv, out_pdf, out_png, out_json, title_suffix=tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
