"""Upload real checkpoint blobs to the project's Hugging Face dataset.

Why this script exists
----------------------
The primary checkpoint artifacts (``checkpoints_tifs/**``) are distributed via
Git LFS on GitHub. GitHub LFS has a per-account *bandwidth budget* that, once
exhausted, blocks fetching even though the pointers still exist in the repo.
Between 2026-04-23 and 2026-04-24 the budget was exceeded, stranding a fresh
RunPod 5090 that held a valid PAT. HANDOFF_2026-04-24.md documents the
incident. This script is the remedy: it pushes real ``.pt`` blobs directly
to the Hugging Face dataset repo ``OneAutumnLeef/dscsp-checkpoints`` (or
its mirror ``dscsp-full-backup``) as flat files, bypassing Git LFS
entirely. Future sessions then restore via the existing
``pull_artifacts_from_hf.sh`` path, with no LFS dependency.

Usage
-----
::

    export HF_TOKEN=hf_...
    python push_checkpoints_to_hf.py                 # uploads the default set
    python push_checkpoints_to_hf.py --dry-run       # list what would upload
    python push_checkpoints_to_hf.py --include "checkpoints_tifs/hardening/*.pt"
    python push_checkpoints_to_hf.py --repo OneAutumnLeef/dscsp-checkpoints

The script refuses to upload files that are still 134-byte LFS pointer stubs,
so you will get a clear error rather than silently pushing useless stubs
back to HF.

Default policy
--------------
By default we upload every ``checkpoints_tifs/**/*.pt`` and
``checkpoints_v5*/**/*.pt`` that is strictly larger than 1 MB, plus the
autopilot state + log files that ride along with them. This avoids
re-uploading the stubs that live on HF for historical reasons, and avoids
clobbering gated v5 weights when that was not the intent.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path
from typing import Iterable, List

DEFAULT_REPO = "OneAutumnLeef/dscsp-checkpoints"
DEFAULT_REPO_TYPE = "dataset"
STUB_THRESHOLD_BYTES = 1_000_000  # files <= this are treated as LFS stubs

DEFAULT_INCLUDES = [
    "checkpoints_tifs/**/*.pt",
    "checkpoints_v5/**/*.pt",
    "checkpoints_v5_clean/**/*.pt",
    "checkpoints_v5_quality/**/*.pt",
    "outputs_tifs/autopilot_state.json",
    "outputs_tifs/MAJOR_LOG.md",
    "outputs_tifs_eval/**/*.json",
    "outputs_v5_ablation/ablation_results.json",
    "outputs_v5_analysis/**/*.json",
]


def _iter_candidates(root: Path, patterns: Iterable[str]) -> List[Path]:
    """Resolve glob patterns relative to ``root``. Deduplicated, sorted."""
    seen: set[Path] = set()
    for pat in patterns:
        for path in sorted(root.glob(pat)):
            if path.is_file():
                seen.add(path.resolve())
    return sorted(seen)


def _classify(paths: List[Path]) -> tuple[List[Path], List[Path], List[Path]]:
    """Partition candidates into (real_blobs, stubs, other)."""
    real, stubs, other = [], [], []
    for p in paths:
        size = p.stat().st_size
        if p.suffix == ".pt":
            if size <= STUB_THRESHOLD_BYTES:
                stubs.append(p)
            else:
                real.append(p)
        else:
            other.append(p)
    return real, stubs, other


def main() -> int:
    parser = argparse.ArgumentParser(description="Push real checkpoint blobs to HF.")
    parser.add_argument("--repo", default=os.environ.get("HF_ARTIFACT_REPO_ID", DEFAULT_REPO))
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE)
    parser.add_argument("--include", action="append", default=None,
                        help="Glob(s) to upload (repeatable). Defaults upload the full set.")
    parser.add_argument("--root", default=".", help="Repository root (default: cwd).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded, then exit.")
    parser.add_argument("--allow-stubs", action="store_true",
                        help="Upload 134-byte LFS pointer files too (dangerous).")
    parser.add_argument("--token", default=None,
                        help="HF token. Defaults to $HF_TOKEN. Do NOT commit.")
    parser.add_argument("--commit-message", default="Mirror real checkpoint blobs from 5090 retrain")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: HF_TOKEN is not set. Pass --token or export HF_TOKEN.",
              file=sys.stderr)
        return 2

    root = Path(args.root).resolve()
    patterns = args.include or DEFAULT_INCLUDES
    paths = _iter_candidates(root, patterns)
    if not paths:
        print("No files matched.", file=sys.stderr)
        return 1

    real, stubs, other = _classify(paths)

    print(f"[push] root={root}")
    print(f"[push] repo={args.repo} ({args.repo_type})")
    print(f"[push] matched: {len(paths)}  real={len(real)}  stubs={len(stubs)}  other={len(other)}")
    total_mb = sum(p.stat().st_size for p in real + other) / (1024 * 1024)
    print(f"[push] upload size (real + other): {total_mb:.1f} MB")

    if stubs and not args.allow_stubs:
        print("[push] skipping LFS stub files:")
        for p in stubs:
            print(f"        {p.relative_to(root)}  ({p.stat().st_size} bytes)")

    uploads: List[Path] = real + other
    if args.allow_stubs:
        uploads += stubs

    if not uploads:
        print("[push] nothing to upload after filtering stubs. Exiting.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("[push] dry-run. Files that WOULD upload:")
        for p in uploads:
            print(f"        {p.relative_to(root)}  {p.stat().st_size}")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub is required. pip install -r requirements.txt",
              file=sys.stderr)
        return 3

    api = HfApi(token=token)
    print(f"[push] uploading {len(uploads)} files ...")
    for idx, src in enumerate(uploads, start=1):
        rel = src.relative_to(root).as_posix()
        print(f"  [{idx}/{len(uploads)}] {rel}  ({src.stat().st_size / 1024:.1f} KB)")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=rel,
            repo_id=args.repo,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
        )
    print(f"[push] done. Mirror: https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
