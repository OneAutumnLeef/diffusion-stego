#!/usr/bin/env python3
"""Pull checkpoints/outputs artifacts from Hugging Face dataset repos.

Usage:
  python pull_artifacts_from_hf.py [target_dir]

Env overrides:
  HF_ARTIFACT_REPO_ID            Primary repo id
  HF_ARTIFACT_FALLBACK_REPO_IDS  Comma-separated fallback repo ids
  HF_ARTIFACT_REPO_TYPE          Repo type (default: dataset)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List


ALLOW_PATTERNS = [
    "checkpoints_tifs/**",
    "checkpoints_v4/**",
    "checkpoints_v4_5090/**",
    "checkpoints_v4_5090_final/**",
    "checkpoints_v4_probe/**",
    "checkpoints_v5/**",
    "checkpoints_v5_clean/**",
    "checkpoints_v5_quality/**",
    "outputs_tifs/**",
    "outputs_tifs_ablation/**",
    "outputs_tifs_eval/**",
    "outputs_tifs_v5/**",
    "outputs_v4/**",
    "outputs_v4_probe/**",
    "outputs_v5/**",
    "outputs_v5_ablation/**",
    "outputs_v5_analysis/**",
    "outputs_v5_clean/**",
    "outputs_v5_quality/**",
]


def _build_repo_candidates(primary: str, fallback_csv: str) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for repo_id in [primary, *fallback_csv.split(",")]:
        cleaned = repo_id.strip()
        if not cleaned or cleaned in seen:
            continue
        ordered.append(cleaned)
        seen.add(cleaned)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull artifacts from Hugging Face.")
    parser.add_argument("target_dir", nargs="?", default=".")
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_ARTIFACT_REPO_ID", "OneAutumnLeef/dscsp-full-backup"),
    )
    parser.add_argument(
        "--fallback-repo-ids",
        default=os.getenv("HF_ARTIFACT_FALLBACK_REPO_IDS", "OneAutumnLeef/dscsp-checkpoints"),
    )
    parser.add_argument(
        "--repo-type",
        default=os.getenv("HF_ARTIFACT_REPO_TYPE", "dataset"),
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print(
            "ERROR: huggingface_hub is required. Install with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    target_dir = args.target_dir
    repo_candidates = _build_repo_candidates(args.repo_id, args.fallback_repo_ids)
    os.makedirs(target_dir, exist_ok=True)

    print(f"Target directory: {os.path.abspath(target_dir)}")
    print(f"Repo candidates: {', '.join(repo_candidates)}")

    last_error: Exception | None = None
    for repo_id in repo_candidates:
        print(f"Attempting pull from {repo_id} ({args.repo_type})")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=args.repo_type,
                local_dir=target_dir,
                allow_patterns=ALLOW_PATTERNS,
            )
            print(f"Artifact pull complete using {repo_id}.")
            return 0
        except Exception as exc:
            last_error = exc
            print(f"WARNING: pull failed for {repo_id}: {exc}", file=sys.stderr)

    print("ERROR: all artifact repo candidates failed.", file=sys.stderr)
    if last_error is not None:
        print(f"Last error: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())