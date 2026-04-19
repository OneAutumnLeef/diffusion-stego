#!/usr/bin/env bash
set -euo pipefail

# Pull checkpoints and outputs from Hugging Face dataset repository.
# Usage: bash pull_artifacts_from_hf.sh [target_dir]

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash pull_artifacts_from_hf.sh [target_dir]"
    echo ""
    echo "Downloads checkpoints and outputs from Hugging Face dataset repo"
    echo "(default target_dir: current directory)."
    echo ""
    echo "Optional env overrides:"
    echo "  HF_ARTIFACT_REPO_ID   (default: OneAutumnLeef/dscsp-checkpoints)"
    echo "  HF_ARTIFACT_REPO_TYPE (default: dataset)"
    exit 0
fi

TARGET_DIR="${1:-.}"
REPO_ID="${HF_ARTIFACT_REPO_ID:-OneAutumnLeef/dscsp-checkpoints}"
REPO_TYPE="${HF_ARTIFACT_REPO_TYPE:-dataset}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found." >&2
  exit 1
fi

python3 - "$REPO_ID" "$REPO_TYPE" "$TARGET_DIR" <<'PY'
import os
import sys

repo_id = sys.argv[1]
repo_type = sys.argv[2]
target_dir = sys.argv[3]

allow_patterns = [
    "checkpoints_v5/**",
    "checkpoints_v5_clean/**",
    "checkpoints_v5_quality/**",
    "checkpoints_v4/**",
    "checkpoints_v4_5090/**",
    "checkpoints_v4_5090_final/**",
    "checkpoints_tifs/**",
    "outputs_v5/**",
    "outputs_v5_clean/**",
    "outputs_v5_ablation/**",
    "outputs_v5_quality/**",
    "outputs_v4/**",
    "outputs_v4_probe/**",
    "outputs_tifs/**",
    "outputs_tifs_eval/**",
    "outputs_tifs_v5/**",
]

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    print("ERROR: huggingface_hub is required. Install with: pip install -r requirements.txt", file=sys.stderr)
    raise

os.makedirs(target_dir, exist_ok=True)
print(f"Pulling artifacts from {repo_id} ({repo_type}) into {os.path.abspath(target_dir)}")

snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir=target_dir,
    allow_patterns=allow_patterns,
)

print("Artifact pull complete.")
PY

echo "Done."
