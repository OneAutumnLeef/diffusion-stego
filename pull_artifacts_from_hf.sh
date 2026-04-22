#!/usr/bin/env bash
set -euo pipefail

# Cross-platform wrapper: delegate logic to Python utility.
# Usage: bash pull_artifacts_from_hf.sh [target_dir]

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        echo "Usage: bash pull_artifacts_from_hf.sh [target_dir]"
        echo ""
        echo "Downloads checkpoints and outputs from Hugging Face dataset repo"
        echo "(default target_dir: current directory)."
        echo ""
        echo "Optional env overrides:"
        echo "  HF_ARTIFACT_REPO_ID            (default: OneAutumnLeef/dscsp-full-backup)"
        echo "  HF_ARTIFACT_FALLBACK_REPO_IDS  (default: OneAutumnLeef/dscsp-checkpoints)"
        echo "  HF_ARTIFACT_REPO_TYPE          (default: dataset)"
        exit 0
fi

TARGET_DIR="${1:-.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "ERROR: python3/python not found." >&2
    exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/pull_artifacts_from_hf.py" "$TARGET_DIR"
