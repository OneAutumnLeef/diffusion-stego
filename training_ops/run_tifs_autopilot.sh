#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p outputs_tifs

python3 -u training_ops/tifs_autopilot.py "$@" | tee -a outputs_tifs/autopilot_console.log
