#!/usr/bin/env bash
set -euo pipefail
cd "/root/diffusion-stego"
python3 -u training_ops/tifs_autopilot.py --max-cycles 11 --steps-per-cycle 1200 --eval-samples 200 --adaptive-train 300 --adaptive-test 100 --cross-dataset-samples 200 --cross-dataset-every 1 --min-ablation-variants 3 
