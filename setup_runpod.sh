#!/bin/bash
# ============================================================================
# DSCSP RunPod Setup Script — RTX 5090 (32GB VRAM)
# ============================================================================
# Run this ONCE after launching a RunPod instance with PyTorch template.
#
# Recommended RunPod setup:
#   Template: "RunPod Pytorch 2.4.0" (or newer)
#   GPU: RTX 5090 (32GB VRAM)
#   Disk: 50GB+ (model weights + diffusion model + checkpoints)
#   Region: Any (US-East recommended for speed)
#
# Steps:
#   1. Launch RunPod pod with above config
#   2. Upload project files (rsync/scp) to /workspace/dscsp/
#   3. SSH in and run: chmod +x setup_runpod.sh && ./setup_runpod.sh
#   4. Start training: ./run_5090.sh
# ============================================================================

set -e

PROJ_DIR="/workspace/dscsp"

echo "============================================"
echo "DSCSP RunPod Setup — RTX 5090"
echo "============================================"

# ─── 1. System info ─────────────────────────────────────────────────────────
echo ""
echo "[1/7] System Info"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# ─── 2. Install dependencies ────────────────────────────────────────────────
echo ""
echo "[2/7] Installing dependencies..."
pip install --quiet --upgrade \
    torch torchvision \
    "diffusers>=0.25.0" \
    "accelerate>=0.25.0" \
    "lpips>=0.1.4" \
    scikit-learn \
    scikit-image \
    scipy \
    torchmetrics \
    numpy \
    Pillow \
    tqdm

echo "  Dependencies installed."

# ─── 3. Verify project files ────────────────────────────────────────────────
echo ""
echo "[3/7] Verifying project files..."

if [ ! -d "$PROJ_DIR" ]; then
    echo "  ERROR: Project directory $PROJ_DIR not found!"
    echo ""
    echo "  Please upload project files first. From your LOCAL machine:"
    echo ""
    echo "    rsync -avz --exclude '__pycache__' --exclude '*.pt' \\"
    echo "      --exclude 'outputs*' --exclude 'checkpoints*' \\"
    echo "      --exclude 'tmpclaude*' --exclude 'eval_*' \\"
    echo "      . root@<RUNPOD_IP>:$PROJ_DIR/"
    echo ""
    echo "  Then optionally upload latest checkpoint:"
    echo "    scp checkpoints_v4/dscsp_v4_step_XXXXX.pt root@<RUNPOD_IP>:$PROJ_DIR/checkpoints_v4/"
    echo ""
    exit 1
fi

cd "$PROJ_DIR"

# Check required files
REQUIRED_FILES=(
    "train_v4.py"
    "train_v4_5090.py"
    "evaluate_v4.py"
    "dscsp/__init__.py"
    "dscsp/models/controller_v4.py"
    "dscsp/models/extractor.py"
    "dscsp/models/diffusion_wrapper.py"
    "dscsp/augmentations/differentiable.py"
    "dscsp/losses/composite_loss.py"
    "dscsp/utils/metrics.py"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  MISSING: $f"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "  ERROR: Missing required files. Upload them and re-run."
    exit 1
fi
echo "  All required files present."

# ─── 4. Pre-download diffusion model ────────────────────────────────────────
echo ""
echo "[4/7] Pre-downloading diffusion model (google/ddpm-celebahq-256)..."
python3 -c "
from diffusers import DDPMPipeline
print('  Downloading DDPM CelebA-HQ 256...')
pipe = DDPMPipeline.from_pretrained('google/ddpm-celebahq-256')
print('  Model cached successfully.')
"

# ─── 5. Create output directories ─────────────────────────────────────────
echo ""
echo "[5/7] Creating output directories..."
mkdir -p outputs_v4_5090
mkdir -p checkpoints_v4_5090
mkdir -p checkpoints_v4
echo "  Done."

# ─── 6. Verify GPU + imports ──────────────────────────────────────────────
echo ""
echo "[6/7] Verifying GPU + imports..."
python3 -c "
import torch
import torchvision
import diffusers
import lpips
import sklearn
import skimage
import scipy
import torchmetrics

print(f'  torch:       {torch.__version__}')
print(f'  torchvision: {torchvision.__version__}')
print(f'  diffusers:   {diffusers.__version__}')
print(f'  CUDA avail:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:         {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:        {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print('  All imports OK!')
"

# ─── 7. Create run script ─────────────────────────────────────────────────
echo ""
echo "[7/7] Creating run script..."
cat > run_5090.sh << 'RUNEOF'
#!/bin/bash
# ============================================================================
# DSCSP v4 Training + Evaluation — RTX 5090
# ============================================================================
# Runs the full pipeline: train → evaluate → steganalysis
#
# Usage:
#   ./run_5090.sh                          # Train from scratch
#   ./run_5090.sh --resume checkpoint.pt   # Resume from checkpoint
# ============================================================================

set -e
cd /workspace/dscsp

RESUME_ARG=""
if [ "$1" == "--resume" ] && [ -n "$2" ]; then
    RESUME_ARG="$2"
    echo "Resuming from checkpoint: $RESUME_ARG"
fi

echo "============================================"
echo "DSCSP v4 Training — RTX 5090 (32GB VRAM)"
echo "============================================"
echo ""
echo "Config overrides for 5090:"
echo "  batch_size: 4 → 16"
echo "  base_channels: 48 → 64"
echo "  lr_controller: 1e-4 → 2e-4"
echo "  lr_extractor: 3e-4 → 6e-4"
echo "  jpeg_weight: 0.6 → 0.7"
echo "  robustness_max_prob: 0.85 → 0.9"
echo ""

# If resuming, patch the resume path into train_v4_5090.py temporarily
if [ -n "$RESUME_ARG" ]; then
    python3 -c "
import train_v4
from train_v4_5090 import OVERRIDES
config = {**train_v4.CONFIG, **OVERRIDES, 'resume_path': '$RESUME_ARG'}
print('Starting with resume_path:', config['resume_path'])
train_v4.train(config)
" 2>&1 | tee train_v4_5090.log
else
    python3 -u train_v4_5090.py 2>&1 | tee train_v4_5090.log
fi

echo ""
echo "============================================"
echo "Training complete. Running evaluation..."
echo "============================================"

# Find the latest checkpoint
LATEST_CKPT=$(ls -t checkpoints_v4_5090/dscsp_v4_step_*.pt 2>/dev/null | head -1)
FINAL_CKPT="checkpoints_v4_5090/dscsp_v4_final.pt"

if [ -f "$FINAL_CKPT" ]; then
    CKPT="$FINAL_CKPT"
elif [ -n "$LATEST_CKPT" ]; then
    CKPT="$LATEST_CKPT"
else
    echo "ERROR: No checkpoint found!"
    exit 1
fi

echo "Using checkpoint: $CKPT"

python3 -u evaluate_v4.py \
    --checkpoint "$CKPT" \
    --num_samples 1000 \
    --output eval_v4_5090_results.json \
    2>&1 | tee eval_v4_5090.log

echo ""
echo "============================================"
echo "All done!"
echo "============================================"
echo "Results: eval_v4_5090_results.json"
echo "Training log: train_v4_5090.log"
echo "Eval log: eval_v4_5090.log"
RUNEOF
chmod +x run_5090.sh

echo ""
echo "============================================"
echo "SETUP COMPLETE!"
echo "============================================"
echo ""
echo "Quick Start:"
echo "  ./run_5090.sh                                    # Train from scratch"
echo "  ./run_5090.sh --resume checkpoints_v4/dscsp_v4_step_012000.pt  # Resume"
echo ""
echo "Monitor training:"
echo "  tail -f train_v4_5090.log"
echo "  ls -la checkpoints_v4_5090/"
echo "  ls -la outputs_v4_5090/"
echo ""
echo "Estimated training time: ~1.5 hours for 50K steps"
echo "  (vs ~7.5 hours on RTX 3060)"
echo ""
