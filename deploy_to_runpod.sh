#!/bin/bash
# ============================================================================
# Deploy DSCSP project to RunPod
# ============================================================================
# Run this from your LOCAL machine (Windows/Git Bash or WSL) to transfer
# all project files to a RunPod instance.
#
# Usage:
#   ./deploy_to_runpod.sh <RUNPOD_SSH>
#   ./deploy_to_runpod.sh root@xxx.xxx.xxx.xxx
#   ./deploy_to_runpod.sh root@xxx.xxx.xxx.xxx --with-checkpoint
#
# Prerequisites:
#   - RunPod pod running with SSH access
#   - SSH key configured (or password ready)
# ============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: ./deploy_to_runpod.sh <SSH_TARGET> [--with-checkpoint]"
    echo ""
    echo "Example:"
    echo "  ./deploy_to_runpod.sh root@123.45.67.89"
    echo "  ./deploy_to_runpod.sh root@123.45.67.89 --with-checkpoint"
    echo ""
    echo "SSH target format: user@host (from RunPod SSH connection info)"
    exit 1
fi

SSH_TARGET="$1"
WITH_CHECKPOINT="$2"
REMOTE_DIR="/workspace/dscsp"

echo "============================================"
echo "DSCSP → RunPod Deployment"
echo "============================================"
echo "Target: $SSH_TARGET:$REMOTE_DIR"
echo ""

# ─── 1. Create remote directory ──────────────────────────────────────────────
echo "[1/4] Creating remote directory..."
ssh "$SSH_TARGET" "mkdir -p $REMOTE_DIR"

# ─── 2. Transfer project files (excluding large/temp files) ─────────────────
echo "[2/4] Transferring project files..."
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.claude' \
    --exclude 'outputs*' \
    --exclude 'checkpoints*' \
    --exclude 'eval_clean' \
    --exclude 'eval_stego' \
    --exclude 'tmpclaude*' \
    --exclude '*.pt' \
    --exclude '*.png' \
    --exclude '*.log' \
    --exclude '%OUTDIR%' \
    --exclude '*.pdf' \
    --exclude '*.docx' \
    --exclude '*.aux' \
    --exclude '*.bbl' \
    --exclude '*.blg' \
    --exclude '*.out' \
    --exclude '*.synctex*' \
    ./ "$SSH_TARGET:$REMOTE_DIR/"

echo "  Project files transferred."

# ─── 3. Optionally transfer latest checkpoint ───────────────────────────────
if [ "$WITH_CHECKPOINT" == "--with-checkpoint" ]; then
    echo "[3/4] Transferring latest checkpoint..."

    # Find latest v4 checkpoint
    LATEST=$(ls -t checkpoints_v4/dscsp_v4_step_*.pt 2>/dev/null | head -1)

    if [ -n "$LATEST" ]; then
        ssh "$SSH_TARGET" "mkdir -p $REMOTE_DIR/checkpoints_v4"
        echo "  Uploading $LATEST (~210MB)..."
        rsync -avz --progress "$LATEST" "$SSH_TARGET:$REMOTE_DIR/checkpoints_v4/"
        echo "  Checkpoint transferred."
        CKPT_NAME=$(basename "$LATEST")
    else
        echo "  No v4 checkpoints found locally. Skipping."
    fi
else
    echo "[3/4] Skipping checkpoint transfer (use --with-checkpoint to include)"
fi

# ─── 4. Run setup on remote ─────────────────────────────────────────────────
echo "[4/4] Running setup on RunPod..."
ssh "$SSH_TARGET" "cd $REMOTE_DIR && chmod +x setup_runpod.sh && ./setup_runpod.sh"

echo ""
echo "============================================"
echo "DEPLOYMENT COMPLETE!"
echo "============================================"
echo ""
echo "SSH into your RunPod:"
echo "  ssh $SSH_TARGET"
echo ""
echo "Start training:"
echo "  cd $REMOTE_DIR"

if [ -n "$CKPT_NAME" ]; then
    echo "  ./run_5090.sh --resume checkpoints_v4/$CKPT_NAME"
else
    echo "  ./run_5090.sh"
fi

echo ""
echo "Monitor:"
echo "  tail -f $REMOTE_DIR/train_v4_5090.log"
echo ""
echo "Download results when done:"
echo "  scp $SSH_TARGET:$REMOTE_DIR/eval_v4_5090_results.json ."
echo "  scp $SSH_TARGET:$REMOTE_DIR/checkpoints_v4_5090/dscsp_v4_final.pt checkpoints_v4_5090/"
echo ""
