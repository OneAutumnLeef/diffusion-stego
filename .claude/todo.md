# DSCSP Todo List

## Completed
- [x] Implement ContentAdaptiveController (U-Net, ~3.5M params) in controller_v4.py
- [x] Create train_v4.py (JPEG-heavy, spectral norm disc, spectral loss, 50K steps)
- [x] Create evaluate_v4.py (1000+ images, FID, steganalysis AUC, JPEG sweep)
- [x] V4 training on RTX 3060 — stopped at step 16000/50000 (32%)
- [x] Prepare RunPod 5090 deployment (setup_runpod.sh, deploy_to_runpod.sh, train_v4_5090.py)
- [x] Clean project directory (removed ~300 temp files, legacy v1/v2/v3 scripts/checkpoints)
- [x] Set up RunPod 5090 environment (`pip install -r requirements.txt`)
- [x] Fix stale `dscsp.models` package exports after legacy cleanup so fresh-host v4 imports work
- [x] Verify RTX 5090 + CUDA + v4 training/eval imports on RunPod
- [x] Start from-scratch v4 5090 training run (`python -u train_v4_5090.py 2>&1 | tee train_v4_5090.log`)
- [x] Stop 5090 training run on user request and preserve artifacts on disk
- [x] Set up Git LFS tracking for `checkpoints_v4_5090/*.pt` and stage saved checkpoints for GitHub
- [x] Diagnose failed 5090 run (BER stuck ~0.50): root cause was simultaneous aggressive scaling (Lesson 13)
- [x] Fix train_v4_5090.py: conservative overrides (batch_size=8, base_channels=48, proven LRs)
- [x] Add signal handler to train_v4.py (SIGINT/SIGTERM → emergency checkpoint, Lesson 14)
- [x] Add TF32 optimization to train_v4_5090.py startup
- [x] Add per-step progress.json for stdout-independent monitoring (Lesson 5)

## Pending (in order)
1. [ ] Launch fresh conservative 5090 run: `python -u train_v4_5090.py 2>&1 | tee train_v4_5090.log`
2. [ ] Monitor via `outputs_v4_5090/progress.json` and checkpoints; expect ~0.20s/step → ~2.8 hrs for 50K steps
3. [ ] After run: evaluate with `python evaluate_v4.py --checkpoint checkpoints_v4_5090/dscsp_v4_final.pt`
4. [ ] Analyze v4 results against ALL target metrics
5. [ ] If metrics not met: iterate (adjust max_perturbation, loss weights, JPEG schedule, architecture)
6. [ ] Capacity scaling curve (64, 128, 256, 384 bits)
7. [ ] Baseline comparison table (v4 vs literature)
8. [ ] Update paper_ieee.tex with v4 results
9. [ ] Final paper review and polishing

## Current Saved State
- Previous failed 5090 run artifacts in `checkpoints_v4_5090/` (steps 2000, 4000, 6000)
  - Those checkpoints used base_channels=64; incompatible with corrected config (base_channels=48)
  - Start fresh — do NOT resume from these
- Fixed `train_v4_5090.py` uses conservative overrides; ready to launch
