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

## Pending (in order)
1. [ ] Analyze stopped 5090 run artifacts (`train_v4_5090.log`, `checkpoints_v4_5090/`, `outputs_v4_5090/`) and decide on next config only if user wants to resume experimentation later
2. [ ] Evaluate v4 only if a satisfactory final checkpoint is produced in a future run
3. [ ] Analyze v4 results against ALL target metrics
4. [ ] If metrics not met: iterate (adjust max_perturbation, loss weights, JPEG schedule, architecture)
5. [ ] Capacity scaling curve (64, 128, 256, 384 bits)
6. [ ] Baseline comparison table (v4 vs literature)
7. [ ] Update paper_ieee.tex with v4 results
8. [ ] Final paper review and polishing

## Current Saved State
- Stopped 5090 run artifacts preserved in `train_v4_5090.log`, `checkpoints_v4_5090/`, and `outputs_v4_5090/`
- Saved checkpoints available at steps 2000, 4000, and 6000
- Latest logged step before interruption: ~7200 (not checkpointed)
- `checkpoints_v4_5090/*.pt` are now intended to be versioned via Git LFS rather than normal Git
