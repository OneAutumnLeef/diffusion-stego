# DSCSP Todo List

## Completed
- [x] Implement ContentAdaptiveController (U-Net, ~3.5M params) in controller_v4.py
- [x] Create train_v4.py (JPEG-heavy, spectral norm disc, spectral loss, 50K steps)
- [x] Create evaluate_v4.py (1000+ images, FID, steganalysis AUC, JPEG sweep)
- [x] V4 training on RTX 3060 — stopped at step 16000/50000 (32%)
- [x] Prepare RunPod 5090 deployment (setup_runpod.sh, deploy_to_runpod.sh, train_v4_5090.py)
- [x] Clean project directory (removed ~300 temp files, legacy v1/v2/v3 scripts/checkpoints)

## Pending (in order)
1. [ ] Deploy to RunPod 5090 and resume training from step 16000 checkpoint
2. [ ] Complete v4 training (50K steps) on RunPod
3. [ ] Evaluate v4: `python evaluate_v4.py --checkpoint checkpoints_v4/dscsp_v4_final.pt --num_samples 1000`
4. [ ] Analyze v4 results against ALL target metrics
5. [ ] If metrics not met: iterate (adjust max_perturbation, loss weights, JPEG schedule, architecture)
6. [ ] Capacity scaling curve (64, 128, 256, 384 bits)
7. [ ] Baseline comparison table (v4 vs literature)
8. [ ] Update paper_ieee.tex with v4 results
9. [ ] Final paper review and polishing
