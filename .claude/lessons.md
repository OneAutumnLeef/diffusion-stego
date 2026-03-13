# DSCSP Lessons Learned

## Lesson 1: Tiled Grid is Fundamentally Detectable (AUC)
**Problem**: v2 (lambda_adv=0.1) got AUC=0.9986. v3 (lambda_adv=0.5, max_pert=0.15, 3x disc steps) got AUC=1.0.
**Root Cause**: The 8x8 tiled pattern (64 cells of 32x32 pixels) creates hard boundaries every 32 pixels that any classifier trivially detects via high-pass filtering. No amount of adversarial training can hide periodic spatial artifacts.
**Lesson**: Content-independent, spatially-periodic perturbation CANNOT achieve low AUC. Must use content-adaptive encoding.

## Lesson 2: Lower max_perturbation Hurts Robustness
**Problem**: v3 (max_pert=0.15) had HIGHER PSNR (44.3 vs 40.9) but WORSE JPEG robustness (Q80 BER: 0.21 vs 0.11) and crop robustness.
**Root Cause**: Smaller perturbation amplitude = weaker signal = easier destroyed by attacks.
**Lesson**: Fundamental PSNR vs robustness tradeoff. For JPEG Q75 BER < 0.01, need sufficient signal strength.

## Lesson 3: Orthogonality Regularization is Critical
**Finding**: no_ortho ablation: PSNR dropped from 35.1 to 24.3 (-10.8 dB!)
**Lesson**: Orthogonality loss is the MOST IMPORTANT regularizer for quality. Without it, bit patterns collapse.

## Lesson 4: Curriculum Has Nuanced Effect
**Finding**: no_curriculum variant: PSNR was 37.6 vs baseline 35.1 (+2.5 dB)
**Lesson**: Curriculum helps robustness convergence but slightly hurts final visual quality.

## Lesson 5: conda run Buffers Output
**Workaround**: Monitor via output files (checkpoints, result JSONs, image timestamps) rather than stdout.

## Lesson 6: JPEG STE Gradient Quality
**Problem**: DiffJPEG STE at low quality (Q50-Q75) has noisy/biased gradients.
**Implication**: Train with JPEG at very high probability so model adapts despite noisy gradients. Content-adaptive embedding should concentrate signal in low/mid frequencies that survive JPEG.

## Lesson 7: v4 Training Speed
**Observation**: v4 at batch=4 runs at ~0.55s/step (measured, slower than initial 0.36s estimate — likely due to 3 disc steps after step 1000).
50K steps ≈ 7.5 hours on RTX 3060.

## Lesson 8: Memory Estimation for Model Design
**Rule of thumb** (256x256, RTX 3060 6GB):
- Simple parameter-based models: ~200MB → batch=8
- U-Net (~5M params): ~1.5-2GB → batch=4-6
- ResNet-18 extractor: ~1GB
- Need ~1GB headroom for optimizer states

## Lesson 9: Ablation Process Stability
**Problem**: Ablation study (PID 1468) died silently after completing 4/7 variants.
**Lesson**: Long-running multi-variant training should checkpoint progress and be restartable. Always verify process is still alive before assuming it's computing.

## Lesson 10: Fresh-Host Import Verification Catches Cleanup Regressions
**Problem**: After legacy cleanup, `dscsp/models/__init__.py` still imported removed `controller.py`, which broke `train_v4.py` imports on the new RunPod host before training could start.
**Lesson**: After deleting legacy modules, always run import smoke tests for `train_v4.py`, `train_v4_5090.py`, and `evaluate_v4.py` on a fresh environment. Package-level `__init__` files can fail even when direct module paths still look valid.

## Lesson 11: First-Run Startup on RunPod Includes Expected Weight Downloads
**Observation**: The first fresh-host launch downloaded both Hugging Face DDPM weights and LPIPS VGG16 weights before step 1. Startup looked idle for tens of seconds but was normal.
**Lesson**: On new instances, use `train_v4_5090.log` to distinguish expected one-time weight downloads from an actual hang.

## Lesson 12: Early RTX 5090 Throughput Changes at Step 1000
**Observation**: The from-scratch 5090 config (`batch_size=16`, `base_channels=64`) ran at ~0.29s/step before adversarial training, then ~0.34-0.35s/step just after step 1000 when 3 discriminator updates kicked in. Early ETA was roughly 4.0-4.8 hours for 50K steps, excluding initial model downloads.
**Lesson**: Use post-step-1000 speed, not warmup speed, when estimating wall-clock time and choosing checkpoint/monitoring cadence for the 5090 run.

## Lesson 13: The Aggressive 5090 Override Can Stall Message Learning
**Observation**: The from-scratch 5090 run with `batch_size=16`, `base_channels=64`, higher learning rates, and JPEG-heavy training never escaped the random-guess regime. By the saved step-6000 checkpoint and interrupted log at ~step 7200, BER was still ~0.49-0.51 and message BCE was still ~0.693.
**Lesson**: More GPU headroom does not justify scaling model width, learning rate, and attack intensity simultaneously. For this project, preserve the proven optimization dynamics first and use extra VRAM primarily for safer batch-size increases or monitoring convenience.

## Lesson 14: Interrupted Runs Only Preserve the Last Checkpoint
**Observation**: The user-requested stop was sent during training, but only checkpoints at steps 2000, 4000, and 6000 were recoverable afterward. Later progress appeared in `train_v4_5090.log` up to ~step 7200 and preview images up to step 7000, but there was no checkpoint beyond 6000.
**Lesson**: Without explicit signal handling, manual interruption preserves logs and images but loses in-memory optimization state after the last checkpoint. If frequent manual stopping is expected, add signal-triggered checkpoint saving or shorten checkpoint intervals.

## Lesson 15: Run Artifacts Need Git LFS Before GitHub Push
**Observation**: Each saved 5090 checkpoint was about 246 MB, which exceeds GitHub's normal 100 MB file limit. Regular Git push is unsuitable for these `.pt` artifacts.
**Lesson**: Track checkpoint binaries with Git LFS from the start (`checkpoints_v4_5090/*.pt`) and keep ordinary Git for code, notes, and lightweight metadata only.
