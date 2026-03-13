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
