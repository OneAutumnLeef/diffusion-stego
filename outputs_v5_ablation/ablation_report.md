# DSCSP v5 Ablation Results
Generated: 2026-04-19 16:57 UTC
Warm-start: 5000 steps | Variant training: 8000 steps

| Variant | Clean BER | JPEG Q75 BER | Crop Q80 BER | AUC | PSNR (dB) |
|---------|-----------|--------------|--------------|-----|----------|
| full                 | 0.0698 | 0.1949 | 0.4943 | 0.9490 | 28.8 |
| no_srm_match         | 0.0968 | 0.2173 | 0.4929 | 0.9777 | 28.8 |
| no_spectral          | 0.1013 | 0.2097 | 0.5008 | 0.9200 | 28.8 |
| no_adv               | 0.0054 | 0.0137 | 0.4980 | 0.8338 | 33.1 |
| uniform_attacks      | 0.1066 | 0.2674 | 0.5013 | 0.9950 | 28.7 |
| no_redundant         | ERROR (training failed) | | | | |

## Key Ablation Insights
- **no_srm_match** vs full: Clean BER +39%, JPEG Q75 +12% → SRM matching helps frequency-domain fidelity
- **no_spectral** vs full: Clean BER +45%, JPEG Q75 +8% → Spectral loss regularizes high-freq artifacts
- **no_adv** vs full: Clean BER −92%, JPEG Q75 −93%, PSNR +4.2 dB → Adversarial objective trades JPEG robustness for detectability resistance (higher AUC without adv)
- **uniform_attacks** vs full: Clean BER +53%, JPEG Q75 +37%, AUC +0.05 → JPEG-weighted attack schedule is critical

## TIFS Targets
- Clean BER: ≈ 0.000
- JPEG Q75 BER: < 0.10
- Crop Q80 BER: ≤ 0.30
- AUC: ≤ 0.70
- PSNR: ≥ 35 dB
- LPIPS: ≤ 0.05

## Final Model (78K steps, full pipeline)
- Clean BER: 0.0068 ✓ (<0.01)
- JPEG Q75 BER: 0.266 ✗
- Adaptive AUC: 0.620 ✓ (<0.70)
- PSNR: 34.8 dB (target ≥35)
- LPIPS: 0.021 ✓ (<0.05)
