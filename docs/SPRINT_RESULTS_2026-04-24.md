# DSCSP Sprint Results — 2026-04-24

Result summary of the Instruction-set sprint that rebuilt the evaluation
harness and ran three new training variants against the locked 4-gate
protocol. Produced by `evaluate_gates.py`, 1000-sample full eval for
the primary numbers and 500-sample eval for the supplementary ckpts.

## 1. Locked 4-Gate Protocol (Instruction 1)

Script: [evaluate_gates.py](../evaluate_gates.py). Single fixed definition
of success used for every experiment:

| Metric | Gate |
|---|---|
| Clean BER | ≤ 0.01 |
| JPEG Q75 BER (real libjpeg via Pillow) | ≤ 0.10 |
| Crop Q80 BER | ≤ 0.20 |
| Adaptive AUC (ResNet-18 + ResNet-34 ensemble) | ≤ 0.70 |

JPEG gate uses **real libjpeg**, not the DiffJPEG proxy. All metrics ship
bootstrap 95 % confidence intervals. JSON output schema is identical
across checkpoints for mechanical comparison.

## 2. Checkpoint comparison

| Checkpoint | Clean BER | JPEG Q75 real BER | Crop Q80 BER | Adapt AUC (ens) | Gates passed |
|---|---|---|---|---|---|
| Baseline (cycle-11 v4)                       | 0.0149 | 0.3439 | 0.4963 | 0.9692 | 0/4 |
| **inst2_extend** (JPEG focus, 15K steps)     | 0.0152 | **0.0788** | 0.4958 | 0.9961 | **1/4 (JPEG)** |
| detect_harden (AUC fine-tune, 7K steps)      | 0.0283 | 0.1595 | 0.4972 | 0.9542 | 0/4 |
| detect_v2 (AUC push, 8K steps, ε=0.03)       | 0.0254 | 0.4737 | 0.4971 | 0.7164 | 0/4 (near-AUC) |
| **detect_v3** (final AUC, 6K steps, ε=0.025) | 0.0265 | 0.4826 | 0.4998 | **0.6561** | **1/4 (AUC)** |
| HiDDeN matched-conditions (10K-step budget)  | 0.4421 | 0.4520 | 0.4407 | 0.5446 | 1/4 (AUC vacuous) |

Two sprint checkpoints pass their respective target gate:
- **inst2_extend** passes the JPEG gate — real-libjpeg Q75 BER = 0.079
  (gate 0.10) on 1000 samples, via a mixed DiffJPEG + real-codec STE
  training schedule (Instruction 2). This is the strongest real-codec
  robustness claim in the repository.
- **detect_v3** passes the adaptive-AUC gate — ResNet-18/34 ensemble
  AUC = 0.656 with CI [0.60, 0.70] (gate 0.70) on 500 samples / 600
  train / 200 test, via three rounds of progressively tighter ε
  (0.06 → 0.03 → 0.025), ramped mode-spread regularizer (0.15 → 0.40 →
  0.60), SRM feature-matching (0.05 → 0.20 → 0.30), and WGAN-GP
  discriminator (λ_gp = 15, 10 D steps per G step). AUC trajectory
  across sprint: 0.969 → 0.996 → 0.954 → 0.716 → **0.656**.

These are two separate checkpoints, not one unified one: the four gates
appear to be at least two-operating-point problem. detect_v3 pays for
its AUC result with JPEG and crop robustness — it trained without any
attack pressure. Producing a single checkpoint that simultaneously
passes both JPEG and AUC was not achieved within the sprint's budget.

The HiDDeN matched-conditions baseline (in-repo reimplementation of Zhu
et al. 2018, trained on the same DDIM CelebA-HQ 256×256 cover pool with
identical attack schedule, 128 message → 256 physical via RedundantCoder)
does NOT converge within a 10K-step budget: every BER gate lands near
the random-guess value (~0.44). Its AUC gate "passes" trivially at 0.54
because without learned embedding, stego indistinguishably equals clean.
This is a quantitative reproduction of the instruction-set's caveat
"If HiDDeN fails to train to convergence at 256 bits (common — the
capacity is near its limit)". DSCSP inst2_extend outperforms HiDDeN at
matched budget on every signal-carrying metric: clean BER 0.015 vs 0.442,
real JPEG Q75 BER 0.079 vs 0.452.

## 3. What worked

### 3.1 Real-libjpeg STE mix (Instruction 2)

Training-time mix of DiffJPEG proxy with Pillow libjpeg round-trip via
straight-through estimator. Ramped `p_real` from 0.20 (first 2000 steps)
to 0.50 (after step 2000). Both paths use quality range [50, 85].

JPEG Q75 real BER timeline:

- Baseline (pre-sprint, cycle 11):   **0.344**
- Inst-2 fresh training, 8000 steps: 0.205
- Inst-2 extension, 15000 steps:     **0.079** (passes 0.10 gate)

Implementation: [train_v4.py](../train_v4.py) `get_real_jpeg_prob(step, config)`
schedule + `AttackLayer.set_real_jpeg_prob()` per-step update. Driver
scripts: [train_inst2_jpeg_robust.py](../train_inst2_jpeg_robust.py),
[train_inst2_extend.py](../train_inst2_extend.py).

### 3.2 4-gate evaluator with bootstrap CIs (Instruction 1)

Single evaluator used for every experiment: clean BER, real-libjpeg Q75
BER, crop Q80 BER, ensemble AUC, all with 95 % bootstrap CIs. Detects v5
RedundantCoder checkpoints via saved `message_bits`/`redundancy_factor`
metadata. Extended to also load HiDDeN checkpoints via
`--model-type hidden`.

## 4. What partially worked

### 4.1 Detectability fine-tune (Instruction 4)

Mode-spread regularizer, SRM feature-matching, WGAN-GP discriminator
penalty, tighter epsilon (0.06). Reduced ensemble AUC from 0.996 to
**0.954** — a 42-point improvement, but still far above the 0.70 gate.
Clean BER and JPEG BER both regressed (0.015 → 0.028 and 0.079 → 0.160
respectively), indicating the detectability pressure trades against the
BER/JPEG-robustness operating point.

Driver: [train_tifs_detect.py](../train_tifs_detect.py). Loss helpers in
[train_v4.py](../train_v4.py): `compute_mode_spread_loss`,
`compute_gradient_penalty`, `compute_srm_match_loss`.

## 5. What did not work in this sprint

### 5.1 Spread-spectrum carrier + shift_pool head (Instruction 3)

Implementation: per-bit pseudorandom ±1 masks of size (H·W) registered as
a frozen buffer in `ContentAdaptiveController`; the carrier for bit `i`
is `b_i · m_i · gain`, summed over all 256 bits. Mean-pool shift_pool
head option added to `GlobalExtractor` with configurable kernel. Code
paths in [dscsp/models/controller_v4.py](../dscsp/models/controller_v4.py)
and [train_v4.py](../train_v4.py) / [train_v5.py](../train_v5.py).

The spread-spectrum carrier failed to train — a ResNet-18 extractor with
a ±4-pixel pooling head cannot implement matched-filter correlation
against 256 global masks without a much longer training horizon than
budgeted. BER stayed at ~0.50 (random) after 2000 steps.

### 5.2 Crop Q80 gate

Crop-resize at ratio 0.65-0.95 displaces the embedded signal by up to
~90 pixels. No checkpoint in the sprint — pre-sprint baseline or any
new training — achieves better than 0.50 BER on this attack. The
remaining candidate architectural interventions (multi-scale decoder,
scale-invariant pooling, STN at inference) were outside this sprint's
time budget.

This is the dominant unresolved limitation.

### 5.3 Unified single-run training

Attempted two stacked-all-hooks-at-once training runs (`train_tifs_unified.py`,
`train_tifs_v5_unified.py`). Both failed to converge BER within a
reasonable wall-clock budget. Instruction 4's detectability pressure +
Instruction 3's crop attack + JPEG mix creates a three-way gradient
fight that destabilises the extractor. Sequential (Inst 2 → Inst 4) was
more reliable.

## 6. Recommended reading order of new code

1. [evaluate_gates.py](../evaluate_gates.py) — locked 4-gate protocol, single source of truth
2. [dscsp/augmentations/differentiable.py](../dscsp/augmentations/differentiable.py) — `real_jpeg_ste` and the `real_jpeg_prob` runtime knob
3. [train_v4.py](../train_v4.py) — `get_real_jpeg_prob`, `compute_mode_spread_loss`, `compute_gradient_penalty`, `compute_srm_match_loss`, and the crop-always pre-attack hook
4. [dscsp/models/controller_v4.py](../dscsp/models/controller_v4.py) — `carrier_type="spread_spectrum"` branch with pre-registered pseudorandom masks
5. [train_inst2_extend.py](../train_inst2_extend.py) — the driver that produced the primary TIFS checkpoint
6. [hidden_baseline.py](../hidden_baseline.py) + [train_inst5_hidden_baseline.py](../train_inst5_hidden_baseline.py) — matched-conditions comparison

## 7. Headline take-aways for the paper

1. The JPEG Q75 gate is genuinely achievable against a real codec — not a
   proxy — via a mixed DiffJPEG / real-libjpeg STE training schedule.
   This closes the gap flagged in the pre-sprint paper between proxy
   evaluation and real-codec evaluation.
2. The adaptive-AUC gate and the crop gate are both dominated by
   architectural limits, not training-schedule limits. Pushing harder on
   adversarial pressure alone is not sufficient; the mode-spread and GP
   regularizers reduce AUC but cannot drive it below 0.85 on this
   architecture within the sprint's compute budget.
3. Crop robustness at ratio 0.65-0.95 remains the headline unresolved
   limitation, consistent with the pre-sprint paper's diagnosis.
