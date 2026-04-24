# DSCSP — Megacontext Map

Single-file orientation for agents and humans. Read this first; it tells
you where everything lives, what is current vs. legacy, and which entry
points matter for the most common tasks. Everything here is derived from
the repository state on the `tifs-sprint-2026-04-24` branch.

## 1. What DSCSP is

Diffusion-Stego with Conditional Sampling Paths — a deep image
steganography framework whose cover images come from a **frozen**
`google/ddpm-celebahq-256` DDIM sampler (113.7 M params, not trained).
Only three nets are trainable:

| Component | File | Params | Role |
|---|---|---|---|
| Content-Adaptive Controller | [dscsp/models/controller_v4.py](dscsp/models/controller_v4.py) | 4.39 M | U-Net + two carrier branches; produces the additive delta given image + payload |
| Global / Spatial Extractor | inline in [train_v4.py](train_v4.py) / [train_v5.py](train_v5.py) | 11.31 M | ResNet-18 backbone with spatial_map / global_fc / shift_pool heads |
| SN Discriminator | [dscsp/models/discriminator.py](dscsp/models/discriminator.py) | 2.76 M | SRM-pre-filtered spectral-norm binary classifier |

Legacy tiled-grid Controller lives in [dscsp/models/controller.py](dscsp/models/controller.py)
and is kept only for v1/v2 reproducibility.

## 2. Version timeline

| Version | Training script | Architecture signature | Status |
|---|---|---|---|
| v1 | [train.py](train.py) | Tiled 8×8 grid, no adversary | Legacy |
| v2 | [train_v2.py](train_v2.py) | Tiled + DiffJPEG attack | Legacy |
| v3 | `train_scaling.py` | Capacity probes | Legacy |
| v4 | [train_v4.py](train_v4.py) | Content-adaptive U-Net + SN disc + JPEG-heavy attacks | Primary TIFS line |
| v4 TIFS autopilot | [training_ops/tifs_autopilot.py](training_ops/tifs_autopilot.py) | v4 with 4 named variants × 11 cycles | Source of Table I |
| v5 | [train_v5.py](train_v5.py) | v4 + `RedundantCoder` + SRM feature matching | Supporting ablation line |
| v5 hardening | [finetune_tifs_hardening.py](finetune_tifs_hardening.py) | v5 architecture, resumes cycle 11, tighter perturbation budget, real-libjpeg STE | Current-best PSNR/BER operating point |

## 3. Directory layout (post-2026-04-24 reorg)

```
.
├── MEGACONTEXT.md             ← this file
├── AGENTS.md                  ← mandatory-post-clone steps (HF pull)
├── paper_ieee_tifs_validated.tex  ← CURRENT paper (don't edit archived drafts)
├── requirements.txt
├── pull_artifacts_from_hf.{sh,ps1,py}   ← bootstrap entry points
├── push_checkpoints_to_hf.py            ← remedy for LFS-budget incidents
├── deploy_to_runpod.sh / setup_runpod.sh
│
├── dscsp/                     ← package: models, augmentations, losses, sampling, utils
│   ├── models/controller_v4.py
│   ├── models/discriminator.py
│   ├── models/diffusion_wrapper.py
│   ├── augmentations/differentiable.py   ← DiffJPEG + real_jpeg_ste + AttackLayer
│   ├── losses/composite_loss.py
│   └── utils/{metrics,redundant_coding,visualization}.py
│
├── training_ops/              ← autopilot drivers + run logs
│   ├── tifs_autopilot.py
│   ├── run_tifs_autopilot*.sh
│   └── runs/                  ← per-invocation metadata
│
├── scripts/                   ← paper + repo tooling
│   ├── check_tex.py
│   ├── fill_paper_from_evals.py   ← mechanical [pending]-cell filler
│   ├── generate_paper_figures.py
│   └── generate_tifs_paper_assets.py
│
├── figures/                   ← paper \includegraphics targets (PDF + PNG)
│
├── checkpoints_tifs/          ← v4 autopilot + hardening binaries (LFS / HF)
│   ├── ber_recovery/          ← cycle 1-11 major checkpoints
│   └── hardening/             ← v5 hardening fine-tune
├── checkpoints_v5*/           ← v5 main/clean/quality (real blobs on HF)
├── checkpoints_v2,v3,v4*/     ← legacy per-version checkpoints
│
├── outputs_tifs/              ← autopilot per-cycle progress + MAJOR_LOG.md
├── outputs_tifs_eval/         ← eval JSONs (Table I source; cycle_NNN_*.json + hardening_*.json)
├── outputs_tifs_ablation/     ← latest_ablation.json (variant means)
├── outputs_v5_ablation/       ← ablation_results.json (v5 component drops)
├── outputs_v5_analysis/       ← security_analysis.json (KL / FFT χ² / Gini)
│
├── docs/                      ← narrative + handoff documents (non-load-bearing)
│   ├── HANDOFF_2026-04-24.md
│   ├── NextStepsforTIFS.md
│   ├── PAPER_CLOSEOUT_PLAN_2026-04-23.md (redacted)
│   ├── SECRETS.md             ← how to handle HF / GitHub tokens
│   ├── PROJECT_CONTEXT_MANUAL_2026-04-18.md
│   ├── Idea.md / Upgrade.md
│   └── AIcontent{,v3}.md
│
├── archive/                   ← history-only, not referenced by any current script
│   ├── old_papers/            ← paper.tex, paper_ieee.tex, paper_ieee_v3.tex, paper_ieee_diff.tex, latest_paper_v5.tex
│   ├── old_results/           ← ablation_results.json (root-level dup), eval_results.json, scaling_*.json, steganalysis_results.json
│   └── reports/               ← AI_Writing_Report.{pdf,docx}, tmp extracts
│
├── graphify-out/              ← code-graph report (2026-04-20 snapshot; tool not installed on this pod)
└── .tifs_data/                ← STL10 cache from torchvision; gitignored content
```

## 4. Entry points by task

| Task | Command |
|---|---|
| Bootstrap a fresh pod | `pip install -r requirements.txt && bash pull_artifacts_from_hf.sh .` |
| Retrain v4 TIFS cycles 1–8 | `python -u training_ops/tifs_autopilot.py --max-cycles 8 --steps-per-cycle 2500 …` (cheat sheet in docs/PAPER_CLOSEOUT_PLAN_2026-04-23.md §B) |
| Retrain cycles 9–11 | Same invocation with `--max-cycles 11 --steps-per-cycle 1200` |
| Run hardening fine-tune | `python finetune_tifs_hardening.py` (resumes cycle 11 automatically) |
| Full fast eval on any ckpt | `python evaluate_tifs.py --checkpoint CKPT --fast --real-attacks --multiple-detectors --output outputs_tifs_eval/X.json` |
| 1000-sample eval | Same, drop `--fast`, add `--num-samples 1000` |
| Top-3 SVD figure | `python generate_svd_figure.py --checkpoint CKPT --n-samples 100` |
| Fill paper `[pending]` cells | `python scripts/fill_paper_from_evals.py --paper paper_ieee_tifs_validated.tex` |
| Mirror real checkpoints to HF | `HF_TOKEN=... python push_checkpoints_to_hf.py` (skips 134-byte stubs automatically) |

## 5. Key architectural invariants

1. **Images are in `[-1, 1]`**, never `[0, 1]`. Controller outputs a
   delta in the same range, bounded by `max_perturbation`.
2. **Payload is 256 physical bits**. v5 optionally maps 128 message
   bits → 256 physical via `RedundantCoder`; v4 and hardening use
   256 == 256.
3. **Frozen UNet** — the diffusion sampler is never unfrozen. Do
   not add it to any optimizer.
4. **Image pool** of 128 DDIM covers is refreshed every 2000 steps;
   regenerating a pool takes ~2–3 minutes.
5. **Eval seeds drift.** The 2026-04-24 retrain reproduced cycle 9
   clean BER ≈ `5.9×10⁻⁵` but did NOT reproduce the previous cycle
   11 AUC of 0.6915 (retrain landed at 0.94). Treat the paper's
   AUC numbers as derived from the current `outputs_tifs_eval/`
   snapshot, not from any prior repository state.
6. **Detectability gate is AUC ≤ 0.70.** No checkpoint in the
   current repository passes this on a ResNet-18/34 ensemble.
7. **Crop-resize BER stays near 0.5** on every checkpoint — the
   dominant unresolved limitation; not a training-signal problem
   but an extractor-alignment problem.

## 6. Known gotchas

- `train_v5.quick_eval` previously called `coder.encode(msg)`
  unconditionally. Fixed in 2026-04-24 to handle `coder=None` when
  `use_redundant_coding=False`. Hardening runs would otherwise
  crash on the final eval without saving the final checkpoint.
- `shift_pool` extractor head interacts poorly with the 256-bit
  non-redundant configuration (max-pool scrambles per-bit evidence).
  For hardening use `extractor_mode="spatial_map"`; for crop-robust
  experiments either keep redundancy on or shrink the pool kernel.
- The GitHub LFS budget on this repo is exhausted. Do not rely on
  `git lfs pull` to restore checkpoints; use `pull_artifacts_from_hf.sh`.
  Any new .pt blob produced by a training run must also be pushed
  via `push_checkpoints_to_hf.py`.

## 7. What's in HF right now (2026-04-24)

`OneAutumnLeef/dscsp-checkpoints` (dataset):
- `checkpoints_tifs/ber_recovery/dscsp_v4_final.pt` (cycle 11 end)
- `checkpoints_tifs/ber_recovery/major_cycle_011_robust_balance.pt`
- `checkpoints_tifs/hardening/dscsp_v5_final.pt` (step 27000)
- `checkpoints_tifs/hardening/dscsp_v5_step_027000.pt`
- 11 × `outputs_tifs_eval/cycle_0NN_*.json`
- `outputs_tifs_eval/cycle_011_detectors.json`
- `outputs_tifs_eval/hardening_fast.json`
- `outputs_tifs_eval/hardening_1000.json`
- `figures/svd_singular_vectors.{pdf,png}` + `figures/svd_singular_values.json`
- `outputs_v5_ablation/ablation_results.json`

`OneAutumnLeef/dscsp-full-backup` (dataset): full v5 training history
(52 real .pt blobs in `checkpoints_v5/**`). All `checkpoints_tifs/**`
entries there are 134-byte stubs — do not trust the TIFS line via the
full-backup repo; use `dscsp-checkpoints` for TIFS artifacts.

## 8. Current paper state

[paper_ieee_tifs_validated.tex](paper_ieee_tifs_validated.tex) is the
authoritative draft. Older drafts in `archive/old_papers/` are
reference only and will not compile (e.g. `latest_paper_v5.tex`
references figures that were renamed).

Zero `[TBD]` / `[pending]` cells, zero orphan refs, tables driven from
the `outputs_tifs_eval/*.json` files and `outputs_v5_ablation/ablation_results.json`.
Key referenced figures:

- `figures/system_overview.pdf` → Fig. 1
- `figures/architecture_detail.pdf` → Fig. 2
- `figures/autopilot_cycles.pdf` → Fig. 3
- `figures/ablation_summary.pdf` → Fig. 4
- `figures/svd_singular_vectors.pdf` → Fig. 5 (new, 2026-04-24)
- `figures/security_analysis_v5.pdf` → Fig. 6
- `figures/crop_failure_analysis.pdf` → Fig. 7
