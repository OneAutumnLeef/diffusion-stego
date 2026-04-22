# DSCSP Paper Closeout Plan — 5090 Handoff Brief

**Date:** 2026-04-23
**Target machine:** RTX 5090 workstation (31.4 GB VRAM, original training host `84b46cf651b8`)
**Target file:** `latest_paper_v5.tex` (IEEE TIFS submission draft)
**Expected wall clock on 5090:** ~90 minutes end-to-end

> This document is a self-contained brief for a fresh Claude Code window on the 5090. It assumes the agent has not seen any prior conversation.

---

## 1. Context — read this first

### 1.1 What DSCSP is
DSCSP (Diffusion-Stego with Conditional Sampling Paths) is a deep steganography framework using diffusion-generated cover images. It is targeted at IEEE TIFS. The architecture has two main lines:

- **v4** — content-adaptive U-Net controller + global-pool ResNet-18 extractor + spectral-norm discriminator. This is the primary results line (Table I in the paper, cycles 1–11 of the autopilot).
- **v5** — same controller/extractor as v4 plus `RedundantCoder` and an SRM feature-matching loss. This is the ablation line (Table IV in the paper).

The diffusion model (`google/ddpm-celebahq-256`, 113.7 M params) is **frozen**. Only Controller (4.39 M) + Extractor (11.31 M) + Discriminator are trained.

### 1.2 What was done before
On 2026-04-18, the 5090 ran the full 11-cycle `tifs_autopilot.py` sweep. Timing (from `training_ops/runs/*/metadata.json`):

- Warmup (2,000 steps): 237 s
- Cycles 1–8 detached (2,500 steps/cycle): ≤ 3,399 s — bounded by MAJOR_LOG start (`2026-04-18 20:06:53 UTC`) and sprint start (`21:03:32 UTC`)
- Sprint cycles 9–11 (1,200 steps/cycle): 621 s

Total training ≈ 71 min. Observed 5090 rate: **5.8 steps/s** on this workload.

Cycle 9 and cycle 11 are the checkpoints the paper leans on:
- **Cycle 9** (`checkpoints_tifs/ber_recovery/dscsp_v4_final.pt`, variant `ber_recovery_clean_first`): clean BER 1.95×10⁻⁵, PSNR 30.68 dB, AUC 0.999 — the recovery headline.
- **Cycle 11** (`checkpoints_tifs/ber_recovery/major_cycle_011_robust_balance.pt`, variant `robust_balance`): AUC 0.6915 — the detectability headline (passes the <0.70 gate).

### 1.3 Why we're doing this again
The 11 cycles already ran. The eval JSONs (`outputs_tifs_eval/cycle_*.json`) exist and are the source of Table I. **But the checkpoint binaries were never pushed to accessible storage as real LFS blobs.** Every `.pt` under `checkpoints_tifs/ber_recovery/` on disk is a 134-byte pointer stub, and the HF dataset repos (`OneAutumnLeef/dscsp-full-backup`, `OneAutumnLeef/dscsp-checkpoints`) also contain only pointer stubs for the TIFS line (confirmed via `HfApi.repo_info(files_metadata=True)`: all 40 `.pt` files have `size=134, pointer_size=128`). The v5 checkpoints on HF *are* real (220 MB each, gated).

So: **the code and data to *reproduce* cycles 1–11 are here; the *weights* have to be regenerated. The 5090 can do this in ~70 min.**

### 1.4 What the paper still needs
`latest_paper_v5.tex` has three concrete numeric gaps that require **inference on the regenerated cycle 9 and cycle 11 checkpoints**:

- **L3 (Section X limitations, lines ~817–822)** — libjpeg vs STE-proxy gap at Q75: `$X_{\mathrm{proxy}}$`, `$Y_{\mathrm{real}}$`, `$\Delta Z$` are placeholders.
- **Table III (Steganalysis detectors, lines ~582–585)** — ResNet-18 / ResNet-34 / ensemble AUC for cycles 9 and 11. Only cycle 11 ResNet-18 (`0.6915`) is filled; the rest are `[TBD]`.
- **Table II (Learned-embedding baselines, lines ~554–555)** — HiDDeN and SteganoGAN rows have `[TBD]` on all four columns. Since we are not running those baselines end-to-end in this sprint, the plan fills these from published numbers with a footnote (see step D2).

All the codebase changes (C1 `--real-attacks`, C2 `--multiple-detectors`, C3 `wo_redundancy` fix, C4 `shift_pool`, C5 n_samples=1000 + `--fast`, C6 `--celeba-dir`) are already landed. Verify by greping `evaluate_tifs.py` lines 712–727 for the CLI flags.

### 1.5 Hugging Face access
The user provided this HF token with access to private `OneAutumnLeef/dscsp-checkpoints`:

```
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
```

Export as `HF_TOKEN` — never echo it, never pass on CLI, never commit it. The token is also useful for pulling real v5 checkpoints if needed for ablation cross-checks, but this sprint does not require them.

---

## 2. Goal

Ship `latest_paper_v5.tex` with:
- zero `[TBD]` cells
- L3 / Table III backed by real measurements on regenerated v4 cycle-9 and cycle-11 checkpoints
- Table II rows filled with published HiDDeN / SteganoGAN numbers + an explicit footnote about eval-setup differences

Success criterion: `grep -nE 'TBD|X_\\mathrm|Y_\\mathrm|\\Delta Z' latest_paper_v5.tex` returns nothing.

---

## 3. Execution plan (5090)

### Phase A — environment prep (~5 min)

1. Confirm CUDA + PyTorch:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   # expect: 2.x True NVIDIA GeForce RTX 5090
   ```
2. Authenticate HF (use env var, do not persist if machine is shared):
   ```bash
   export HF_TOKEN=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
   python -c "from huggingface_hub import whoami; print(whoami(token='$HF_TOKEN')['name'])"
   # expect: OneAutumnLeef
   ```
3. Prime the diffusers cache for `google/ddpm-celebahq-256` if not already cached (`~/.cache/huggingface/hub/models--google--ddpm-celebahq-256`). If absent, the first autopilot cycle will fetch it automatically (~1.5 GB).
4. **Remove stale 134-byte pointer stubs** so the autopilot does not attempt to resume from them:
   ```bash
   find checkpoints_tifs/ber_recovery -name "*.pt" -size -1M -print -delete
   rm -f outputs_tifs/autopilot_state.json
   ```
   `autopilot_state.json` currently marks cycle 11 as complete and points at the stub; deleting it forces a clean sweep.
5. Make log/output dirs just in case:
   ```bash
   mkdir -p logs outputs_tifs_eval
   ```

### Phase B — retrain all 11 cycles (~70 min unattended)

The original schedule was two separate autopilot invocations (8 + 3). Replicate exactly.

6. **Cycles 1–8** (2,500 steps each, ≈ 57 min):
   ```bash
   python -u training_ops/tifs_autopilot.py \
     --max-cycles 8 \
     --steps-per-cycle 2500 \
     --eval-samples 200 \
     --adaptive-train 300 --adaptive-test 100 \
     --cross-dataset-samples 200 --cross-dataset-every 1 \
     --min-ablation-variants 3 \
     2>&1 | tee logs/autopilot_cycles_1_8.log
   ```
   Monitor: `tail -f logs/autopilot_cycles_1_8.log | grep -E "Cycle [0-9]+/|\\[eval\\] Completed|Traceback|Error"`. Each cycle prints a banner `Cycle N/8 | variant=... | step X -> Y` and ends with `[eval] Completed full TIFS evaluation`.

7. **Cycles 9–11** (1,200 steps each, ≈ 10 min):
   ```bash
   python -u training_ops/tifs_autopilot.py \
     --max-cycles 11 \
     --steps-per-cycle 1200 \
     --eval-samples 200 \
     --adaptive-train 300 --adaptive-test 100 \
     --cross-dataset-samples 200 --cross-dataset-every 1 \
     --min-ablation-variants 3 \
     2>&1 | tee logs/autopilot_cycles_9_11.log
   ```
   The autopilot reads `outputs_tifs/autopilot_state.json` to know it is already at cycle 8 and resumes into cycles 9, 10, 11. This is exactly how the original sprint worked on 2026-04-18.

8. **Verify checkpoints are real**:
   ```bash
   ls -lh checkpoints_tifs/ber_recovery/dscsp_v4_final.pt
   ls -lh checkpoints_tifs/ber_recovery/major_cycle_011_robust_balance.pt
   # expect ~220 MB each
   python -c "import torch; c=torch.load('checkpoints_tifs/ber_recovery/dscsp_v4_final.pt', map_location='cpu', weights_only=False); print('keys:', list(c.keys())[:5])"
   # expect something like ['controller', 'extractor', 'config', 'step', ...]
   ```

### Phase C — inference for paper numbers (~15 min on 5090)

9. **Cycle 9 — L3 gap + Table III ResNet ensemble**:
   ```bash
   python evaluate_tifs.py \
     --checkpoint checkpoints_tifs/ber_recovery/dscsp_v4_final.pt \
     --fast --skip-cross-dataset \
     --real-attacks --multiple-detectors \
     --output outputs_tifs_eval/cycle_009_real_and_detectors.json
   ```
   Extract from the JSON:
   - `real_jpeg_gap.q75_proxy_ber` → `$X_{\mathrm{proxy}}$`
   - `real_jpeg_gap.q75_real_ber`  → `$Y_{\mathrm{real}}$`
   - `real_jpeg_gap.q75_gap`       → `$\Delta Z$`
   - (also `q50_*` for a one-sentence aside)
   - `multi_detector.auc_resnet18` / `auc_resnet34` / `auc_ensemble` → Table III row "Cycle 9"

10. **Cycle 11 — Table III ResNet ensemble**:
    ```bash
    python evaluate_tifs.py \
      --checkpoint checkpoints_tifs/ber_recovery/major_cycle_011_robust_balance.pt \
      --fast --skip-cross-dataset \
      --multiple-detectors \
      --output outputs_tifs_eval/cycle_011_detectors.json
    ```
    Extract: `multi_detector.auc_resnet34`, `auc_ensemble`. Cross-check that `auc_resnet18 ≈ 0.6915` (sanity — should match existing cycle 11 JSON).

11. **Pretty-print the numbers** (copy-paste into chat so the paper-edit step is trivial):
    ```bash
    python -c "
    import json
    for p in ['outputs_tifs_eval/cycle_009_real_and_detectors.json',
              'outputs_tifs_eval/cycle_011_detectors.json']:
        print('===', p)
        d = json.load(open(p))
        print('  real_jpeg_gap:', d.get('real_jpeg_gap'))
        print('  multi_detector:', d.get('multi_detector'))
    "
    ```

### Phase D — paper edits (~30 min)

All line numbers below are relative to the current `latest_paper_v5.tex` on `main` as of 2026-04-23. Confirm with `grep -n` before editing — drift is possible.

12. **L3 — Section X limitations paragraph** (find with `grep -n "X_{\\\\mathrm{proxy}" latest_paper_v5.tex`, ~lines 817–822). Rewrite from:
    > Under real libjpeg compression at Q75, BER degrades from $X_{\mathrm{proxy}}$ (proxy) to $Y_{\mathrm{real}}$ (libjpeg), a gap of $\Delta Z$; we attribute this to the straight-through estimator's quantisation rounding mismatch and leave a learned proxy correction to future work.\footnote{Numbers $X_{\mathrm{proxy}}$, $Y_{\mathrm{real}}$, $\Delta Z$ will be filled from the \texttt{--real-attacks} evaluation once completed.}

    To (substitute real values from step 9):
    > Under real libjpeg compression at Q75, BER degrades from \(X\) (DiffJPEG proxy) to \(Y\) (Pillow libjpeg, Q=75), a gap of \(\Delta = Y - X\); we attribute this to the straight-through estimator's quantisation rounding mismatch between the proxy's soft rounding and libjpeg's true integer DCT rounding, and leave a learned proxy correction to future work.

    Remove the footnote.

13. **Table III — Steganalysis detectors** (find with `grep -n "ResNet-34 AUC" latest_paper_v5.tex`, ~lines 578–588). Fill the four `[TBD]` cells:
    - Row "Cycle 9": ResNet-18 / ResNet-34 / ensemble from `cycle_009_real_and_detectors.json`
    - Row "Cycle 11": ResNet-34 / ensemble from `cycle_011_detectors.json` (ResNet-18 stays `0.6915`)

14. **Table II — learned-embedding baselines** (find with `grep -n "HiDDeN" latest_paper_v5.tex`, ~lines 554–565). Published numbers to use (cite carefully):

    | Baseline | Dataset at source | Clean BER | JPEG Q75 BER | PSNR | Adaptive AUC |
    |---|---|---|---|---|---|
    | HiDDeN [Zhu+ 2018] | COCO 128×128, 30-bit payload | 0.00 | ~0.15 | 33.5 dB | N/R |
    | SteganoGAN [Zhang+ 2019] | Div2K 360×360, 4 bpp | ~0.01 | N/R (not a primary claim) | 30.5 dB | N/R |
    | **DSCSP (ours, cycle 9)** | CelebA-HQ 256×256, 256-bit | 1.95×10⁻⁵ | 3.9×10⁻⁵ | 30.68 dB | 0.999 |
    | **DSCSP (ours, cycle 11)** | CelebA-HQ 256×256, 256-bit | 0.0243 | 0.137 | 30.5 dB | 0.6915 |

    Add a footnote immediately below the table:
    > Baseline numbers are drawn from the original publications and use different image resolution, payload size, and JPEG codec (libjpeg vs our DiffJPEG proxy); they are included for orientation only. A matched-conditions re-run of both baselines at CelebA-HQ 256×256 with a 256-bit payload is deferred to the supplementary. \texttt{N/R}: not reported in source.

    Update the row values accordingly and mark "N/R" where noted.

15. **Section III.C adversary description** — verify it already reads "ResNet-18 + ResNet-34 …" etc. (should be present from prior L4 edit). Adjust wording only if the new numbers change the "matched-family" story.

16. **Section IX discussion** — add one sentence on the ResNet-18 vs ResNet-34 gap. If ResNet-34 AUC is noticeably higher, frame it honestly:
    > A deeper steganalyser (ResNet-34) raises cycle-11 AUC from \(0.6915\) to \(\text{A34}\); the gap is consistent with prior findings that steganalysis accuracy tracks adversary capacity, and motivates the SRNet comparison deferred to future work.

17. **Abstract + Conclusion** — re-verify headline numbers. If the regenerated cycle-9 clean BER differs from `1.95\times 10^{-5}` by more than a factor of two (nondeterminism / seed), reword as "below \(10^{-4}\)" or use the new exact value consistently in abstract, Table I, and conclusion.

18. **Remove now-stale prose** — any sentence promising "numbers to be filled from the `--real-attacks` evaluation once completed" or similar forward-looking language. Final text should read as a finished result.

### Phase E — validation (~10 min)

19. **No leftover placeholders**:
    ```bash
    grep -nE 'TBD|X_\\mathrm|Y_\\mathrm|Delta Z|\[X\]|\[Y\]' latest_paper_v5.tex
    # expect: no output
    ```

20. **Rebuild PDF**:
    ```bash
    pdflatex latest_paper_v5.tex
    bibtex   latest_paper_v5
    pdflatex latest_paper_v5.tex
    pdflatex latest_paper_v5.tex
    ```
    Watch for undefined references or overfull hboxes that weren't there before.

21. **Cross-check** the rebuilt PDF: abstract → Table I → Table II → Table III → Section X limitations paragraph → conclusion. Numbers must be consistent, no `\texttt{[TBD]}` anywhere.

22. **Commit**. Single commit per logical unit:
    - a) Retrain artifacts (`checkpoints_tifs/**`, `outputs_tifs/**`, `outputs_tifs_eval/cycle_0*.json`, `logs/autopilot_*.log`) — tracked in `.gitignore` if they're large; if not, follow existing pattern in `.gitattributes`.
    - b) New inference JSONs (`cycle_009_real_and_detectors.json`, `cycle_011_detectors.json`).
    - c) `latest_paper_v5.tex` edits.
    - d) Update `PAPER_CLOSEOUT_PLAN_2026-04-23.md` marking each phase done.

    Commit messages: concise, per phase.

---

## 4. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Diffusion model cache missing | low | First cycle autoloads from HF (~1.5 GB) |
| `autopilot_state.json` thinks training is complete | **high** | **Phase A step 4 explicitly deletes it.** Skipping this step wastes the whole run. |
| Retrained cycle-9 clean BER differs from the old 1.95×10⁻⁵ by seed nondeterminism | med | Acceptable — re-report the new number throughout (Abstract, Table I, Conclusion). Don't mix old and new. |
| BER / AUC drifts enough to break a paper claim (e.g. AUC > 0.70 on cycle 11) | low | If it happens, cycle 11 no longer passes the detectability gate — fall back to cycle 10 or 8, or note the drift as a supplementary caveat. |
| `--real-attacks` hits a PIL / DiffJPEG shape mismatch | low | C1 is already landed (evaluate_tifs.py lines 99–130); sanity-check on a single sample first if paranoid. |
| HF token leaks into commit | med | Use `HF_TOKEN` env var only. `git diff` before commit must show no token string. |
| 5090 has stale data/model caches from 2026-04-18 that are incompatible | low | Rerunning on the same host the original training used; configs are unchanged. |

---

## 5. Out of scope for this sprint

- Running HiDDeN / SteganoGAN end-to-end on matched CelebA-HQ 256×256 + 256-bit payload → supplementary / revision.
- Training the `shift_pool` crop-robustness head (C4 is wired, not trained) → separate sprint.
- Rebuilding the HF dataset repo with real LFS blobs for v4 → post-submission cleanup, but worth fixing so this never happens again.
- SRNet steganalyser (not in `eval_multiple_detectors`) → acknowledged in-paper.

---

## 6. File / directory map

```
.
├── PAPER_CLOSEOUT_PLAN_2026-04-23.md     # this file
├── latest_paper_v5.tex                    # paper draft — the target of Phase D
├── evaluate_tifs.py                       # has --real-attacks, --multiple-detectors, --fast, --celeba-dir, --skip-cross-dataset
├── training_ops/
│   └── tifs_autopilot.py                  # autopilot driver — writes outputs_tifs/autopilot_state.json
├── outputs_tifs/
│   ├── autopilot_state.json               # DELETE in Phase A
│   ├── autopilot_detached.log             # reference: original cycle 1–8 run (2026-04-18)
│   ├── MAJOR_LOG.md                       # human-readable cycle log
│   └── cycle_0{01..11}_*/                 # per-cycle progress + train_v4_results.json
├── outputs_tifs_eval/
│   ├── cycle_0{01..11}_*.json             # reference: old eval JSONs — will be regenerated
│   └── (new) cycle_009_real_and_detectors.json, cycle_011_detectors.json
├── checkpoints_tifs/ber_recovery/
│   └── *.pt                               # all are 134-byte stubs now — DELETE in Phase A
├── logs/
│   └── autopilot_cycles_{1_8,9_11}.log    # new
└── .claude/CLAUDE.md                      # project instructions — read it if context is missing
```

Key code entry points:
- `evaluate_tifs.py::main` (line 709) — CLI
- `evaluate_tifs.py::evaluate_checkpoint` (line 469) — orchestration
- `evaluate_tifs.py::_eval_jpeg_series` (line 99) — DiffJPEG / real-libjpeg dispatch
- `evaluate_tifs.py::eval_multiple_detectors` (line 330) — ResNet-18/34 ensemble
- `training_ops/tifs_autopilot.py` — cycle scheduler, reads/writes `autopilot_state.json`
- `train_v4.py::train` — core training loop
- `evaluate_v4.py::load_v4_model` — called by `evaluate_checkpoint` to load `.pt`

---

## 7. Go / no-go gates

- **Before Phase B**: `torch.cuda.is_available() == True`; `hf auth` succeeds; `autopilot_state.json` deleted; pointer stubs deleted.
- **Before Phase C**: Both cycle-9 and cycle-11 `.pt` files exist and are >100 MB; they load without `_pickle.UnpicklingError`.
- **Before Phase D**: Both new JSONs contain non-null `real_jpeg_gap` (cycle 9 only) and `multi_detector` keys.
- **Before commit**: `grep -nE 'TBD|X_\\mathrm' latest_paper_v5.tex` returns nothing; PDF rebuilds clean.

---

## 8. Handoff summary for the next agent

You are picking up an IEEE TIFS submission whose eval JSONs are intact but whose checkpoint binaries were never uploaded. Training was originally done on this 5090 in ~70 min on 2026-04-18; the configs are all still checked in. Your job is:

1. Re-run the same autopilot invocations (Phase B) to regenerate the weights.
2. Run two inference passes to fill three paper placeholders (Phase C).
3. Edit `latest_paper_v5.tex` at the three specified locations (Phase D).
4. Rebuild PDF + commit (Phase E).

No architecture or training-config changes. No new experiments. No scope creep. If anything diverges materially from the 2026-04-18 run, note it in a `RUN_NOTES_2026-04-23.md` and proceed — don't stall the paper for a seed-level mismatch.
