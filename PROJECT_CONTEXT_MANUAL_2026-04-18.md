# Project Context Manual (2026-04-18)

## 1) Purpose of this document
This is a full handover and operating manual for the current state of this repository after all work performed on 2026-04-18.

Use this file before editing code, launching training, running autopilot, or changing Git/LFS policies.

---

## 2) Current repository state (authoritative snapshot)
- Repository: diffusion-stego
- Branch: main
- Current HEAD: ec75d76
- Remote: origin/main is aligned to ec75d76
- Working tree status:
  - Two local untracked files remain:
    - .tifs_data/stl10_binary.tar.gz (about 2.5 GB)
    - .tifs_data/stl10_binary/unlabeled_X.bin (about 2.6 GB)

Reason these are still local:
- GitHub LFS has a hard per-file upload limit of 2 GB.
- Both files exceed that limit and cannot be pushed as-is.

---

## 3) Project objective and context
Project objective:
- Diffusion-based image steganography system intended for publication-quality results.

Core success dimensions used in current automation:
- Clean BER
- JPEG Q75 BER
- Crop Q80 BER
- Adaptive steganalysis AUC
- Cross-dataset checks
- Ablation coverage

Current hard gates in automation (training_ops/tifs_autopilot.py):
- detectability_auc_max: 0.70
- crop80_ber_max: 0.35
- clean_ber_max: 0.10
- jpeg_q75_ber_max: 0.20

Key unresolved scientific blocker right now:
- Crop robustness remains close to random (~0.495 to 0.502 BER) despite large improvements in clean/JPEG BER and partial detectability success.

---

## 4) What changed today (complete timeline)

### 4.1 Commit timeline (chronological summary)
Today commits on main:
- 7fecdeb: added structured training operations workflow and logging scaffolding.
- ee67e3d, 4499ec2, 3eaaa0d, 3e40260: warmup run logging updates.
- 8e56c12: added autonomous TIFS optimization/evaluation pipeline.
- e91e9df: added detached autopilot controls.
- 7e82e9c: improved foreground status detection.
- 4ffcd0b: improved monitoring visibility and evaluation progress logging.
- ed92972: added one-command log watcher.
- 30c572f: major BER-path architecture/training rework.
- 47bc5d5: hardened training writes against missing output directories.
- 8e75f4c: sprint artifacts, evaluation fixes, chat history archive.
- ec75d76: tracked almost all artifacts (except pycache and over-2GB STL10 files).

### 4.2 Training and evaluation run chronology
From training_ops records and logs:
- initial probe failed due missing dependency:
  - ModuleNotFoundError: diffusers
- dependency install run succeeded.
- one-step probe succeeded but BER near random (~0.5).
- 2000-step warmup succeeded but BER remained near chance, motivating major logic/framework changes.
- autopilot 8-cycle run completed without meeting all required targets.
- sprint extension (cycles 9-11 equivalent) completed with evaluation fixes validated.

### 4.3 Key outcome of the day
- Major framework changes broke the strict BER=0.50 deadlock in key phases.
- Detectability target was achieved in cycle 11 (AUC 0.6915), but crop robustness is still far from target.
- Cross-dataset evaluation previously had logic/data-loader faults and is now functioning with valid numeric outputs.

---

## 5) High-impact code changes made today

### 5.1 Core training/model path
Files:
- dscsp/models/controller_v4.py
- train_v4.py
- evaluate_v4.py

High-level changes introduced:
- Payload-carrier controller path improvements.
- Bootstrap phases for extractor/controller in training.
- Resume/load compatibility hardening for architecture evolution.
- Output/checkpoint directory recreation before writes to avoid crash on missing directories.

### 5.2 TIFS evaluation pipeline fixes
File:
- evaluate_tifs.py

Critical fixes:
- Cross-dataset loaders now instantiate CIFAR10/STL10 with tensor transforms.
- DataLoader worker count for cross-dataset load path set to 0 (stability-focused).
- Cross-dataset pass criterion now requires valid metric payload entries, not just a non-empty dictionary.

Before fix:
- Cross-dataset could be marked passed even when entries contained only errors.

After fix:
- Cross-dataset pass reflects actual evaluable results.

### 5.3 Automation and observability
Files:
- training_ops/tifs_autopilot.py
- training_ops/tifs_autopilot_status.sh
- training_ops/tifs_autopilot_stop.sh
- training_ops/tifs_watch_output.sh
- training_ops/run_tifs_autopilot.sh
- training_ops/run_tifs_autopilot_detached.sh

Operational improvements:
- Better state visibility (cycle/phase/active variant, etc.).
- Detached execution support and stop/status controls.
- Fast one-command output streaming helper.
- Milestone logging in outputs_tifs/MAJOR_LOG.md.

---

## 6) Metric history and current model quality

### 6.1 Autopilot state after latest completed cycle
Source: outputs_tifs/autopilot_state.json
- cycle: 11
- phase: stopped
- stopped_reason: max_cycles_reached
- best tracker (by clean BER logic) points to cycle 9 variant ber_recovery_clean_first

### 6.2 Cycle-by-cycle summary (cycles 1 to 11)
Format: cycle | variant | clean BER | JPEG Q75 BER | crop Q80 BER | adaptive AUC

- 1 | ber_recovery_clean_first | 0.007458 | 0.010083 | 0.497742 | 0.960556
- 2 | ber_recovery_curriculum  | 0.018311 | 0.168079 | 0.501550 | 0.839583
- 3 | robust_balance           | 0.060266 | 0.257544 | 0.502637 | 0.858958
- 4 | detectability_harden     | 0.113025 | 0.292212 | 0.498438 | 0.782917
- 5 | ber_recovery_clean_first | 0.000024 | 0.000220 | 0.500171 | 0.998958
- 6 | ber_recovery_curriculum  | 0.012915 | 0.162512 | 0.499072 | 0.985694
- 7 | robust_balance           | 0.038428 | 0.246753 | 0.497363 | 0.843125
- 8 | detectability_harden     | 0.156970 | 0.305151 | 0.501538 | 0.749167
- 9 | ber_recovery_clean_first | 0.000020 | 0.000039 | 0.499883 | 0.999000
- 10| ber_recovery_curriculum  | 0.008965 | 0.010527 | 0.498457 | 0.998700
- 11| robust_balance           | 0.024316 | 0.136973 | 0.495469 | 0.691500

Interpretation:
- BER focus gates are often met in several cycles.
- Detectability gate is met in cycle 11.
- Crop robustness gate remains unpassed in all cycles.

### 6.3 Final cycle 11 detailed eval highlights
Source: outputs_tifs_eval/cycle_011_robust_balance.json
- clean BER: 0.024316 (pass for 0.10 gate)
- clean PSNR: 27.80 (below 35 dB publication-style quality preference)
- JPEG Q75 BER: 0.136973 (pass for 0.20 gate)
- crop Q80 BER: 0.495469 (fail for 0.35 gate)
- adaptive AUC: 0.6915 (pass for 0.70 gate)
- cross-dataset available and valid:
  - CIFAR10 clean BER 0.006309, crop80 BER 0.502012
  - STL10 clean BER 0.123750, crop80 BER 0.500664

---

## 7) Error and warning ledger (what matters)

### 7.1 Resolved issues
1. Cross-dataset DataLoader TypeError with PIL collation in evaluator
- Resolved via transform-on-dataset path and safer loader behavior.

2. Cross-dataset gating false positives
- Resolved by validating payload structure before setting pass status.

3. Missing output directory write failure
- Resolved by proactively creating required write directories in training flow.

4. Console visibility and status ambiguity
- Resolved with watcher/status/detached scripts and phase-aware state updates.

### 7.2 Remaining warnings/non-blockers
1. Diffusion loader warning about safetensors fallback appears repeatedly.
- It has not blocked completion.
- Keep in mind for reproducibility/security hardening.

2. Multiprocessing teardown assertions occurred in previous runs.
- Reduced by evaluation path changes.

### 7.3 Outstanding hard blocker
- Crop robustness remains near chance-level BER and is the main scientific/engineering blocker.

---

## 8) Git, ignore, and LFS policy as of now

### 8.1 Ignore policy
Current .gitignore intentionally ignores only Python cache artifacts:
- __pycache__/
- *.pyc
- *.pyo

Everything else is now intended to be trackable.

### 8.2 LFS policy
Current .gitattributes includes:
- checkpoints_v4_5090/*.pt via LFS
- checkpoints_v4_5090_final/**/*.pt via LFS
- all *.pt via LFS
- all *.bin via LFS
- all *.tar.gz via LFS

### 8.3 Practical GitHub limit caveat
Even with LFS enabled, GitHub rejects files larger than 2 GB each.

Current local exceptions (cannot be pushed as-is):
- .tifs_data/stl10_binary.tar.gz (~2.5 GB)
- .tifs_data/stl10_binary/unlabeled_X.bin (~2.6 GB)

If those must be remotely stored, options are:
- external object storage (S3/GCS/RunPod volume snapshots),
- split archives/chunks with reconstruction tooling,
- dataset redownload script instead of storing binary blobs in repo.

---

## 9) Operational runbook before touching anything

### 9.1 Pre-flight checklist
1. Verify branch and head:
- git branch --show-current
- git rev-parse --short HEAD

2. Verify LFS is installed and active:
- git lfs version
- git lfs ls-files

3. Verify current autopilot state:
- bash training_ops/tifs_autopilot_status.sh

4. Check if there are local large untracked files:
- git status --short

### 9.2 Live monitoring commands
- foreground run:
  - bash training_ops/run_tifs_autopilot.sh [args]
- detached run:
  - bash training_ops/run_tifs_autopilot_detached.sh [args]
- status:
  - bash training_ops/tifs_autopilot_status.sh
- stop:
  - bash training_ops/tifs_autopilot_stop.sh
- watch output:
  - bash training_ops/tifs_watch_output.sh

### 9.3 Logged run pattern
- bash training_ops/run_logged.sh <tag> <command> [args...]

Run artifacts appear in:
- training_ops/runs/<run_id>/

Summary records should be updated in:
- training_ops/records/
- training_ops/TRAINING_LOG.md

### 9.4 Safe Git workflow for this repository now
Because many large artifacts are now tracked:
1. Do not run blind add/push from weak network environments.
2. Prefer checking what is staged before commit:
- git status --short
- git diff --cached --name-status
3. Expect large LFS transfer volumes.
4. Remember 2 GB per-file GitHub LFS hard limit.

---

## 10) File map that matters most

Core training and model files:
- train_v4.py
- dscsp/models/controller_v4.py
- dscsp/models/extractor.py
- dscsp/models/discriminator.py
- dscsp/models/diffusion_wrapper.py

Evaluation:
- evaluate_tifs.py
- evaluate_v4.py

Automation and operations:
- training_ops/tifs_autopilot.py
- training_ops/run_tifs_autopilot.sh
- training_ops/run_tifs_autopilot_detached.sh
- training_ops/tifs_autopilot_status.sh
- training_ops/tifs_autopilot_stop.sh
- training_ops/tifs_watch_output.sh
- training_ops/run_logged.sh

Progress and historical records:
- training_ops/TRAINING_LOG.md
- training_ops/records/20260418_sprint_plan_execution.md
- outputs_tifs/MAJOR_LOG.md
- outputs_tifs/autopilot_state.json

Chat/session archive:
- training_ops/chat_history_archive/README.md
- training_ops/chat_history_archive/current_session_4d7e8de5.jsonl

---

## 11) What to do next (recommended)
Priority order:
1. Crop-first robustness plan
- Increase crop attack pressure earlier.
- Add explicit crop-consistency objective.
- Keep BER path stable while forcing spatial perturbation resilience.

2. Preserve detectability gains from cycle 11
- Keep robust_balance-like detectability pressure that reached AUC 0.6915.
- Avoid reverting to configurations that push AUC back near 0.99.

3. Improve PSNR while holding BER/AUC
- Cycle 11 reached target AUC and BER gates but PSNR was low (~27.8).
- Tune perturbation schedule/loss balance to recover quality without regressing crop and AUC.

4. Keep cross-dataset checks enabled frequently
- Current fix is validated; continue per-cycle checks during further tuning.

---

## 12) Absolute do-not-ignore warnings for future contributors
1. If you change ignore/LFS rules, re-check what gets staged immediately.
2. Do not assume all local files are pushable; check >2GB files before push.
3. Crop BER is the central blocker; improvements in clean BER alone can be misleading.
4. A model can meet BER and AUC gates yet still fail robustness/publication readiness due to crop behavior and image quality.

---

## 13) Quick orientation for a new engineer (5-minute version)
1. Read this file first.
2. Run status script and inspect outputs_tifs/autopilot_state.json.
3. Review outputs_tifs_eval/cycle_011_robust_balance.json.
4. Read training_ops/tifs_autopilot.py REQUIRED gates and variant settings.
5. Start from crop-first experiments, not clean-BER-only variants.
6. Maintain run records in training_ops/TRAINING_LOG.md and training_ops/records.

---

## 14) End-of-day reality check
What is strong now:
- End-to-end automation and observability are materially better.
- Evaluator cross-dataset path is corrected and producing valid outputs.
- Detectability target has been reached in at least one cycle.
- Most historical artifacts are now in the remote repository with LFS.

What is still not solved:
- Crop robustness remains near random.
- Two oversized STL10 files remain local due GitHub hard limits.

This is the exact point where the next day should begin.
