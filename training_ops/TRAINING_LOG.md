# DSCSP Training Logbook

Use this file as the top-level chronological ledger.

## Entry Template

- Date (UTC):
- Run ID:
- Stage/Goal:
- Command:
- Status: success | failed | interrupted
- Key Metrics:
- Errors:
- Learnings:
- Next Steps:
- Linked Record:

## Entries

- Date (UTC): 2026-04-18
- Run ID: bootstrap_logging_setup
- Stage/Goal: establish reproducible training logging process
- Command: repository update only (no model step yet)
- Status: success
- Key Metrics: N/A
- Errors: none
- Learnings: existing scripts provide raw progress but not a durable decision log
- Next Steps: execute first logged training probe and record outcomes
- Linked Record: training_ops/records/20260418_bootstrap_logging_setup.md

- Date (UTC): 2026-04-18
- Run ID: 20260418_190031_initial_probe
- Stage/Goal: low-cost training startup probe
- Command: `bash training_ops/run_logged.sh initial_probe ... train_v4.train(cfg)`
- Status: failed
- Key Metrics: none (startup failure before first step)
- Errors: `ModuleNotFoundError: No module named 'diffusers'`
- Learnings: environment provisioning must be completed before any training run
- Next Steps: install `requirements.txt`, rerun probe, then begin staged training
- Linked Record: training_ops/records/20260418_190031_initial_probe.md

- Date (UTC): 2026-04-18
- Run ID: 20260418_190108_deps_install
- Stage/Goal: dependency provisioning
- Command: `bash training_ops/run_logged.sh deps_install python3 -m pip install -r requirements.txt`
- Status: success
- Key Metrics: required packages installed (diffusers/accelerate/lpips/sklearn/scipy/torchmetrics)
- Errors: none
- Learnings: environment is now training-ready from package perspective
- Next Steps: rerun minimal probe and verify end-to-end model step
- Linked Record: training_ops/records/20260418_190108_deps_install.md

- Date (UTC): 2026-04-18
- Run ID: 20260418_190133_initial_probe_after_deps
- Stage/Goal: end-to-end one-step training validation
- Command: `bash training_ops/run_logged.sh initial_probe_after_deps ... train_v4.train(cfg)`
- Status: success
- Key Metrics: step BER=0.5039, final BER=0.500762, final PSNR=57.0 dB
- Errors: none (warnings captured)
- Learnings: training pipeline executes correctly on RTX 5090 in this environment
- Next Steps: launch longer staged run and track trend metrics beyond random-guess regime
- Linked Record: training_ops/records/20260418_190133_initial_probe_after_deps.md

- Date (UTC): 2026-04-18
- Run ID: 20260418_190432_warmup_2000
- Stage/Goal: substantive warmup training (2000 steps)
- Command: `bash training_ops/run_logged.sh warmup_2000 ... train_v4.train(cfg)`
- Status: running
- Key Metrics: pending first periodic logging line
- Errors: none so far (non-fatal diffusers warning only)
- Learnings: process is stable enough to start long-form logged training
- Next Steps: monitor first metrics/checkpoint events and update with trend data
- Linked Record: training_ops/records/20260418_190432_warmup_2000.md
