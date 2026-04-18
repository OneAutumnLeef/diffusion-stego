# Sprint Plan and Execution (2026-04-18)

## Objective
Run a focused sprint cycle batch after fixing evaluation validity issues, then summarize outcomes and next actions.

## Scope
- Fix cross-dataset evaluation data pipeline and pass/fail gating.
- Run sprint autopilot cycles with reduced cost for fast iteration.
- Record resulting metrics and blockers.

## Sprint Configuration
- Command:
  - `python3 -u training_ops/tifs_autopilot.py --max-cycles 11 --steps-per-cycle 1200 --eval-samples 200 --adaptive-train 300 --adaptive-test 100 --cross-dataset-samples 200 --cross-dataset-every 1 --min-ablation-variants 3`
- Rationale:
  - Continues 3 additional cycles beyond the completed 8-cycle run (cycles 9 to 11).
  - Uses cycle-every cross-dataset checks to immediately validate the loader/gating fix.
  - Uses reduced sample counts for faster turnaround while preserving trend signal.

## Execution Checklist
- [x] Patch evaluator cross-dataset loader and pass gating.
- [x] Run 3 sprint cycles (cycles 9-11 equivalent sequence).
- [x] Capture final metrics and error scan.
- [x] Write post-sprint recommendations.

## Execution Results
- Run ID: 20260418_210332_sprint_cycles_9_11
- Duration: 621s
- Status: success
- Command:
  - `bash training_ops/run_logged.sh sprint_cycles_9_11 python3 -u training_ops/tifs_autopilot.py --max-cycles 11 --steps-per-cycle 1200 --eval-samples 200 --adaptive-train 300 --adaptive-test 100 --cross-dataset-samples 200 --cross-dataset-every 1 --min-ablation-variants 3`

### Cycle Metrics (Sprint)
- Cycle 9 (ber_recovery_clean_first): clean BER 0.000020, JPEG Q75 BER 0.000039, crop Q80 BER 0.499883, adaptive AUC 0.9990
- Cycle 10 (ber_recovery_curriculum): clean BER 0.008965, JPEG Q75 BER 0.010527, crop Q80 BER 0.498457, adaptive AUC 0.9987
- Cycle 11 (robust_balance): clean BER 0.024316, JPEG Q75 BER 0.136973, crop Q80 BER 0.495469, adaptive AUC 0.6915

### Error Scan
- Fixed: cross-dataset DataLoader PIL collation failures are no longer present.
- Fixed: cross-dataset gate now requires valid metric payloads and does not pass on error-only entries.
- Remaining non-blocking warning: safetensors file fallback warning from diffusion model cache resolution.

## Conclusion
- Sprint objective completed: plan created, fixes implemented, sprint cycles executed to completion.
- Publication blockers still open: crop robustness remains near chance and only detectability reached target in cycle 11.

## Next Actions
1. Run a crop-first variant sweep (increase crop pressure earlier and add crop-consistency objective).
2. Keep cross-dataset checks every cycle until at least one full run is clean of warnings/errors.
3. Narrow adversarial pressure ramp in robust_balance to preserve BER while improving AUC.
