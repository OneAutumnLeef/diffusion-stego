# Training Operations

This folder standardizes how training runs are executed and documented.

## Goals

- Capture exact commands and runtime metadata for each run.
- Keep full raw logs out of git while preserving concise run records in git.
- Log errors, learnings, and next steps after every significant run.

## Workflow

1. Launch a run through `training_ops/run_logged.sh`.
2. Inspect run artifacts under `training_ops/runs/<run_id>/` (ignored by git).
3. Write a concise record in `training_ops/records/`.
4. Append a short summary entry in `training_ops/TRAINING_LOG.md`.
5. Commit and push when a significant milestone is reached.

## Run Command

```bash
bash training_ops/run_logged.sh <run_tag> <command> [args...]
```

Example:

```bash
bash training_ops/run_logged.sh s1_probe python3 -u train_v4_final_iterated.py --start-stage 1 --skip-quick-eval
```

## Output Layout

- `training_ops/runs/<run_id>/metadata.json`: git hash, branch, host, timing, status.
- `training_ops/runs/<run_id>/command.sh`: exact replay command.
- `training_ops/runs/<run_id>/stdout.log`: full console output.
- `training_ops/runs/<run_id>/error_extract.log`: quick filtered error lines.
- `training_ops/records/<run_id>.md`: curated, committable summary.
