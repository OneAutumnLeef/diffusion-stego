# Run Record: 20260418_190133_initial_probe_after_deps

- Date (UTC): 2026-04-18
- Run ID: 20260418_190133_initial_probe_after_deps
- Objective: confirm post-install training initialization and logging

## Command

```bash
bash training_ops/run_logged.sh initial_probe_after_deps python3 -u -c "import copy, train_v4; cfg = copy.deepcopy(train_v4.CONFIG); cfg.update({'total_steps': 1, 'batch_size': 1, 'pool_size': 4, 'pool_refresh_interval': 100000, 'log_interval': 1, 'save_image_interval': 100000, 'checkpoint_interval': 100000, 'output_dir': 'outputs_v4_probe', 'checkpoint_dir': 'checkpoints_v4_probe', 'use_adversarial': False, 'lambda_adv': 0.0, 'robustness_start_step': 100000, 'use_combined_attacks': False}); train_v4.train(cfg)"
```

## Status

- success (exit code 0)

## Key Metrics

- Device: NVIDIA GeForce RTX 5090 (31.4 GB)
- Train step 1/1:
  - BER: 0.5039
  - PSNR: 37.4 dB
  - msg loss: 0.7115
  - perc loss: 0.1473
  - spec loss: 0.3837
- Final eval (200 samples):
  - Average BER: 0.500762
  - Average PSNR: 57.0 dB

## Warnings and Non-Fatal Issues

- Diffusers warning about missing safetensors, then fallback to pickle serialization.
- `torch.tensor(sourceTensor)` construction warning in `dscsp/models/diffusion_wrapper.py`.
- Torchvision deprecation warnings from LPIPS dependency path.

## Learnings

- End-to-end training now runs in this environment and generates checkpoints/results.
- BER at one step is random-guess as expected; this confirms plumbing, not model quality.
- Warning cleanup in `dscsp/models/diffusion_wrapper.py` should be addressed to keep logs clean.

## Next Steps

1. Start a longer logged warmup run (stage-oriented, not one-step probe).
2. Add a warning cleanup patch for `dscsp/models/diffusion_wrapper.py` tensor construction.
3. After first significant staged milestone, run quick eval and record BER/PSNR/JPEG deltas.
