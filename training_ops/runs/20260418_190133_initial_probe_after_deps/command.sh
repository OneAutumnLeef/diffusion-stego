#!/usr/bin/env bash
set -euo pipefail
cd "/root/diffusion-stego"
python3 -u -c import\ copy\,\ train_v4\;\ cfg\ =\ copy.deepcopy\(train_v4.CONFIG\)\;\ cfg.update\(\{\'total_steps\':\ 1\,\ \'batch_size\':\ 1\,\ \'pool_size\':\ 4\,\ \'pool_refresh_interval\':\ 100000\,\ \'log_interval\':\ 1\,\ \'save_image_interval\':\ 100000\,\ \'checkpoint_interval\':\ 100000\,\ \'output_dir\':\ \'outputs_v4_probe\'\,\ \'checkpoint_dir\':\ \'checkpoints_v4_probe\'\,\ \'use_adversarial\':\ False\,\ \'lambda_adv\':\ 0.0\,\ \'robustness_start_step\':\ 100000\,\ \'use_combined_attacks\':\ False\}\)\;\ train_v4.train\(cfg\) 
