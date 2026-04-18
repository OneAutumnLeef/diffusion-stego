#!/usr/bin/env bash
set -euo pipefail
cd "/root/diffusion-stego"
python3 -u -c import\ copy\,\ train_v4\;\ cfg\ =\ copy.deepcopy\(train_v4.CONFIG\)\;\ cfg.update\(\{\'total_steps\':\ 2000\,\ \'output_dir\':\ \'outputs_v4\'\,\ \'checkpoint_dir\':\ \'checkpoints_v4\'\}\)\;\ train_v4.train\(cfg\) 
