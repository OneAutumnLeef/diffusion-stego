# Agent Instructions

## Mandatory Post-Clone Step

This repository does not store large checkpoints and outputs in git.
After cloning, restore artifacts from Hugging Face before running training, finetuning, evaluation, or ablation.

1. Install dependencies:
   pip install -r requirements.txt
2. Pull artifacts:
   bash pull_artifacts_from_hf.sh .
3. Verify key directories are populated:
   checkpoints_v5/
   checkpoints_v5_clean/
   checkpoints_v5_quality/
   outputs_v5_ablation/

Hugging Face source of truth:
- Repo id: OneAutumnLeef/dscsp-checkpoints
- Repo type: dataset

If access is required, authenticate first:
- hf auth login
