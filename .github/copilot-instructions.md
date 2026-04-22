## graphify

Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` if it exists.
If `graphify-out/wiki/index.md` exists, navigate it for deep questions.
Type `/graphify` in Copilot Chat to build or update the knowledge graph.

## artifact bootstrap (required after clone)

This repository stores large checkpoints and outputs on Hugging Face, not in git.

Before running training, fine-tuning, evaluation, or ablation scripts, always restore artifacts:

1. Install dependencies from `requirements.txt`.
2. Run `bash pull_artifacts_from_hf.sh .` from repository root.
   Windows PowerShell alternative: `./pull_artifacts_from_hf.ps1 .`
3. Confirm directories such as `checkpoints_v5/`, `checkpoints_v5_clean/`, `checkpoints_v5_quality/`, and `outputs_v5_ablation/` exist and are populated.

Hugging Face source of truth:

- repo id: `OneAutumnLeef/dscsp-full-backup`
- fallback repo id: `OneAutumnLeef/dscsp-checkpoints`
- repo type: `dataset`

If authentication is required, use `hf auth login` before downloading.
