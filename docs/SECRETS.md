# Secrets handling

This project uses external services that require tokens:

- **Hugging Face** (`HF_TOKEN`) — to pull/push checkpoint blobs from
  `OneAutumnLeef/dscsp-checkpoints` and related dataset repos.
- **GitHub** (`GH_TOKEN` or `GITHUB_TOKEN`) — only needed if you still
  want to `git lfs pull` from the repo directly. Since the LFS budget
  incident of 2026-04-24, the Hugging Face mirror is the primary
  distribution path and GitHub LFS is not required for routine work.

## Rules

1. **Never commit a token to this repo.** Not in code, not in docs,
   not in commit messages, not in shell history files, not in
   notebooks.
2. **Never paste a token into a PR title/body or GitHub issue.**
3. **Never echo a token on the command line** — it lands in shell
   history and tool logs. Always pass via an environment variable
   set once at shell init time, e.g. in a local untracked
   `~/.secrets/dscsp.env` sourced manually, or via the pod's
   secret-manager injection.
4. **Redact tokens in any file you commit.** Substitute the placeholder
   `<REDACTED: see docs/SECRETS.md>` and reference this document.
5. **Rotate immediately** if a token is ever committed. Prior
   commits still hold the historical value. Rotation revokes access
   to the exposed value; for full purge, use `git filter-repo`
   followed by a coordinated force-push to `main`.

## Historical incident — 2026-04-24

An HF token was committed to `PAPER_CLOSEOUT_PLAN_2026-04-23.md` on
legacy commit `ce8d238 complete code`. The token remained in repository history
until the `tifs-sprint-2026-04-24` branch redacted it in the working
tree. The owner rotated the exposed HF token; GitHub classic PATs used
during the sprint were also scheduled for rotation. Future
`pull_artifacts_from_hf.sh` / `push_checkpoints_to_hf.py` invocations
must source the new token from an untracked location.

If you need to purge a token from history entirely after rotation:

```bash
# destructive — coordinate with every collaborator before running
# Put the rotated value in a temp file; never inline on the command line.
TMP=$(mktemp)
printf '%s==><REDACTED>\n' "$EXPOSED_TOKEN" > "$TMP"   # set EXPOSED_TOKEN in env
git filter-repo --replace-text "$TMP"
shred -u "$TMP"
git push --force origin main
```
