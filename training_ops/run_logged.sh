#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash training_ops/run_logged.sh <run_tag> <command> [args...]"
    echo "Example: bash training_ops/run_logged.sh s1_probe python3 -u train_v4.py"
    exit 1
fi

RUN_TAG="$1"
shift

json_escape() {
    local s="$1"
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    printf '%s' "$s"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$REPO_ROOT/training_ops/runs"
mkdir -p "$RUN_ROOT"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_ID="${TIMESTAMP}_${RUN_TAG}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

BRANCH="$(cd "$REPO_ROOT" && git branch --show-current || echo unknown)"
COMMIT="$(cd "$REPO_ROOT" && git rev-parse HEAD || echo unknown)"
DIRTY_COUNT="$(cd "$REPO_ROOT" && git status --porcelain | wc -l | tr -d ' ')"
HOSTNAME_VAL="$(hostname)"
START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"
CMD_DISPLAY="$(printf '%q ' "$@")"
CMD_JSON="$(json_escape "$CMD_DISPLAY")"
BRANCH_JSON="$(json_escape "$BRANCH")"
COMMIT_JSON="$(json_escape "$COMMIT")"
HOSTNAME_JSON="$(json_escape "$HOSTNAME_VAL")"
START_JSON="$(json_escape "$START_ISO")"

cat > "$RUN_DIR/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
$CMD_DISPLAY
EOF
chmod +x "$RUN_DIR/command.sh"

echo "=================================================================="
echo "Starting logged run: $RUN_ID"
echo "Run dir: $RUN_DIR"
echo "Command: $CMD_DISPLAY"
echo "=================================================================="

set +e
(
    cd "$REPO_ROOT"
    "$@"
) 2>&1 | tee "$RUN_DIR/stdout.log"
EXIT_CODE=${PIPESTATUS[0]}
set -e

END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_EPOCH="$(date +%s)"
DURATION_S="$((END_EPOCH - START_EPOCH))"
END_JSON="$(json_escape "$END_ISO")"

STATUS="failed"
if [[ "$EXIT_CODE" -eq 0 ]]; then
    STATUS="success"
fi

cat > "$RUN_DIR/metadata.json" <<EOF
{
    "run_id": "$(json_escape "$RUN_ID")",
    "run_tag": "$(json_escape "$RUN_TAG")",
    "branch": "$BRANCH_JSON",
    "commit": "$COMMIT_JSON",
  "dirty_count": $DIRTY_COUNT,
    "host": "$HOSTNAME_JSON",
    "start_utc": "$START_JSON",
    "end_utc": "$END_JSON",
  "duration_s": $DURATION_S,
  "exit_code": $EXIT_CODE,
  "status": "$STATUS",
    "command": "$CMD_JSON"
}
EOF

grep -nEi "traceback|runtimeerror|error|exception|cuda out of memory|out of memory|failed|nan" \
    "$RUN_DIR/stdout.log" \
    > "$RUN_DIR/error_extract.log" || true

echo ""
echo "Run finished: $RUN_ID"
echo "Status: $STATUS (exit code $EXIT_CODE)"
echo "Artifacts: $RUN_DIR"

exit "$EXIT_CODE"
