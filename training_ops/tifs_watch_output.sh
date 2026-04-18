#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FG_LOG="outputs_tifs/autopilot_console.log"
DET_LOG="outputs_tifs/autopilot_detached.log"
PID_FILE="outputs_tifs/autopilot_detached.pid"

LOG_FILE=""

if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$PID" ]] && ps -p "$PID" > /dev/null 2>&1 && [[ -f "$DET_LOG" ]]; then
        LOG_FILE="$DET_LOG"
    fi
fi

if [[ -z "$LOG_FILE" && -f "$FG_LOG" ]]; then
    LOG_FILE="$FG_LOG"
fi

if [[ -z "$LOG_FILE" ]]; then
    echo "No autopilot log found yet."
    echo "Start a run with: bash training_ops/run_tifs_autopilot.sh ..."
    exit 1
fi

echo "Streaming: $LOG_FILE"
echo "(Ctrl+C to stop viewing)"

tail -n 120 -F "$LOG_FILE"
