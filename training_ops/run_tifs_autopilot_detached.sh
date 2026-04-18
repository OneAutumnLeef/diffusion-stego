#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p outputs_tifs
PID_FILE="outputs_tifs/autopilot_detached.pid"
LOG_FILE="outputs_tifs/autopilot_detached.log"

if [[ -f "$PID_FILE" ]]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${OLD_PID}" ]] && ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Autopilot already running with PID $OLD_PID"
        echo "Log: $LOG_FILE"
        exit 0
    fi
fi

nohup python3 -u training_ops/tifs_autopilot.py "$@" >> "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "Started detached TIFS autopilot"
echo "PID: $NEW_PID"
echo "PID file: $PID_FILE"
echo "Log: $LOG_FILE"
