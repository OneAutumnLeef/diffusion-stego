#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PID_FILE="outputs_tifs/autopilot_detached.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file found. Nothing to stop."
    exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$PID" ]]; then
    echo "PID file is empty."
    rm -f "$PID_FILE"
    exit 0
fi

if ps -p "$PID" > /dev/null 2>&1; then
    kill "$PID"
    echo "Sent SIGTERM to PID $PID"
else
    echo "Process $PID is not running."
fi

rm -f "$PID_FILE"
