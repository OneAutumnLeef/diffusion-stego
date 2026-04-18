#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PID_FILE="outputs_tifs/autopilot_detached.pid"
STATE_FILE="outputs_tifs/autopilot_state.json"
LOG_FILE="outputs_tifs/autopilot_detached.log"

if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$PID" ]] && ps -p "$PID" > /dev/null 2>&1; then
        echo "Autopilot status: RUNNING"
        echo "PID: $PID"
    else
        echo "Autopilot status: NOT RUNNING (stale PID file)"
    fi
else
    echo "Autopilot status: NOT RUNNING"
fi

if [[ -f "$STATE_FILE" ]]; then
    echo "State file: $STATE_FILE"
    python3 - <<'PY'
import json
p='outputs_tifs/autopilot_state.json'
with open(p,'r',encoding='utf-8') as f:
    s=json.load(f)
print('cycle:', s.get('cycle'))
print('finished:', s.get('finished'))
print('best:', s.get('best', {}))
PY
else
    echo "State file: not found"
fi

if [[ -f "$LOG_FILE" ]]; then
    echo "Log file: $LOG_FILE"
fi
