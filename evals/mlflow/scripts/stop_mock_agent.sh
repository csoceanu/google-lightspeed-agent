#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$PROJECT_DIR/.mock_agent.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file found -- agent may not be running."
    exit 0
fi

AGENT_PID=$(cat "$PID_FILE")

if kill -0 "$AGENT_PID" 2>/dev/null; then
    echo "Stopping mock agent (PID $AGENT_PID) ..."
    kill "$AGENT_PID"
    # Wait up to 5s for clean exit.
    for i in $(seq 1 5); do
        if ! kill -0 "$AGENT_PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    # Force kill if still alive.
    if kill -0 "$AGENT_PID" 2>/dev/null; then
        kill -9 "$AGENT_PID" 2>/dev/null || true
    fi
    echo "Mock agent stopped."
else
    echo "Agent process $AGENT_PID is not running."
fi

rm -f "$PID_FILE"
