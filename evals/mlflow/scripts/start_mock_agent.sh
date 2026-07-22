#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$PROJECT_DIR/.mock_agent.pid"
PORT="${MOCK_AGENT_PORT:-8888}"
HOST="${MOCK_AGENT_HOST:-127.0.0.1}"
MAX_WAIT=15

if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Mock agent already running (PID $OLD_PID)"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

echo "Starting mock A2A agent on $HOST:$PORT ..."

python3 "$PROJECT_DIR/mock_a2a_agent.py" \
    --host "$HOST" \
    --port "$PORT" \
    --env-file "$PROJECT_DIR/.env" \
    --log-level INFO &

AGENT_PID=$!
echo "$AGENT_PID" > "$PID_FILE"

echo "Waiting for agent to become healthy ..."
for i in $(seq 1 "$MAX_WAIT"); do
    if curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
        echo "Mock agent is ready (PID $AGENT_PID, http://$HOST:$PORT)"
        exit 0
    fi
    if ! kill -0 "$AGENT_PID" 2>/dev/null; then
        echo "ERROR: Agent process died during startup"
        rm -f "$PID_FILE"
        exit 1
    fi
    sleep 1
done

echo "ERROR: Agent did not become healthy within ${MAX_WAIT}s"
kill "$AGENT_PID" 2>/dev/null || true
rm -f "$PID_FILE"
exit 1
