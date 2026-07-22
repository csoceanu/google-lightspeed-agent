#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env for GEMINI_API_KEY.
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

PORT="${MOCK_AGENT_PORT:-8888}"
HOST="${MOCK_AGENT_HOST:-127.0.0.1}"
ENDPOINT="http://$HOST:$PORT"
CONCURRENCY="${EVAL_CONCURRENCY:-3}"
TIMEOUT="${EVAL_TIMEOUT:-120}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$PROJECT_DIR/results}"

# ── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Cleaning up ..."
    bash "$SCRIPT_DIR/stop_mock_agent.sh"
}
trap cleanup EXIT INT TERM

# ── 1. Start the mock agent ──────────────────────────────────────────────────
echo "============================================================"
echo "  LIGHTSPEED EVALUATION BENCHMARK"
echo "============================================================"
echo ""

bash "$SCRIPT_DIR/start_mock_agent.sh"
echo ""

# ── 2. Quick smoke test ──────────────────────────────────────────────────────
echo "Smoke test: sending a test question to the agent ..."
SMOKE_RESPONSE=$(curl -sf -X POST "$ENDPOINT/chat" \
    -H "Content-Type: application/json" \
    -d '{"message": "What is Red Hat Enterprise Linux?"}' 2>&1) || {
    echo "ERROR: Smoke test failed. Agent may not be responding."
    echo "Response: $SMOKE_RESPONSE"
    exit 1
}
echo "Smoke test passed. Agent responded."
echo ""

# ── 3. Run the evaluation ────────────────────────────────────────────────────
echo "Running evaluation against $ENDPOINT ..."
echo "  Dataset    : $PROJECT_DIR/eval_dataset.json"
echo "  Output dir : $OUTPUT_DIR"
echo "  Workers    : $CONCURRENCY"
echo "  Timeout    : ${TIMEOUT}s"
echo ""

# Pass through any extra args (e.g. --category vulnerability --difficulty easy)
FREE_FORM_STRATEGY="${EVAL_FREE_FORM_STRATEGY:-gemini}"

python3 "$PROJECT_DIR/eval_runner.py" \
    --endpoint "$ENDPOINT" \
    --concurrency "$CONCURRENCY" \
    --timeout "$TIMEOUT" \
    --output-dir "$OUTPUT_DIR" \
    --dataset "$PROJECT_DIR/eval_dataset.json" \
    --free-form-strategy "$FREE_FORM_STRATEGY" \
    "$@"

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  BENCHMARK COMPLETE"
echo "============================================================"
echo ""
echo "  Results in : $OUTPUT_DIR/"
echo "  (see runner log above for the exact timestamped directory)"
echo ""

exit $EXIT_CODE
