#!/usr/bin/env bash
# Simulates the a2a-lightspeed-eval Tekton step locally.
#
# Usage:
#   # Against a running agent (local or remote):
#   AGENT_ENDPOINT=http://localhost:8000 ./mlflow_eval/test_pipeline_step.sh
#
#   # With MLflow logging:
#   AGENT_ENDPOINT=http://localhost:8000 \
#   MLFLOW_TRACKING_URI=http://localhost:5000 \
#   ./mlflow_eval/test_pipeline_step.sh
#
# Required env vars (same as Tekton step):
#   AGENT_ENDPOINT          - A2A agent URL
#   OPENAI_API_KEY          - LLM credentials (or set to "mock" for LiteLLM proxy)
#   OPENAI_BASE_URL         - LiteLLM proxy URL (e.g. http://litellm:4000)
#   MLFLOW_GENAI_JUDGE_DEFAULT_MODEL - Judge model (e.g. openai:/claude-sonnet)
#
# Optional:
#   MLFLOW_TRACKING_URI     - MLflow server URL
#   LLM_JUDGE_MODEL         - Pipeline-style judge model var
#   LLM_BASE_URL            - Pipeline-style LLM base URL var

set -euo pipefail

# ── Defaults (match pipeline params) ──────────────────────────────
AGENT_ENDPOINT="${AGENT_ENDPOINT:-http://localhost:8000}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"
RESULTS_DIR="${RESULTS_DIR:-./test-results}"

# Map pipeline vars to MLflow vars if not already set
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${LLM_BASE_URL:-}}"
export MLFLOW_GENAI_JUDGE_DEFAULT_MODEL="${MLFLOW_GENAI_JUDGE_DEFAULT_MODEL:-${LLM_JUDGE_MODEL:+openai:/${LLM_JUDGE_MODEL#openai/}}}"

mkdir -p "$RESULTS_DIR"

echo "=== Lightspeed Dataset Evaluation (local test) ==="
echo "Agent:  $AGENT_ENDPOINT"
echo "Judge:  ${MLFLOW_GENAI_JUDGE_DEFAULT_MODEL:-not set (code-based scorers only)}"
echo "MLflow: $MLFLOW_TRACKING_URI"
echo ""

# ── Check agent health ────────────────────────────────────────────
if curl -sf "$AGENT_ENDPOINT/.well-known/agent.json" > /dev/null 2>&1; then
  AGENT_NAME=$(curl -sf "$AGENT_ENDPOINT/.well-known/agent.json" | python3 -c "import sys,json;print(json.load(sys.stdin).get('name','unknown'))" 2>/dev/null)
  echo "Agent OK: $AGENT_NAME"
else
  echo "WARNING: Agent not reachable at $AGENT_ENDPOINT"
  echo "Start the agent first, or set AGENT_ENDPOINT to a running instance."
  exit 1
fi

# ── Install lightspeed-eval if needed ─────────────────────────────
if ! python3 -c "import mlflow_eval" 2>/dev/null; then
  echo "Installing lightspeed-eval..."
  pip install --quiet "lightspeed-eval @ git+https://github.com/ccamacho/lightspeed-dataset.git" 2>&1 | tail -1
fi

# ── Run evaluation ────────────────────────────────────────────────
EVAL_ARGS="--agent-endpoint $AGENT_ENDPOINT --per-type 1 --tracking-uri $MLFLOW_TRACKING_URI"

python3 -m mlflow_eval.run_eval $EVAL_ARGS 2>&1 | tee "$RESULTS_DIR/lightspeed-eval.log"

echo "=== Done. Results in $RESULTS_DIR/lightspeed-eval.log ==="
