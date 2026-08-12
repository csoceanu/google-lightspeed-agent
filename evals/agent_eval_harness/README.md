# AEH Evaluation

Evaluation pipeline for the Lightspeed Agent using [Agent-Eval-Harness (AEH)](https://github.com/opendatahub-io/agent-eval-harness). Sends questions to a deployed agent via A2A, then scores responses using deterministic checks and LLM-as-a-judge scorers.

## How It Works

```
Developer laptop (on VPN)
    │
    ├──► Agent (A2A endpoint) ── sends evaluation questions
    │         │
    │         └──► Traces flow to MLflow automatically (agent-side)
    │
    ├──► Judge model (VPN-backed) ── scores agent responses
    │
    └──► MLflow server ── stores evaluation results and traces
```

1. AEH creates isolated workspaces per test case
2. CLI runner sends each question to the agent's A2A endpoint
3. Collects the agent's text response
4. Runs judges (deterministic + LLM) against each response
5. Exports results and traces to MLflow

## Prerequisites

- [Agent-Eval-Harness](https://github.com/opendatahub-io/agent-eval-harness) cloned and installed (`pip install -e .`)
- VPN access to the OpenShift cluster (agent + MLflow + judge model endpoints)
- A valid Bearer token for the agent (from Red Hat SSO)

## Configuration

### Environment Variables

```bash
# Agent endpoint
export A2A_AGENT_URL="https://lightspeed-agent-<namespace>.apps.<cluster>/"
export A2A_AUTH_TOKEN="<bearer-token>"
export A2A_INSECURE_TLS="true"

# Judge model (VPN-backed, for LLM judges)
export JUDGE_API_KEY="<api-key>"
export JUDGE_API_URL="<model-endpoint>/v1"
export JUDGE_MODEL="<model-id>"

# MLflow (for result export)
export MLFLOW_TRACKING_URI="https://mlflow-<namespace>.apps.<cluster>/"
export MLFLOW_TRACKING_INSECURE_TLS="true"
```

### Data Privacy

LLM-as-a-judge scorers send agent responses (which may contain CVEs, host names, advisor details) to the judge model for scoring. Use a self-hosted model to keep data within your network.

## Usage

### Run evaluation (local)

Set `PYTHONPATH` to the AEH repo root, then run each step:

```bash
AEH_PATH="/path/to/agent-eval-harness"

# 1. Create workspaces
PYTHONPATH="$AEH_PATH" python3 $AEH_PATH/skills/eval-run/scripts/workspace.py \
  --config evals/agent_eval_harness/eval.yaml --run-id <run-id>

# 2. Execute (send questions to agent)
PYTHONPATH="$AEH_PATH" python3 $AEH_PATH/skills/eval-run/scripts/execute.py \
  --config evals/agent_eval_harness/eval.yaml \
  --workspace /tmp/agent-eval/<run-id> \
  --output eval/runs/lightspeed-agent-evaluation/<run-id> \
  --run-id <run-id> --agent cli --model <agent-model>

# 3. Collect artifacts
PYTHONPATH="$AEH_PATH" python3 $AEH_PATH/skills/eval-run/scripts/collect.py \
  --config evals/agent_eval_harness/eval.yaml \
  --workspace /tmp/agent-eval/<run-id> \
  --output eval/runs/lightspeed-agent-evaluation/<run-id>

# 4. Score with judges
PYTHONPATH="$AEH_PATH" python3 $AEH_PATH/skills/eval-run/scripts/score.py judges \
  --config evals/agent_eval_harness/eval.yaml --run-id <run-id>

# 5. Export to MLflow (optional, after reviewing results)
PYTHONPATH="$AEH_PATH" python3 $AEH_PATH/skills/eval-mlflow/scripts/log_results.py \
  --run-id <run-id> --config evals/agent_eval_harness/eval.yaml
```

Results are saved locally in `eval/runs/lightspeed-agent-evaluation/<run-id>/summary.yaml`.

## Judges

### Deterministic

| Judge | What it checks |
|-------|---------------|
| `safety` | Tool name leakage and code generation in responses |
| `response_received` | Valid response received (not empty, not error) |

### LLM-as-judge

These judges call the VPN-backed judge model via `llm_judge.py`.

| Judge | What it evaluates |
|-------|------------------|
| `correctness` | Are the key facts from the expected response present? |
| `relevance` | Does the response address the user's question? |
| `expected_behavior` | Did the agent follow all behavioral constraints? |
| `error_handling` | Does the agent handle errors gracefully? |

All LLM judges return `yes`/`no` verdicts with rationales.

## Dataset

Test cases are in `cases/` as AEH case directories. Each case has:

- `input.yaml` — the question, category, scenario type, difficulty
- `annotations.yaml` — expected tools, expected response, expected behavior

## Viewing Results

- **Local:** `eval/runs/lightspeed-agent-evaluation/<run-id>/summary.yaml`
- **MLflow:** After export, results appear in the `lightspeed-agent-aeh-evals` experiment with per-question traces and judge assessments
