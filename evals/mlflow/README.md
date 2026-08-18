# MLflow Evaluation

Evaluation pipeline for the Lightspeed Agent using [MLflow GenAI Evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/). Sends questions to a deployed agent via A2A, then scores responses using deterministic checks and LLM-as-a-judge scorers.

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
    └──► MLflow server ── stores evaluation results
```

1. Load 8 curated evaluation questions from `mlflow_eval/cases/`
2. Send each question to the agent's A2A endpoint with a Bearer token
3. Collect the agent's text response
4. Run scorers (deterministic + LLM judges) against each response
5. Log results to the MLflow experiment

## Prerequisites

- Python virtual environment with eval dependencies
- VPN access to the OpenShift cluster (agent + MLflow + judge model endpoints)
- A valid Bearer token for the agent (from Red Hat SSO)
- `litellm` installed (`pip install -r requirements.txt`)

## Configuration

### Judge Model (required)

LLM-as-a-judge scorers need a model to evaluate responses. The script refuses to run without a judge model set, to prevent evaluation data from being sent to external cloud providers.

```bash
export MLFLOW_GENAI_JUDGE_DEFAULT_MODEL="openai:/Qwen/Qwen3-14B"
export OPENAI_BASE_URL="https://<judge-model-endpoint>/v1"
export OPENAI_API_KEY="<api-key>"
```

### Data Privacy

LLM-as-a-judge scorers send agent responses (which may contain CVEs, host names, advisor details) to the judge model for scoring. Use a self-hosted model to keep data within your network.

### SSL/TLS for Internal Clusters

OpenShift clusters and internal model endpoints typically use self-signed certificates that Python doesn't trust by default. To skip certificate verification (connections remain encrypted — only the CA check is skipped):

```bash
export MLFLOW_TRACKING_INSECURE_TLS=true
```

This covers all HTTPS calls: MLflow tracking, agent A2A requests, and judge model calls.

For proper certificate verification instead, export the cluster's CA certificate:

```bash
# Extract the OpenShift ingress CA
oc extract configmap/router-ca -n openshift-ingress-operator \
    --keys=ca-bundle.crt --to=/tmp/

# Point Python at it
export REQUESTS_CA_BUNDLE=/tmp/ca-bundle.crt
```

If the agent and judge are on different clusters, concatenate both CA certs into a single bundle file.

## Usage

```bash
# Set environment variables
export MLFLOW_GENAI_JUDGE_DEFAULT_MODEL="openai:/Qwen/Qwen3-14B"
export OPENAI_BASE_URL="https://<judge-endpoint>/v1"
export OPENAI_API_KEY="<api-key>"
export MLFLOW_TRACKING_INSECURE_TLS=true

# Run evaluation
python -u -m mlflow_eval.run_eval \
    --agent-endpoint https://lightspeed-agent-<namespace>.apps.<cluster>/ \
    --agent-token "<bearer-token>" \
    --tracking-uri https://mlflow-<namespace>.apps.<cluster>/ \
    --judge-model "openai:/Qwen/Qwen3-14B"
```

Use `python -u` for unbuffered output to see progress in real time.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--agent-endpoint` | `http://localhost:8000` | Agent A2A endpoint |
| `--agent-token` | (required) | Bearer token for agent authentication |
| `--judge-model` | (required) | Judge model URI, falls back to `MLFLOW_GENAI_JUDGE_DEFAULT_MODEL` env var |
| `--tracking-uri` | `sqlite:///mlflow.db` | MLflow tracking server URI |
| `--experiment` | `lightspeed-eval` | MLflow experiment name |
| `--limit` | all | Max number of questions |
| `--category` | all | Filter by category (e.g. vulnerability, inventory) |
| `--difficulty` | all | Filter by difficulty (easy, medium, hard) |

## Scorers

### Deterministic (no LLM needed)

| Scorer | What it checks | Score |
|--------|----------------|-------|
| `response_received` | Valid response (non-empty, no error, ≥10 chars) | 0 or 1 |
| `answer_correctness` | Factually correct against ground truth, dispatched by question type (binary, single_select, multiple_select, substring_match, exact_match, ordered_list) | 0.0-1.0 |

### Trace-based (queries MLflow, no LLM needed)

| Scorer | What it checks | Score |
|--------|----------------|-------|
| `tool_call_correctness` | Did the agent actually invoke the correct MCP tools? Queries agent MLflow traces for TOOL-type spans. | yes / partial / no / unknown |

### LLM Judge (require judge model)

All LLM judges use the model specified via `--judge-model` or `MLFLOW_GENAI_JUDGE_DEFAULT_MODEL`. The model is passed explicitly to each scorer to prevent fallback to external providers.

| Scorer | What it checks | Score |
|--------|----------------|-------|
| `correctness` | (MLflow built-in) Are expected facts from `expected_response` supported by the output? | yes / no |
| `relevance_to_query` | (MLflow built-in) Does the response address the question? | yes / no |
| `expectations_guidelines` | (MLflow built-in) Does the response follow per-question behavioral guidelines from `expected_behavior`? | yes / no |
| `safety` | No tool name leakage, no code generation, domain boundaries, no internal API disclosure | yes / no |
| `error_handling` | No raw errors/stack traces, honest failure acknowledgment, suggests alternatives, professional tone | yes / no |

## Dataset

Test cases are in `mlflow_eval/cases/` as case directories. Each case has:

- `input.yaml` — the question, category, scenario type, difficulty
- `annotations.yaml` — expected tools, expected response, expected behavior

The full 257-question dataset is in `eval_dataset.json` for broader coverage.

## Viewing Results

Results are logged to the MLflow experiment specified by `--experiment`. Open the MLflow UI to see per-question scores, compare evaluation runs, and drill into individual results.

## Running Tests

```bash
python -m pytest tests/ -v
```
