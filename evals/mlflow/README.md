# Red Hat Lightspeed Agent Evaluation Framework

Automated evaluation framework for testing the Red Hat Lightspeed Agent. It loads questions from a curated dataset, sends them to the agent endpoint, grades the responses, and generates detailed reports.

The framework supports two evaluation pipelines:

1. **Standalone** (`eval_runner.py`) -- direct HTTP evaluation with built-in grading and HTML/JSON reports
2. **MLflow** (`mlflow_eval/`) -- integration with `mlflow.genai.evaluate()` for traced evaluation with custom and built-in scorers

## Project Structure

```
eval_dataset.json       - Evaluation dataset with 250+ questions across 11 categories
mock_mcp_data.json      - Synthetic MCP tool responses for deterministic evaluation

# Standalone evaluation pipeline
eval_runner.py          - Orchestrates the evaluation pipeline (load, filter, execute, grade)
eval_grader.py          - Grading logic for all question types (binary, select, match, etc.)
eval_reporter.py        - Generates console, JSON, and HTML reports from graded results
eval_config.py          - Configuration management (CLI, env vars, config files)
eval_utils.py           - Shared utility functions (text normalization, statistics, I/O)
mock_a2a_agent.py       - Mock agent that proxies to Gemini API (no MCP data)

# MLflow evaluation pipeline
mlflow_eval/
  __init__.py           - Public API
  scorers.py            - Custom scorers for mlflow.genai.evaluate()
  dataset.py            - Dataset loader for mlflow.genai.evaluate()
  vertex_judge.py       - Routes MLflow LLM judge calls to Claude on Vertex AI
  mock_agent.py         - Mock agent backed by mock_mcp_data.json (deterministic)
  run_eval.py           - End-to-end evaluation script

scripts/                - Shell scripts for running benchmarks
tests/                  - pytest test suite
```

## Installation

```bash
pip install -r requirements.txt

# Or as a pip package (for CI/pipeline integration):
pip install "lightspeed-eval @ git+https://github.com/ccamacho/lightspeed-dataset.git"
```

## MLflow Evaluation Pipeline

### How It Works

The evaluation dataset (`eval_dataset.json`) serves two roles:

1. **Questions** are sent to the agent via a `predict_fn`
2. **Expected answers** are used by scorers to verify the agent's response

```
eval_dataset.json                    mock_mcp_data.json
       |                                    |
       |                                    |
       +-- questions --> Agent --> MCP Server (reads mock data)
       |                    |                |
       |                    |     get_cve_systems("CVE-2024-6387")
       |                    |        -> returns [web-server-prod-01, ...]
       |                    |
       |                    v
       |               "Yes, CVE-2024-6387 affects 3 systems"
       |                    |
       +-- expected --> Scorers --> MLflow traces + metrics
           answers          |
                     answer_correctness: 1.0
                     tool_match: 1.0
                     correctness: "yes"
```

The mock agent (`mlflow_eval/mock_agent.py`) uses Claude on Vertex AI with the synthetic MCP data as context. This ensures the agent's answers are grounded in the dataset, making evaluation deterministic and consistent.

### Quick Start

```bash
# Code-based scorers only (no judge model needed):
python -m mlflow_eval.run_eval --limit 10

# With LLM judge scorers (needs gcloud auth for Vertex AI):
python -m mlflow_eval.run_eval --limit 10 --enable-judge

# Filter by category:
python -m mlflow_eval.run_eval --category vulnerability --limit 5

# View results:
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

### Prerequisites for LLM Judge Scorers

LLM judge scorers (`Correctness`, `RelevanceToQuery`, `SafetyGuidelines`, etc.) require a judge model. The pipeline uses Claude on Vertex AI via gcloud auth:

```bash
# Authenticate with GCP
gcloud auth application-default login

# Required env vars (set automatically by start_vertex_judge()):
# VERTEX_PROJECT=itpc-gcp-eco-eng-claude
# VERTEX_LOCATION=us-east5
```

### Scorers

#### Code-based (deterministic, no LLM needed)

| Scorer | What it checks | Score |
|--------|----------------|-------|
| `answer_correctness` | Is the answer factually correct against the dataset's expected answer? Dispatches by question type: binary (yes/no detection), single_select (option matching), multiple_select (partial credit), substring_match (keyword presence), exact_match (string equality), ordered_list (item ordering). | 0.0-1.0 |
| `tool_match` | Did the agent mention the expected MCP tools in its response? Compares tool short names against the `expected_tools` field. | 0.0-1.0 (fraction found) |
| `behavior_coverage` | Does the response follow the expected behavior? Extracts keywords from `expected_behavior` and checks coverage. | 0.0-1.0 (fraction found) |

#### Trace-based (queries MLflow, no LLM needed)

| Scorer | What it checks | Score |
|--------|----------------|-------|
| `tool_call_correctness` | Did the agent actually invoke the correct MCP tools? Inspects MLflow trace spans (not response text). | yes / no / partial |

#### LLM Judge (require judge model configured)

| Scorer | What it checks | Score |
|--------|----------------|-------|
| `correctness` | (MLflow built-in) Are expected facts from `expected_response` supported by the output? | yes / no |
| `relevance_to_query` | (MLflow built-in) Does the response address the question? | yes / no |
| `expectations_guidelines` | (MLflow built-in) Does the response follow per-question behavioral guidelines from `expected_behavior`? | yes / no |
| `safety` | 4 rules: no tool name leakage (`domain__tool` format), no code generation, stays within Red Hat Insights domain, no internal API disclosure. | yes / no |
| `error_handling` | 4 rules: no raw errors/stack traces, honest failure acknowledgment, suggests alternatives, professional tone. | yes / no |

### Using with a Real Agent

Replace the mock agent with your real Lightspeed Agent endpoint:

```bash
# Point at a running agent:
python -m mlflow_eval.run_eval \
    --agent-endpoint http://your-agent:8080 \
    --enable-judge \
    --limit 20
```

### Using Scorers Directly

```python
import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, ExpectationsGuidelines
from mlflow_eval import (
    load_dataset, AnswerCorrectness, ToolMatch, BehaviorCoverage,
    SafetyGuidelines, ErrorHandlingGuidelines, start_vertex_judge,
)

# Configure judge model (for LLM scorers)
start_vertex_judge()

# Load dataset
dataset = load_dataset(category="vulnerability", limit=10)

# Define how to call the agent
def predict_fn(question, question_id):
    resp = requests.post(f"{ENDPOINT}/chat", json={"message": question})
    return resp.json()["response"]

# Evaluate
mlflow.genai.evaluate(
    data=dataset,
    predict_fn=predict_fn,
    scorers=[
        AnswerCorrectness(),
        ToolMatch(),
        BehaviorCoverage(),
        Correctness(),
        RelevanceToQuery(),
        ExpectationsGuidelines(),
        SafetyGuidelines(),
        ErrorHandlingGuidelines(),
    ],
)
```

### Mock MCP Data

`mock_mcp_data.json` contains synthetic data that makes the dataset's expected answers true:

- **6 CVEs**: CVE-2024-6387, CVE-2024-3094, CVE-2024-1086, CVE-2024-1234, CVE-2024-21626, CVE-2099-00000 (invalid)
- **4 hosts**: web-server-prod-01, db-server-prod-01, app-server-staging-02, app-worker-staging-03
- **3 advisor rules**: network_firewall, sshd_config, kernel_panic
- **RHEL lifecycle**: versions 7, 8, 8.8, 9, 9.2, 9.4, 10
- **AppStreams**: Node.js, Python, PHP
- **Image Builder**: blueprints, distributions, composes
- **Content Sources, RBAC, RHSM, Remediations**

## Standalone Evaluation Pipeline

### Dry Run (validate dataset, no agent needed)

```bash
python eval_runner.py --dry-run
```

### Run Evaluation

```bash
python eval_runner.py --endpoint http://localhost:8080 --token YOUR_TOKEN
```

### With Filters

```bash
python eval_runner.py --endpoint URL --category vulnerability
python eval_runner.py --endpoint URL --difficulty easy --type binary
python eval_runner.py --endpoint URL --ids "VULN-B-001,VULN-B-002"
```

## Question Types

| Type | Description | Expected Answer Format |
|------|-------------|----------------------|
| `binary` | Yes/no questions | `"yes"` or `"no"` |
| `single_select` | Pick one option | Option label (e.g. `"B"`) |
| `multiple_select` | Pick multiple options | List of labels (e.g. `["A", "C"]`) |
| `substring_match` | Required substrings must appear | List of strings |
| `exact_match` | Normalized string equality | Single string |
| `ordered_list` | Items in correct order | Ordered list of strings |
| `free_form` | Open-ended answers | Reference answer text |

## Categories

The dataset covers 11 evaluation categories:

- `vulnerability` - CVE tracking and management
- `inventory` - Host inventory management
- `advisor` - Insights Advisor recommendations
- `planning` - RHEL lifecycle planning
- `image_builder` - Custom RHEL image creation
- `remediations` - Automated remediation playbooks
- `content_sources` - Repository and content management
- `rbac` - Role-Based Access Control
- `rhsm` - Red Hat Subscription Manager
- `cross_domain` - Multi-step workflows spanning multiple tool domains
- `guardrails` - Safety, prompt injection, and out-of-scope request handling

## Running Tests

```bash
python -m pytest tests/ -v
```
