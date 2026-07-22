# Red Hat Lightspeed Agent Evaluation Framework

Automated evaluation framework for testing the Red Hat Lightspeed Agent. It loads questions from a curated dataset, sends them to the agent endpoint, grades the responses, and generates detailed reports.

## Project Structure

```
eval_runner.py      - Orchestrates the evaluation pipeline (load, filter, execute, grade)
eval_grader.py      - Grading logic for all question types (binary, select, match, etc.)
eval_reporter.py    - Generates console, JSON, and HTML reports from graded results
eval_config.py      - Configuration management (CLI, env vars, config files)
eval_utils.py       - Shared utility functions (text normalization, statistics, I/O)
eval_dataset.json   - Evaluation dataset with 250+ questions across 9 categories
tests/              - pytest test suite
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

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
# Run only vulnerability questions
python eval_runner.py --endpoint URL --category vulnerability

# Run only easy binary questions
python eval_runner.py --endpoint URL --difficulty easy --type binary

# Run specific question IDs
python eval_runner.py --endpoint URL --ids "VULN-B-001,VULN-B-002"

# Filter by tags
python eval_runner.py --endpoint URL --tags "cve,security"
```

### Resume a Previous Run

```bash
python eval_runner.py --endpoint URL --resume results.json --output results.json
```

## Question Types

The grader supports seven question types:

| Type | Description | Expected Answer Format |
|------|-------------|----------------------|
| `binary` | Yes/no questions | `"yes"` or `"no"` |
| `single_select` | Pick one option | Option label (e.g. `"B"`) |
| `multiple_select` | Pick multiple options | List of labels (e.g. `["A", "C"]`) |
| `substring_match` | Required substrings must appear | List of strings |
| `exact_match` | Normalized string equality | Single string |
| `ordered_list` | Items in correct order | Ordered list of strings |
| `free_form` | Open-ended answers | Reference answer text |

### Free-form Evaluation

Free-form questions support two selectable evaluation strategies:

**LLM-as-Judge** -- sends the question, reference answer, and agent response to a judge LLM that scores correctness, relevance, and completeness (0-1 each):

```bash
python eval_runner.py --endpoint URL \
  --free-form-strategy llm_judge \
  --judge-endpoint http://judge-llm:8080/v1/chat/completions \
  --judge-model gpt-4o \
  --judge-pass-threshold 0.7
```

**Semantic Similarity** -- computes cosine similarity between embeddings of the reference and agent responses:

```bash
pip install sentence-transformers
python eval_runner.py --endpoint URL \
  --free-form-strategy semantic_similarity \
  --embedding-model all-MiniLM-L6-v2 \
  --similarity-pass-threshold 0.75
```

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

## Reports

### Generate Reports

```python
from eval_reporter import EvalReporter, QuestionResult

reporter = EvalReporter(results=results_list)
paths = reporter.generate_all(output_dir="./reports", console=True)
```

### Report Formats

- **Console** - ASCII table printed to stdout
- **JSON** - Machine-readable with full breakdown
- **HTML** - Interactive report with charts, filtering, and sorting

## Configuration

Configuration sources (highest priority wins):

1. CLI arguments
2. Environment variables (`LIGHTSPEED_ENDPOINT`, `LIGHTSPEED_TOKEN`, etc.)
3. Config file (`eval_config.yaml` or `eval_config.json`)
4. Defaults

### Environment Variables

```bash
export LIGHTSPEED_ENDPOINT=http://localhost:8080
export LIGHTSPEED_TOKEN=your-token
export LIGHTSPEED_CONCURRENCY=4
export LIGHTSPEED_TIMEOUT=120
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Verify Imports

```bash
python -c "from eval_grader import Grader; from eval_utils import normalize_text; from eval_config import EvalConfig; from eval_reporter import EvalReporter; print('All imports OK')"
```
