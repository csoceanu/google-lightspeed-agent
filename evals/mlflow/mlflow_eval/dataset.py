"""Load evaluation cases for mlflow.genai.evaluate().

Single source of truth: the ``cases/`` directory in AEH format.
Each case has ``input.yaml`` (question) and ``annotations.yaml``
(ground truth). The same cases are used by both our MLflow pipeline
and the AEH pipeline.

Fallback: ``eval_dataset.json`` can still be loaded for the full 257-question
set with filtering. The ``cases/`` directory holds the curated deterministic
subset (1 per question type).
"""

import json
from pathlib import Path
from typing import Any

import yaml


CASES_DIR = Path(__file__).resolve().parent / "cases"
DATASET_PATH = Path(__file__).resolve().parent.parent / "eval_dataset.json"


def load_cases(cases_dir: str | None = None) -> list[dict[str, Any]]:
    """Load evaluation cases from the ``cases/`` directory (AEH format).

    Each case directory contains:
    - ``input.yaml``: prompt, category, scenario_type, difficulty
    - ``annotations.yaml``: expected_response, expected_tools,
      expected_behavior, question_type

    Returns:
        List of dicts with ``inputs`` and ``expectations`` keys,
        ready for ``mlflow.genai.evaluate(data=...)``.
    """
    cases_path = Path(cases_dir) if cases_dir else CASES_DIR
    if not cases_path.is_dir():
        raise FileNotFoundError(f"Cases directory not found: {cases_path}")

    dataset = []
    for case_dir in sorted(cases_path.iterdir()):
        if not case_dir.is_dir():
            continue
        input_file = case_dir / "input.yaml"
        annotations_file = case_dir / "annotations.yaml"
        if not input_file.exists():
            continue

        with open(input_file, "r", encoding="utf-8") as f:
            inp = yaml.safe_load(f)
        ann = {}
        if annotations_file.exists():
            with open(annotations_file, "r", encoding="utf-8") as f:
                ann = yaml.safe_load(f) or {}

        question_id = case_dir.name.replace("case-", "")

        expected_behavior = ann.get("expected_behavior", "")
        if isinstance(expected_behavior, list):
            expected_behavior = " ".join(expected_behavior)

        expectations: dict[str, Any] = {
            "question_type": ann.get("question_type", ""),
            "expected_response": ann.get("expected_response", ""),
            "expected_tools": ann.get("expected_tools", []),
            "expected_behavior": expected_behavior,
            "category": inp.get("category", ""),
            "difficulty": inp.get("difficulty", ""),
        }
        if ann.get("options") is not None:
            expectations["options"] = ann["options"]
        if expected_behavior:
            guidelines = ann.get("expected_behavior", [])
            if isinstance(guidelines, str):
                guidelines = [guidelines]
            expectations["guidelines"] = guidelines

        dataset.append({
            "inputs": {
                "question": inp["prompt"],
                "question_id": question_id,
            },
            "expectations": expectations,
        })

    return dataset


def load_dataset(
    path: str | None = None,
    category: str | None = None,
    difficulty: str | None = None,
    ids: list[str] | None = None,
    limit: int | None = None,
    per_type: int | None = None,
) -> list[dict[str, Any]]:
    """Load eval_dataset.json for ``mlflow.genai.evaluate()``.

    For the curated deterministic subset, use :func:`load_cases` instead.
    This function loads the full 257-question dataset with filtering.

    Args:
        path: Path to eval_dataset.json. Defaults to the file in this repo.
        category: Filter by category (e.g. ``"vulnerability"``).
        difficulty: Filter by difficulty (e.g. ``"easy"``).
        ids: Only include these question IDs.
        limit: Max number of questions to return.
        per_type: Pick this many questions per question_type (deterministic).

    Returns:
        List of dicts with ``inputs`` and ``expectations`` keys.
    """
    if path is None:
        path = str(DATASET_PATH)

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, dict):
        items = raw.get("questions", raw.get("items", raw.get("data", [])))
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Dataset must be a JSON array or object")

    if category:
        items = [q for q in items if q.get("category", "").lower() == category.lower()]
    if difficulty:
        items = [q for q in items if q.get("difficulty", "").lower() == difficulty.lower()]
    if ids:
        id_set = set(ids)
        items = [q for q in items if q["id"] in id_set]

    if per_type is not None:
        by_type: dict[str, list] = {}
        for q in items:
            qt = q.get("question_type", "unknown")
            by_type.setdefault(qt, []).append(q)
        sampled = []
        for qt in sorted(by_type):
            sampled.extend(by_type[qt][:per_type])
        items = sampled
    elif limit:
        items = items[:limit]

    dataset = []
    for q in items:
        expectations: dict[str, Any] = {
            "question_type": q.get("question_type", ""),
            "expected_response": q.get("expected_answer", ""),
            "expected_tools": q.get("expected_tools", []),
            "expected_behavior": q.get("expected_behavior", ""),
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
        }
        if q.get("options") is not None:
            expectations["options"] = q["options"]
        behavior = q.get("expected_behavior", "")
        if behavior:
            expectations["guidelines"] = [behavior]

        dataset.append({
            "inputs": {
                "question": q["question"],
                "question_id": q["id"],
            },
            "expectations": expectations,
        })

    return dataset
