"""Tests that validate the evaluation dataset (eval_dataset.json).

These tests load the dataset once and verify structural integrity,
referential consistency, and minimum coverage requirements.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Constants -- canonical allowed values
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "id",
    "category",
    "question",
    "question_type",
    "expected_answer",
    "expected_tools",
    "scenario_type",
    "expected_behavior",
    "difficulty",
    "tags",
]

VALID_QUESTION_TYPES = {
    "binary",
    "single_select",
    "multiple_select",
    "substring_match",
    "exact_match",
    "ordered_list",
    "free_form",
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}

VALID_SCENARIO_TYPES = {
    "single_tool",
    "multi_step",
    "no_tool",
    "error_handling",
    "pagination",
}

VALID_CATEGORIES = {
    "image_builder",
    "rhsm",
    "vulnerability",
    "remediations",
    "advisor",
    "inventory",
    "content_sources",
    "rbac",
    "planning",
    "cross_domain",
    "guardrails",
}

# All known MCP tool names in both bare and prefixed forms.
# The dataset uses prefixed names (e.g., "vulnerability__get_cves").
_BARE_TOOL_NAMES = {
    "get_blueprint_details", "get_blueprints", "get_compose_details",
    "get_composes", "get_distributions", "get_openapi", "get_org_id",
    "blueprint_compose", "create_blueprint", "update_blueprint",
    "get_activation_key", "get_activation_keys",
    "explain_cves", "get_cve", "get_cve_systems", "get_cves",
    "get_system_cves", "get_systems",
    "create_vuln_playbook",
    "get_active_rules", "get_hosts_details_for_rule",
    "get_hosts_hitting_a_rule", "get_recommendations_stats",
    "get_rule_by_text_search", "get_rule_details", "get_rule_from_node_id",
    "find_host_by_name", "get_host_details", "get_host_system_profile",
    "get_host_tags", "list_hosts",
    "list_repositories",
    "get_all_access",
    "get_appstreams_lifecycle", "get_relevant_appstreams",
    "get_relevant_rhel_lifecycle", "get_relevant_upcoming",
    "get_rhel_lifecycle", "get_upcoming_changes",
}

_TOOL_PREFIXES = [
    "vulnerability__", "inventory__", "advisor__", "planning__",
    "remediations__", "image_builder__", "image-builder__",
    "content-sources__", "content_sources__", "rbac__", "rhsm__",
]

VALID_TOOL_NAMES = set(_BARE_TOOL_NAMES)
for prefix in _TOOL_PREFIXES:
    for name in _BARE_TOOL_NAMES:
        VALID_TOOL_NAMES.add(f"{prefix}{name}")

MIN_TOTAL_QUESTIONS = 250
MIN_QUESTIONS_PER_CATEGORY = 5
MIN_QUESTIONS_PER_TYPE = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATASET_PATH = Path(__file__).resolve().parent.parent / "eval_dataset.json"


@pytest.fixture(scope="module")
def dataset() -> List[Dict[str, Any]]:
    """Load the evaluation dataset and return the list of question dicts."""
    assert DATASET_PATH.exists(), f"Dataset file not found: {DATASET_PATH}"
    with open(DATASET_PATH, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Accept both a bare list and an object wrapping the list.
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("questions", "items", "data"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
    pytest.fail("Dataset must be a JSON array or an object with a 'questions' key")


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

class TestRequiredFields:
    """Every question must contain all required fields."""

    def test_all_questions_have_required_fields(
        self, dataset: List[Dict[str, Any]]
    ) -> None:
        missing_report: List[str] = []
        for idx, q in enumerate(dataset):
            qid = q.get("id", f"<index {idx}>")
            missing = [f for f in REQUIRED_FIELDS if f not in q]
            if missing:
                missing_report.append(f"{qid}: missing {missing}")
        assert not missing_report, (
            "Questions with missing required fields:\n"
            + "\n".join(missing_report)
        )


class TestUniqueIds:
    """All question IDs must be unique."""

    def test_ids_are_unique(self, dataset: List[Dict[str, Any]]) -> None:
        ids = [q["id"] for q in dataset if "id" in q]
        duplicates = [qid for qid, count in Counter(ids).items() if count > 1]
        assert not duplicates, f"Duplicate question IDs found: {duplicates}"


class TestNoDuplicateQuestions:
    """No two questions should share the exact same question text."""

    def test_no_duplicate_question_text(
        self, dataset: List[Dict[str, Any]]
    ) -> None:
        texts = [q["question"] for q in dataset if "question" in q]
        seen: Dict[str, str] = {}
        duplicates: List[str] = []
        for q in dataset:
            text = q.get("question", "")
            qid = q.get("id", "?")
            if text in seen:
                duplicates.append(
                    f"'{qid}' duplicates '{seen[text]}' -- "
                    f"text: {text[:80]}..."
                )
            else:
                seen[text] = qid
        assert not duplicates, (
            "Duplicate question texts found:\n" + "\n".join(duplicates)
        )


# ---------------------------------------------------------------------------
# Enum / value validity tests
# ---------------------------------------------------------------------------

class TestQuestionTypes:
    """All question_type values must belong to the valid set."""

    def test_valid_question_types(
        self, dataset: List[Dict[str, Any]]
    ) -> None:
        invalid: List[str] = []
        for q in dataset:
            qt = q.get("question_type", "")
            if qt not in VALID_QUESTION_TYPES:
                invalid.append(f"{q.get('id', '?')}: '{qt}'")
        assert not invalid, (
            f"Invalid question_type values (allowed: {sorted(VALID_QUESTION_TYPES)}):\n"
            + "\n".join(invalid)
        )


class TestCategories:
    """All category values must belong to the valid set."""

    def test_valid_categories(self, dataset: List[Dict[str, Any]]) -> None:
        invalid: List[str] = []
        for q in dataset:
            cat = q.get("category", "")
            if cat not in VALID_CATEGORIES:
                invalid.append(f"{q.get('id', '?')}: '{cat}'")
        assert not invalid, (
            f"Invalid category values (allowed: {sorted(VALID_CATEGORIES)}):\n"
            + "\n".join(invalid)
        )


class TestDifficulties:
    """All difficulty values must belong to the valid set."""

    def test_valid_difficulties(self, dataset: List[Dict[str, Any]]) -> None:
        invalid: List[str] = []
        for q in dataset:
            diff = q.get("difficulty", "")
            if diff not in VALID_DIFFICULTIES:
                invalid.append(f"{q.get('id', '?')}: '{diff}'")
        assert not invalid, (
            f"Invalid difficulty values (allowed: {sorted(VALID_DIFFICULTIES)}):\n"
            + "\n".join(invalid)
        )


class TestScenarioTypes:
    """All scenario_type values must belong to the valid set."""

    def test_valid_scenario_types(
        self, dataset: List[Dict[str, Any]]
    ) -> None:
        invalid: List[str] = []
        for q in dataset:
            st = q.get("scenario_type", "")
            if st not in VALID_SCENARIO_TYPES:
                invalid.append(f"{q.get('id', '?')}: '{st}'")
        assert not invalid, (
            f"Invalid scenario_type values (allowed: {sorted(VALID_SCENARIO_TYPES)}):\n"
            + "\n".join(invalid)
        )


# ---------------------------------------------------------------------------
# Type-specific answer constraints
# ---------------------------------------------------------------------------

class TestBinaryQuestions:
    """Binary questions must have 'yes' or 'no' as expected_answer."""

    def test_binary_expected_answer(
        self, dataset: List[Dict[str, Any]]
    ) -> None:
        invalid: List[str] = []
        for q in dataset:
            if q.get("question_type") == "binary":
                ans = q.get("expected_answer")
                if not isinstance(ans, str) or ans.lower() not in ("yes", "no"):
                    invalid.append(
                        f"{q.get('id', '?')}: expected_answer={ans!r}"
                    )
        assert not invalid, (
            "Binary questions must have 'yes' or 'no' as expected_answer:\n"
            + "\n".join(invalid)
        )


class TestSelectQuestions:
    """single_select and multiple_select questions must have options."""

    def test_select_questions_have_options(
        self, dataset: List[Dict[str, Any]]
    ) -> None:
        invalid: List[str] = []
        for q in dataset:
            qt = q.get("question_type", "")
            if qt in ("single_select", "multiple_select"):
                options = q.get("options")
                if options is None:
                    invalid.append(
                        f"{q.get('id', '?')} ({qt}): options is null"
                    )
                elif isinstance(options, dict) and len(options) < 2:
                    invalid.append(
                        f"{q.get('id', '?')} ({qt}): options has "
                        f"{len(options)} item(s), need >= 2"
                    )
                elif isinstance(options, list) and len(options) < 2:
                    invalid.append(
                        f"{q.get('id', '?')} ({qt}): options has "
                        f"{len(options)} item(s), need >= 2"
                    )
        assert not invalid, (
            "Select questions must have non-null options with >= 2 items:\n"
            + "\n".join(invalid)
        )


class TestSubstringMatchQuestions:
    """substring_match questions must have a list as expected_answer."""

    def test_substring_match_expected_answer_is_list(
        self, dataset: List[Dict[str, Any]]
    ) -> None:
        invalid: List[str] = []
        for q in dataset:
            if q.get("question_type") == "substring_match":
                ans = q.get("expected_answer")
                if not isinstance(ans, list):
                    invalid.append(
                        f"{q.get('id', '?')}: expected_answer type is "
                        f"{type(ans).__name__}, expected list"
                    )
        assert not invalid, (
            "substring_match questions must have a list as expected_answer:\n"
            + "\n".join(invalid)
        )


# ---------------------------------------------------------------------------
# Tool name validation
# ---------------------------------------------------------------------------

class TestExpectedTools:
    """expected_tools must only contain known MCP tool names."""

    def test_expected_tools_are_valid(
        self, dataset: List[Dict[str, Any]]
    ) -> None:
        invalid: List[str] = []
        for q in dataset:
            tools = q.get("expected_tools", [])
            if not isinstance(tools, list):
                invalid.append(
                    f"{q.get('id', '?')}: expected_tools is not a list "
                    f"(got {type(tools).__name__})"
                )
                continue
            bad = [t for t in tools if t not in VALID_TOOL_NAMES]
            if bad:
                invalid.append(f"{q.get('id', '?')}: unknown tools {bad}")
        assert not invalid, (
            "Questions with invalid expected_tools:\n" + "\n".join(invalid)
        )


# ---------------------------------------------------------------------------
# Tags validation
# ---------------------------------------------------------------------------

class TestTags:
    """Tags must be non-empty lists."""

    def test_tags_are_nonempty_lists(
        self, dataset: List[Dict[str, Any]]
    ) -> None:
        invalid: List[str] = []
        for q in dataset:
            tags = q.get("tags")
            if not isinstance(tags, list) or len(tags) == 0:
                invalid.append(
                    f"{q.get('id', '?')}: tags={tags!r}"
                )
        assert not invalid, (
            "Every question must have a non-empty list of tags:\n"
            + "\n".join(invalid)
        )


# ---------------------------------------------------------------------------
# Coverage / minimum count tests
# ---------------------------------------------------------------------------

class TestMinimumCounts:
    """The dataset must meet minimum size and distribution requirements."""

    def test_minimum_total_questions(
        self, dataset: List[Dict[str, Any]]
    ) -> None:
        assert len(dataset) >= MIN_TOTAL_QUESTIONS, (
            f"Dataset has {len(dataset)} questions, "
            f"need at least {MIN_TOTAL_QUESTIONS}"
        )

    def test_minimum_questions_per_category(
        self, dataset: List[Dict[str, Any]]
    ) -> None:
        counts = Counter(q.get("category", "") for q in dataset)
        under: List[str] = []
        for cat in VALID_CATEGORIES:
            if counts.get(cat, 0) < MIN_QUESTIONS_PER_CATEGORY:
                under.append(f"{cat}: {counts.get(cat, 0)}")
        assert not under, (
            f"Categories with fewer than {MIN_QUESTIONS_PER_CATEGORY} questions:\n"
            + "\n".join(under)
        )

    def test_minimum_questions_per_question_type(
        self, dataset: List[Dict[str, Any]]
    ) -> None:
        counts = Counter(q.get("question_type", "") for q in dataset)
        under: List[str] = []
        for qt in VALID_QUESTION_TYPES:
            if counts.get(qt, 0) < MIN_QUESTIONS_PER_TYPE:
                under.append(f"{qt}: {counts.get(qt, 0)}")
        assert not under, (
            f"Question types with fewer than {MIN_QUESTIONS_PER_TYPE} questions:\n"
            + "\n".join(under)
        )
