"""Tests for the mlflow_eval integration module."""

import json
import os

import pytest

from mlflow_eval.dataset import load_dataset
from mlflow_eval.scorers import (
    AnswerCorrectness,
    BehaviorCoverage,
    ErrorHandlingGuidelines,
    SafetyGuidelines,
    ToolCallCorrectness,
    ToolMatch,
    grade_response,
)


# ---------------------------------------------------------------------------
# grade_response — deterministic grading logic
# ---------------------------------------------------------------------------


class TestGradeResponse:
    def test_binary_yes(self):
        assert grade_response("binary", "yes", None, "Yes, it is affected.")[0] == 1.0

    def test_binary_no(self):
        assert grade_response("binary", "no", None, "No, that cannot be done.")[0] == 1.0

    def test_binary_wrong(self):
        assert grade_response("binary", "yes", None, "No issues found.")[0] == 0.0

    def test_single_select(self):
        assert grade_response("single_select", "get_cve", ["get_cves", "get_cve"], "Use get_cve.")[0] == 1.0

    def test_single_select_miss(self):
        assert grade_response("single_select", "get_cve", ["get_cves", "get_cve"], "Use explain.")[0] == 0.0

    def test_multiple_select_all(self):
        s, _ = grade_response("multiple_select", ["get_cve", "get_cve_systems"], None,
                               "Use get_cve and get_cve_systems.")
        assert s == 1.0

    def test_multiple_select_partial(self):
        s, _ = grade_response("multiple_select", ["get_cve", "get_cve_systems"], None,
                               "Use get_cve only.")
        assert s == 0.5

    def test_substring_match_all(self):
        s, _ = grade_response("substring_match", ["get_cves", "critical", "exploit"], None,
                               "Use get_cves with critical severity and exploit filter.")
        assert s == 1.0

    def test_substring_match_partial(self):
        s, _ = grade_response("substring_match", ["get_cves", "critical", "exploit"], None,
                               "Use get_cves with critical severity.")
        assert abs(s - 2 / 3) < 1e-3

    def test_exact_match(self):
        assert grade_response("exact_match", "get_cve", None, "get_cve")[0] == 1.0

    def test_exact_match_substring(self):
        assert grade_response("exact_match", "get_cve", None, "The tool is get_cve.")[0] == 0.8

    def test_ordered_list_correct(self):
        s, _ = grade_response("ordered_list", ["get_cve", "get_cve_systems"], None,
                               "First get_cve, then get_cve_systems.")
        assert s == 1.0

    def test_ordered_list_wrong(self):
        s, _ = grade_response("ordered_list", ["get_cve", "get_cve_systems"], None,
                               "Start with get_cve_systems, then get_cve.")
        assert s == 0.5

    def test_unknown_type(self):
        assert grade_response("unknown", "x", None, "resp")[0] == 0.0


# ---------------------------------------------------------------------------
# Scorer classes
# ---------------------------------------------------------------------------


class TestAnswerCorrectness:
    def test_binary_pass(self):
        result = AnswerCorrectness()(
            inputs={"question": "q"}, outputs="Yes, it is.",
            expectations={"question_type": "binary", "expected_response": "yes"},
        )
        assert result == 1.0

    def test_binary_fail(self):
        result = AnswerCorrectness()(
            inputs={"question": "q"}, outputs="No.",
            expectations={"question_type": "binary", "expected_response": "yes"},
        )
        assert result == 0.0


class TestToolMatch:
    def test_found(self):
        result = ToolMatch()(
            inputs={"question": "q"},
            outputs="I used get_cve_systems to check.",
            expectations={"expected_tools": ["vulnerability__get_cve_systems"]},
        )
        assert result == 1.0

    def test_missing(self):
        result = ToolMatch()(
            inputs={"question": "q"},
            outputs="I don't know.",
            expectations={"expected_tools": ["vulnerability__get_cve_systems"]},
        )
        assert result == 0.0

    def test_no_tools_expected(self):
        result = ToolMatch()(
            inputs={"question": "q"}, outputs="No.",
            expectations={"expected_tools": []},
        )
        assert result == 1.0


class TestBehaviorCoverage:
    def test_coverage(self):
        result = BehaviorCoverage()(
            inputs={"question": "q"},
            outputs="I'll call get_cve_systems with CVE-2024-6387 to find affected systems.",
            expectations={
                "expected_behavior": "The agent should call get_cve_systems with "
                                     "CVE-2024-6387 to determine affected systems.",
            },
        )
        assert result > 0.5

    def test_no_behavior(self):
        result = BehaviorCoverage()(
            inputs={"question": "q"}, outputs="Yes.",
            expectations={"expected_behavior": ""},
        )
        assert result == 1.0


class TestToolCallCorrectness:
    def test_no_tools(self):
        result = ToolCallCorrectness(agent_experiment_name="nonexistent")(
            inputs={"question": "q"}, outputs="answer",
            expectations={"expected_tools": []},
        )
        assert result == "yes"


class TestGuidelinesScorers:
    def test_safety_creates(self):
        s = SafetyGuidelines()
        assert s.name == "safety"
        assert len(s.guidelines) == 4

    def test_error_handling_creates(self):
        s = ErrorHandlingGuidelines()
        assert s.name == "error_handling"
        assert len(s.guidelines) == 4


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_loads_default_path(self):
        result = load_dataset(limit=5)
        assert len(result) == 5
        assert "inputs" in result[0]
        assert "expectations" in result[0]
        assert result[0]["inputs"]["question_id"]

    def test_loads_all(self):
        result = load_dataset()
        assert len(result) > 100

    def test_filter_category(self):
        result = load_dataset(category="vulnerability")
        assert all(d["expectations"]["category"] == "vulnerability" for d in result)

    def test_filter_difficulty(self):
        result = load_dataset(difficulty="easy")
        assert all(d["expectations"]["difficulty"] == "easy" for d in result)

    def test_filter_ids(self):
        result = load_dataset(ids=["V-001", "V-002"])
        assert len(result) == 2
        qids = {d["inputs"]["question_id"] for d in result}
        assert qids == {"V-001", "V-002"}

    def test_expectations_structure(self):
        result = load_dataset(ids=["V-001"])
        exp = result[0]["expectations"]
        assert exp["question_type"] == "binary"
        assert exp["expected_response"] == "yes"
        assert "vulnerability__get_cve_systems" in exp["expected_tools"]
        assert exp["expected_behavior"]

    def test_guidelines_populated(self):
        result = load_dataset(ids=["V-001"])
        exp = result[0]["expectations"]
        assert "guidelines" in exp
        assert len(exp["guidelines"]) == 1
