"""Shared fixtures for eval_grader tests."""

from __future__ import annotations

import pytest

from eval_grader import BehaviorGrader, Grader, SafetyGrader, ToolGrader


# ---------------------------------------------------------------------------
# Grader instances
# ---------------------------------------------------------------------------

@pytest.fixture
def grader() -> Grader:
    """Return a fresh core Grader instance."""
    return Grader()


@pytest.fixture
def tool_grader() -> ToolGrader:
    """Return a fresh ToolGrader instance."""
    return ToolGrader()


@pytest.fixture
def safety_grader() -> SafetyGrader:
    """Return a fresh SafetyGrader instance."""
    return SafetyGrader()


@pytest.fixture
def behavior_grader() -> BehaviorGrader:
    """Return a fresh BehaviorGrader instance."""
    return BehaviorGrader()


# ---------------------------------------------------------------------------
# Sample questions
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_question_yes() -> dict:
    return {"question_type": "BINARY", "expected_answer": "yes"}


@pytest.fixture
def binary_question_no() -> dict:
    return {"question_type": "BINARY", "expected_answer": "no"}


@pytest.fixture
def single_select_question() -> dict:
    return {
        "question_type": "SINGLE_SELECT",
        "expected_answer": "B",
        "options": {
            "A": "Ubuntu",
            "B": "Red Hat Enterprise Linux",
            "C": "Fedora",
            "D": "CentOS",
        },
    }


@pytest.fixture
def multiple_select_question() -> dict:
    return {
        "question_type": "MULTIPLE_SELECT",
        "expected_answer": ["A", "C"],
        "options": {
            "A": "dnf update",
            "B": "apt-get upgrade",
            "C": "yum update",
            "D": "pacman -Syu",
        },
    }


@pytest.fixture
def substring_match_question() -> dict:
    return {
        "question_type": "SUBSTRING_MATCH",
        "expected_answer": ["Red Hat", "Enterprise Linux", "subscription"],
    }


@pytest.fixture
def exact_match_question() -> dict:
    return {
        "question_type": "EXACT_MATCH",
        "expected_answer": "Red Hat Enterprise Linux 9",
    }


@pytest.fixture
def ordered_list_question() -> dict:
    return {
        "question_type": "ORDERED_LIST",
        "expected_answer": ["install", "configure", "deploy"],
    }
