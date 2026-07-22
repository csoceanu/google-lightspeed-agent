"""Tests for eval_reporter.py -- report generation for the evaluation framework."""

from __future__ import annotations

import json
import os
from typing import List

import pytest

from eval_reporter import (
    EvalReporter,
    QuestionResult,
    ReportData,
    _bar,
    _breakdown_dict,
    _pct,
    _percentile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    qid: str = "Q-001",
    passed: bool = True,
    category: str = "vulnerability",
    difficulty: str = "easy",
    question_type: str = "binary",
    scenario_type: str = "single_tool",
    latency_ms: float = 500.0,
    question_text: str = "Sample question?",
) -> QuestionResult:
    """Convenience factory for QuestionResult."""
    return QuestionResult(
        question_id=qid,
        question_text=question_text,
        category=category,
        question_type=question_type,
        difficulty=difficulty,
        scenario_type=scenario_type,
        passed=passed,
        expected_answer="expected",
        actual_answer="actual" if passed else "wrong",
        grade_details="" if passed else "Did not match",
        latency_ms=latency_ms,
    )


def _mixed_results() -> List[QuestionResult]:
    """Return a list with a mix of passing and failing results."""
    return [
        _make_result("Q-001", passed=True, category="vulnerability", difficulty="easy", latency_ms=400),
        _make_result("Q-002", passed=True, category="vulnerability", difficulty="medium", latency_ms=800),
        _make_result("Q-003", passed=False, category="inventory", difficulty="hard", latency_ms=1200),
        _make_result("Q-004", passed=False, category="advisor", difficulty="easy", latency_ms=600),
        _make_result("Q-005", passed=True, category="advisor", difficulty="medium", latency_ms=900),
    ]


# =========================================================================
# Console report
# =========================================================================


class TestConsoleReport:
    """Tests for generate_console_report()."""

    def test_console_report_contains_header(self):
        reporter = EvalReporter(results=_mixed_results())
        report = reporter.generate_console_report()
        assert "LIGHTSPEED AGENT EVALUATION REPORT" in report

    def test_console_report_contains_overall_score(self):
        reporter = EvalReporter(results=_mixed_results())
        report = reporter.generate_console_report()
        assert "OVERALL SCORE" in report
        # 3 out of 5 passed -> 60.0%
        assert "3/5 passed" in report
        assert "60.0%" in report

    def test_console_report_contains_category_breakdown(self):
        reporter = EvalReporter(results=_mixed_results())
        report = reporter.generate_console_report()
        assert "BY CATEGORY" in report
        assert "vulnerability" in report
        assert "advisor" in report

    def test_console_report_lists_failed_questions(self):
        reporter = EvalReporter(results=_mixed_results())
        report = reporter.generate_console_report()
        assert "FAILED QUESTIONS (2)" in report
        assert "Q-003" in report
        assert "Q-004" in report

    def test_console_report_all_passed_message(self):
        results = [_make_result("Q-001", passed=True)]
        reporter = EvalReporter(results=results)
        report = reporter.generate_console_report()
        assert "All questions passed!" in report

    def test_console_report_timing_section(self):
        reporter = EvalReporter(results=_mixed_results())
        report = reporter.generate_console_report()
        assert "TIMING STATISTICS" in report
        assert "Average" in report


# =========================================================================
# JSON report
# =========================================================================


class TestJsonReport:
    """Tests for generate_json_report() and write_json_report()."""

    def test_json_report_is_valid_json(self):
        reporter = EvalReporter(results=_mixed_results())
        raw = reporter.generate_json_report()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_json_report_summary_counts(self):
        reporter = EvalReporter(results=_mixed_results())
        parsed = json.loads(reporter.generate_json_report())
        summary = parsed["summary"]
        assert summary["total_questions"] == 5
        assert summary["passed"] == 3
        assert summary["failed"] == 2
        assert summary["score"] == pytest.approx(60.0)

    def test_json_report_contains_results_list(self):
        reporter = EvalReporter(results=_mixed_results())
        parsed = json.loads(reporter.generate_json_report())
        assert len(parsed["results"]) == 5

    def test_json_report_has_report_version(self):
        reporter = EvalReporter(results=_mixed_results())
        parsed = json.loads(reporter.generate_json_report())
        assert parsed["report_version"] == "1.0"

    def test_write_json_report_creates_file(self, tmp_path):
        reporter = EvalReporter(results=_mixed_results())
        out = str(tmp_path / "report.json")
        returned_path = reporter.write_json_report(out)
        assert returned_path == out
        assert os.path.isfile(out)
        parsed = json.loads(open(out, encoding="utf-8").read())
        assert parsed["summary"]["total_questions"] == 5


# =========================================================================
# HTML report
# =========================================================================


class TestHtmlReport:
    """Tests for generate_html_report() and write_html_report()."""

    def test_html_report_is_valid_html_document(self):
        reporter = EvalReporter(results=_mixed_results())
        html = reporter.generate_html_report()
        assert html.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_html_report_contains_title(self):
        reporter = EvalReporter(results=_mixed_results())
        html = reporter.generate_html_report()
        assert "Lightspeed Agent Evaluation Report" in html

    def test_html_report_contains_score(self):
        reporter = EvalReporter(results=_mixed_results())
        html = reporter.generate_html_report()
        assert "60.0%" in html

    def test_html_report_contains_result_rows(self):
        reporter = EvalReporter(results=_mixed_results())
        html = reporter.generate_html_report()
        assert "Q-001" in html
        assert "Q-003" in html

    def test_html_report_escapes_special_characters(self):
        results = [
            _make_result(
                qid="Q-XSS",
                question_text='Is <script>alert("xss")</script> safe?',
                passed=False,
            )
        ]
        reporter = EvalReporter(results=results)
        html = reporter.generate_html_report()
        # The raw script tag must NOT appear; it should be escaped
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_write_html_report_creates_file(self, tmp_path):
        reporter = EvalReporter(results=_mixed_results())
        out = str(tmp_path / "report.html")
        returned_path = reporter.write_html_report(out)
        assert returned_path == out
        assert os.path.isfile(out)


# =========================================================================
# Empty results
# =========================================================================


class TestEmptyResults:
    """Tests for reporter behaviour when no results are provided."""

    def test_console_report_with_empty_results(self):
        reporter = EvalReporter(results=[])
        report = reporter.generate_console_report()
        assert "LIGHTSPEED AGENT EVALUATION REPORT" in report
        assert "0/0 passed" in report

    def test_json_report_with_empty_results(self):
        reporter = EvalReporter(results=[])
        parsed = json.loads(reporter.generate_json_report())
        assert parsed["summary"]["total_questions"] == 0
        assert parsed["summary"]["score"] == 0.0

    def test_html_report_with_empty_results(self):
        reporter = EvalReporter(results=[])
        html = reporter.generate_html_report()
        assert "<!DOCTYPE html>" in html
        assert "0.0%" in html


# =========================================================================
# ReportData aggregation
# =========================================================================


class TestBuildReportData:
    """Tests for build_report_data() aggregation logic."""

    def test_score_is_percentage(self):
        reporter = EvalReporter(results=_mixed_results())
        data = reporter.build_report_data()
        assert data.score == pytest.approx(60.0)

    def test_category_breakdown_keys(self):
        reporter = EvalReporter(results=_mixed_results())
        data = reporter.build_report_data()
        assert "vulnerability" in data.category_breakdown
        assert "inventory" in data.category_breakdown
        assert "advisor" in data.category_breakdown

    def test_timing_populated(self):
        reporter = EvalReporter(results=_mixed_results())
        data = reporter.build_report_data()
        assert data.timing["avg_ms"] > 0
        assert "p50_ms" in data.timing
        assert "p95_ms" in data.timing
        assert "p99_ms" in data.timing

    def test_add_result_incremental(self):
        reporter = EvalReporter()
        reporter.add_result(_make_result("Q-A", passed=True))
        reporter.add_result(_make_result("Q-B", passed=False))
        data = reporter.build_report_data()
        assert data.total_questions == 2
        assert data.passed == 1
        assert data.failed == 1

    def test_set_config_reflected_in_report(self):
        reporter = EvalReporter(results=_mixed_results())
        reporter.set_config({"endpoint": "https://example.com", "timestamp": "2025-01-01T00:00:00Z"})
        data = reporter.build_report_data()
        assert data.config["endpoint"] == "https://example.com"
        assert data.config["timestamp"] == "2025-01-01T00:00:00Z"

    def test_timestamp_auto_generated_when_missing(self):
        reporter = EvalReporter(results=_mixed_results())
        data = reporter.build_report_data()
        assert "timestamp" in data.config


# =========================================================================
# Helper functions
# =========================================================================


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_pct_normal(self):
        assert _pct(3, 5).strip() == "60.0%"

    def test_pct_zero_denominator(self):
        assert "N/A" in _pct(0, 0)

    def test_bar_full(self):
        bar = _bar(1.0, 10)
        assert "=" * 10 in bar

    def test_bar_empty(self):
        bar = _bar(0.0, 10)
        assert "=" not in bar

    def test_bar_partial(self):
        bar = _bar(0.5, 10)
        assert ">" in bar

    def test_breakdown_dict_groups_correctly(self):
        results = [
            _make_result("Q1", passed=True, category="cat_a"),
            _make_result("Q2", passed=False, category="cat_a"),
            _make_result("Q3", passed=True, category="cat_b"),
        ]
        bd = _breakdown_dict(results, "category")
        assert bd["cat_a"]["total"] == 2
        assert bd["cat_a"]["passed"] == 1
        assert bd["cat_a"]["failed"] == 1
        assert bd["cat_b"]["total"] == 1
        assert bd["cat_b"]["passed"] == 1

    def test_percentile_helper_single_value(self):
        assert _percentile([42.0], 50) == 42.0

    def test_percentile_helper_empty(self):
        assert _percentile([], 50) == 0.0


# =========================================================================
# generate_all
# =========================================================================


class TestGenerateAll:
    """Tests for the generate_all() convenience method."""

    def test_generate_all_creates_json_and_html(self, tmp_path):
        reporter = EvalReporter(results=_mixed_results())
        paths = reporter.generate_all(output_dir=str(tmp_path), console=False)
        assert "json" in paths
        assert "html" in paths
        assert os.path.isfile(paths["json"])
        assert os.path.isfile(paths["html"])

    def test_generate_all_with_console_true(self, tmp_path, capsys):
        """Cover line 897: generate_all with console=True prints the report."""
        reporter = EvalReporter(results=_mixed_results())
        paths = reporter.generate_all(output_dir=str(tmp_path), console=True)
        captured = capsys.readouterr()
        assert "LIGHTSPEED AGENT EVALUATION REPORT" in captured.out
        assert os.path.isfile(paths["json"])
        assert os.path.isfile(paths["html"])


# =========================================================================
# Coverage gap tests
# =========================================================================


class TestPrintConsoleReport:
    """Cover line 267: print_console_report() calls print()."""

    def test_print_console_report_outputs_to_stdout(self, capsys):
        reporter = EvalReporter(results=_mixed_results())
        reporter.print_console_report()
        captured = capsys.readouterr()
        assert "LIGHTSPEED AGENT EVALUATION REPORT" in captured.out
        assert "OVERALL SCORE" in captured.out


class TestJsonSerializeDatetimeAndFallback:
    """Cover lines 279-283: _serialize handles datetime and fallback str()."""

    def test_datetime_in_config_serialized(self):
        from datetime import datetime, timezone

        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        reporter = EvalReporter(
            results=[_make_result("Q-DT")],
            config={"timestamp": ts},
        )
        raw = reporter.generate_json_report()
        parsed = json.loads(raw)
        assert parsed["config"]["timestamp"] == "2025-06-15T12:00:00+00:00"

    def test_non_serializable_object_falls_back_to_str(self):
        class Custom:
            def __str__(self):
                return "custom_value"

        reporter = EvalReporter(
            results=[_make_result("Q-OBJ")],
            config={"custom_obj": Custom()},
        )
        raw = reporter.generate_json_report()
        parsed = json.loads(raw)
        assert parsed["config"]["custom_obj"] == "custom_value"

    def test_dataclass_in_config_serialized_via_asdict(self):
        inner = QuestionResult(question_id="nested", question_text="test q")
        reporter = EvalReporter(
            results=[_make_result("Q-DC")],
            config={"nested_dc": inner},
        )
        raw = reporter.generate_json_report()
        parsed = json.loads(raw)
        assert parsed["config"]["nested_dc"]["question_id"] == "nested"


class TestDonutSvgZeroCategory:
    """Cover lines 354-355: _donut_svg skips categories with 0 total."""

    def test_zero_total_category_skipped_in_donut(self):
        # Build a ReportData with one real category and one zero category.
        results = [_make_result("Q-Z1", passed=True, category="real_cat")]
        reporter = EvalReporter(results=results)
        data = reporter.build_report_data()
        # Inject a zero-total category into the breakdown.
        data.category_breakdown["empty_cat"] = {"total": 0, "passed": 0, "failed": 0}
        # Generate HTML -- the donut SVG is invoked internally.
        html = reporter.generate_html_report(data)
        assert "<!DOCTYPE html>" in html
        # The empty category should appear in the legend but not cause errors.
        assert "real_cat" in html


class TestGaugeSvgLowScore:
    """Cover line 469: _gauge_svg else branch where score < 60."""

    def test_gauge_uses_red_for_low_score(self):
        # 1 pass out of 5 -> score = 20.0%
        results = [
            _make_result("Q-L1", passed=True),
            _make_result("Q-L2", passed=False),
            _make_result("Q-L3", passed=False),
            _make_result("Q-L4", passed=False),
            _make_result("Q-L5", passed=False),
        ]
        reporter = EvalReporter(results=results)
        html = reporter.generate_html_report()
        # score = 20.0%, gauge should use color #F44336 (red)
        assert "#F44336" in html
        assert "20.0%" in html


class TestDemoFunction:
    """Cover lines 908-952: the _demo() function."""

    def test_demo_generates_files(self, tmp_path, monkeypatch):
        from eval_reporter import _demo

        # Monkeypatch generate_all to use tmp_path instead of /tmp/eval_demo,
        # and capture the call to verify it ran.
        calls = []
        original_generate_all = EvalReporter.generate_all

        def patched_generate_all(self, output_dir, **kwargs):
            return original_generate_all(self, output_dir=str(tmp_path), **kwargs)

        monkeypatch.setattr(EvalReporter, "generate_all", patched_generate_all)
        _demo()

        # Verify demo output files exist.
        json_file = tmp_path / "eval_report.json"
        html_file = tmp_path / "eval_report.html"
        assert json_file.is_file()
        assert html_file.is_file()

        # Verify the JSON has 50 results from the demo.
        parsed = json.loads(json_file.read_text())
        assert parsed["summary"]["total_questions"] == 50
