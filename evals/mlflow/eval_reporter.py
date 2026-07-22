"""
Evaluation Reporter Module for Red Hat Lightspeed Agent.

Generates console, JSON, and HTML reports from graded evaluation results.
Uses only Python standard library -- no external dependencies.
"""

from __future__ import annotations

import html as html_mod
import json
import math
import os
import statistics
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    """Result for a single evaluated question."""

    question_id: str
    question_text: str
    category: str = "unknown"
    question_type: str = "unknown"
    difficulty: str = "unknown"
    scenario_type: str = "unknown"
    passed: bool = False
    expected_answer: Any = None
    actual_answer: Any = None
    grade_details: str = ""
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportData:
    """Aggregated report data produced by the grading pipeline."""

    total_questions: int = 0
    passed: int = 0
    failed: int = 0
    score: float = 0.0
    results: List[QuestionResult] = field(default_factory=list)
    category_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)
    type_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)
    difficulty_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)
    scenario_type_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _percentile(data: List[float], pct: float) -> float:
    """Return the *pct*-th percentile of *data* (0-100 scale)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (pct / 100.0) * (len(sorted_data) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def _bar(fraction: float, width: int = 30) -> str:
    """Return a simple ASCII bar: [=====>          ]."""
    filled = int(round(fraction * width))
    empty = width - filled
    arrow = ">" if 0 < filled < width else ""
    bar_body = "=" * max(filled - 1, 0) + arrow + " " * empty
    if filled == width:
        bar_body = "=" * width
    if filled == 0:
        bar_body = " " * width
    return f"[{bar_body}]"


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "  N/A"
    return f"{num / den * 100:5.1f}%"


def _breakdown_dict(results: List[QuestionResult], attr: str) -> Dict[str, Dict[str, int]]:
    """Build {value: {total, passed, failed}} from *results* grouped by *attr*."""
    buckets: Dict[str, Dict[str, int]] = {}
    for r in results:
        key = getattr(r, attr, "unknown") or "unknown"
        if key not in buckets:
            buckets[key] = {"total": 0, "passed": 0, "failed": 0}
        buckets[key]["total"] += 1
        if r.passed:
            buckets[key]["passed"] += 1
        else:
            buckets[key]["failed"] += 1
    return buckets


# ---------------------------------------------------------------------------
# EvalReporter
# ---------------------------------------------------------------------------

class EvalReporter:
    """Generate console, JSON, and HTML reports from graded evaluation data."""

    def __init__(self, results: Optional[List[QuestionResult]] = None,
                 config: Optional[Dict[str, Any]] = None):
        self._results: List[QuestionResult] = results or []
        self._config: Dict[str, Any] = config or {}

    # -- public helpers to add results incrementally -------------------------

    def add_result(self, result: QuestionResult) -> None:
        self._results.append(result)

    def set_config(self, config: Dict[str, Any]) -> None:
        self._config = config

    # -- build ReportData ----------------------------------------------------

    def build_report_data(self) -> ReportData:
        """Compute all aggregates and return a ``ReportData`` instance."""
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        failed = total - passed
        score = (passed / total * 100.0) if total else 0.0

        latencies = [r.latency_ms for r in self._results if r.latency_ms > 0]
        timing: Dict[str, float] = {}
        if latencies:
            timing["avg_ms"] = round(statistics.mean(latencies), 2)
            timing["p50_ms"] = round(_percentile(latencies, 50), 2)
            timing["p95_ms"] = round(_percentile(latencies, 95), 2)
            timing["p99_ms"] = round(_percentile(latencies, 99), 2)
        else:
            timing = {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

        config = dict(self._config)
        if "timestamp" not in config:
            config["timestamp"] = datetime.now(timezone.utc).isoformat()

        return ReportData(
            total_questions=total,
            passed=passed,
            failed=failed,
            score=round(score, 2),
            results=list(self._results),
            category_breakdown=_breakdown_dict(self._results, "category"),
            type_breakdown=_breakdown_dict(self._results, "question_type"),
            difficulty_breakdown=_breakdown_dict(self._results, "difficulty"),
            scenario_type_breakdown=_breakdown_dict(self._results, "scenario_type"),
            timing=timing,
            config=config,
        )

    # -----------------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------------

    def generate_console_report(self, data: Optional[ReportData] = None) -> str:
        """Return a plain-text report suitable for terminal output."""
        if data is None:
            data = self.build_report_data()

        lines: List[str] = []
        w = 78  # content width

        def hline(char: str = "-") -> str:
            return char * (w + 2)

        def section(title: str) -> None:
            lines.append("")
            lines.append(hline("="))
            lines.append(f" {title}")
            lines.append(hline("="))

        # -- header ----------------------------------------------------------
        lines.append(hline("="))
        lines.append(" LIGHTSPEED AGENT EVALUATION REPORT")
        lines.append(hline("="))
        ts = data.config.get("timestamp", "N/A")
        endpoint = data.config.get("endpoint", "N/A")
        lines.append(f" Timestamp : {ts}")
        lines.append(f" Endpoint  : {endpoint}")
        lines.append("")

        # -- overall score ---------------------------------------------------
        section("OVERALL SCORE")
        bar = _bar(data.score / 100.0, 40)
        lines.append(f"  {data.passed}/{data.total_questions} passed   "
                      f"{data.score:5.1f}%  {bar}")
        lines.append("")

        # -- breakdown helper ------------------------------------------------
        def _render_breakdown(title: str, breakdown: Dict[str, Dict[str, int]]) -> None:
            section(title)
            if not breakdown:
                lines.append("  (no data)")
                return
            # determine column widths
            max_key = max((len(k) for k in breakdown), default=10)
            col = max(max_key, 10)
            header = f"  {'Category':<{col}}  {'Total':>6}  {'Pass':>6}  {'Fail':>6}  {'Rate':>7}  Bar"
            lines.append(header)
            lines.append("  " + "-" * (col + 42))
            for key in sorted(breakdown):
                b = breakdown[key]
                rate = _pct(b["passed"], b["total"])
                frac = b["passed"] / b["total"] if b["total"] else 0
                bar_str = _bar(frac, 20)
                lines.append(f"  {key:<{col}}  {b['total']:>6}  {b['passed']:>6}  "
                              f"{b['failed']:>6}  {rate:>7}  {bar_str}")

        _render_breakdown("BY CATEGORY", data.category_breakdown)
        _render_breakdown("BY QUESTION TYPE", data.type_breakdown)
        _render_breakdown("BY DIFFICULTY", data.difficulty_breakdown)
        _render_breakdown("BY SCENARIO TYPE", data.scenario_type_breakdown)

        # -- timing ----------------------------------------------------------
        section("TIMING STATISTICS")
        if data.timing and data.timing.get("avg_ms", 0) > 0:
            lines.append(f"  Average : {data.timing['avg_ms']:>10.2f} ms")
            lines.append(f"  p50     : {data.timing['p50_ms']:>10.2f} ms")
            lines.append(f"  p95     : {data.timing['p95_ms']:>10.2f} ms")
            lines.append(f"  p99     : {data.timing['p99_ms']:>10.2f} ms")
        else:
            lines.append("  (no timing data)")

        # -- failed questions ------------------------------------------------
        failed_results = [r for r in data.results if not r.passed]
        section(f"FAILED QUESTIONS ({len(failed_results)})")
        if not failed_results:
            lines.append("  All questions passed!")
        else:
            for i, r in enumerate(failed_results, 1):
                lines.append(f"  {i}. [{r.question_id}] ({r.category} / "
                              f"{r.difficulty} / {r.question_type})")
                # wrap long question text
                wrapped = textwrap.fill(r.question_text, width=70,
                                        initial_indent="     Q: ",
                                        subsequent_indent="        ")
                lines.append(wrapped)
                lines.append(f"     Expected : {r.expected_answer}")
                lines.append(f"     Actual   : {r.actual_answer}")
                if r.grade_details:
                    lines.append(f"     Details  : {r.grade_details}")
                lines.append("")

        lines.append(hline("="))
        lines.append("")
        return "\n".join(lines)

    def print_console_report(self, data: Optional[ReportData] = None) -> None:
        """Print the console report to stdout."""
        print(self.generate_console_report(data))

    # -----------------------------------------------------------------------
    # JSON report
    # -----------------------------------------------------------------------

    def generate_json_report(self, data: Optional[ReportData] = None) -> str:
        """Return a JSON string containing the full report."""
        if data is None:
            data = self.build_report_data()

        def _serialize(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)

        payload = {
            "report_version": "1.0",
            "summary": {
                "total_questions": data.total_questions,
                "passed": data.passed,
                "failed": data.failed,
                "score": data.score,
            },
            "category_breakdown": data.category_breakdown,
            "type_breakdown": data.type_breakdown,
            "difficulty_breakdown": data.difficulty_breakdown,
            "scenario_type_breakdown": data.scenario_type_breakdown,
            "timing": data.timing,
            "config": data.config,
            "results": [asdict(r) for r in data.results],
        }
        return json.dumps(payload, indent=2, default=_serialize)

    def write_json_report(self, path: str, data: Optional[ReportData] = None) -> str:
        """Write the JSON report to *path* and return the path."""
        content = self.generate_json_report(data)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    # -----------------------------------------------------------------------
    # HTML report
    # -----------------------------------------------------------------------

    def generate_html_report(self, data: Optional[ReportData] = None) -> str:
        """Return a self-contained HTML report string."""
        if data is None:
            data = self.build_report_data()

        esc = html_mod.escape

        # -- SVG helpers -----------------------------------------------------

        def _donut_svg(breakdown: Dict[str, Dict[str, int]], title: str,
                       size: int = 200) -> str:
            """Render an SVG donut chart for *breakdown*."""
            colors = [
                "#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0",
                "#00BCD4", "#795548", "#607D8B", "#E91E63", "#CDDC39",
                "#3F51B5", "#009688",
            ]
            total = sum(b["total"] for b in breakdown.values())
            if total == 0:
                return f'<svg width="{size}" height="{size}"></svg>'

            cx, cy, r = size // 2, size // 2, size // 2 - 10
            inner_r = r * 0.55
            parts: List[str] = []
            parts.append(f'<svg width="{size}" height="{size + 90}" '
                         f'viewBox="0 0 {size} {size + 90}" '
                         f'xmlns="http://www.w3.org/2000/svg">')
            parts.append(f'<text x="{cx}" y="16" text-anchor="middle" '
                         f'font-size="14" font-weight="bold" fill="#333">'
                         f'{esc(title)}</text>')

            offset_y = 24
            angle_start = -90.0
            idx = 0
            for key in sorted(breakdown):
                b = breakdown[key]
                frac = b["total"] / total
                angle_sweep = frac * 360.0
                if angle_sweep == 0:
                    idx += 1
                    continue

                large = 1 if angle_sweep > 180 else 0
                a1 = math.radians(angle_start)
                a2 = math.radians(angle_start + angle_sweep)

                x1o = cx + r * math.cos(a1)
                y1o = cy + r * math.sin(a1) + offset_y
                x2o = cx + r * math.cos(a2)
                y2o = cy + r * math.sin(a2) + offset_y

                x1i = cx + inner_r * math.cos(a2)
                y1i = cy + inner_r * math.sin(a2) + offset_y
                x2i = cx + inner_r * math.cos(a1)
                y2i = cy + inner_r * math.sin(a1) + offset_y

                color = colors[idx % len(colors)]
                path = (f'M {x1o:.2f} {y1o:.2f} '
                        f'A {r} {r} 0 {large} 1 {x2o:.2f} {y2o:.2f} '
                        f'L {x1i:.2f} {y1i:.2f} '
                        f'A {inner_r} {inner_r} 0 {large} 0 {x2i:.2f} {y2i:.2f} Z')
                parts.append(f'<path d="{path}" fill="{color}" stroke="#fff" '
                             f'stroke-width="1"/>')
                angle_start += angle_sweep
                idx += 1

            # center label
            parts.append(f'<text x="{cx}" y="{cy + offset_y + 5}" '
                         f'text-anchor="middle" font-size="20" '
                         f'font-weight="bold" fill="#333">{total}</text>')
            parts.append(f'<text x="{cx}" y="{cy + offset_y + 20}" '
                         f'text-anchor="middle" font-size="10" '
                         f'fill="#666">total</text>')

            # legend
            legend_y = size + offset_y + 10
            lx = 10
            idx = 0
            for key in sorted(breakdown):
                b = breakdown[key]
                color = colors[idx % len(colors)]
                row_y = legend_y + idx * 16
                parts.append(f'<rect x="{lx}" y="{row_y - 9}" width="10" '
                             f'height="10" rx="2" fill="{color}"/>')
                label = f"{esc(key)} ({b['total']})"
                parts.append(f'<text x="{lx + 14}" y="{row_y}" '
                             f'font-size="11" fill="#555">{label}</text>')
                idx += 1

            svg_height = legend_y + idx * 16 + 10
            parts[0] = (f'<svg width="{size}" height="{svg_height}" '
                        f'viewBox="0 0 {size} {svg_height}" '
                        f'xmlns="http://www.w3.org/2000/svg">')
            parts.append("</svg>")
            return "\n".join(parts)

        def _bar_chart_svg(breakdown: Dict[str, Dict[str, int]], title: str,
                           width: int = 360, bar_h: int = 22) -> str:
            """Render a horizontal bar chart SVG for *breakdown*."""
            if not breakdown:
                return ""
            margin_left = 120
            chart_w = width - margin_left - 20
            max_total = max((b["total"] for b in breakdown.values()), default=1)
            keys = sorted(breakdown)
            svg_h = 30 + len(keys) * (bar_h + 6) + 10

            parts: List[str] = []
            parts.append(f'<svg width="{width}" height="{svg_h}" '
                         f'xmlns="http://www.w3.org/2000/svg">')
            parts.append(f'<text x="{width // 2}" y="18" text-anchor="middle" '
                         f'font-size="14" font-weight="bold" fill="#333">'
                         f'{esc(title)}</text>')

            y = 34
            for key in keys:
                b = breakdown[key]
                total_w = (b["total"] / max_total) * chart_w if max_total else 0
                pass_w = (b["passed"] / max_total) * chart_w if max_total else 0

                parts.append(f'<text x="{margin_left - 6}" y="{y + bar_h // 2 + 4}" '
                             f'text-anchor="end" font-size="11" fill="#555">'
                             f'{esc(key)}</text>')
                # background (total)
                parts.append(f'<rect x="{margin_left}" y="{y}" '
                             f'width="{total_w:.1f}" height="{bar_h}" '
                             f'rx="3" fill="#E0E0E0"/>')
                # foreground (passed)
                if pass_w > 0:
                    parts.append(f'<rect x="{margin_left}" y="{y}" '
                                 f'width="{pass_w:.1f}" height="{bar_h}" '
                                 f'rx="3" fill="#4CAF50"/>')
                # label
                rate = b["passed"] / b["total"] * 100 if b["total"] else 0
                parts.append(f'<text x="{margin_left + total_w + 4:.1f}" '
                             f'y="{y + bar_h // 2 + 4}" font-size="11" '
                             f'fill="#333">{b["passed"]}/{b["total"]} '
                             f'({rate:.0f}%)</text>')
                y += bar_h + 6

            parts.append("</svg>")
            return "\n".join(parts)

        # -- score gauge SVG -------------------------------------------------
        def _gauge_svg(score: float) -> str:
            size = 180
            cx, cy = size // 2, size // 2 + 10
            r = 70
            circumference = 2 * math.pi * r
            half_circ = circumference / 2
            dash = score / 100.0 * half_circ
            gap = half_circ - dash

            if score >= 80:
                color = "#4CAF50"
            elif score >= 60:
                color = "#FF9800"
            else:
                color = "#F44336"

            return f"""<svg width="{size}" height="{size // 2 + 50}"
                        viewBox="0 0 {size} {size // 2 + 50}"
                        xmlns="http://www.w3.org/2000/svg">
              <path d="M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}"
                    fill="none" stroke="#E0E0E0" stroke-width="14"
                    stroke-linecap="round"/>
              <path d="M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}"
                    fill="none" stroke="{color}" stroke-width="14"
                    stroke-linecap="round"
                    stroke-dasharray="{dash:.2f} {gap:.2f}"/>
              <text x="{cx}" y="{cy - 14}" text-anchor="middle"
                    font-size="32" font-weight="bold" fill="{color}">
                {score:.1f}%</text>
              <text x="{cx}" y="{cy + 6}" text-anchor="middle"
                    font-size="13" fill="#666">overall score</text>
            </svg>"""

        # -- build HTML ------------------------------------------------------
        timestamp = esc(str(data.config.get("timestamp", "")))
        endpoint = esc(str(data.config.get("endpoint", "")))

        result_rows: List[str] = []
        for r in data.results:
            status_class = "pass" if r.passed else "fail"
            status_label = "PASS" if r.passed else "FAIL"
            detail_html = ""
            if r.grade_details:
                detail_html = f"<p class='detail-text'>{esc(r.grade_details)}</p>"
            expected_str = esc(str(r.expected_answer)) if r.expected_answer is not None else ""
            actual_str = esc(str(r.actual_answer)) if r.actual_answer is not None else ""

            result_rows.append(f"""
            <tr class="result-row {status_class}-row" data-category="{esc(r.category)}"
                data-difficulty="{esc(r.difficulty)}" data-type="{esc(r.question_type)}"
                data-scenario="{esc(r.scenario_type)}" data-status="{status_label}">
              <td><span class="badge {status_class}">{status_label}</span></td>
              <td class="qid">{esc(r.question_id)}</td>
              <td>{esc(r.category)}</td>
              <td>{esc(r.difficulty)}</td>
              <td>{esc(r.question_type)}</td>
              <td>{esc(r.scenario_type)}</td>
              <td class="latency">{r.latency_ms:.0f}</td>
              <td>
                <details>
                  <summary>{esc(r.question_text[:80])}{'...' if len(r.question_text) > 80 else ''}</summary>
                  <div class="detail-body">
                    <p><strong>Question:</strong> {esc(r.question_text)}</p>
                    <p><strong>Expected:</strong> {expected_str}</p>
                    <p><strong>Actual:</strong> {actual_str}</p>
                    {detail_html}
                  </div>
                </details>
              </td>
            </tr>""")

        # timing rows
        timing_html = ""
        if data.timing and data.timing.get("avg_ms", 0) > 0:
            timing_html = f"""
            <div class="card">
              <h3>Timing Statistics</h3>
              <table class="stats-table">
                <tr><td>Average</td><td>{data.timing['avg_ms']:.2f} ms</td></tr>
                <tr><td>p50</td><td>{data.timing['p50_ms']:.2f} ms</td></tr>
                <tr><td>p95</td><td>{data.timing['p95_ms']:.2f} ms</td></tr>
                <tr><td>p99</td><td>{data.timing['p99_ms']:.2f} ms</td></tr>
              </table>
            </div>"""

        # config filters
        filters_used = data.config.get("filters", {})
        filters_str = ", ".join(f"{k}={v}" for k, v in filters_used.items()) if filters_used else "none"

        donut_cat = _donut_svg(data.category_breakdown, "Categories")
        bar_diff = _bar_chart_svg(data.difficulty_breakdown, "By Difficulty")
        bar_type = _bar_chart_svg(data.type_breakdown, "By Question Type")
        bar_scenario = _bar_chart_svg(data.scenario_type_breakdown, "By Scenario Type")
        gauge = _gauge_svg(data.score)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lightspeed Agent Evaluation Report</title>
<style>
  :root {{
    --pass: #4CAF50;
    --fail: #F44336;
    --bg: #f5f6fa;
    --card: #ffffff;
    --border: #e0e0e0;
    --text: #333333;
    --text-light: #666666;
    --shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding: 24px;
  }}
  h1 {{ font-size: 1.6em; margin-bottom: 4px; }}
  h2 {{ font-size: 1.2em; margin-bottom: 12px; color: var(--text); }}
  h3 {{ font-size: 1.05em; margin-bottom: 8px; }}
  .header {{
    background: var(--card);
    border-radius: 10px;
    padding: 24px 32px;
    box-shadow: var(--shadow);
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
  }}
  .header-meta {{ color: var(--text-light); font-size: 0.88em; }}
  .score-section {{
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
  }}
  .score-big {{
    font-size: 3em;
    font-weight: 800;
    line-height: 1;
  }}
  .score-big.high {{ color: var(--pass); }}
  .score-big.mid {{ color: #FF9800; }}
  .score-big.low {{ color: var(--fail); }}
  .score-detail {{ font-size: 0.95em; color: var(--text-light); }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
  }}
  .card {{
    background: var(--card);
    border-radius: 10px;
    padding: 20px;
    box-shadow: var(--shadow);
  }}
  .card svg {{ max-width: 100%; height: auto; }}
  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }}
  .badge.pass {{ background: #E8F5E9; color: var(--pass); }}
  .badge.fail {{ background: #FFEBEE; color: var(--fail); }}
  .results-card {{
    background: var(--card);
    border-radius: 10px;
    padding: 20px;
    box-shadow: var(--shadow);
    margin-bottom: 20px;
    overflow-x: auto;
  }}
  .filter-bar {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 14px;
    align-items: center;
  }}
  .filter-bar label {{ font-size: 0.85em; color: var(--text-light); }}
  .filter-bar select, .filter-bar input {{
    padding: 5px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.88em;
    background: #fff;
  }}
  table.results {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88em;
  }}
  table.results th {{
    text-align: left;
    padding: 8px 10px;
    border-bottom: 2px solid var(--border);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }}
  table.results th:hover {{ background: #f0f0f0; }}
  table.results th::after {{ content: " \\2195"; color: #bbb; font-size: 0.8em; }}
  table.results td {{
    padding: 7px 10px;
    border-bottom: 1px solid #f0f0f0;
    vertical-align: top;
  }}
  .result-row:hover {{ background: #fafbfd; }}
  .fail-row {{ background: #fffafa; }}
  .qid {{ font-family: monospace; font-size: 0.92em; white-space: nowrap; }}
  .latency {{ text-align: right; font-family: monospace; }}
  details > summary {{
    cursor: pointer;
    color: #1976D2;
    font-size: 0.92em;
  }}
  details > summary:hover {{ text-decoration: underline; }}
  .detail-body {{
    margin-top: 8px;
    padding: 10px;
    background: #f9f9f9;
    border-radius: 6px;
    font-size: 0.92em;
    word-break: break-word;
  }}
  .detail-body p {{ margin-bottom: 4px; }}
  .detail-text {{ color: var(--text-light); font-style: italic; }}
  .stats-table {{ width: 100%; }}
  .stats-table td {{
    padding: 6px 10px;
    border-bottom: 1px solid #f0f0f0;
  }}
  .stats-table td:last-child {{
    text-align: right;
    font-family: monospace;
    font-weight: 600;
  }}
  .hidden {{ display: none; }}
  footer {{
    text-align: center;
    color: var(--text-light);
    font-size: 0.82em;
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Lightspeed Agent Evaluation Report</h1>
    <div class="header-meta">
      <div>Endpoint: {endpoint}</div>
      <div>Timestamp: {timestamp}</div>
      <div>Filters: {esc(filters_str)}</div>
    </div>
  </div>
  <div class="score-section">
    {gauge}
  </div>
</div>

<div class="grid">
  <div class="card" style="text-align:center;">
    <h3>Score Overview</h3>
    <div class="score-big {'high' if data.score >= 80 else 'mid' if data.score >= 60 else 'low'}">{data.score:.1f}%</div>
    <div class="score-detail">{data.passed} passed / {data.failed} failed / {data.total_questions} total</div>
  </div>
  {timing_html}
  <div class="card">{donut_cat}</div>
</div>

<div class="grid">
  <div class="card">{bar_diff}</div>
  <div class="card">{bar_type}</div>
  <div class="card">{bar_scenario}</div>
</div>

<div class="results-card">
  <h2>All Results</h2>
  <div class="filter-bar">
    <label>Status:</label>
    <select id="filterStatus">
      <option value="">All</option>
      <option value="PASS">Pass</option>
      <option value="FAIL">Fail</option>
    </select>
    <label>Category:</label>
    <select id="filterCategory">
      <option value="">All</option>
    </select>
    <label>Difficulty:</label>
    <select id="filterDifficulty">
      <option value="">All</option>
    </select>
    <label>Search:</label>
    <input type="text" id="searchBox" placeholder="question id or text...">
  </div>
  <table class="results" id="resultsTable">
    <thead>
      <tr>
        <th data-col="0">Status</th>
        <th data-col="1">ID</th>
        <th data-col="2">Category</th>
        <th data-col="3">Difficulty</th>
        <th data-col="4">Type</th>
        <th data-col="5">Scenario</th>
        <th data-col="6">Latency</th>
        <th data-col="7">Question</th>
      </tr>
    </thead>
    <tbody>
      {"".join(result_rows)}
    </tbody>
  </table>
</div>

<footer>
  Generated by Lightspeed Agent Evaluation Framework
</footer>

<script>
(function() {{
  // populate filter dropdowns
  var rows = document.querySelectorAll('.result-row');
  var cats = new Set(), diffs = new Set();
  rows.forEach(function(r) {{
    cats.add(r.dataset.category);
    diffs.add(r.dataset.difficulty);
  }});
  var catSel = document.getElementById('filterCategory');
  Array.from(cats).sort().forEach(function(c) {{
    var o = document.createElement('option'); o.value = c; o.textContent = c;
    catSel.appendChild(o);
  }});
  var diffSel = document.getElementById('filterDifficulty');
  Array.from(diffs).sort().forEach(function(d) {{
    var o = document.createElement('option'); o.value = d; o.textContent = d;
    diffSel.appendChild(o);
  }});

  function applyFilters() {{
    var status = document.getElementById('filterStatus').value;
    var cat = catSel.value;
    var diff = diffSel.value;
    var search = document.getElementById('searchBox').value.toLowerCase();
    rows.forEach(function(r) {{
      var show = true;
      if (status && r.dataset.status !== status) show = false;
      if (cat && r.dataset.category !== cat) show = false;
      if (diff && r.dataset.difficulty !== diff) show = false;
      if (search && r.textContent.toLowerCase().indexOf(search) === -1) show = false;
      r.classList.toggle('hidden', !show);
    }});
  }}

  document.getElementById('filterStatus').addEventListener('change', applyFilters);
  catSel.addEventListener('change', applyFilters);
  diffSel.addEventListener('change', applyFilters);
  document.getElementById('searchBox').addEventListener('input', applyFilters);

  // sortable columns
  var table = document.getElementById('resultsTable');
  var headers = table.querySelectorAll('th');
  var sortDir = {{}};
  headers.forEach(function(th) {{
    th.addEventListener('click', function() {{
      var col = parseInt(th.dataset.col);
      var dir = sortDir[col] === 'asc' ? 'desc' : 'asc';
      sortDir[col] = dir;
      var tbody = table.querySelector('tbody');
      var rowsArr = Array.from(tbody.querySelectorAll('tr'));
      rowsArr.sort(function(a, b) {{
        var at = a.children[col].textContent.trim();
        var bt = b.children[col].textContent.trim();
        var an = parseFloat(at), bn = parseFloat(bt);
        if (!isNaN(an) && !isNaN(bn)) {{
          return dir === 'asc' ? an - bn : bn - an;
        }}
        return dir === 'asc' ? at.localeCompare(bt) : bt.localeCompare(at);
      }});
      rowsArr.forEach(function(r) {{ tbody.appendChild(r); }});
    }});
  }});
}})();
</script>
</body>
</html>"""
        return html

    def write_html_report(self, path: str, data: Optional[ReportData] = None) -> str:
        """Write the HTML report to *path* and return the path."""
        content = self.generate_html_report(data)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    # -----------------------------------------------------------------------
    # Convenience: generate all reports at once
    # -----------------------------------------------------------------------

    def generate_all(self, output_dir: str,
                     basename: str = "eval_report",
                     console: bool = True) -> Dict[str, str]:
        """Generate JSON and HTML reports in *output_dir*.

        If *console* is ``True`` the console report is also printed to stdout.
        Returns a dict mapping format names to file paths.
        """
        data = self.build_report_data()
        os.makedirs(output_dir, exist_ok=True)

        paths: Dict[str, str] = {}

        json_path = os.path.join(output_dir, f"{basename}.json")
        self.write_json_report(json_path, data)
        paths["json"] = json_path

        html_path = os.path.join(output_dir, f"{basename}.html")
        self.write_html_report(html_path, data)
        paths["html"] = html_path

        if console:
            self.print_console_report(data)

        return paths


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Generate a demo report with synthetic data (for testing)."""
    import random
    random.seed(42)

    categories = ["vulnerability", "inventory", "advisor", "planning", "compliance"]
    types = ["binary", "single_select", "multiple_select",
             "substring_match", "exact_match", "ordered_list"]
    difficulties = ["easy", "medium", "hard"]
    scenarios = ["single_tool", "multi_step", "no_tool", "error_handling", "pagination"]

    results: List[QuestionResult] = []
    for i in range(50):
        cat = random.choice(categories)
        diff = random.choice(difficulties)
        qtype = random.choice(types)
        scenario = random.choice(scenarios)
        passed = random.random() < 0.72
        latency = random.gauss(1200, 400)
        latency = max(200, latency)

        results.append(QuestionResult(
            question_id=f"Q-{i + 1:03d}",
            question_text=f"Sample question {i + 1} about {cat} ({diff})?",
            category=cat,
            question_type=qtype,
            difficulty=diff,
            scenario_type=scenario,
            passed=passed,
            expected_answer="expected_value",
            actual_answer="actual_value" if passed else "wrong_value",
            grade_details="" if passed else "Answer did not match expected value",
            latency_ms=round(latency, 2),
        ))

    reporter = EvalReporter(
        results=results,
        config={
            "endpoint": "https://lightspeed.example.com/api/v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filters": {"category": "all", "difficulty": "all"},
        },
    )

    paths = reporter.generate_all(output_dir="/tmp/eval_demo", console=True)
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")


if __name__ == "__main__":  # pragma: no cover
    _demo()
