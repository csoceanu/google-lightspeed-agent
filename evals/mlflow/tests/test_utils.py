"""Tests for eval_utils.py -- utility functions used across the evaluation framework."""

from __future__ import annotations

import json
import re

import pytest

from eval_utils import (
    normalize_text,
    extract_options_from_response,
    contains_tool_names,
    calculate_percentile,
    format_duration,
    truncate,
    load_json,
    save_json,
    merge_results,
    generate_run_id,
)


# =========================================================================
# normalize_text
# =========================================================================


class TestNormalizeText:
    """Tests for normalize_text()."""

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_lowercases_input(self):
        assert normalize_text("HELLO WORLD") == "hello world"

    def test_collapses_multiple_spaces(self):
        assert normalize_text("hello    world") == "hello world"

    def test_collapses_tabs_and_newlines(self):
        assert normalize_text("hello\t\tworld\n\nfoo") == "hello world foo"

    def test_combined_strip_lower_collapse(self):
        assert normalize_text("  Hello   World  ") == "hello world"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_single_word(self):
        assert normalize_text("  WORD  ") == "word"

    def test_already_normalized(self):
        assert normalize_text("already clean") == "already clean"


# =========================================================================
# truncate
# =========================================================================


class TestTruncate:
    """Tests for truncate()."""

    def test_no_truncation_when_within_limit(self):
        assert truncate("abc", 5) == "abc"

    def test_no_truncation_when_exact_length(self):
        assert truncate("abcde", 5) == "abcde"

    def test_truncates_and_appends_ellipsis(self):
        assert truncate("abcdef", 5) == "ab..."

    def test_max_len_of_three_gives_all_dots(self):
        assert truncate("abcdef", 3) == "..."

    def test_max_len_of_two_gives_dots(self):
        assert truncate("abcdef", 2) == ".."

    def test_max_len_of_one(self):
        assert truncate("abcdef", 1) == "."

    def test_default_max_len_is_200(self):
        short = "x" * 200
        assert truncate(short) == short
        long_text = "x" * 201
        assert truncate(long_text) == "x" * 197 + "..."


# =========================================================================
# extract_options_from_response
# =========================================================================


class TestExtractOptionsFromResponse:
    """Tests for extract_options_from_response()."""

    def test_empty_response_returns_empty(self):
        assert extract_options_from_response("", ["opt1", "opt2"]) == []

    def test_empty_options_returns_empty(self):
        assert extract_options_from_response("some text", []) == []

    def test_letter_prefix_parenthesized(self):
        options = ["Red", "Green", "Blue"]
        response = "I would choose (B) because green is nice."
        result = extract_options_from_response(response, options)
        assert "Green" in result

    def test_letter_prefix_with_closing_paren(self):
        options = ["Red", "Green", "Blue"]
        response = "A) is the best option."
        result = extract_options_from_response(response, options)
        assert "Red" in result

    def test_letter_prefix_with_dot(self):
        options = ["Red", "Green", "Blue"]
        response = "C. is my answer."
        result = extract_options_from_response(response, options)
        assert "Blue" in result

    def test_verbatim_substring_match(self):
        options = ["vulnerability scan", "compliance check"]
        response = "You should run a compliance check on the system."
        result = extract_options_from_response(response, options)
        assert "compliance check" in result

    def test_verbatim_match_is_case_insensitive(self):
        options = ["Vulnerability Scan"]
        response = "perform a vulnerability scan"
        result = extract_options_from_response(response, options)
        assert "Vulnerability Scan" in result

    def test_multiple_matches(self):
        options = ["Red", "Green", "Blue"]
        response = "A) Red and also Blue is nice."
        result = extract_options_from_response(response, options)
        assert "Red" in result
        assert "Blue" in result

    def test_no_match(self):
        options = ["Red", "Green", "Blue"]
        response = "I like yellow."
        result = extract_options_from_response(response, options)
        assert result == []

    def test_no_duplicates_when_letter_and_substring_both_match(self):
        options = ["Red", "Green", "Blue"]
        # "A)" matches letter prefix -> Red, and "Red" is also a substring
        response = "A) Red is the answer."
        result = extract_options_from_response(response, options)
        assert result.count("Red") == 1


# =========================================================================
# contains_tool_names
# =========================================================================


class TestContainsToolNames:
    """Tests for contains_tool_names()."""

    def test_clean_response_returns_empty(self):
        assert contains_tool_names("No issues found.") == []

    def test_detects_single_tool_name(self):
        result = contains_tool_names("See vulnerability__get_cves for details.")
        assert result == ["vulnerability__get_cves"]

    def test_detects_multiple_tool_names(self):
        response = "Used rag__search and bugzilla__get_bug internally."
        result = contains_tool_names(response)
        assert "rag__search" in result
        assert "bugzilla__get_bug" in result

    def test_does_not_false_positive_on_partial_name(self):
        # "vulnerability" alone is not a tool name
        result = contains_tool_names("vulnerability is important")
        assert result == []

    def test_empty_response(self):
        assert contains_tool_names("") == []


# =========================================================================
# calculate_percentile
# =========================================================================


class TestCalculatePercentile:
    """Tests for calculate_percentile()."""

    def test_median_of_odd_list(self):
        assert calculate_percentile([1, 2, 3, 4, 5], 50) == 3.0

    def test_0th_percentile(self):
        assert calculate_percentile([1, 2, 3, 4, 5], 0) == 1.0

    def test_100th_percentile(self):
        assert calculate_percentile([1, 2, 3, 4, 5], 100) == 5.0

    def test_single_element(self):
        assert calculate_percentile([42], 50) == 42.0

    def test_interpolation(self):
        # 25th percentile of [1,2,3,4,5]: k = 0.25*(5-1) = 1.0 -> index 1 -> 2.0
        assert calculate_percentile([1, 2, 3, 4, 5], 25) == 2.0

    def test_interpolation_fractional(self):
        # 30th percentile of [1,2,3,4,5]: k = 0.30*4 = 1.2
        # lo=1 (val=2), hi=2 (val=3), frac=0.2 -> 2 + 0.2*1 = 2.2
        assert calculate_percentile([1, 2, 3, 4, 5], 30) == pytest.approx(2.2)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_percentile([], 50)

    def test_percentile_out_of_range_raises(self):
        with pytest.raises(ValueError, match="0-100"):
            calculate_percentile([1, 2], 101)

    def test_negative_percentile_raises(self):
        with pytest.raises(ValueError, match="0-100"):
            calculate_percentile([1, 2], -1)

    def test_unsorted_input_is_handled(self):
        assert calculate_percentile([5, 1, 3, 2, 4], 50) == 3.0


# =========================================================================
# format_duration
# =========================================================================


class TestFormatDuration:
    """Tests for format_duration()."""

    def test_milliseconds_range(self):
        assert format_duration(432) == "432ms"

    def test_zero_ms(self):
        assert format_duration(0) == "0ms"

    def test_seconds_with_decimal(self):
        assert format_duration(12500) == "12.5s"

    def test_exact_seconds_no_decimal(self):
        assert format_duration(5000) == "5s"

    def test_just_under_one_second(self):
        assert format_duration(999) == "999ms"

    def test_exactly_one_second(self):
        assert format_duration(1000) == "1s"

    def test_minutes_and_seconds(self):
        assert format_duration(204000) == "3m 24s"

    def test_hours_minutes_seconds(self):
        assert format_duration(3_912_000) == "1h 5m 12s"

    def test_exactly_one_minute(self):
        assert format_duration(60_000) == "1m 0s"


# =========================================================================
# load_json / save_json
# =========================================================================


class TestJsonIO:
    """Tests for load_json() and save_json() using tmp_path."""

    def test_save_and_load_dict(self, tmp_path):
        path = str(tmp_path / "data.json")
        payload = {"key": "value", "num": 42}
        save_json(path, payload)
        loaded = load_json(path)
        assert loaded == payload

    def test_save_and_load_list(self, tmp_path):
        path = str(tmp_path / "data.json")
        payload = [1, 2, 3, "hello"]
        save_json(path, payload)
        loaded = load_json(path)
        assert loaded == payload

    def test_save_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "sub" / "deep" / "data.json")
        save_json(path, {"nested": True})
        loaded = load_json(path)
        assert loaded == {"nested": True}

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json(str(tmp_path / "nope.json"))

    def test_load_malformed_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_json(str(path))

    def test_save_uses_indent(self, tmp_path):
        path = tmp_path / "indented.json"
        save_json(str(path), {"a": 1}, indent=4)
        text = path.read_text(encoding="utf-8")
        # 4-space indent means the key line starts with 4 spaces
        assert '    "a": 1' in text

    def test_save_appends_trailing_newline(self, tmp_path):
        path = tmp_path / "nl.json"
        save_json(str(path), {"x": 1})
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")

    def test_roundtrip_unicode(self, tmp_path):
        path = str(tmp_path / "unicode.json")
        payload = {"greeting": "Hola, mundo!"}
        save_json(path, payload)
        loaded = load_json(path)
        assert loaded == payload


# =========================================================================
# merge_results
# =========================================================================


class TestMergeResults:
    """Tests for merge_results()."""

    def test_new_overwrites_existing_by_key(self):
        existing = [{"id": "q1", "score": 0.5}, {"id": "q2", "score": 0.8}]
        new = [{"id": "q2", "score": 0.9}]
        merged = merge_results(existing, new)
        q2 = [r for r in merged if r["id"] == "q2"][0]
        assert q2["score"] == 0.9

    def test_new_items_appended(self):
        existing = [{"id": "q1", "score": 0.5}]
        new = [{"id": "q3", "score": 1.0}]
        merged = merge_results(existing, new)
        assert len(merged) == 2
        assert merged[-1]["id"] == "q3"

    def test_order_preserved(self):
        existing = [{"id": "q1"}, {"id": "q2"}, {"id": "q3"}]
        new = [{"id": "q2", "extra": True}, {"id": "q4"}]
        merged = merge_results(existing, new)
        ids = [r["id"] for r in merged]
        assert ids == ["q1", "q2", "q3", "q4"]

    def test_custom_key(self):
        existing = [{"name": "a", "v": 1}]
        new = [{"name": "a", "v": 2}]
        merged = merge_results(existing, new, key="name")
        assert merged[0]["v"] == 2

    def test_both_empty(self):
        assert merge_results([], []) == []

    def test_existing_empty(self):
        new = [{"id": "x"}]
        assert merge_results([], new) == [{"id": "x"}]

    def test_new_empty(self):
        existing = [{"id": "x"}]
        merged = merge_results(existing, [])
        assert len(merged) == 1
        assert merged[0]["id"] == "x"

    def test_does_not_mutate_originals(self):
        existing = [{"id": "q1", "score": 0.5}]
        new = [{"id": "q1", "score": 0.9}]
        merge_results(existing, new)
        # Original lists should not be modified
        assert existing[0]["score"] == 0.5


# =========================================================================
# generate_run_id
# =========================================================================


class TestGenerateRunId:
    """Tests for generate_run_id()."""

    def test_format_has_three_dash_segments(self):
        rid = generate_run_id()
        parts = rid.split("-")
        assert len(parts) == 3

    def test_starts_with_date_segment(self):
        rid = generate_run_id()
        date_part = rid.split("-")[0]
        assert len(date_part) == 8
        assert date_part.isdigit()

    def test_second_segment_is_time(self):
        rid = generate_run_id()
        time_part = rid.split("-")[1]
        assert len(time_part) == 6
        assert time_part.isdigit()

    def test_third_segment_is_hex(self):
        rid = generate_run_id()
        short_id = rid.split("-")[2]
        assert len(short_id) == 8
        int(short_id, 16)  # raises ValueError if not hex

    def test_uniqueness(self):
        ids = {generate_run_id() for _ in range(50)}
        assert len(ids) == 50

    def test_matches_expected_regex(self):
        rid = generate_run_id()
        pattern = r"^\d{8}-\d{6}-[0-9a-f]{8}$"
        assert re.match(pattern, rid), f"{rid!r} does not match pattern"
