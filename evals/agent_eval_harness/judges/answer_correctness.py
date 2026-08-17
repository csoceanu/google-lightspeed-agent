"""Deterministic correctness check based on question_type and expected_response.

Grades without an LLM by dispatching to the appropriate grading method
for each question type: binary (yes/no detection), single_select (option
matching), multiple_select (all-or-nothing), substring_match (keyword
presence), exact_match (string equality), ordered_list (item ordering).
"""

import re


def _normalize(text):
    return " ".join(text.lower().split())


_AFFIRMATIVE = [
    "yes", "yeah", "yep", "correct", "right", "true",
    "affirmative", "indeed", "absolutely", "certainly",
    "sure", "of course", "confirmed", "it is", "it does", "it can",
]
_NEGATIVE = [
    "no", "nope", "incorrect", "wrong", "false", "negative",
    "not", "never", "none", "it is not", "it does not",
    "it cannot", "it can't", "it doesn't", "it isn't",
]


def _grade_binary(expected, response):
    expected = _normalize(expected)
    if expected not in ("yes", "no"):
        return False, f"Invalid binary expected: {expected}"
    norm = _normalize(response)
    first_sentence = re.split(r"\.\s|\n", norm, maxsplit=1)[0]
    words = first_sentence.split()
    first = words[0].strip("*_,.!:") if words else ""
    aff = sum(1 for w in _AFFIRMATIVE if w in first_sentence)
    neg = sum(1 for w in _NEGATIVE if w in first_sentence)
    if first == "yes":
        aff += 3
    elif first == "no":
        neg += 3
    detected = "yes" if aff > neg else "no" if neg > aff else "ambiguous"
    passed = detected == expected
    return passed, f"Expected '{expected}', detected '{detected}'"


def _grade_single_select(expected, response):
    norm = _normalize(response)
    if _normalize(str(expected)) in norm:
        return True, f"Found '{expected}'"
    return False, f"'{expected}' not found"


def _grade_multiple_select(expected, response):
    if not isinstance(expected, list):
        expected = [expected]
    norm = _normalize(response)
    found = [e for e in expected if _normalize(str(e)) in norm]
    missed = [e for e in expected if _normalize(str(e)) not in norm]
    passed = len(missed) == 0
    return passed, f"Found {len(found)}/{len(expected)}" + (f", missed: {missed}" if missed else "")


def _grade_substring_match(expected, response):
    subs = [expected] if isinstance(expected, str) else list(expected)
    norm = _normalize(response)
    matched = [s for s in subs if _normalize(s) in norm]
    missing = [s for s in subs if _normalize(s) not in norm]
    passed = len(missing) == 0
    return passed, f"{len(matched)}/{len(subs)} substrings" + (f", missing: {missing}" if missing else "")


def _grade_exact_match(expected, response):
    ne = _normalize(str(expected))
    nr = _normalize(response)
    if ne == nr:
        return True, "Exact match"
    if ne in nr:
        return True, f"'{expected}' found in response"
    return False, f"Expected '{expected}'"


def _grade_ordered_list(expected, response):
    items = [s.strip() for s in expected.split(",")] if isinstance(expected, str) else list(expected)
    norm = _normalize(response)
    positions = []
    missing = []
    for item in items:
        pos = norm.find(_normalize(item))
        if pos >= 0:
            positions.append(pos)
        else:
            missing.append(item)
    in_order = all(a < b for a, b in zip(positions, positions[1:]))
    if missing:
        return False, f"Missing: {missing}"
    if not in_order:
        return False, "Items found but in wrong order"
    return True, "All items in correct order"


def _parse_expected(value):
    """Parse expected_response — handles YAML string-serialized lists."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("["):
        try:
            import ast
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


def judge(outputs=None, **kwargs):
    response = outputs.get("files", {}).get("output/response.txt", "")
    ann = outputs.get("annotations", {})
    q_type = ann.get("question_type", "")
    expected = _parse_expected(ann.get("expected_response", ""))

    if not q_type:
        return True, "No question_type defined, skipping deterministic check"

    dispatch = {
        "binary": lambda: _grade_binary(expected, response),
        "single_select": lambda: _grade_single_select(expected, response),
        "multiple_select": lambda: _grade_multiple_select(expected, response),
        "substring_match": lambda: _grade_substring_match(expected, response),
        "exact_match": lambda: _grade_exact_match(expected, response),
        "ordered_list": lambda: _grade_ordered_list(expected, response),
        "free_form": lambda: (True, "Skipped: free-form questions are graded by the LLM correctness judge"),
    }

    handler = dispatch.get(q_type)
    if handler is None:
        return True, f"Unknown question_type '{q_type}', skipping"
    return handler()
