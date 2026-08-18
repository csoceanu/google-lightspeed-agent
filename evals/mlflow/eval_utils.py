"""Utility functions for the Red Hat Lightspeed Agent evaluation framework.

This module collects small, pure-ish helpers used across the evaluation
pipeline: text normalisation, option extraction, result merging,
serialisation, and basic statistics.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Well-known internal tool names that should never leak into user-facing
# responses.  Extend this list as new tools are added to the agent.
# ---------------------------------------------------------------------------
_INTERNAL_TOOL_NAMES: List[str] = [
    "vulnerability__get_cves",
    "vulnerability__get_cve_details",
    "vulnerability__search_cves",
    "knowledge__search",
    "knowledge__get_document",
    "knowledge__list_documents",
    "rag__retrieve",
    "rag__search",
    "errata__get_erratum",
    "errata__search_errata",
    "image__get_image_info",
    "image__list_images",
    "subscription__check_subscription",
    "subscription__list_subscriptions",
    "certification__get_hardware",
    "certification__search_hardware",
    "bugzilla__get_bug",
    "bugzilla__search_bugs",
]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Strip, lowercase, and collapse runs of whitespace to a single space.

    >>> normalize_text("  Hello   World  ")
    'hello world'
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def truncate(text: str, max_len: int = 200) -> str:
    """Return *text* shortened to at most *max_len* characters.

    If truncation occurs the last three characters are replaced with
    ``...`` so the caller can tell at a glance that the value was cut.

    >>> truncate("abcdef", 5)
    'ab...'
    """
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return "." * max_len
    return text[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Option / response analysis
# ---------------------------------------------------------------------------

def extract_options_from_response(
    response: str,
    options: List[str],
) -> List[str]:
    """Identify which *options* the agent selected in its *response*.

    Detection heuristics (checked in order):

    1. **Letter prefix** -- lines like ``A)`` or ``(B)`` or ``A.`` where the
       letter index maps to an entry in *options*.
    2. **Exact substring** -- the full text of an option appears verbatim
       (case-insensitive) in the response.

    Returns a list of matched option strings (preserving the original
    casing from *options*).  The list may be empty if nothing matched.
    """
    if not options or not response:
        return []

    matched: List[str] = []
    response_lower = response.lower()

    # Build a letter -> option lookup (A=0, B=1, ...)
    letter_to_option: Dict[str, str] = {}
    for idx, opt in enumerate(options):
        if idx < 26:
            letter_to_option[chr(ord("a") + idx)] = opt

    # 1. Look for letter-prefix patterns.
    letter_pattern = re.compile(
        r"(?:^|\n)\s*[\(\[]?\s*([A-Za-z])\s*[\)\].\:]",
    )
    for match in letter_pattern.finditer(response):
        letter = match.group(1).lower()
        if letter in letter_to_option:
            opt = letter_to_option[letter]
            if opt not in matched:
                matched.append(opt)

    # 2. Verbatim substring match (case-insensitive).
    for opt in options:
        if opt not in matched and opt.lower() in response_lower:
            matched.append(opt)

    return matched


def contains_tool_names(response: str) -> List[str]:
    """Return the list of internal tool names found in *response*.

    An empty list means the response is clean -- no tool-name leakage
    detected.

    >>> contains_tool_names("See vulnerability__get_cves for details.")
    ['vulnerability__get_cves']
    >>> contains_tool_names("No issues found.")
    []
    """
    found: List[str] = []
    for name in _INTERNAL_TOOL_NAMES:
        if name in response:
            found.append(name)
    return found


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calculate_percentile(values: List[float], percentile: int) -> float:
    """Compute *percentile* (0-100) over a list of numeric *values*.

    Uses the *linear interpolation* method consistent with NumPy's
    default ``np.percentile`` behaviour.

    Raises ``ValueError`` when *values* is empty or *percentile* is out
    of range.

    >>> calculate_percentile([1, 2, 3, 4, 5], 50)
    3.0
    >>> calculate_percentile([1, 2, 3, 4, 5], 0)
    1.0
    >>> calculate_percentile([1, 2, 3, 4, 5], 100)
    5.0
    """
    if not values:
        raise ValueError("cannot compute percentile of an empty list")
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be 0-100, got {percentile}")

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    if n == 1:
        return float(sorted_vals[0])

    # Map percentile to a fractional index in [0, n-1].
    k = (percentile / 100.0) * (n - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(sorted_vals[lo])

    frac = k - lo
    return float(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_duration(ms: float) -> str:
    """Convert a duration in milliseconds to a human-readable string.

    The function picks the most natural unit:

    * ``< 1 000 ms``  -- e.g. ``"432ms"``
    * ``< 60 000 ms`` -- e.g. ``"12.5s"``
    * ``< 3 600 000 ms`` -- e.g. ``"3m 24s"``
    * otherwise       -- e.g. ``"1h 5m 12s"``

    >>> format_duration(432)
    '432ms'
    >>> format_duration(12500)
    '12.5s'
    >>> format_duration(204000)
    '3m 24s'
    >>> format_duration(3_912_000)
    '1h 5m 12s'
    """
    if ms < 1000:
        return f"{int(ms)}ms"

    total_seconds = ms / 1000.0

    if total_seconds < 60:
        # Show one decimal place only when it is meaningful.
        if total_seconds == int(total_seconds):
            return f"{int(total_seconds)}s"
        return f"{total_seconds:.1f}s"

    minutes, secs = divmod(int(total_seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def load_json(path: str) -> Union[Dict[str, Any], List[Any]]:
    """Read and parse a JSON file, returning the deserialised object.

    Raises ``FileNotFoundError`` when *path* does not exist and
    ``json.JSONDecodeError`` on malformed JSON.
    """
    resolved = Path(path).expanduser().resolve()
    return json.loads(resolved.read_text(encoding="utf-8"))


def save_json(
    path: str,
    data: Union[Dict[str, Any], List[Any]],
    *,
    indent: int = 2,
) -> None:
    """Serialise *data* as pretty-printed JSON and write to *path*.

    Parent directories are created automatically if they do not exist.
    """
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(data, indent=indent, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Result merging (resume support)
# ---------------------------------------------------------------------------

def merge_results(
    existing: List[Dict[str, Any]],
    new: List[Dict[str, Any]],
    key: str = "id",
) -> List[Dict[str, Any]]:
    """Merge *new* evaluation results into *existing*, keyed by *key*.

    If a result with the same key value already appears in *existing*,
    the entry from *new* wins (i.e. the latest run takes precedence).
    Ordering follows the original *existing* list with any genuinely new
    items appended at the end.

    >>> existing = [{"id": "q1", "score": 0.5}, {"id": "q2", "score": 0.8}]
    >>> new = [{"id": "q2", "score": 0.9}, {"id": "q3", "score": 1.0}]
    >>> merge_results(existing, new)
    [{'id': 'q1', 'score': 0.5}, {'id': 'q2', 'score': 0.9}, {'id': 'q3', 'score': 1.0}]
    """
    index: Dict[str, int] = {}
    merged: List[Dict[str, Any]] = []

    for item in existing:
        k = item.get(key)
        index[k] = len(merged)
        merged.append(dict(item))

    for item in new:
        k = item.get(key)
        if k in index:
            merged[index[k]] = dict(item)
        else:
            index[k] = len(merged)
            merged.append(dict(item))

    return merged


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """Create a unique, timestamp-based run identifier.

    The format is ``YYYYMMDD-HHMMSS-<short-uuid>`` so that IDs sort
    chronologically and remain human-readable while still being unique
    across concurrent runs.

    >>> rid = generate_run_id()
    >>> len(rid.split("-")) >= 3
    True
    """
    now = datetime.now(tz=timezone.utc)
    ts = now.strftime("%Y%m%d-%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"{ts}-{short_id}"
