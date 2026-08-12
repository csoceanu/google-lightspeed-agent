"""LLM judge helper — calls the judge model via plain chat (no tool calling).

Used by check judges to call the judge model and parse yes/no verdicts.
Retries up to 3 times on empty responses or server errors.
Respects rate limits by waiting between calls (file-based to persist across modules).
"""

import json
import os
import re
import ssl
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

_RATE_FILE = Path(tempfile.gettempdir()) / "llm_judge_last_call"


def _get_last_call_time() -> float:
    try:
        return float(_RATE_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return 0.0


def _set_last_call_time():
    _RATE_FILE.write_text(str(time.time()))


def score(prompt: str, max_tokens: int = 2000, timeout: int = 120) -> tuple:
    """Call the judge model and parse VERDICT/RATIONALE from the response.

    Returns (bool, rationale_string).
    Enforces minimum 11 seconds between calls (6 req/min rate limit).
    """
    api_key = os.environ.get("JUDGE_API_KEY", "")
    api_url = os.environ.get("JUDGE_API_URL", "")
    if not api_url:
        return False, "JUDGE_API_URL not set"
    model = os.environ.get("JUDGE_MODEL", "")
    if not model:
        return False, "JUDGE_MODEL not set"

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    last_error = None
    for attempt in range(3):
        elapsed = time.time() - _get_last_call_time()
        if elapsed < 11:
            time.sleep(11 - elapsed)

        try:
            req = Request(
                f"{api_url.rstrip('/')}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
                data=payload,
            )
            _set_last_call_time()
            result = json.loads(urlopen(req, timeout=timeout, context=ctx).read())
            msg = result["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = (msg.get("reasoning_content")
                         or msg.get("reasoning")
                         or msg.get("provider_specific_fields", {}).get("reasoning_content")
                         or msg.get("provider_specific_fields", {}).get("reasoning")
                         or "")
            text = (content or reasoning).strip()

            if text:
                matches = list(re.finditer(r'VERDICT:\s*(yes|no)', text, re.IGNORECASE))
                if matches:
                    verdict = matches[-1].group(1).lower() == "yes"
                else:
                    verdict = "yes" in text.lower().split("\n")[0]

                last_rat = list(re.finditer(r'RATIONALE:\s*', text))
                if last_rat:
                    rationale = text[last_rat[-1].end():].strip()
                else:
                    rationale = text
                rationale = rationale.split('\n\nReply ONLY:')[0].strip()
                return verdict, rationale

        except (HTTPError, OSError) as e:
            last_error = e

        if attempt < 2:
            time.sleep(15)

    return False, f"Judge error after 3 attempts: {last_error or 'empty response'}"
