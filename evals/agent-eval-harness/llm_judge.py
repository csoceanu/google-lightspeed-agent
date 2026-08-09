"""LLM judge helper — calls the judge model via plain chat (no tool calling).

Used by check judges in eval.yaml to avoid the tool_choice reliability
issue with gpt-oss-120b via LiteLLM.
"""

import json
import os
import re
import ssl
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def score(prompt: str, max_tokens: int = 200, timeout: int = 60) -> tuple:
    """Call the judge model and parse SCORE/RATIONALE from the response.

    Returns (score_int, rationale_string).
    Retries up to 3 times on empty responses or server errors.
    """
    api_key = os.environ.get("JUDGE_API_KEY", "")
    api_url = os.environ.get("JUDGE_API_URL", "")
    if not api_url:
        return 0, "JUDGE_API_URL not set"
    model = os.environ.get("JUDGE_MODEL", "gpt-oss-120b")

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
            result = json.loads(urlopen(req, timeout=timeout, context=ctx).read())
            msg = result["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = (msg.get("reasoning_content")
                         or msg.get("provider_specific_fields", {}).get("reasoning_content")
                         or "")
            text = (content or reasoning).strip()

            if text:
                # Find the LAST SCORE/RATIONALE (skip reasoning preamble)
                matches = list(re.finditer(r'SCORE:\s*(\d)', text))
                score_val = int(matches[-1].group(1)) if matches else 3
                last_rat = list(re.finditer(r'RATIONALE:\s*', text))
                if last_rat:
                    rationale = text[last_rat[-1].end():].strip()
                else:
                    rationale = text
                # Clean up: remove any trailing prompt echoes
                rationale = rationale.split('\n\nReply ONLY:')[0].strip()
                rationale = rationale.split('\n\nScore 1-5')[0].strip()
                return score_val, f"{score_val}/5 — {rationale[:300]}"

        except (HTTPError, OSError) as e:
            last_error = e

        if attempt < 2:
            time.sleep(3)

    return 0, f"Judge error after 3 attempts: {last_error or 'empty response'}"
