#!/usr/bin/env python3
"""A2A runner for Agent-Eval-Harness CLI runner integration.

Sends a question to the Lightspeed Agent via A2A protocol and saves the
response. Designed to be called by AEH's CLI runner — reads question from
args, writes response to output directory.

Environment variables:
    A2A_AGENT_URL    — Agent A2A endpoint URL (required)
    A2A_AUTH_TOKEN   — Bearer token for authentication (required)
    A2A_TIMEOUT      — Request timeout in seconds (default: 180)
    A2A_INSECURE_TLS — Skip TLS verification (default: false)
"""

import argparse
import json
import os
import ssl
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def extract_final_answer(text: str) -> str:
    """Extract only the user-facing answer, stripping Gemini reasoning blocks."""
    if "/*FINAL_ANSWER*/" in text:
        _, _, answer = text.partition("/*FINAL_ANSWER*/")
        return answer.strip()
    return text.strip()


def send_a2a_message(agent_url: str, token: str, message: str,
                     timeout: int, insecure: bool) -> dict:
    """Send a JSON-RPC message/send to the agent and return parsed response."""
    url = agent_url.rstrip("/") + "/"
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": message}],
                "messageId": str(uuid.uuid4()),
            },
        },
    }).encode("utf-8")

    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    request = Request(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
        data=payload,
    )

    result = urlopen(request, timeout=timeout, context=ctx).read()
    return json.loads(result)


def extract_response_text(response: dict) -> str:
    """Extract text parts from A2A response."""
    parts = []
    task_result = response.get("result", {})

    for artifact in task_result.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("kind") == "text" or part.get("type") == "text":
                parts.append(part.get("text", ""))

    status_msg = task_result.get("status", {}).get("message", {})
    if status_msg:
        for part in status_msg.get("parts", []):
            if part.get("kind") == "text" or part.get("type") == "text":
                parts.append(part.get("text", ""))

    text = "\n".join(parts) if parts else "(no text response)"
    return extract_final_answer(text)


def extract_metadata(response: dict) -> dict:
    """Extract token usage and timing metadata from A2A response."""
    task_result = response.get("result", {})
    metadata = task_result.get("metadata", {})
    usage = metadata.get("adk_usage_metadata", {})

    return {
        "prompt_tokens": usage.get("promptTokenCount", 0),
        "completion_tokens": usage.get("candidatesTokenCount", 0),
        "total_tokens": usage.get("totalTokenCount", 0),
        "cached_tokens": usage.get("cachedContentTokenCount", 0),
        "model": metadata.get("adk_app_name", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="A2A runner for AEH")
    parser.add_argument("--question", required=True, help="Question to send")
    parser.add_argument("--output-dir", required=True, help="Directory for output files")
    args = parser.parse_args()

    agent_url = os.environ.get("A2A_AGENT_URL")
    token = os.environ.get("A2A_AUTH_TOKEN")
    timeout = int(os.environ.get("A2A_TIMEOUT", "180"))
    insecure = os.environ.get("A2A_INSECURE_TLS", "false").lower() == "true"

    if not agent_url:
        print("ERROR: A2A_AGENT_URL not set", file=sys.stderr)
        sys.exit(1)
    if not token:
        print("ERROR: A2A_AUTH_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    try:
        response = send_a2a_message(agent_url, token, args.question, timeout, insecure)
        elapsed = time.time() - start

        response_text = extract_response_text(response)
        metadata = extract_metadata(response)

        # Write response text for judges
        (output_dir / "response.txt").write_text(response_text)

        # Write full A2A response for detailed analysis
        (output_dir / "a2a_response.json").write_text(
            json.dumps(response, indent=2, default=str)
        )

        # Write metrics for AEH
        metrics = {
            "duration_s": round(elapsed, 2),
            "token_usage": metadata,
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        print(f"Response received ({elapsed:.1f}s, {len(response_text)} chars)")

    except (HTTPError, URLError, RuntimeError) as e:
        elapsed = time.time() - start
        error_msg = f"[ERROR] {e}"

        (output_dir / "response.txt").write_text(error_msg)
        (output_dir / "metrics.json").write_text(json.dumps({
            "duration_s": round(elapsed, 2),
            "error": str(e),
        }, indent=2))

        print(f"ERROR after {elapsed:.1f}s: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
