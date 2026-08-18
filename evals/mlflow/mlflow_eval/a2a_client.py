"""A2A (Agent-to-Agent) protocol client for the Lightspeed Agent.

The real Lightspeed Agent uses A2A JSON-RPC 2.0 at ``POST /``, not
``POST /chat``.  This module translates simple question strings into
A2A ``message/send`` requests and extracts the text response.

Usage::

    from mlflow_eval.a2a_client import a2a_predict_fn

    predict = a2a_predict_fn("http://localhost:8000")
    answer = predict(question="Is CVE-2024-6387 affecting my systems?",
                     question_id="V-001")
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Callable

import requests

logger = logging.getLogger(__name__)


def send_a2a_message(
    agent_url: str,
    message: str,
    token: str = "dev-token",
    timeout: int = 60,
) -> str:
    """Send an A2A ``message/send`` request and return the text response.

    Args:
        agent_url: Base URL of the agent (e.g. ``http://localhost:8000``).
        message: The question text to send.
        token: Bearer token. Use ``"dev-token"`` when the agent has
            ``SKIP_JWT_VALIDATION=true``.
        timeout: Request timeout in seconds.

    Returns:
        The agent's text response extracted from the A2A task result.
    """
    url = agent_url.rstrip("/") + "/"
    msg_id = str(uuid.uuid4())

    payload = {
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": message}],
                "messageId": msg_id,
            },
        },
    }

    resp = requests.post(
        url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    return _extract_text(data), data


def _extract_text(response: dict) -> str:
    """Extract text from an A2A JSON-RPC response.

    The response structure varies by task state. Common shapes:

    - ``result.status.message.parts[].text`` — direct response
    - ``result.artifacts[].parts[].text`` — artifact-based response
    - ``result.history[-1].parts[].text`` — conversation history
    """
    result = response.get("result", {})

    # Try status message first
    status_msg = result.get("status", {}).get("message", {})
    if status_msg:
        parts = status_msg.get("parts", [])
        texts = [p.get("text", "") for p in parts if p.get("kind") == "text"]
        if texts:
            text = "\n".join(texts)
            return _strip_reasoning(text)

    # Try artifacts — skip adk_thought parts (LLM reasoning, not user-facing)
    for artifact in result.get("artifacts", []):
        parts = artifact.get("parts", [])
        answer_parts = [
            p.get("text", "") for p in parts
            if p.get("kind") == "text"
            and not p.get("metadata", {}).get("adk_thought")
        ]
        if answer_parts:
            return _strip_reasoning("\n".join(answer_parts))
        # Fallback: if all parts are thought parts, use them but strip reasoning
        all_texts = [p.get("text", "") for p in parts if p.get("kind") == "text"]
        if all_texts:
            return _strip_reasoning("\n".join(all_texts))

    # Try history (last message from agent)
    history = result.get("history", [])
    for msg in reversed(history):
        if msg.get("role") != "user":
            parts = msg.get("parts", [])
            texts = [p.get("text", "") for p in parts if p.get("kind") == "text"]
            if texts:
                text = "\n".join(texts)
                return _strip_reasoning(text)

    # Fallback: stringify the whole result
    logger.warning("Could not extract text from A2A response, using raw result")
    return json.dumps(result)


def _strip_reasoning(text: str) -> str:
    """Strip Gemini reasoning markers, keeping only the final answer.

    Gemini models wrap chain-of-thought in ``/*REASONING*/`` and
    ``/*PLANNING*/`` blocks. The actual response is in ``/*FINAL_ANSWER*/``.
    """
    if "/*FINAL_ANSWER*/" in text:
        final = text.split("/*FINAL_ANSWER*/")[-1].strip()
        if final:
            return final
    return text


def a2a_predict_fn(
    agent_url: str = "http://localhost:8000",
    token: str = "dev-token",
    timeout: int = 60,
) -> Callable:
    """Create a ``predict_fn`` compatible with ``mlflow.genai.evaluate()``.

    Returns a function with signature ``(question, question_id) -> str``
    that sends each question to the real Lightspeed Agent via A2A protocol.

    Args:
        agent_url: Base URL of the agent.
        token: Bearer token (use ``"dev-token"`` with ``SKIP_JWT_VALIDATION=true``).
        timeout: Per-question timeout in seconds.

    Usage::

        predict = a2a_predict_fn("http://localhost:8000")

        mlflow.genai.evaluate(
            data=dataset,
            predict_fn=predict,
            scorers=[...],
        )
    """

    traces = {}

    def predict(question: str, question_id: str) -> str:
        logger.info("A2A request %s: %s", question_id, question[:60])
        text, raw = send_a2a_message(agent_url, question, token=token, timeout=timeout)
        traces[question_id] = raw
        return text

    predict.traces = traces
    return predict
