#!/usr/bin/env python3
"""
Mock A2A Agent Server

A lightweight HTTP server that accepts evaluation requests in the same format
as the Red Hat Lightspeed Agent and proxies questions to Google Gemini API.

Endpoints:
    GET  /        - Agent card (capabilities and metadata)
    GET  /health  - Health check
    POST /chat    - Send a question, get a Gemini-powered response

Usage:
    python mock_a2a_agent.py [--port 8888]

    curl -X POST http://localhost:8888/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "What is CVE-2024-1234?"}'
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import aiohttp
from aiohttp import web

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("mock_a2a_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

gemini_api_key_key: web.AppKey[str] = web.AppKey("gemini_api_key", str)
http_session_key: web.AppKey[aiohttp.ClientSession] = web.AppKey(
    "http_session", aiohttp.ClientSession
)

AGENT_CARD = {
    "name": "Mock A2A Agent",
    "description": (
        "A mock Agent-to-Agent server that proxies questions to "
        "Google Gemini API for evaluation purposes."
    ),
    "version": "1.0.0",
    "capabilities": ["chat", "question-answering"],
    "endpoints": {
        "/": "Agent card (this document)",
        "/health": "Health check",
        "/chat": "POST a question and receive an answer",
    },
    "model_backend": f"google/{GEMINI_MODEL}",
}


# ---------------------------------------------------------------------------
# .env loading (with fallback if python-dotenv is not installed)
# ---------------------------------------------------------------------------

def _load_dotenv_manual(dotenv_path: str) -> None:
    """Minimal .env loader: reads KEY=VALUE lines, ignoring comments."""
    path = Path(dotenv_path)
    if not path.is_file():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip optional surrounding quotes.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ.setdefault(key, value)


def load_env(dotenv_path: Optional[str] = None) -> None:
    """Load environment variables from a .env file.

    Tries python-dotenv first; falls back to a simple manual parser.
    """
    if dotenv_path is None:
        dotenv_path = str(Path(__file__).resolve().parent / ".env")

    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        load_dotenv(dotenv_path)
        logger.debug("Loaded .env via python-dotenv")
    except ImportError:
        logger.debug("python-dotenv not installed; using manual .env loader")
        _load_dotenv_manual(dotenv_path)


def get_api_key() -> str:
    """Return the Gemini API key from the environment, or raise."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Provide it via the environment or a "
            ".env file in the project root."
        )
    return key


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

async def ask_gemini(
    question: str,
    api_key: str,
    session: aiohttp.ClientSession,
    timeout: float = 60.0,
) -> str:
    """Send *question* to the Gemini API and return the text answer."""
    url = f"{GEMINI_API_URL}?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": question}]}],
    }

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            body = await resp.json()

            if resp.status != 200:
                error_msg = body.get("error", {}).get("message", resp.reason)
                raise RuntimeError(
                    f"Gemini API error (HTTP {resp.status}): {error_msg}"
                )

            # Extract the text from the first candidate.
            candidates = body.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini returned no candidates")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise RuntimeError("Gemini candidate has no content parts")

            return parts[0].get("text", "")

    except aiohttp.ClientError as exc:
        raise RuntimeError(f"Network error calling Gemini API: {exc}") from exc


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------

async def handle_root(request: web.Request) -> web.Response:
    """GET / - return the agent card."""
    return web.json_response(AGENT_CARD)


async def handle_health(request: web.Request) -> web.Response:
    """GET /health - basic health check."""
    return web.json_response({"status": "ok"})


async def handle_chat(request: web.Request) -> web.Response:
    """POST /chat - accept a question and return a Gemini-powered answer."""
    # Parse the request body.
    try:
        data = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response(
            {"error": "Invalid JSON in request body"},
            status=400,
        )

    message = data.get("message")
    if not message or not isinstance(message, str):
        return web.json_response(
            {"error": "Request must include a non-empty 'message' string"},
            status=400,
        )

    # Retrieve the API key and shared session from the app context.
    api_key: str = request.app[gemini_api_key_key]
    session: aiohttp.ClientSession = request.app[http_session_key]

    try:
        answer = await ask_gemini(message, api_key, session)
        return web.json_response({"response": answer})
    except RuntimeError as exc:
        logger.error("Gemini error: %s", exc)
        return web.json_response(
            {"error": str(exc)},
            status=502,
        )
    except Exception as exc:
        logger.exception("Unexpected error in /chat handler")
        return web.json_response(
            {"error": f"Internal server error: {exc}"},
            status=500,
        )


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

async def on_startup(app: web.Application) -> None:
    """Create a shared aiohttp session on server startup."""
    app[http_session_key] = aiohttp.ClientSession()
    logger.info("HTTP session created")


async def on_cleanup(app: web.Application) -> None:
    """Close the shared aiohttp session on server shutdown."""
    session: aiohttp.ClientSession = app[http_session_key]
    await session.close()
    logger.info("HTTP session closed")


def create_app(api_key: Optional[str] = None) -> web.Application:
    """Build and return the aiohttp Application.

    Parameters
    ----------
    api_key : str, optional
        Gemini API key.  When *None*, the key is read from the environment
        (GEMINI_API_KEY).
    """
    app = web.Application()

    if api_key is None:
        api_key = get_api_key()
    app[gemini_api_key_key] = api_key

    # Routes
    app.router.add_get("/", handle_root)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/chat", handle_chat)

    # Lifecycle hooks
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mock A2A Agent Server -- proxies questions to Google Gemini"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Port to listen on (default: 8888)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        dest="env_file",
        help="Path to .env file (default: .env in the project root)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load .env
    load_env(args.env_file)

    # Validate API key early.
    api_key = get_api_key()
    logger.info("Gemini API key loaded (ends with ...%s)", api_key[-4:])

    app = create_app(api_key=api_key)

    logger.info("Starting Mock A2A Agent on %s:%d", args.host, args.port)
    logger.info("Endpoints: GET /  |  GET /health  |  POST /chat")

    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":  # pragma: no cover
    main()
