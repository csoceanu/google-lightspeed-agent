#!/usr/bin/env python3
"""
Tests for the mock A2A agent server (mock_a2a_agent.py).

Covers:
    - GET /health endpoint
    - GET / endpoint (agent card)
    - POST /chat with mocked Gemini responses
    - Error handling when Gemini is unavailable
    - .env loading fallback
    - ask_gemini function (unit tests with mocked HTTP)
    - Application factory and CLI argument parsing
"""

import json
import os
from unittest import mock

import pytest
import pytest_asyncio
import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

# Ensure the project root is importable.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mock_a2a_agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_api_key():
    return "test-fake-api-key-12345"


@pytest_asyncio.fixture
async def client(fake_api_key):
    """Create an aiohttp test client for the mock agent app."""
    app = mock_a2a_agent.create_app(api_key=fake_api_key)
    async with TestClient(TestServer(app)) as tc:
        yield tc


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_health_returns_ok_status(self, client):
        resp = await client.get("/health")
        body = await resp.json()
        assert body == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_health_content_type_is_json(self, client):
        resp = await client.get("/health")
        assert "application/json" in resp.headers.get("Content-Type", "")


# ---------------------------------------------------------------------------
# GET / (agent card)
# ---------------------------------------------------------------------------

class TestAgentCardEndpoint:
    """Tests for the / (agent card) endpoint."""

    @pytest.mark.asyncio
    async def test_root_returns_200(self, client):
        resp = await client.get("/")
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_root_returns_agent_card(self, client):
        resp = await client.get("/")
        body = await resp.json()
        assert body["name"] == "Mock A2A Agent"
        assert "description" in body
        assert "version" in body

    @pytest.mark.asyncio
    async def test_root_lists_capabilities(self, client):
        resp = await client.get("/")
        body = await resp.json()
        assert "chat" in body["capabilities"]
        assert "question-answering" in body["capabilities"]

    @pytest.mark.asyncio
    async def test_root_lists_endpoints(self, client):
        resp = await client.get("/")
        body = await resp.json()
        assert "/chat" in body["endpoints"]
        assert "/health" in body["endpoints"]
        assert "/" in body["endpoints"]

    @pytest.mark.asyncio
    async def test_root_includes_model_backend(self, client):
        resp = await client.get("/")
        body = await resp.json()
        assert "model_backend" in body
        assert "gemini" in body["model_backend"]


# ---------------------------------------------------------------------------
# POST /chat -- mocked Gemini
# ---------------------------------------------------------------------------

def _make_gemini_response(text: str) -> dict:
    """Build a mock Gemini API JSON response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}],
                }
            }
        ]
    }


class TestChatEndpoint:
    """Tests for the /chat endpoint with mocked Gemini API."""

    @pytest.mark.asyncio
    async def test_chat_returns_gemini_answer(self, client):
        gemini_answer = "This is a test answer from Gemini."

        with mock.patch(
            "mock_a2a_agent.ask_gemini",
            return_value=gemini_answer,
        ):
            resp = await client.post(
                "/chat",
                json={"message": "What is Linux?"},
            )

        assert resp.status == 200
        body = await resp.json()
        assert body["response"] == gemini_answer

    @pytest.mark.asyncio
    async def test_chat_response_key_exists(self, client):
        with mock.patch(
            "mock_a2a_agent.ask_gemini",
            return_value="hello",
        ):
            resp = await client.post(
                "/chat",
                json={"message": "Hi"},
            )
        body = await resp.json()
        assert "response" in body

    @pytest.mark.asyncio
    async def test_chat_rejects_missing_message(self, client):
        resp = await client.post("/chat", json={"text": "oops"})
        assert resp.status == 400
        body = await resp.json()
        assert "error" in body

    @pytest.mark.asyncio
    async def test_chat_rejects_empty_message(self, client):
        resp = await client.post("/chat", json={"message": ""})
        assert resp.status == 400
        body = await resp.json()
        assert "error" in body

    @pytest.mark.asyncio
    async def test_chat_rejects_non_string_message(self, client):
        resp = await client.post("/chat", json={"message": 42})
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_chat_rejects_invalid_json(self, client):
        resp = await client.post(
            "/chat",
            data=b"this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400
        body = await resp.json()
        assert "error" in body

    @pytest.mark.asyncio
    async def test_chat_passes_question_to_gemini(self, client):
        question = "Explain CVE-2024-1234"
        with mock.patch(
            "mock_a2a_agent.ask_gemini",
            return_value="answer",
        ) as mock_gemini:
            await client.post("/chat", json={"message": question})

        mock_gemini.assert_called_once()
        call_args = mock_gemini.call_args
        assert call_args[0][0] == question


# ---------------------------------------------------------------------------
# Error handling -- Gemini unavailable / errors
# ---------------------------------------------------------------------------

class TestChatErrorHandling:
    """Tests for error handling when Gemini API is unavailable or fails."""

    @pytest.mark.asyncio
    async def test_gemini_api_error_returns_502(self, client):
        with mock.patch(
            "mock_a2a_agent.ask_gemini",
            side_effect=RuntimeError("Gemini API error (HTTP 500): internal"),
        ):
            resp = await client.post(
                "/chat",
                json={"message": "test"},
            )

        assert resp.status == 502
        body = await resp.json()
        assert "error" in body

    @pytest.mark.asyncio
    async def test_gemini_network_error_returns_502(self, client):
        with mock.patch(
            "mock_a2a_agent.ask_gemini",
            side_effect=RuntimeError("Network error calling Gemini API: timeout"),
        ):
            resp = await client.post(
                "/chat",
                json={"message": "test"},
            )

        assert resp.status == 502
        body = await resp.json()
        assert "error" in body
        assert "Network error" in body["error"]

    @pytest.mark.asyncio
    async def test_unexpected_error_returns_500(self, client):
        with mock.patch(
            "mock_a2a_agent.ask_gemini",
            side_effect=ValueError("something unexpected"),
        ):
            resp = await client.post(
                "/chat",
                json={"message": "test"},
            )

        assert resp.status == 500
        body = await resp.json()
        assert "error" in body

    @pytest.mark.asyncio
    async def test_gemini_no_candidates_returns_502(self, client):
        with mock.patch(
            "mock_a2a_agent.ask_gemini",
            side_effect=RuntimeError("Gemini returned no candidates"),
        ):
            resp = await client.post(
                "/chat",
                json={"message": "test"},
            )

        assert resp.status == 502


# ---------------------------------------------------------------------------
# ask_gemini unit tests (with mocked HTTP)
# ---------------------------------------------------------------------------

class TestAskGeminiFunction:
    """Unit tests for the ask_gemini function itself."""

    @pytest.mark.asyncio
    async def test_ask_gemini_extracts_text(self):
        gemini_response = _make_gemini_response("The answer is 42.")

        mock_resp = mock.AsyncMock()
        mock_resp.status = 200
        mock_resp.json = mock.AsyncMock(return_value=gemini_response)
        mock_resp.__aenter__ = mock.AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = mock.AsyncMock(return_value=False)

        mock_session = mock.AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = mock.MagicMock(return_value=mock_resp)

        result = await mock_a2a_agent.ask_gemini(
            "What is the meaning of life?", "fake-key", mock_session
        )
        assert result == "The answer is 42."

    @pytest.mark.asyncio
    async def test_ask_gemini_raises_on_http_error(self):
        error_body = {"error": {"message": "Quota exceeded"}}

        mock_resp = mock.AsyncMock()
        mock_resp.status = 429
        mock_resp.reason = "Too Many Requests"
        mock_resp.json = mock.AsyncMock(return_value=error_body)
        mock_resp.__aenter__ = mock.AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = mock.AsyncMock(return_value=False)

        mock_session = mock.AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = mock.MagicMock(return_value=mock_resp)

        with pytest.raises(RuntimeError, match="Quota exceeded"):
            await mock_a2a_agent.ask_gemini("test", "fake-key", mock_session)

    @pytest.mark.asyncio
    async def test_ask_gemini_raises_on_empty_candidates(self):
        empty_response = {"candidates": []}

        mock_resp = mock.AsyncMock()
        mock_resp.status = 200
        mock_resp.json = mock.AsyncMock(return_value=empty_response)
        mock_resp.__aenter__ = mock.AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = mock.AsyncMock(return_value=False)

        mock_session = mock.AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = mock.MagicMock(return_value=mock_resp)

        with pytest.raises(RuntimeError, match="no candidates"):
            await mock_a2a_agent.ask_gemini("test", "fake-key", mock_session)

    @pytest.mark.asyncio
    async def test_ask_gemini_raises_on_no_parts(self):
        no_parts_response = {
            "candidates": [{"content": {"parts": []}}]
        }

        mock_resp = mock.AsyncMock()
        mock_resp.status = 200
        mock_resp.json = mock.AsyncMock(return_value=no_parts_response)
        mock_resp.__aenter__ = mock.AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = mock.AsyncMock(return_value=False)

        mock_session = mock.AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = mock.MagicMock(return_value=mock_resp)

        with pytest.raises(RuntimeError, match="no content parts"):
            await mock_a2a_agent.ask_gemini("test", "fake-key", mock_session)

    @pytest.mark.asyncio
    async def test_ask_gemini_raises_on_network_error(self):
        mock_session = mock.AsyncMock(spec=aiohttp.ClientSession)
        mock_cm = mock.AsyncMock()
        mock_cm.__aenter__ = mock.AsyncMock(
            side_effect=aiohttp.ClientError("connection refused")
        )
        mock_cm.__aexit__ = mock.AsyncMock(return_value=False)
        mock_session.post = mock.MagicMock(return_value=mock_cm)

        with pytest.raises(RuntimeError, match="Network error"):
            await mock_a2a_agent.ask_gemini("test", "fake-key", mock_session)


# ---------------------------------------------------------------------------
# .env loading fallback
# ---------------------------------------------------------------------------

class TestEnvLoading:
    """Tests for the .env file loading (both python-dotenv and manual)."""

    def test_manual_dotenv_loader_reads_key_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_TEST_KEY=my_test_value\n")

        os.environ.pop("MY_TEST_KEY", None)

        mock_a2a_agent._load_dotenv_manual(str(env_file))
        assert os.environ.get("MY_TEST_KEY") == "my_test_value"

        os.environ.pop("MY_TEST_KEY", None)

    def test_manual_dotenv_ignores_comments(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# This is a comment\nTEST_VAR=hello\n")

        os.environ.pop("TEST_VAR", None)
        mock_a2a_agent._load_dotenv_manual(str(env_file))
        assert os.environ.get("TEST_VAR") == "hello"

        os.environ.pop("TEST_VAR", None)

    def test_manual_dotenv_ignores_empty_lines(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("\n\nTEST_EMPTY=present\n\n")

        os.environ.pop("TEST_EMPTY", None)
        mock_a2a_agent._load_dotenv_manual(str(env_file))
        assert os.environ.get("TEST_EMPTY") == "present"

        os.environ.pop("TEST_EMPTY", None)

    def test_manual_dotenv_strips_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('QUOTED_VAR="some value"\nSINGLE_QUOTED=\'other\'\n')

        os.environ.pop("QUOTED_VAR", None)
        os.environ.pop("SINGLE_QUOTED", None)

        mock_a2a_agent._load_dotenv_manual(str(env_file))
        assert os.environ.get("QUOTED_VAR") == "some value"
        assert os.environ.get("SINGLE_QUOTED") == "other"

        os.environ.pop("QUOTED_VAR", None)
        os.environ.pop("SINGLE_QUOTED", None)

    def test_manual_dotenv_does_not_overwrite_existing(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_VAR=from_file\n")

        os.environ["EXISTING_VAR"] = "from_env"
        mock_a2a_agent._load_dotenv_manual(str(env_file))
        assert os.environ.get("EXISTING_VAR") == "from_env"

        os.environ.pop("EXISTING_VAR", None)

    def test_manual_dotenv_nonexistent_file(self, tmp_path):
        # Should not raise -- just silently return.
        mock_a2a_agent._load_dotenv_manual(str(tmp_path / "nonexistent"))

    def test_get_api_key_raises_when_missing(self):
        os.environ.pop("GEMINI_API_KEY", None)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
            mock_a2a_agent.get_api_key()

    def test_get_api_key_returns_key_when_set(self):
        os.environ["GEMINI_API_KEY"] = "test-key-123"
        try:
            assert mock_a2a_agent.get_api_key() == "test-key-123"
        finally:
            os.environ.pop("GEMINI_API_KEY", None)

    def test_load_env_uses_manual_when_dotenv_missing(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MANUAL_LOAD_TEST=works\n")

        os.environ.pop("MANUAL_LOAD_TEST", None)

        # Simulate python-dotenv not being installed.
        with mock.patch.dict("sys.modules", {"dotenv": None}):
            mock_a2a_agent.load_env(str(env_file))

        assert os.environ.get("MANUAL_LOAD_TEST") == "works"
        os.environ.pop("MANUAL_LOAD_TEST", None)

    def test_manual_dotenv_skips_lines_without_equals(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("NO_EQUALS_LINE\nGOOD_KEY=good_val\n")

        os.environ.pop("GOOD_KEY", None)
        mock_a2a_agent._load_dotenv_manual(str(env_file))
        assert os.environ.get("GOOD_KEY") == "good_val"
        os.environ.pop("GOOD_KEY", None)

    def test_load_env_default_path(self):
        with mock.patch.object(mock_a2a_agent, "_load_dotenv_manual") as m, \
             mock.patch.dict("sys.modules", {"dotenv": None}):
            mock_a2a_agent.load_env(None)
        called_path = m.call_args[0][0]
        assert called_path.endswith(".env")

    def test_load_env_uses_dotenv_when_available(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("DOTENV_TEST=val\n")
        mock_dotenv = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"dotenv": mock_dotenv}):
            mock_a2a_agent.load_env(str(env_file))
        mock_dotenv.load_dotenv.assert_called_once_with(str(env_file))


# ---------------------------------------------------------------------------
# create_app factory
# ---------------------------------------------------------------------------

class TestCreateApp:
    """Tests for the application factory."""

    def test_create_app_with_explicit_key(self):
        app = mock_a2a_agent.create_app(api_key="explicit-key")
        assert app[mock_a2a_agent.gemini_api_key_key] == "explicit-key"

    def test_create_app_reads_key_from_env(self):
        os.environ["GEMINI_API_KEY"] = "env-key-456"
        try:
            app = mock_a2a_agent.create_app()
            assert app[mock_a2a_agent.gemini_api_key_key] == "env-key-456"
        finally:
            os.environ.pop("GEMINI_API_KEY", None)

    def test_create_app_raises_without_key(self):
        os.environ.pop("GEMINI_API_KEY", None)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
            mock_a2a_agent.create_app()

    def test_create_app_has_all_routes(self, fake_api_key):
        app = mock_a2a_agent.create_app(api_key=fake_api_key)
        routes = {
            r.resource.canonical
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        }
        assert "/" in routes
        assert "/health" in routes
        assert "/chat" in routes


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    """Tests for the CLI argument parser."""

    def test_default_port(self):
        args = mock_a2a_agent.parse_args([])
        assert args.port == 8888

    def test_custom_port(self):
        args = mock_a2a_agent.parse_args(["--port", "9999"])
        assert args.port == 9999

    def test_default_host(self):
        args = mock_a2a_agent.parse_args([])
        assert args.host == "0.0.0.0"

    def test_custom_host(self):
        args = mock_a2a_agent.parse_args(["--host", "127.0.0.1"])
        assert args.host == "127.0.0.1"

    def test_log_level(self):
        args = mock_a2a_agent.parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_env_file(self):
        args = mock_a2a_agent.parse_args(["--env-file", "/tmp/.env.test"])
        assert args.env_file == "/tmp/.env.test"


# ---------------------------------------------------------------------------
# main() function
# ---------------------------------------------------------------------------

class TestMainFunction:
    """Tests for the main() CLI entry point."""

    def test_main_starts_server(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("GEMINI_API_KEY=test-key-main\n")

        with mock.patch("mock_a2a_agent.web.run_app") as mock_run, \
             mock.patch.dict("sys.modules", {"dotenv": None}):
            mock_a2a_agent.main([
                "--port", "9999",
                "--host", "127.0.0.1",
                "--env-file", str(env_file),
                "--log-level", "WARNING",
            ])

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["port"] == 9999
        assert call_kwargs[1]["host"] == "127.0.0.1"

    def test_main_raises_without_api_key(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")

        os.environ.pop("GEMINI_API_KEY", None)
        with mock.patch.dict("sys.modules", {"dotenv": None}), \
             pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            mock_a2a_agent.main(["--env-file", str(env_file)])
