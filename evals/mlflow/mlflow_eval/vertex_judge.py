"""Start a local OpenAI-compatible proxy that routes to Claude on Vertex AI.

MLflow's built-in ``vertex_ai:/`` judge provider has a bug with Anthropic
models (missing ``anthropic_version``).  This module works around it by
running a tiny HTTP server that accepts OpenAI-format requests and
forwards them to Vertex AI via ``litellm``.

Usage::

    from mlflowplug.vertex_judge import start_vertex_judge

    start_vertex_judge(
        project="itpc-gcp-eco-eng-claude",
        region="us-east5",
        model="claude-sonnet-4@20250514",
    )
    # Sets MLFLOW_GENAI_JUDGE_DEFAULT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
    # Now all MLflow LLM-judge scorers route through Claude on Vertex AI.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def start_vertex_judge(
    project: str = "itpc-gcp-eco-eng-claude",
    region: str = "us-east5",
    model: str = "claude-sonnet-4@20250514",
    port: int = 4111,
) -> None:
    """Start a local proxy and configure MLflow to use it as the judge.

    The proxy translates OpenAI chat-completion requests into
    ``litellm.completion()`` calls routed to ``vertex_ai/<model>``.
    Authentication uses Application Default Credentials (gcloud).

    Args:
        project: GCP project ID.
        region: Vertex AI region where the model is available.
        model: Anthropic model ID on Vertex AI.
        port: Local port for the proxy (default 4111).
    """
    import litellm

    litellm.vertex_project = project
    litellm.vertex_location = region
    vertex_model = f"vertex_ai/{model}"

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            try:
                resp = litellm.completion(
                    model=vertex_model,
                    messages=body.get("messages", []),
                    max_tokens=body.get("max_tokens", 4096),
                    tools=body.get("tools"),
                    tool_choice=body.get("tool_choice"),
                    temperature=body.get("temperature"),
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp.model_dump()).encode())
            except Exception as exc:
                error_body = json.dumps({"error": {"message": str(exc)}})
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(error_body.encode())

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    model_alias = model.split("@")[0]
    os.environ["OPENAI_API_KEY"] = "vertex-proxy"
    os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    os.environ["MLFLOW_GENAI_JUDGE_DEFAULT_MODEL"] = f"openai:/{model_alias}"
