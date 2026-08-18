#!/usr/bin/env python3
"""
Full-stack evaluation: MLflow → mock API → real MCP → real agent → eval.

The agent sends OTel traces to MLflow. The eval pipeline calls the agent,
scores responses with code-based scorers, and logs metrics to MLflow.

Usage:
    PYTHONPATH=. python -m mlflow_eval.run_full_stack --per-type 1
    PYTHONPATH=. python -m mlflow_eval.run_full_stack --per-type 2 --enable-judge
"""

import argparse
import os
import subprocess
import sys
import time

os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "True"

import requests


def wait_for_health(url: str, label: str, timeout: int = 60) -> bool:
    for i in range(timeout):
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                print(f"  {label} ready")
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"  ERROR: {label} failed to start after {timeout}s")
    return False


def main():
    parser = argparse.ArgumentParser(description="Full-stack Lightspeed evaluation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-type", type=int, default=1,
                        help="Questions per question_type (default: 1)")
    parser.add_argument("--category", default=None)
    parser.add_argument("--enable-judge", action="store_true")
    parser.add_argument("--experiment", default="lightspeed-full-eval")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--mock-api-port", type=int, default=9000)
    parser.add_argument("--mcp-port", type=int, default=8080)
    parser.add_argument("--agent-port", type=int, default=8000)
    parser.add_argument("--mlflow-port", type=int, default=5000)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mcp_dir = os.path.join(repo_root, "dependencies", "insights-mcp")
    agent_dir = os.path.join(repo_root, "dependencies", "google-lightspeed-agent")

    procs = []

    try:
        print("=" * 60)
        print("STARTING FULL STACK")
        print("=" * 60)

        # ── 0. Start MLflow server (accepts OTel traces from agent) ──
        print(f"\n0. MLflow Server (port {args.mlflow_port})...")
        mlflow_server = subprocess.Popen(
            [sys.executable, "-m", "mlflow", "server",
             "--backend-store-uri", args.tracking_uri,
             "--host", "127.0.0.1", "--port", str(args.mlflow_port),
             "--no-serve-artifacts"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        procs.append(("MLflow Server", mlflow_server))
        if not wait_for_health(f"http://127.0.0.1:{args.mlflow_port}/health", "MLflow Server", 15):
            raise SystemExit(1)

        mlflow_url = f"http://127.0.0.1:{args.mlflow_port}"

        # ── 1. Start mock Insights API ───────────────────────────────
        print(f"\n1. Mock Insights API (port {args.mock_api_port})...")
        mock_api = subprocess.Popen(
            [sys.executable, "-m", "mlflow_eval.mock_insights_api",
             "--port", str(args.mock_api_port)],
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": repo_root},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        procs.append(("Mock API", mock_api))
        if not wait_for_health(
            f"http://127.0.0.1:{args.mock_api_port}/api/vulnerability/v1/vulnerabilities/cves",
            "Mock API", 10,
        ):
            raise SystemExit(1)

        # ── 2. Start real MCP server ─────────────────────────────────
        print(f"\n2. MCP Server (port {args.mcp_port})...")
        toolsets = "vulnerability,inventory,advisor,planning,content-sources,rbac,rhsm,remediations"
        mcp = subprocess.Popen(
            ["uv", "run", "insights-mcp", "--toolset", toolsets,
             "http", "--host", "127.0.0.1", "--port", str(args.mcp_port)],
            cwd=mcp_dir,
            env={
                **os.environ,
                "INSIGHTS_BASE_URL": f"http://127.0.0.1:{args.mock_api_port}",
                "INSIGHTS_SSO_BASE_URL": "http://localhost:0",
            },
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        procs.append(("MCP Server", mcp))
        time.sleep(10)
        if mcp.poll() is not None:
            stderr = mcp.stderr.read(1000).decode()
            print(f"  ERROR: MCP Server died: {stderr[:300]}")
            raise SystemExit(1)
        print("  MCP Server ready")

        # ── 3. Start real Lightspeed Agent (with OTel → MLflow) ──────
        print(f"\n3. Lightspeed Agent (port {args.agent_port})...")
        agent_python = os.path.join(agent_dir, ".venv-agent", "bin", "python")
        if not os.path.exists(agent_python):
            agent_python = sys.executable
        agent = subprocess.Popen(
            [agent_python, "-m", "lightspeed_agent.main"],
            cwd=agent_dir,
            env={
                **os.environ,
                "MCP_TRANSPORT_MODE": "http",
                "MCP_SERVER_URL": f"http://127.0.0.1:{args.mcp_port}",
                "SKIP_JWT_VALIDATION": "true",
                "SKIP_ORDER_VALIDATION": "true",
                "SESSION_BACKEND": "memory",
                "LOG_FORMAT": "text",
                "AGENT_PORT": str(args.agent_port),
                "LLM_PROVIDER": "litellm",
                "LLM_MODEL": "vertex_ai/claude-sonnet-4@20250514",
                "VERTEXAI_PROJECT": "itpc-gcp-eco-eng-claude",
                "VERTEXAI_LOCATION": "us-east5",
                "RATE_LIMIT_REDIS_URL": "redis://localhost:6379/0",
                "OTEL_ENABLED": "true",
                "OTEL_EXPORTER_TYPE": "otlp-http",
                "OTEL_EXPORTER_OTLP_HTTP_ENDPOINT": mlflow_url,
                "OTEL_EXPORTER_OTLP_ENDPOINT": mlflow_url,
            },
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        procs.append(("Agent", agent))
        if not wait_for_health(
            f"http://127.0.0.1:{args.agent_port}/.well-known/agent.json",
            "Agent", 60,
        ):
            stderr = agent.stderr.read(2000).decode() if agent.poll() is not None else ""
            if stderr:
                print(f"  Agent stderr: {stderr[:500]}")
            raise SystemExit(1)

        # ── 4. Run evaluation ────────────────────────────────────────
        print(f"\n4. Running evaluation...")
        print("=" * 60)

        import mlflow

        sys.path.insert(0, repo_root)
        from mlflow_eval.a2a_client import a2a_predict_fn
        from mlflow_eval.dataset import load_dataset
        from mlflow_eval.dataset import load_cases
        from mlflow_eval.scorers import AnswerCorrectness, BehaviorCoverage, ResponseReceived, ToolMatch

        mlflow.set_tracking_uri(mlflow_url)
        mlflow.set_experiment(args.experiment)
        mlflow.autolog(disable=True)

        if args.limit or args.category:
            from mlflow_eval.dataset import load_dataset
            dataset = load_dataset(
                category=args.category, limit=args.limit, per_type=args.per_type,
            )
        else:
            dataset = load_cases()
        print(f"Loaded {len(dataset)} questions")

        predict = a2a_predict_fn(
            agent_url=f"http://127.0.0.1:{args.agent_port}",
            token="dev-token",
        )

        # Call the agent for each question and collect responses.
        # Agent traces go to MLflow via OTel independently.
        print("Calling agent...")
        for d in dataset:
            qid = d["inputs"]["question_id"]
            q = d["inputs"]["question"]
            try:
                answer = predict(question=q, question_id=qid)
                d["outputs"] = answer
                if qid in predict.traces:
                    d["expectations"]["a2a_trace"] = predict.traces[qid]
                print(f"  {qid}: {answer[:60]}...")
            except Exception as exc:
                d["outputs"] = f"ERROR: {exc}"
                print(f"  {qid}: ERROR - {exc}")

        scorers = [ResponseReceived(), AnswerCorrectness(), ToolMatch(), BehaviorCoverage()]

        if args.enable_judge:
            from mlflow.genai.scorers import Correctness, ExpectationsGuidelines, RelevanceToQuery
            from mlflow_eval.vertex_judge import start_vertex_judge
            start_vertex_judge()
            scorers.extend([
                Correctness(), RelevanceToQuery(), ExpectationsGuidelines(),
            ])
            print("\nLLM judges enabled")

        # Evaluate with pre-collected outputs — no predict_fn needed.
        print("\nScoring...")
        result = mlflow.genai.evaluate(
            data=dataset,
            scorers=scorers,
        )

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for k, v in sorted(result.metrics.items()):
            if isinstance(v, float):
                bar = "#" * int(v * 20)
                print(f"  {k:45s}  {v:.3f}  {bar}")
            else:
                print(f"  {k:45s}  {v}")
        print("=" * 60)
        print(f"\n  MLflow UI: {mlflow_url}")
        print(f"  Agent traces visible in Traces tab\n")

    finally:
        print("\nShutting down...")
        for label, proc in reversed(procs):
            proc.kill()
            proc.wait()
            print(f"  {label} stopped")
        print("Done.")


if __name__ == "__main__":
    main()
