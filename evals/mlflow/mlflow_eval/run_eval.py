#!/usr/bin/env python3
"""
MLflow evaluation pipeline for the Lightspeed Agent.

Sends questions to a deployed agent via A2A, then scores responses using
deterministic checks and LLM-as-a-judge scorers. Results are logged to MLflow.

Data privacy: LLM-as-a-judge scorers send agent responses to the judge model
for scoring. A self-hosted judge model (--judge-model) is required to prevent
evaluation data from being sent to external cloud providers.

Usage:
    # Set judge model (required — VPN-backed model for real Insights data):
    export MLFLOW_GENAI_JUDGE_DEFAULT_MODEL="openai:/Qwen/Qwen3-14B"
    export OPENAI_BASE_URL="https://<judge-endpoint>/v1"
    export OPENAI_API_KEY="<api-key>"

    # Skip TLS verification for internal clusters:
    export MLFLOW_TRACKING_INSECURE_TLS=true

    # Run evaluation:
    python -m mlflow_eval.run_eval \\
        --agent-endpoint https://<agent-endpoint> \\
        --agent-token <bearer-token> \\
        --tracking-uri https://<mlflow-endpoint> \\
        --judge-model "openai:/Qwen/Qwen3-14B"
"""

import argparse
import os
import sys

os.environ.setdefault("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "True")

import requests


def _apply_tls_patch():
    """Disable TLS certificate verification for internal OpenShift clusters.

    Connections remain encrypted — only the CA check is skipped.
    Connections remain encrypted — only the CA check is skipped.
    """
    if os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").lower() == "true":
        import requests.adapters
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        _original_send = requests.adapters.HTTPAdapter.send

        def _patched_send(self, request, **kwargs):
            kwargs["verify"] = False
            return _original_send(self, request, **kwargs)

        requests.adapters.HTTPAdapter.send = _patched_send


def main():
    parser = argparse.ArgumentParser(
        description="Run MLflow evaluation against the Lightspeed Agent"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of questions")
    parser.add_argument("--per-type", type=int, default=1,
                        help="Questions per question_type (default: 1)")
    parser.add_argument("--category", default=None,
                        help="Filter by category (e.g. vulnerability, inventory)")
    parser.add_argument("--difficulty", default=None,
                        help="Filter by difficulty (easy, medium, hard)")

    agent_group = parser.add_argument_group("agent")
    agent_group.add_argument("--agent-endpoint", default="http://localhost:8000",
                             help="Agent URL (default: http://localhost:8000)")
    agent_group.add_argument("--agent-token", default="",
                             help="Bearer token for agent authentication (required)")

    scorer_group = parser.add_argument_group("scorers")
    scorer_group.add_argument("--judge-model", default=None,
                              help="Judge model URI (e.g. openai:/Qwen/Qwen3-14B). "
                              "Falls back to MLFLOW_GENAI_JUDGE_DEFAULT_MODEL env var.")

    mlflow_group = parser.add_argument_group("mlflow")
    mlflow_group.add_argument("--experiment", default="lightspeed-eval",
                              help="MLflow experiment name (default: lightspeed-eval)")
    mlflow_group.add_argument("--tracking-uri", default="sqlite:///mlflow.db",
                              help="MLflow tracking URI")
    args = parser.parse_args()

    # ── Safeguard: require token for agent authentication ─────────────
    if not args.agent_token:
        print(
            "ERROR: --agent-token required. Provide a valid Bearer token "
            "from Red Hat SSO for agent authentication.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Safeguard: require judge model for data privacy ──────────────
    judge_model = args.judge_model or os.environ.get("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL", "")
    if not judge_model:
        print(
            "ERROR: Judge model not set. Pass --judge-model or set "
            "MLFLOW_GENAI_JUDGE_DEFAULT_MODEL. A self-hosted judge model is "
            "required to prevent evaluation data from being sent to external "
            "cloud providers.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Apply TLS patch for internal clusters ────────────────────────
    _apply_tls_patch()

    # ── Rate limiting for judge model ────────────────────────────────
    # Self-hosted judge models may have rate limits. Limit concurrent scorer
    # threads so MLflow doesn't fire all judge calls in parallel.
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")

    import mlflow
    from mlflow.genai.scorers import Correctness, ExpectationsGuidelines, RelevanceToQuery
    from mlflow_eval.a2a_client import a2a_predict_fn
    from mlflow_eval.dataset import load_cases, load_dataset
    from mlflow_eval.scorers import (
        AnswerCorrectness,
        ErrorHandlingGuidelines,
        ResponseReceived,
        SafetyGuidelines,
        ToolCallCorrectness,
    )

    # ── Configure MLflow ──────────────────────────────────────────────
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    mlflow.autolog(disable=True)
    print(f"Judge model: {judge_model}")

    # ── Load dataset ──────────────────────────────────────────────────
    # Default: load curated cases from cases/ (single source of truth)
    # Use --full-dataset to load all 257 questions from eval_dataset.json
    if args.limit or args.category or args.difficulty:
        dataset = load_dataset(
            category=args.category,
            difficulty=args.difficulty,
            limit=args.limit,
            per_type=args.per_type,
        )
    else:
        dataset = load_cases()
    print(f"Loaded {len(dataset)} questions")
    cats = sorted(set(
        d["expectations"]["category"]
        for d in dataset if d["expectations"].get("category")
    ))
    print(f"Categories: {cats}")

    # ── Check agent is reachable ──────────────────────────────────────
    endpoint = args.agent_endpoint
    print(f"\nAgent: {endpoint}")
    try:
        r = requests.get(f"{endpoint.rstrip('/')}/.well-known/agent.json", timeout=5)
        if r.status_code == 200:
            card = r.json()
            print(f"  Name: {card.get('name', 'unknown')}")
    except Exception:
        print("  WARNING: Agent not reachable. Make sure it's running.")

    # ── Collect agent responses ───────────────────────────────────────
    predict = a2a_predict_fn(
        agent_url=endpoint,
        token=args.agent_token,
    )

    print("\nCalling agent...")
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

    # ── Build scorer list ─────────────────────────────────────────────
    scorers = [
        # Deterministic (no LLM needed, run first)
        ResponseReceived(),
        AnswerCorrectness(),
        # LLM judges (require judge model)
        Correctness(model=judge_model),
        RelevanceToQuery(model=judge_model),
        ExpectationsGuidelines(model=judge_model),
        SafetyGuidelines(model=judge_model),
        ErrorHandlingGuidelines(model=judge_model),
        # Trace-based (queries MLflow for actual tool calls)
        ToolCallCorrectness(
            agent_experiment_name=args.experiment,
        ),
    ]

    print(f"Scorers: {[s.name for s in scorers]}\n")

    # ── Evaluate ──────────────────────────────────────────────────────
    print("Scoring...")
    result = mlflow.genai.evaluate(
        data=dataset,
        scorers=scorers,
    )

    # ── Print results ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    for k, v in sorted(result.metrics.items()):
        if isinstance(v, float):
            bar = "#" * int(v * 20)
            print(f"  {k:45s}  {v:.3f}  {bar}")
        else:
            print(f"  {k:45s}  {v}")
    print("=" * 65)
    print(f"\n  mlflow ui --backend-store-uri {args.tracking_uri}")
    print(f"  http://localhost:5000\n")


if __name__ == "__main__":
    main()
