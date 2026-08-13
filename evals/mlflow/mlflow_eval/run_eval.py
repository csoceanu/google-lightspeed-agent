#!/usr/bin/env python3
"""
MLflow evaluation pipeline for the Lightspeed Agent.

Sends questions from eval_dataset.json to the agent, grades responses
using LLM judge scorers (default) and optional code-based scorers,
and logs results to MLflow.

Usage:
    # Against a running agent (default: LLM judges):
    python -m mlflow_eval.run_eval --agent-endpoint http://localhost:8000

    # With code-based scorers too (needs mock data alignment):
    python -m mlflow_eval.run_eval --agent-endpoint http://localhost:8000 --enable-code-scorers

    # Full stack (starts mock API + MCP + agent locally):
    python -m mlflow_eval.run_full_stack --per-type 1

    # View results:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import os
import sys

os.environ.setdefault("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "True")

import requests


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
    agent_group.add_argument("--agent-token", default="dev-token",
                             help="Bearer token for A2A auth (default: dev-token)")

    scorer_group = parser.add_argument_group("scorers")
    scorer_group.add_argument("--enable-code-scorers", action="store_true",
                              help="Enable code-based scorers (answer_correctness, tool_match, behavior_coverage)")
    scorer_group.add_argument("--judge-model", default=None,
                              help="Override judge model (e.g. openai:/gpt-4o)")

    mlflow_group = parser.add_argument_group("mlflow")
    mlflow_group.add_argument("--experiment", default="lightspeed-eval",
                              help="MLflow experiment name (default: lightspeed-eval)")
    mlflow_group.add_argument("--tracking-uri", default="sqlite:///mlflow.db",
                              help="MLflow tracking URI")
    args = parser.parse_args()

    import mlflow
    from mlflow.genai.scorers import Correctness, ExpectationsGuidelines, RelevanceToQuery
    from mlflow_eval.a2a_client import a2a_predict_fn
    from mlflow_eval.dataset import load_cases, load_dataset
    from mlflow_eval.scorers import (
        AnswerCorrectness,
        BehaviorCoverage,
        ErrorHandlingGuidelines,
        ResponseReceived,
        SafetyGuidelines,
        ToolMatch,
    )

    # ── Configure MLflow ──────────────────────────────────────────────
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    mlflow.autolog(disable=True)

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
            print(f"  {qid}: {answer[:60]}...")
        except Exception as exc:
            d["outputs"] = f"ERROR: {exc}"
            print(f"  {qid}: ERROR - {exc}")

    # ── Build scorer list ─────────────────────────────────────────────
    # LLM judges are the default — they evaluate quality even without
    # mock data alignment. Code-based scorers are optional.
    scorers = [
        ResponseReceived(),
        Correctness(),
        RelevanceToQuery(),
        ExpectationsGuidelines(),
        SafetyGuidelines(model=args.judge_model),
        ErrorHandlingGuidelines(model=args.judge_model),
    ]

    if args.enable_code_scorers:
        scorers.extend([
            AnswerCorrectness(),
            ToolMatch(),
            BehaviorCoverage(),
        ])
        print("Code-based scorers enabled")

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
