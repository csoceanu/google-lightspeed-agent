#!/usr/bin/env python3
"""
Evaluation runner for the Red Hat Lightspeed Agent.

Loads questions from eval_dataset.json, sends each to the agent endpoint,
collects responses, passes them to the grader, and generates results.

Usage:
    python eval_runner.py --endpoint URL --token TOKEN \
        [--category vulnerability] [--difficulty easy] [--type binary] \
        [--tags cve,kernel] [--ids V-001,V-002] \
        [--concurrency 5] [--timeout 30] \
        [--dry-run] [--output results.json] [--resume results.json]
"""

import json
import asyncio
import argparse
import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

import aiohttp

from eval_utils import generate_run_id
from eval_grader import (
    Grader,
    GradeResult,
    LLMJudgeStrategy,
    SemanticSimilarityStrategy,
)
from eval_reporter import EvalReporter, QuestionResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("eval_runner")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Question:
    """A single evaluation question loaded from the dataset."""

    id: str
    question: str
    expected_answer: Any
    category: str = ""
    difficulty: str = ""
    question_type: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Question":
        return cls(
            id=data["id"],
            question=data["question"],
            expected_answer=data.get("expected_answer"),
            category=data.get("category", ""),
            difficulty=data.get("difficulty", ""),
            question_type=data.get("question_type", ""),
            tags=data.get("tags", []),
            metadata={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "id",
                    "question",
                    "expected_answer",
                    "category",
                    "difficulty",
                    "question_type",
                    "tags",
                }
            },
        )


@dataclass
class EvalResult:
    """Result of evaluating a single question."""

    question_id: str
    question_text: str
    expected_answer: Any
    category: str = ""
    difficulty: str = ""
    question_type: str = ""
    scenario_type: str = ""
    agent_response: Optional[str] = None
    grade: Optional[str] = None
    score: Optional[float] = None
    feedback: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Agent client
# ---------------------------------------------------------------------------

class AgentClient:
    """HTTP client that sends questions to the Lightspeed Agent endpoint."""

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def ask(
        self, question: str, session: aiohttp.ClientSession
    ) -> str:
        """Send *question* to the agent and return the text response.

        Retries on transient HTTP errors (429, 500, 502, 503, 504) and
        connection-level failures, with exponential back-off.
        """
        url = f"{self.base_url}/chat"
        payload = {"message": question}
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                ) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        # Accept common response shapes.
                        if isinstance(body, dict):
                            return str(
                                body.get("response")
                                or body.get("answer")
                                or body.get("message")
                                or body.get("text")
                                or json.dumps(body)
                            )
                        return str(body)

                    # Retry on transient server / rate-limit errors.
                    if resp.status in {429, 500, 502, 503, 504}:
                        text = await resp.text()
                        last_exc = RuntimeError(
                            f"HTTP {resp.status}: {text[:300]}"
                        )
                        logger.warning(
                            "Attempt %d/%d for question failed with HTTP %d, "
                            "retrying in %.1fs ...",
                            attempt,
                            self.max_retries,
                            resp.status,
                            self.retry_backoff * attempt,
                        )
                        await asyncio.sleep(self.retry_backoff * attempt)
                        continue

                    # Non-retryable HTTP error.
                    text = await resp.text()
                    raise RuntimeError(
                        f"Agent returned HTTP {resp.status}: {text[:500]}"
                    )

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "Attempt %d/%d connection error: %s, retrying ...",
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    await asyncio.sleep(self.retry_backoff * attempt)
                    continue

        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

class EvalRunner:
    """Orchestrates loading, filtering, executing, grading, and reporting."""

    def __init__(
        self,
        dataset_path: Path,
        client: AgentClient,
        grader: Grader,
        concurrency: int = 5,
        output_path: Optional[Path] = None,
        resume_path: Optional[Path] = None,
        dry_run: bool = False,
    ) -> None:
        self.dataset_path = dataset_path
        self.client = client
        self.grader = grader
        self.concurrency = concurrency
        self.output_path = output_path
        self.resume_path = resume_path
        self.dry_run = dry_run

        self.questions: List[Question] = []
        self.results: List[EvalResult] = []
        self._completed_ids: Set[str] = set()

    # -- dataset loading & validation --------------------------------------

    def load_dataset(self) -> None:
        """Load and validate the dataset from *self.dataset_path*."""
        logger.info("Loading dataset from %s", self.dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        with open(self.dataset_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        if isinstance(raw, dict):
            items = raw.get("questions", raw.get("items", raw.get("data", [])))
        elif isinstance(raw, list):
            items = raw
        else:
            raise ValueError("Dataset must be a JSON array or object with a questions key")

        if not items:
            raise ValueError("Dataset contains no questions")

        seen_ids: Set[str] = set()
        for idx, entry in enumerate(items):
            if "id" not in entry:
                raise ValueError(f"Question at index {idx} is missing an 'id' field")
            if "question" not in entry:
                raise ValueError(
                    f"Question '{entry['id']}' is missing a 'question' field"
                )
            if entry["id"] in seen_ids:
                raise ValueError(f"Duplicate question id: {entry['id']}")
            seen_ids.add(entry["id"])
            self.questions.append(Question.from_dict(entry))

        logger.info("Loaded %d questions", len(self.questions))

    # -- filtering ---------------------------------------------------------

    def apply_filters(
        self,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        question_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Narrow *self.questions* to those matching the supplied filters."""
        before = len(self.questions)

        if ids:
            id_set = set(ids)
            self.questions = [q for q in self.questions if q.id in id_set]
            missing = id_set - {q.id for q in self.questions}
            if missing:
                logger.warning("Requested IDs not found in dataset: %s", missing)

        if category:
            self.questions = [
                q
                for q in self.questions
                if q.category.lower() == category.lower()
            ]

        if difficulty:
            self.questions = [
                q
                for q in self.questions
                if q.difficulty.lower() == difficulty.lower()
            ]

        if question_type:
            self.questions = [
                q
                for q in self.questions
                if q.question_type.lower() == question_type.lower()
            ]

        if tags:
            tag_set = {t.lower() for t in tags}
            self.questions = [
                q
                for q in self.questions
                if tag_set & {t.lower() for t in q.tags}
            ]

        logger.info(
            "Filters applied: %d -> %d questions", before, len(self.questions)
        )

    # -- resume ------------------------------------------------------------

    def load_resume(self) -> None:
        """Load previously completed results so we can skip them."""
        if not self.resume_path or not self.resume_path.exists():
            return
        logger.info("Resuming from %s", self.resume_path)
        with open(self.resume_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        prev_results = data if isinstance(data, list) else data.get("results", [])
        for r in prev_results:
            qid = r.get("question_id", "")
            if qid and r.get("error") is None:
                self._completed_ids.add(qid)
                self.results.append(
                    EvalResult(
                        question_id=r["question_id"],
                        question_text=r.get("question_text", ""),
                        expected_answer=r.get("expected_answer"),
                        category=r.get("category", ""),
                        difficulty=r.get("difficulty", ""),
                        question_type=r.get("question_type", ""),
                        scenario_type=r.get("scenario_type", ""),
                        agent_response=r.get("agent_response"),
                        grade=r.get("grade"),
                        score=r.get("score"),
                        feedback=r.get("feedback"),
                        error=r.get("error"),
                        duration_seconds=r.get("duration_seconds", 0.0),
                        timestamp=r.get("timestamp", ""),
                    )
                )
        logger.info(
            "Resumed %d previously completed results", len(self._completed_ids)
        )

    # -- single question evaluation ----------------------------------------

    async def _evaluate_one(
        self,
        question: Question,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
    ) -> EvalResult:
        """Evaluate a single question: call the agent, then the grader."""
        result = EvalResult(
            question_id=question.id,
            question_text=question.question,
            expected_answer=question.expected_answer,
            category=question.category,
            difficulty=question.difficulty,
            question_type=question.question_type,
            scenario_type=question.metadata.get("scenario_type", ""),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        t0 = time.monotonic()

        async with semaphore:
            try:
                logger.info("Evaluating %s ...", question.id)
                response = await self.client.ask(question.question, session)
                result.agent_response = response
                result.duration_seconds = round(time.monotonic() - t0, 3)

                # Grade the response.
                question_dict = {
                    "question": question.question,
                    "question_type": question.question_type,
                    "expected_answer": question.expected_answer,
                    **question.metadata,
                }
                grade_result: GradeResult = self.grader.grade(
                    question_dict, response,
                )
                result.grade = "pass" if grade_result.passed else "fail"
                result.score = grade_result.score
                result.feedback = grade_result.feedback

            except Exception as exc:
                result.duration_seconds = round(time.monotonic() - t0, 3)
                result.error = f"{type(exc).__name__}: {exc}"
                logger.error("Error evaluating %s: %s", question.id, result.error)

        return result

    # -- save results incrementally ----------------------------------------

    def _save_results(self) -> None:
        """Write current results to the output file."""
        if not self.output_path:
            return

        summary = self._build_summary()
        payload = {
            "summary": summary,
            "results": [r.to_dict() for r in self.results],
        }
        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

    # -- summary -----------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        """Build an aggregate summary of all results collected so far."""
        total = len(self.results)
        errors = sum(1 for r in self.results if r.error)
        graded = [r for r in self.results if r.score is not None]
        avg_score = (
            round(sum(r.score for r in graded) / len(graded), 4)
            if graded
            else None
        )
        durations = [r.duration_seconds for r in self.results if r.duration_seconds > 0]
        avg_duration = (
            round(sum(durations) / len(durations), 3) if durations else None
        )

        # Per-category breakdown.
        by_category: Dict[str, List[float]] = {}
        for r in graded:
            # Look up category from the original question list.
            cat = ""
            for q in self.questions:
                if q.id == r.question_id:
                    cat = q.category
                    break
            by_category.setdefault(cat or "unknown", []).append(r.score)  # type: ignore[arg-type]

        category_scores = {
            cat: round(sum(scores) / len(scores), 4)
            for cat, scores in by_category.items()
        }

        return {
            "total_questions": total,
            "errors": errors,
            "graded": len(graded),
            "average_score": avg_score,
            "average_duration_seconds": avg_duration,
            "scores_by_category": category_scores,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # -- dry-run -----------------------------------------------------------

    def dry_run_report(self) -> Dict[str, Any]:
        """Validate dataset and print what *would* be evaluated."""
        categories = sorted({q.category for q in self.questions if q.category})
        difficulties = sorted({q.difficulty for q in self.questions if q.difficulty})
        types = sorted({q.question_type for q in self.questions if q.question_type})
        all_tags = sorted(
            {t for q in self.questions for t in q.tags}
        )
        report = {
            "mode": "dry-run",
            "total_questions": len(self.questions),
            "categories": categories,
            "difficulties": difficulties,
            "question_types": types,
            "tags": all_tags,
            "question_ids": [q.id for q in self.questions],
            "skipped_resume": len(self._completed_ids),
        }
        return report

    # -- main entry point --------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        """Execute the full evaluation pipeline and return the summary."""
        self.load_dataset()
        # Filters and resume are expected to be applied before calling run()
        # via the caller (see main()), but load_resume is idempotent.

        if self.dry_run:
            report = self.dry_run_report()
            logger.info("Dry-run report:\n%s", json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
            return report

        # Determine which questions still need evaluation.
        pending = [
            q for q in self.questions if q.id not in self._completed_ids
        ]
        logger.info(
            "Running evaluation: %d pending, %d already completed",
            len(pending),
            len(self._completed_ids),
        )

        if not pending:
            logger.info("Nothing to evaluate -- all questions already completed.")
            summary = self._build_summary()
            self._print_console_report(summary)
            self._save_results()
            return summary

        semaphore = asyncio.Semaphore(self.concurrency)

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._evaluate_one(q, session, semaphore) for q in pending
            ]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                self.results.append(result)
                # Incremental save after each result.
                self._save_results()
                self._log_progress(result)

        summary = self._build_summary()
        self._print_console_report(summary)
        self._save_results()
        return summary

    # -- console output ----------------------------------------------------

    def _log_progress(self, result: EvalResult) -> None:
        completed = len(self.results)
        total = len(self.questions)
        status = "OK" if result.error is None else "ERROR"
        grade_info = (
            f"grade={result.grade} score={result.score}"
            if result.grade is not None
            else "not graded"
        )
        logger.info(
            "[%d/%d] %s %s (%s) %.1fs",
            completed,
            total,
            result.question_id,
            status,
            grade_info,
            result.duration_seconds,
        )

    @staticmethod
    def _print_console_report(summary: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Total questions : {summary['total_questions']}")
        print(f"  Graded          : {summary['graded']}")
        print(f"  Errors          : {summary['errors']}")
        print(f"  Average score   : {summary['average_score']}")
        print(f"  Avg duration    : {summary['average_duration_seconds']}s")
        if summary.get("scores_by_category"):
            print("  Scores by category:")
            for cat, score in sorted(summary["scores_by_category"].items()):
                print(f"    {cat:20s} : {score}")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluations against the Red Hat Lightspeed Agent."
    )

    # Required (unless --dry-run).
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Base URL of the agent endpoint (e.g. http://localhost:8080).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Bearer token for agent authentication.",
    )

    # Filters.
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter questions by category.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="Filter questions by difficulty (e.g. easy, medium, hard).",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        dest="question_type",
        help="Filter questions by question_type (e.g. binary, descriptive).",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated list of tags to filter by.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated list of question IDs to evaluate.",
    )

    # Execution.
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of parallel agent requests (default: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds per agent request (default: 30).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate the dataset and show what would run, without calling the agent.",
    )

    # Output.
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        dest="output_dir",
        help="Base directory for results. Each run creates a timestamped "
             "subdirectory containing results.json and reports (default: results).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a previous results JSON to resume from.",
    )

    # Logging.
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    # Dataset path.
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the evaluation dataset JSON (default: eval_dataset.json next to this script).",
    )

    # Free-form evaluation strategy.
    ff = parser.add_argument_group("free-form evaluation")
    ff.add_argument(
        "--free-form-strategy",
        type=str,
        default=None,
        choices=["llm_judge", "semantic_similarity", "gemini"],
        dest="free_form_strategy",
        help="Strategy for evaluating free_form questions.",
    )
    ff.add_argument(
        "--judge-endpoint",
        type=str,
        default=None,
        dest="judge_endpoint",
        help="HTTP endpoint for the judge LLM (used with llm_judge).",
    )
    ff.add_argument(
        "--judge-model",
        type=str,
        default=None,
        dest="judge_model",
        help="Model name for the judge LLM.",
    )
    ff.add_argument(
        "--judge-token",
        type=str,
        default=None,
        dest="judge_token",
        help="Auth token for the judge LLM endpoint.",
    )
    ff.add_argument(
        "--judge-pass-threshold",
        type=float,
        default=0.7,
        dest="judge_pass_threshold",
        help="Minimum avg score to pass for llm_judge (default: 0.7).",
    )
    ff.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        dest="embedding_model",
        help="Model name for embeddings (semantic_similarity strategy).",
    )
    ff.add_argument(
        "--similarity-pass-threshold",
        type=float,
        default=0.75,
        dest="similarity_pass_threshold",
        help="Minimum cosine similarity to pass (default: 0.75).",
    )

    return parser.parse_args(argv)


def _generate_reports(runner: EvalRunner, args: argparse.Namespace) -> None:
    """Convert runner results to QuestionResult objects and generate reports."""
    question_results = []
    for r in runner.results:
        question_results.append(
            QuestionResult(
                question_id=r.question_id,
                question_text=r.question_text,
                category=r.category or "unknown",
                question_type=r.question_type or "unknown",
                difficulty=r.difficulty or "unknown",
                scenario_type=r.scenario_type or "unknown",
                passed=r.grade == "pass",
                expected_answer=r.expected_answer,
                actual_answer=r.agent_response,
                grade_details=r.feedback or "",
                latency_ms=r.duration_seconds * 1000.0,
            )
        )

    reporter = EvalReporter(
        results=question_results,
        config={
            "endpoint": args.endpoint or "",
            "filters": {
                "category": args.category,
                "difficulty": args.difficulty,
                "question_type": args.question_type,
            },
        },
    )
    paths = reporter.generate_all(output_dir=args.report_dir, console=True)
    for fmt, path in paths.items():
        logger.info("Report written: %s -> %s", fmt, path)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Configure logging.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve dataset path.
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        dataset_path = Path(__file__).resolve().parent / "eval_dataset.json"

    # Validate endpoint requirement outside dry-run.
    if not args.dry_run and not args.endpoint:
        raise SystemExit(
            "ERROR: --endpoint is required unless --dry-run is specified."
        )

    # Build collaborators.
    client = AgentClient(
        base_url=args.endpoint or "",
        token=args.token,
        timeout=args.timeout,
    )

    # Build free-form strategy if requested.
    free_form_strategy = None
    if args.free_form_strategy == "llm_judge":
        import requests as _requests
        judge_ep = args.judge_endpoint
        judge_tok = args.judge_token
        judge_model = args.judge_model or "default"
        if not judge_ep:
            raise SystemExit(
                "ERROR: --judge-endpoint is required with --free-form-strategy=llm_judge"
            )

        def _llm_judge_client(prompt: str) -> str:
            headers = {"Content-Type": "application/json"}
            if judge_tok:
                headers["Authorization"] = f"Bearer {judge_tok}"
            resp = _requests.post(
                judge_ep,
                json={"model": judge_model, "messages": [{"role": "user", "content": prompt}]},
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "choices" in data:
                return data["choices"][0]["message"]["content"]
            return str(data.get("response", data.get("message", data)))

        free_form_strategy = LLMJudgeStrategy(
            llm_client=_llm_judge_client,
            model_name=judge_model,
            pass_threshold=args.judge_pass_threshold,
        )

    elif args.free_form_strategy == "semantic_similarity":
        emb_model = args.embedding_model or "all-MiniLM-L6-v2"
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer(emb_model)

            def _embed_fn(text: str) -> list:
                return _st_model.encode(text).tolist()

        except ImportError:
            raise SystemExit(
                "ERROR: semantic_similarity strategy requires the "
                "'sentence-transformers' package. Install with:\n"
                "  pip install sentence-transformers"
            )

        free_form_strategy = SemanticSimilarityStrategy(
            embed_fn=_embed_fn,
            model_name=emb_model,
            pass_threshold=args.similarity_pass_threshold,
        )

    elif args.free_form_strategy == "gemini":
        import requests as _requests

        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if not gemini_key:
            raise SystemExit(
                "ERROR: GEMINI_API_KEY environment variable is required "
                "with --free-form-strategy=gemini"
            )

        gemini_judge_model = args.judge_model or "gemini-2.5-flash"

        def _gemini_llm_client(prompt: str) -> str:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{gemini_judge_model}:generateContent?key={gemini_key}"
            )
            resp = _requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

        free_form_strategy = LLMJudgeStrategy(
            llm_client=_gemini_llm_client,
            model_name=gemini_judge_model,
            pass_threshold=args.judge_pass_threshold,
        )

    grader = Grader(free_form_strategy=free_form_strategy)

    run_id = generate_run_id()
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run ID: %s", run_id)
    logger.info("Output directory: %s", run_dir)

    output_path = run_dir / "results.json"
    resume_path = Path(args.resume) if args.resume else None

    runner = EvalRunner(
        dataset_path=dataset_path,
        client=client,
        grader=grader,
        concurrency=args.concurrency,
        output_path=output_path,
        resume_path=resume_path,
        dry_run=args.dry_run,
    )

    # Pre-run steps that need the dataset loaded first.
    runner.load_dataset()

    # Apply filters.
    tag_list = [t.strip() for t in args.tags.split(",")] if args.tags else None
    id_list = [i.strip() for i in args.ids.split(",")] if args.ids else None
    runner.apply_filters(
        category=args.category,
        difficulty=args.difficulty,
        question_type=args.question_type,
        tags=tag_list,
        ids=id_list,
    )

    # Resume.
    runner.load_resume()

    # Run -- reuse the already-loaded dataset (run() will skip re-loading
    # because questions are already populated).  We override load_dataset to
    # be a no-op for this invocation to avoid double-loading.
    runner.load_dataset = lambda: None  # type: ignore[assignment]

    asyncio.run(runner.run())

    # Generate reports.
    if not args.dry_run and runner.results:
        args.report_dir = str(run_dir)
        _generate_reports(runner, args)


if __name__ == "__main__":  # pragma: no cover
    main()
