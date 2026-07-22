"""Configuration module for the Red Hat Lightspeed Agent evaluation framework.

Provides EvalConfig, a dataclass that holds every tunable knob for an
evaluation run.  Values are resolved in this priority order (highest wins):

    CLI arguments  >  environment variables  >  config file  >  defaults

Typical usage::

    config = EvalConfig.from_all_sources(cli_args=parsed_namespace)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Filter sub-config
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    """Controls which questions from the dataset are included in the run."""

    category: Optional[str] = None
    difficulty: Optional[str] = None
    question_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

@dataclass
class FreeFormConfig:
    """Configuration for the free-form question evaluation strategy."""

    strategy: str = "none"
    """Which strategy to use: ``"llm_judge"``, ``"semantic_similarity"``,
    or ``"none"`` (raises an error if a FREE_FORM question is encountered)."""

    judge_endpoint: str = ""
    """HTTP endpoint for the judge LLM (used with ``llm_judge``)."""

    judge_model: str = ""
    """Model name / identifier for the judge LLM."""

    judge_token: str = ""
    """Auth token for the judge LLM endpoint."""

    judge_pass_threshold: float = 0.7
    """Minimum average score to pass (for ``llm_judge``)."""

    embedding_model: str = ""
    """Model name for the embedding model (used with ``semantic_similarity``)."""

    similarity_pass_threshold: float = 0.75
    """Minimum cosine similarity to pass (for ``semantic_similarity``)."""


@dataclass
class EvalConfig:
    """Central configuration for a single evaluation run.

    Every field carries a sensible default so the dataclass can be
    instantiated with zero arguments for quick local testing.
    """

    # Connection
    endpoint: str = "http://localhost:8080"
    token: str = ""

    # Execution
    concurrency: int = 4
    timeout_seconds: int = 120

    # I/O paths
    output_path: str = "eval_results.json"
    resume_path: Optional[str] = None

    # Filtering
    filters: FilterConfig = field(default_factory=FilterConfig)

    # Free-form evaluation
    free_form: FreeFormConfig = field(default_factory=FreeFormConfig)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ``ValueError`` when any field is out of acceptable range."""
        url_pattern = re.compile(
            r"^https?://"
            r"(?:[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+)$"
        )
        if not url_pattern.match(self.endpoint):
            raise ValueError(
                f"endpoint must be a valid HTTP(S) URL, got: {self.endpoint!r}"
            )
        if self.concurrency < 1:
            raise ValueError(
                f"concurrency must be >= 1, got: {self.concurrency}"
            )
        if self.timeout_seconds < 1:
            raise ValueError(
                f"timeout_seconds must be >= 1, got: {self.timeout_seconds}"
            )

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalConfig":
        """Build an ``EvalConfig`` from a flat or nested dictionary.

        Keys that do not map to a field are silently ignored so that
        config files can carry extra metadata without breaking the
        loader.
        """
        filters_raw = data.get("filters", {})
        if isinstance(filters_raw, dict):
            tags = filters_raw.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            ids = filters_raw.get("ids", [])
            if isinstance(ids, str):
                ids = [i.strip() for i in ids.split(",") if i.strip()]
            filters = FilterConfig(
                category=filters_raw.get("category"),
                difficulty=filters_raw.get("difficulty"),
                question_type=filters_raw.get("question_type"),
                tags=tags,
                ids=ids,
            )
        else:
            filters = FilterConfig()

        ff_raw = data.get("free_form", {})
        if isinstance(ff_raw, dict):
            free_form = FreeFormConfig(
                strategy=str(ff_raw.get("strategy", "none")),
                judge_endpoint=str(ff_raw.get("judge_endpoint", "")),
                judge_model=str(ff_raw.get("judge_model", "")),
                judge_token=str(ff_raw.get("judge_token", "")),
                judge_pass_threshold=float(ff_raw.get("judge_pass_threshold", 0.7)),
                embedding_model=str(ff_raw.get("embedding_model", "")),
                similarity_pass_threshold=float(ff_raw.get("similarity_pass_threshold", 0.75)),
            )
        else:
            free_form = FreeFormConfig()

        return cls(
            endpoint=str(data.get("endpoint", cls.endpoint)),
            token=str(data.get("token", cls.token)),
            concurrency=int(data.get("concurrency", cls.concurrency)),
            timeout_seconds=int(
                data.get("timeout_seconds", cls.timeout_seconds)
            ),
            output_path=str(data.get("output_path", cls.output_path)),
            resume_path=data.get("resume_path"),
            filters=filters,
            free_form=free_form,
        )

    @classmethod
    def from_config_file(cls, path: str) -> "EvalConfig":
        """Load configuration from a YAML or JSON file.

        The file extension determines the parser (``.yaml`` / ``.yml``
        for YAML, everything else is tried as JSON).

        Raises ``FileNotFoundError`` if the path does not exist.
        """
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Config file not found: {resolved}")

        text = resolved.read_text(encoding="utf-8")
        if resolved.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError(
                    "PyYAML is required to load YAML config files. "
                    "Install it with: pip install pyyaml"
                )
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)

        return cls.from_dict(data)

    @classmethod
    def from_env(cls) -> Dict[str, Any]:
        """Read recognised environment variables and return a dict.

        Only variables that are *actually set* appear in the returned
        mapping, so the caller can layer them on top of earlier
        sources without clobbering values with ``None``.
        """
        env_map: Dict[str, Any] = {}

        endpoint = os.environ.get("LIGHTSPEED_ENDPOINT")
        if endpoint is not None:
            env_map["endpoint"] = endpoint

        token = os.environ.get("LIGHTSPEED_TOKEN")
        if token is not None:
            env_map["token"] = token

        concurrency = os.environ.get("LIGHTSPEED_CONCURRENCY")
        if concurrency is not None:
            env_map["concurrency"] = int(concurrency)

        timeout = os.environ.get("LIGHTSPEED_TIMEOUT")
        if timeout is not None:
            env_map["timeout_seconds"] = int(timeout)

        output = os.environ.get("LIGHTSPEED_OUTPUT_PATH")
        if output is not None:
            env_map["output_path"] = output

        resume = os.environ.get("LIGHTSPEED_RESUME_PATH")
        if resume is not None:
            env_map["resume_path"] = resume

        ff_strategy = os.environ.get("LIGHTSPEED_FREE_FORM_STRATEGY")
        if ff_strategy is not None:
            env_map.setdefault("free_form", {})["strategy"] = ff_strategy

        judge_endpoint = os.environ.get("LIGHTSPEED_JUDGE_ENDPOINT")
        if judge_endpoint is not None:
            env_map.setdefault("free_form", {})["judge_endpoint"] = judge_endpoint

        judge_model = os.environ.get("LIGHTSPEED_JUDGE_MODEL")
        if judge_model is not None:
            env_map.setdefault("free_form", {})["judge_model"] = judge_model

        judge_token = os.environ.get("LIGHTSPEED_JUDGE_TOKEN")
        if judge_token is not None:
            env_map.setdefault("free_form", {})["judge_token"] = judge_token

        embedding_model = os.environ.get("LIGHTSPEED_EMBEDDING_MODEL")
        if embedding_model is not None:
            env_map.setdefault("free_form", {})["embedding_model"] = embedding_model

        return env_map

    @classmethod
    def from_cli_args(
        cls, args: Optional[argparse.Namespace] = None
    ) -> Dict[str, Any]:
        """Extract config-relevant values from parsed CLI arguments.

        ``None`` values (arguments that were not supplied on the command
        line) are dropped so they do not override lower-priority
        sources.
        """
        if args is None:
            return {}

        mapping: Dict[str, Any] = {}
        direct_keys = [
            "endpoint",
            "token",
            "concurrency",
            "timeout_seconds",
            "output_path",
            "resume_path",
        ]
        for key in direct_keys:
            val = getattr(args, key, None)
            if val is not None:
                mapping[key] = val

        # Filters live under a nested dict.
        filter_keys = ["category", "difficulty", "question_type", "tags", "ids"]
        filters: Dict[str, Any] = {}
        for key in filter_keys:
            val = getattr(args, key, None)
            if val is not None:
                filters[key] = val
        if filters:
            mapping["filters"] = filters

        # Free-form config lives under a nested dict.
        ff_keys = [
            "free_form_strategy", "judge_endpoint", "judge_model",
            "judge_token", "judge_pass_threshold",
            "embedding_model", "similarity_pass_threshold",
        ]
        ff: Dict[str, Any] = {}
        for key in ff_keys:
            val = getattr(args, key, None)
            if val is not None:
                mapped_key = key.replace("free_form_", "")
                ff[mapped_key] = val
        if ff:
            mapping["free_form"] = ff

        return mapping

    @classmethod
    def from_all_sources(
        cls,
        config_file: Optional[str] = None,
        cli_args: Optional[argparse.Namespace] = None,
    ) -> "EvalConfig":
        """Merge every configuration source in priority order.

        Resolution order (highest priority wins)::

            CLI args  >  env vars  >  config file  >  dataclass defaults

        After merging, the resulting config is validated.
        """
        # 1. Start from defaults.
        merged: Dict[str, Any] = {}

        # 2. Layer in config file (if provided or auto-detected).
        if config_file is None:
            for candidate in ("eval_config.yaml", "eval_config.json"):
                if Path(candidate).is_file():
                    config_file = candidate
                    break

        if config_file is not None:
            file_cfg = cls.from_config_file(config_file)
            merged.update(
                {
                    "endpoint": file_cfg.endpoint,
                    "token": file_cfg.token,
                    "concurrency": file_cfg.concurrency,
                    "timeout_seconds": file_cfg.timeout_seconds,
                    "output_path": file_cfg.output_path,
                    "resume_path": file_cfg.resume_path,
                    "filters": {
                        "category": file_cfg.filters.category,
                        "difficulty": file_cfg.filters.difficulty,
                        "question_type": file_cfg.filters.question_type,
                        "tags": file_cfg.filters.tags,
                        "ids": file_cfg.filters.ids,
                    },
                    "free_form": {
                        "strategy": file_cfg.free_form.strategy,
                        "judge_endpoint": file_cfg.free_form.judge_endpoint,
                        "judge_model": file_cfg.free_form.judge_model,
                        "judge_token": file_cfg.free_form.judge_token,
                        "judge_pass_threshold": file_cfg.free_form.judge_pass_threshold,
                        "embedding_model": file_cfg.free_form.embedding_model,
                        "similarity_pass_threshold": file_cfg.free_form.similarity_pass_threshold,
                    },
                }
            )

        # 3. Layer in environment variables.
        env_vals = cls.from_env()
        merged.update(env_vals)

        # 4. Layer in CLI arguments (highest priority).
        cli_vals = cls.from_cli_args(cli_args)
        # Filters need a sub-merge so CLI filter flags do not wipe out
        # file-level filter values.
        if "filters" in cli_vals:
            existing_filters = merged.get("filters", {})
            if isinstance(existing_filters, dict):
                existing_filters.update(cli_vals.pop("filters"))
                merged["filters"] = existing_filters
            else:
                merged["filters"] = cli_vals.pop("filters")
        if "free_form" in cli_vals:
            existing_ff = merged.get("free_form", {})
            if isinstance(existing_ff, dict):
                existing_ff.update(cli_vals.pop("free_form"))
                merged["free_form"] = existing_ff
            else:
                merged["free_form"] = cli_vals.pop("free_form")
        merged.update(cli_vals)

        config = cls.from_dict(merged)
        config.validate()
        return config

    # ------------------------------------------------------------------
    # CLI argument parser (convenience)
    # ------------------------------------------------------------------
    @staticmethod
    def build_arg_parser(
        parser: Optional[argparse.ArgumentParser] = None,
    ) -> argparse.ArgumentParser:
        """Add eval-config flags to an ``argparse.ArgumentParser``.

        If *parser* is ``None`` a fresh one is created and returned.
        """
        if parser is None:
            parser = argparse.ArgumentParser(
                description="Red Hat Lightspeed Agent Evaluation Framework"
            )

        conn = parser.add_argument_group("connection")
        conn.add_argument(
            "--endpoint",
            help="Lightspeed Agent HTTP(S) endpoint URL",
        )
        conn.add_argument(
            "--token",
            help="Authentication token for the endpoint",
        )

        exe = parser.add_argument_group("execution")
        exe.add_argument(
            "--concurrency",
            type=int,
            help="Number of parallel evaluation workers (default: 4)",
        )
        exe.add_argument(
            "--timeout-seconds",
            dest="timeout_seconds",
            type=int,
            help="Per-question timeout in seconds (default: 120)",
        )

        io_grp = parser.add_argument_group("input / output")
        io_grp.add_argument(
            "--output-path",
            dest="output_path",
            help="Path for the JSON results file",
        )
        io_grp.add_argument(
            "--resume-path",
            dest="resume_path",
            help="Path to a previous results file to resume from",
        )
        io_grp.add_argument(
            "--config-file",
            dest="config_file",
            help="Path to eval_config.yaml or eval_config.json",
        )

        flt = parser.add_argument_group("filters")
        flt.add_argument("--category", help="Only run this category")
        flt.add_argument("--difficulty", help="Only run this difficulty level")
        flt.add_argument(
            "--question-type",
            dest="question_type",
            help="Only run this question type",
        )
        flt.add_argument(
            "--tags",
            help="Comma-separated list of tags to include",
        )
        flt.add_argument(
            "--ids",
            help="Comma-separated list of question IDs to run",
        )

        ff = parser.add_argument_group("free-form evaluation")
        ff.add_argument(
            "--free-form-strategy",
            dest="free_form_strategy",
            choices=["none", "llm_judge", "semantic_similarity"],
            help="Strategy for evaluating free-form questions",
        )
        ff.add_argument(
            "--judge-endpoint",
            dest="judge_endpoint",
            help="HTTP endpoint for the judge LLM (llm_judge strategy)",
        )
        ff.add_argument(
            "--judge-model",
            dest="judge_model",
            help="Model name for the judge LLM",
        )
        ff.add_argument(
            "--judge-token",
            dest="judge_token",
            help="Auth token for the judge LLM endpoint",
        )
        ff.add_argument(
            "--judge-pass-threshold",
            dest="judge_pass_threshold",
            type=float,
            help="Minimum avg score to pass for llm_judge (default: 0.7)",
        )
        ff.add_argument(
            "--embedding-model",
            dest="embedding_model",
            help="Model name for embeddings (semantic_similarity strategy)",
        )
        ff.add_argument(
            "--similarity-pass-threshold",
            dest="similarity_pass_threshold",
            type=float,
            help="Minimum cosine similarity to pass (default: 0.75)",
        )

        return parser
