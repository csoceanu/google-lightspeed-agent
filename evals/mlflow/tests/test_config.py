"""Comprehensive tests for eval_config.py -- targeting 100% coverage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

from eval_config import EvalConfig, FilterConfig, FreeFormConfig


# ===================================================================
# FilterConfig defaults
# ===================================================================


class TestFilterConfigDefaults:
    def test_default_values(self):
        fc = FilterConfig()
        assert fc.category is None
        assert fc.difficulty is None
        assert fc.question_type is None
        assert fc.tags == []
        assert fc.ids == []

    def test_custom_values(self):
        fc = FilterConfig(
            category="networking",
            difficulty="hard",
            question_type="mcq",
            tags=["a", "b"],
            ids=["q1", "q2"],
        )
        assert fc.category == "networking"
        assert fc.difficulty == "hard"
        assert fc.question_type == "mcq"
        assert fc.tags == ["a", "b"]
        assert fc.ids == ["q1", "q2"]


# ===================================================================
# FreeFormConfig defaults
# ===================================================================


class TestFreeFormConfigDefaults:
    def test_default_values(self):
        ff = FreeFormConfig()
        assert ff.strategy == "none"
        assert ff.judge_endpoint == ""
        assert ff.judge_model == ""
        assert ff.judge_token == ""
        assert ff.judge_pass_threshold == 0.7
        assert ff.embedding_model == ""
        assert ff.similarity_pass_threshold == 0.75

    def test_custom_values(self):
        ff = FreeFormConfig(
            strategy="llm_judge",
            judge_endpoint="http://judge:8080",
            judge_model="gpt-4",
            judge_token="tok",
            judge_pass_threshold=0.9,
            embedding_model="emb-model",
            similarity_pass_threshold=0.85,
        )
        assert ff.strategy == "llm_judge"
        assert ff.judge_endpoint == "http://judge:8080"
        assert ff.judge_model == "gpt-4"
        assert ff.judge_token == "tok"
        assert ff.judge_pass_threshold == 0.9
        assert ff.embedding_model == "emb-model"
        assert ff.similarity_pass_threshold == 0.85


# ===================================================================
# EvalConfig defaults
# ===================================================================


class TestEvalConfigDefaults:
    def test_default_values(self):
        cfg = EvalConfig()
        assert cfg.endpoint == "http://localhost:8080"
        assert cfg.token == ""
        assert cfg.concurrency == 4
        assert cfg.timeout_seconds == 120
        assert cfg.output_path == "eval_results.json"
        assert cfg.resume_path is None
        assert isinstance(cfg.filters, FilterConfig)
        assert isinstance(cfg.free_form, FreeFormConfig)


# ===================================================================
# EvalConfig.validate()
# ===================================================================


class TestValidate:
    def test_valid_http_url(self):
        cfg = EvalConfig(endpoint="http://localhost:8080")
        cfg.validate()  # should not raise

    def test_valid_https_url(self):
        cfg = EvalConfig(endpoint="https://api.example.com/v1")
        cfg.validate()  # should not raise

    def test_invalid_url_no_scheme(self):
        cfg = EvalConfig(endpoint="not-a-url")
        with pytest.raises(ValueError, match="endpoint must be a valid HTTP"):
            cfg.validate()

    def test_invalid_url_empty(self):
        cfg = EvalConfig(endpoint="")
        with pytest.raises(ValueError, match="endpoint must be a valid HTTP"):
            cfg.validate()

    def test_concurrency_zero(self):
        cfg = EvalConfig(concurrency=0)
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            cfg.validate()

    def test_concurrency_negative(self):
        cfg = EvalConfig(concurrency=-1)
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            cfg.validate()

    def test_timeout_zero(self):
        cfg = EvalConfig(timeout_seconds=0)
        with pytest.raises(ValueError, match="timeout_seconds must be >= 1"):
            cfg.validate()

    def test_timeout_negative(self):
        cfg = EvalConfig(timeout_seconds=-5)
        with pytest.raises(ValueError, match="timeout_seconds must be >= 1"):
            cfg.validate()


# ===================================================================
# EvalConfig.from_dict()
# ===================================================================


class TestFromDict:
    def test_empty_dict_gives_defaults(self):
        cfg = EvalConfig.from_dict({})
        assert cfg.endpoint == "http://localhost:8080"
        assert cfg.token == ""
        assert cfg.concurrency == 4
        assert cfg.timeout_seconds == 120
        assert cfg.output_path == "eval_results.json"
        assert cfg.resume_path is None
        assert cfg.filters.tags == []
        assert cfg.free_form.strategy == "none"

    def test_top_level_scalars(self):
        cfg = EvalConfig.from_dict(
            {
                "endpoint": "https://my.server",
                "token": "secret",
                "concurrency": 8,
                "timeout_seconds": 300,
                "output_path": "out.json",
                "resume_path": "prev.json",
            }
        )
        assert cfg.endpoint == "https://my.server"
        assert cfg.token == "secret"
        assert cfg.concurrency == 8
        assert cfg.timeout_seconds == 300
        assert cfg.output_path == "out.json"
        assert cfg.resume_path == "prev.json"

    def test_nested_filters_dict(self):
        cfg = EvalConfig.from_dict(
            {
                "filters": {
                    "category": "security",
                    "difficulty": "easy",
                    "question_type": "mcq",
                    "tags": ["t1", "t2"],
                    "ids": ["id1"],
                }
            }
        )
        assert cfg.filters.category == "security"
        assert cfg.filters.difficulty == "easy"
        assert cfg.filters.question_type == "mcq"
        assert cfg.filters.tags == ["t1", "t2"]
        assert cfg.filters.ids == ["id1"]

    def test_filters_tags_as_csv_string(self):
        cfg = EvalConfig.from_dict({"filters": {"tags": "a, b , c"}})
        assert cfg.filters.tags == ["a", "b", "c"]

    def test_filters_tags_as_string_with_empty_parts(self):
        cfg = EvalConfig.from_dict({"filters": {"tags": "a,,, b, "}})
        assert cfg.filters.tags == ["a", "b"]

    def test_filters_ids_as_csv_string(self):
        cfg = EvalConfig.from_dict({"filters": {"ids": "x1 , x2"}})
        assert cfg.filters.ids == ["x1", "x2"]

    def test_filters_ids_as_string_with_empty_parts(self):
        cfg = EvalConfig.from_dict({"filters": {"ids": ",, id1 , "}})
        assert cfg.filters.ids == ["id1"]

    def test_filters_non_dict_fallback(self):
        cfg = EvalConfig.from_dict({"filters": "not-a-dict"})
        assert cfg.filters == FilterConfig()

    def test_filters_non_dict_none_fallback(self):
        cfg = EvalConfig.from_dict({"filters": None})
        assert cfg.filters == FilterConfig()

    def test_free_form_dict(self):
        cfg = EvalConfig.from_dict(
            {
                "free_form": {
                    "strategy": "llm_judge",
                    "judge_endpoint": "http://j:9090",
                    "judge_model": "m1",
                    "judge_token": "jt",
                    "judge_pass_threshold": 0.8,
                    "embedding_model": "emb",
                    "similarity_pass_threshold": 0.9,
                }
            }
        )
        assert cfg.free_form.strategy == "llm_judge"
        assert cfg.free_form.judge_endpoint == "http://j:9090"
        assert cfg.free_form.judge_model == "m1"
        assert cfg.free_form.judge_token == "jt"
        assert cfg.free_form.judge_pass_threshold == 0.8
        assert cfg.free_form.embedding_model == "emb"
        assert cfg.free_form.similarity_pass_threshold == 0.9

    def test_free_form_non_dict_fallback(self):
        cfg = EvalConfig.from_dict({"free_form": 42})
        assert cfg.free_form == FreeFormConfig()

    def test_free_form_non_dict_none_fallback(self):
        cfg = EvalConfig.from_dict({"free_form": None})
        assert cfg.free_form == FreeFormConfig()

    def test_extra_keys_ignored(self):
        cfg = EvalConfig.from_dict({"unknown_key": "whatever", "endpoint": "http://x"})
        assert cfg.endpoint == "http://x"


# ===================================================================
# EvalConfig.from_config_file()
# ===================================================================


class TestFromConfigFile:
    def test_json_file(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text(
            json.dumps(
                {
                    "endpoint": "https://prod.example.com",
                    "concurrency": 16,
                    "filters": {"category": "ops"},
                }
            )
        )
        cfg = EvalConfig.from_config_file(str(p))
        assert cfg.endpoint == "https://prod.example.com"
        assert cfg.concurrency == 16
        assert cfg.filters.category == "ops"

    def test_yaml_file_with_mock(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("endpoint: https://yaml.example.com\nconcurrency: 2\n")

        fake_yaml = mock.MagicMock()
        fake_yaml.safe_load.return_value = {
            "endpoint": "https://yaml.example.com",
            "concurrency": 2,
        }

        with mock.patch("eval_config.yaml", fake_yaml):
            cfg = EvalConfig.from_config_file(str(p))
        assert cfg.endpoint == "https://yaml.example.com"
        assert cfg.concurrency == 2
        fake_yaml.safe_load.assert_called_once()

    def test_yml_extension(self, tmp_path):
        p = tmp_path / "cfg.yml"
        p.write_text("endpoint: https://yml.example.com\n")

        fake_yaml = mock.MagicMock()
        fake_yaml.safe_load.return_value = {
            "endpoint": "https://yml.example.com",
        }

        with mock.patch("eval_config.yaml", fake_yaml):
            cfg = EvalConfig.from_config_file(str(p))
        assert cfg.endpoint == "https://yml.example.com"

    def test_yaml_safe_load_returns_none(self, tmp_path):
        """An empty YAML file produces None from safe_load; should not crash."""
        p = tmp_path / "empty.yaml"
        p.write_text("")

        fake_yaml = mock.MagicMock()
        fake_yaml.safe_load.return_value = None

        with mock.patch("eval_config.yaml", fake_yaml):
            cfg = EvalConfig.from_config_file(str(p))
        # Falls back to defaults because the dict is empty.
        assert cfg.endpoint == "http://localhost:8080"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            EvalConfig.from_config_file(str(tmp_path / "nonexistent.json"))

    def test_yaml_unavailable_raises_import_error(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("endpoint: http://x\n")

        with mock.patch("eval_config.yaml", None):
            with pytest.raises(ImportError, match="PyYAML is required"):
                EvalConfig.from_config_file(str(p))

    def test_yaml_unavailable_yml_extension(self, tmp_path):
        p = tmp_path / "cfg.yml"
        p.write_text("endpoint: http://x\n")

        with mock.patch("eval_config.yaml", None):
            with pytest.raises(ImportError, match="PyYAML is required"):
                EvalConfig.from_config_file(str(p))


# ===================================================================
# EvalConfig.from_env()
# ===================================================================


class TestFromEnv:
    def test_no_env_vars_set(self, monkeypatch):
        # Clear all relevant vars.
        for var in (
            "LIGHTSPEED_ENDPOINT",
            "LIGHTSPEED_TOKEN",
            "LIGHTSPEED_CONCURRENCY",
            "LIGHTSPEED_TIMEOUT",
            "LIGHTSPEED_OUTPUT_PATH",
            "LIGHTSPEED_RESUME_PATH",
            "LIGHTSPEED_FREE_FORM_STRATEGY",
            "LIGHTSPEED_JUDGE_ENDPOINT",
            "LIGHTSPEED_JUDGE_MODEL",
            "LIGHTSPEED_JUDGE_TOKEN",
            "LIGHTSPEED_EMBEDDING_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)
        result = EvalConfig.from_env()
        assert result == {}

    def test_all_env_vars_set(self, monkeypatch):
        monkeypatch.setenv("LIGHTSPEED_ENDPOINT", "http://env:1234")
        monkeypatch.setenv("LIGHTSPEED_TOKEN", "env-token")
        monkeypatch.setenv("LIGHTSPEED_CONCURRENCY", "12")
        monkeypatch.setenv("LIGHTSPEED_TIMEOUT", "60")
        monkeypatch.setenv("LIGHTSPEED_OUTPUT_PATH", "env_out.json")
        monkeypatch.setenv("LIGHTSPEED_RESUME_PATH", "env_resume.json")
        monkeypatch.setenv("LIGHTSPEED_FREE_FORM_STRATEGY", "llm_judge")
        monkeypatch.setenv("LIGHTSPEED_JUDGE_ENDPOINT", "http://judge-env")
        monkeypatch.setenv("LIGHTSPEED_JUDGE_MODEL", "judge-m")
        monkeypatch.setenv("LIGHTSPEED_JUDGE_TOKEN", "judge-t")
        monkeypatch.setenv("LIGHTSPEED_EMBEDDING_MODEL", "emb-env")

        result = EvalConfig.from_env()
        assert result["endpoint"] == "http://env:1234"
        assert result["token"] == "env-token"
        assert result["concurrency"] == 12
        assert result["timeout_seconds"] == 60
        assert result["output_path"] == "env_out.json"
        assert result["resume_path"] == "env_resume.json"
        ff = result["free_form"]
        assert ff["strategy"] == "llm_judge"
        assert ff["judge_endpoint"] == "http://judge-env"
        assert ff["judge_model"] == "judge-m"
        assert ff["judge_token"] == "judge-t"
        assert ff["embedding_model"] == "emb-env"

    def test_partial_env_vars(self, monkeypatch):
        for var in (
            "LIGHTSPEED_ENDPOINT",
            "LIGHTSPEED_TOKEN",
            "LIGHTSPEED_CONCURRENCY",
            "LIGHTSPEED_TIMEOUT",
            "LIGHTSPEED_OUTPUT_PATH",
            "LIGHTSPEED_RESUME_PATH",
            "LIGHTSPEED_FREE_FORM_STRATEGY",
            "LIGHTSPEED_JUDGE_ENDPOINT",
            "LIGHTSPEED_JUDGE_MODEL",
            "LIGHTSPEED_JUDGE_TOKEN",
            "LIGHTSPEED_EMBEDDING_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)

        monkeypatch.setenv("LIGHTSPEED_ENDPOINT", "http://partial")
        monkeypatch.setenv("LIGHTSPEED_CONCURRENCY", "3")
        result = EvalConfig.from_env()
        assert result == {"endpoint": "http://partial", "concurrency": 3}

    def test_only_free_form_strategy(self, monkeypatch):
        for var in (
            "LIGHTSPEED_ENDPOINT",
            "LIGHTSPEED_TOKEN",
            "LIGHTSPEED_CONCURRENCY",
            "LIGHTSPEED_TIMEOUT",
            "LIGHTSPEED_OUTPUT_PATH",
            "LIGHTSPEED_RESUME_PATH",
            "LIGHTSPEED_JUDGE_ENDPOINT",
            "LIGHTSPEED_JUDGE_MODEL",
            "LIGHTSPEED_JUDGE_TOKEN",
            "LIGHTSPEED_EMBEDDING_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)

        monkeypatch.setenv("LIGHTSPEED_FREE_FORM_STRATEGY", "semantic_similarity")
        result = EvalConfig.from_env()
        assert result == {"free_form": {"strategy": "semantic_similarity"}}

    def test_multiple_free_form_vars_share_dict(self, monkeypatch):
        """All free-form env vars should end up in the same nested dict."""
        for var in (
            "LIGHTSPEED_ENDPOINT",
            "LIGHTSPEED_TOKEN",
            "LIGHTSPEED_CONCURRENCY",
            "LIGHTSPEED_TIMEOUT",
            "LIGHTSPEED_OUTPUT_PATH",
            "LIGHTSPEED_RESUME_PATH",
        ):
            monkeypatch.delenv(var, raising=False)

        monkeypatch.setenv("LIGHTSPEED_FREE_FORM_STRATEGY", "llm_judge")
        monkeypatch.setenv("LIGHTSPEED_JUDGE_ENDPOINT", "http://jj")
        monkeypatch.setenv("LIGHTSPEED_JUDGE_MODEL", "jm")
        monkeypatch.setenv("LIGHTSPEED_JUDGE_TOKEN", "jt")
        monkeypatch.setenv("LIGHTSPEED_EMBEDDING_MODEL", "em")

        result = EvalConfig.from_env()
        ff = result["free_form"]
        assert ff["strategy"] == "llm_judge"
        assert ff["judge_endpoint"] == "http://jj"
        assert ff["judge_model"] == "jm"
        assert ff["judge_token"] == "jt"
        assert ff["embedding_model"] == "em"


# ===================================================================
# EvalConfig.from_cli_args()
# ===================================================================


class TestFromCliArgs:
    def test_none_args_returns_empty(self):
        assert EvalConfig.from_cli_args(None) == {}

    def test_all_direct_fields(self):
        ns = argparse.Namespace(
            endpoint="http://cli",
            token="cli-tok",
            concurrency=5,
            timeout_seconds=60,
            output_path="cli_out.json",
            resume_path="cli_resume.json",
        )
        result = EvalConfig.from_cli_args(ns)
        assert result["endpoint"] == "http://cli"
        assert result["token"] == "cli-tok"
        assert result["concurrency"] == 5
        assert result["timeout_seconds"] == 60
        assert result["output_path"] == "cli_out.json"
        assert result["resume_path"] == "cli_resume.json"

    def test_none_direct_fields_dropped(self):
        ns = argparse.Namespace(
            endpoint=None,
            token=None,
            concurrency=None,
            timeout_seconds=None,
            output_path=None,
            resume_path=None,
        )
        result = EvalConfig.from_cli_args(ns)
        assert result == {}

    def test_filter_fields(self):
        ns = argparse.Namespace(
            endpoint=None,
            token=None,
            concurrency=None,
            timeout_seconds=None,
            output_path=None,
            resume_path=None,
            category="networking",
            difficulty="hard",
            question_type="mcq",
            tags="t1,t2",
            ids="id1,id2",
            free_form_strategy=None,
            judge_endpoint=None,
            judge_model=None,
            judge_token=None,
            judge_pass_threshold=None,
            embedding_model=None,
            similarity_pass_threshold=None,
        )
        result = EvalConfig.from_cli_args(ns)
        assert result["filters"]["category"] == "networking"
        assert result["filters"]["difficulty"] == "hard"
        assert result["filters"]["question_type"] == "mcq"
        assert result["filters"]["tags"] == "t1,t2"
        assert result["filters"]["ids"] == "id1,id2"

    def test_free_form_fields(self):
        ns = argparse.Namespace(
            endpoint=None,
            token=None,
            concurrency=None,
            timeout_seconds=None,
            output_path=None,
            resume_path=None,
            category=None,
            difficulty=None,
            question_type=None,
            tags=None,
            ids=None,
            free_form_strategy="llm_judge",
            judge_endpoint="http://j",
            judge_model="jm",
            judge_token="jt",
            judge_pass_threshold=0.8,
            embedding_model="em",
            similarity_pass_threshold=0.9,
        )
        result = EvalConfig.from_cli_args(ns)
        ff = result["free_form"]
        assert ff["strategy"] == "llm_judge"
        assert ff["judge_endpoint"] == "http://j"
        assert ff["judge_model"] == "jm"
        assert ff["judge_token"] == "jt"
        assert ff["judge_pass_threshold"] == 0.8
        assert ff["embedding_model"] == "em"
        assert ff["similarity_pass_threshold"] == 0.9

    def test_missing_attributes_skipped(self):
        """Namespace with no config-relevant attributes at all."""
        ns = argparse.Namespace()
        result = EvalConfig.from_cli_args(ns)
        assert result == {}

    def test_mixed_present_and_absent(self):
        ns = argparse.Namespace(
            endpoint="http://mixed",
            token=None,
            concurrency=None,
            timeout_seconds=None,
            output_path=None,
            resume_path=None,
            category="cat1",
            difficulty=None,
            question_type=None,
            tags=None,
            ids=None,
            free_form_strategy=None,
            judge_endpoint=None,
            judge_model=None,
            judge_token=None,
            judge_pass_threshold=None,
            embedding_model=None,
            similarity_pass_threshold=None,
        )
        result = EvalConfig.from_cli_args(ns)
        assert result["endpoint"] == "http://mixed"
        assert result["filters"] == {"category": "cat1"}
        assert "free_form" not in result


# ===================================================================
# EvalConfig.from_all_sources()
# ===================================================================


class TestFromAllSources:
    def test_defaults_only(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)
        cfg = EvalConfig.from_all_sources()
        assert cfg.endpoint == "http://localhost:8080"
        assert cfg.concurrency == 4

    def test_explicit_config_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)
        p = tmp_path / "my.json"
        p.write_text(json.dumps({"endpoint": "http://file:5555", "concurrency": 7}))
        cfg = EvalConfig.from_all_sources(config_file=str(p))
        assert cfg.endpoint == "http://file:5555"
        assert cfg.concurrency == 7

    def test_auto_detect_yaml(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)
        p = tmp_path / "eval_config.yaml"
        p.write_text("")

        fake_yaml = mock.MagicMock()
        fake_yaml.safe_load.return_value = {"endpoint": "http://auto-yaml"}

        with mock.patch("eval_config.yaml", fake_yaml):
            cfg = EvalConfig.from_all_sources()
        assert cfg.endpoint == "http://auto-yaml"

    def test_auto_detect_json(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)
        p = tmp_path / "eval_config.json"
        p.write_text(json.dumps({"endpoint": "http://auto-json"}))
        cfg = EvalConfig.from_all_sources()
        assert cfg.endpoint == "http://auto-json"

    def test_auto_detect_yaml_preferred_over_json(self, monkeypatch, tmp_path):
        """If both eval_config.yaml and eval_config.json exist, yaml wins."""
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)
        (tmp_path / "eval_config.yaml").write_text("")
        (tmp_path / "eval_config.json").write_text(
            json.dumps({"endpoint": "http://json-should-lose"})
        )
        fake_yaml = mock.MagicMock()
        fake_yaml.safe_load.return_value = {"endpoint": "http://yaml-wins"}

        with mock.patch("eval_config.yaml", fake_yaml):
            cfg = EvalConfig.from_all_sources()
        assert cfg.endpoint == "http://yaml-wins"

    def test_env_overrides_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"endpoint": "http://file", "concurrency": 2}))
        monkeypatch.setenv("LIGHTSPEED_ENDPOINT", "http://env-override")

        cfg = EvalConfig.from_all_sources(config_file=str(p))
        assert cfg.endpoint == "http://env-override"
        assert cfg.concurrency == 2  # file value preserved

    def test_cli_overrides_env_and_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"endpoint": "http://file", "concurrency": 2}))
        monkeypatch.setenv("LIGHTSPEED_ENDPOINT", "http://env")
        ns = argparse.Namespace(
            endpoint="http://cli-wins",
            token=None,
            concurrency=None,
            timeout_seconds=None,
            output_path=None,
            resume_path=None,
        )
        cfg = EvalConfig.from_all_sources(config_file=str(p), cli_args=ns)
        assert cfg.endpoint == "http://cli-wins"
        assert cfg.concurrency == 2  # from file

    def test_merge_filters_from_file_and_cli(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)
        p = tmp_path / "cfg.json"
        p.write_text(
            json.dumps(
                {
                    "filters": {
                        "category": "security",
                        "difficulty": "medium",
                    }
                }
            )
        )
        ns = argparse.Namespace(
            endpoint=None,
            token=None,
            concurrency=None,
            timeout_seconds=None,
            output_path=None,
            resume_path=None,
            category=None,
            difficulty="hard",
            question_type=None,
            tags=None,
            ids=None,
            free_form_strategy=None,
            judge_endpoint=None,
            judge_model=None,
            judge_token=None,
            judge_pass_threshold=None,
            embedding_model=None,
            similarity_pass_threshold=None,
        )
        cfg = EvalConfig.from_all_sources(config_file=str(p), cli_args=ns)
        # CLI overwrites difficulty but file category is preserved.
        assert cfg.filters.category == "security"
        assert cfg.filters.difficulty == "hard"

    def test_merge_free_form_from_file_and_cli(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)
        p = tmp_path / "cfg.json"
        p.write_text(
            json.dumps(
                {
                    "free_form": {
                        "strategy": "llm_judge",
                        "judge_model": "file-model",
                    }
                }
            )
        )
        ns = argparse.Namespace(
            endpoint=None,
            token=None,
            concurrency=None,
            timeout_seconds=None,
            output_path=None,
            resume_path=None,
            category=None,
            difficulty=None,
            question_type=None,
            tags=None,
            ids=None,
            free_form_strategy=None,
            judge_endpoint=None,
            judge_model="cli-model",
            judge_token=None,
            judge_pass_threshold=None,
            embedding_model=None,
            similarity_pass_threshold=None,
        )
        cfg = EvalConfig.from_all_sources(config_file=str(p), cli_args=ns)
        assert cfg.free_form.strategy == "llm_judge"  # from file
        assert cfg.free_form.judge_model == "cli-model"  # from CLI

    def test_filters_non_dict_existing_branch(self, monkeypatch, tmp_path):
        """Cover the else branch on line 382: existing_filters is not a dict."""
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)

        # Manually simulate: merged["filters"] is not a dict before CLI merge.
        ns = argparse.Namespace(
            category="cat1",
            difficulty=None,
            question_type=None,
            tags=None,
            ids=None,
            endpoint=None,
            token=None,
            concurrency=None,
            timeout_seconds=None,
            output_path=None,
            resume_path=None,
            free_form_strategy=None,
            judge_endpoint=None,
            judge_model=None,
            judge_token=None,
            judge_pass_threshold=None,
            embedding_model=None,
            similarity_pass_threshold=None,
        )
        # Patch from_env to inject a non-dict filters value.
        with mock.patch.object(
            EvalConfig, "from_env", return_value={"filters": "not-a-dict"}
        ):
            cfg = EvalConfig.from_all_sources(cli_args=ns)
        assert cfg.filters.category == "cat1"

    def test_free_form_non_dict_existing_branch(self, monkeypatch, tmp_path):
        """Cover the else branch on line 389: existing_ff is not a dict."""
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)

        ns = argparse.Namespace(
            category=None,
            difficulty=None,
            question_type=None,
            tags=None,
            ids=None,
            endpoint=None,
            token=None,
            concurrency=None,
            timeout_seconds=None,
            output_path=None,
            resume_path=None,
            free_form_strategy="llm_judge",
            judge_endpoint=None,
            judge_model=None,
            judge_token=None,
            judge_pass_threshold=None,
            embedding_model=None,
            similarity_pass_threshold=None,
        )
        # Patch from_env to inject a non-dict free_form value.
        with mock.patch.object(
            EvalConfig, "from_env", return_value={"free_form": "not-a-dict"}
        ):
            cfg = EvalConfig.from_all_sources(cli_args=ns)
        assert cfg.free_form.strategy == "llm_judge"

    def test_validation_is_called(self, monkeypatch, tmp_path):
        """from_all_sources should call validate and raise on bad data."""
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)
        monkeypatch.setenv("LIGHTSPEED_ENDPOINT", "not-a-url")
        with pytest.raises(ValueError, match="endpoint must be a valid HTTP"):
            EvalConfig.from_all_sources()

    def test_no_config_file_no_auto_detect(self, monkeypatch, tmp_path):
        """No config file provided and none auto-detected."""
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)
        cfg = EvalConfig.from_all_sources(config_file=None)
        assert cfg.endpoint == "http://localhost:8080"

    @staticmethod
    def _clear_env(monkeypatch):
        for var in (
            "LIGHTSPEED_ENDPOINT",
            "LIGHTSPEED_TOKEN",
            "LIGHTSPEED_CONCURRENCY",
            "LIGHTSPEED_TIMEOUT",
            "LIGHTSPEED_OUTPUT_PATH",
            "LIGHTSPEED_RESUME_PATH",
            "LIGHTSPEED_FREE_FORM_STRATEGY",
            "LIGHTSPEED_JUDGE_ENDPOINT",
            "LIGHTSPEED_JUDGE_MODEL",
            "LIGHTSPEED_JUDGE_TOKEN",
            "LIGHTSPEED_EMBEDDING_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)


# ===================================================================
# EvalConfig.build_arg_parser()
# ===================================================================


class TestBuildArgParser:
    def test_creates_fresh_parser(self):
        parser = EvalConfig.build_arg_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_uses_existing_parser(self):
        existing = argparse.ArgumentParser(description="Existing")
        returned = EvalConfig.build_arg_parser(parser=existing)
        assert returned is existing

    def test_all_args_registered(self):
        parser = EvalConfig.build_arg_parser()
        # Parse with all arguments.
        args = parser.parse_args(
            [
                "--endpoint",
                "http://x",
                "--token",
                "tok",
                "--concurrency",
                "2",
                "--timeout-seconds",
                "30",
                "--output-path",
                "o.json",
                "--resume-path",
                "r.json",
                "--config-file",
                "c.yaml",
                "--category",
                "cat",
                "--difficulty",
                "easy",
                "--question-type",
                "mcq",
                "--tags",
                "t1,t2",
                "--ids",
                "id1",
                "--free-form-strategy",
                "llm_judge",
                "--judge-endpoint",
                "http://je",
                "--judge-model",
                "jm",
                "--judge-token",
                "jt",
                "--judge-pass-threshold",
                "0.8",
                "--embedding-model",
                "em",
                "--similarity-pass-threshold",
                "0.9",
            ]
        )
        assert args.endpoint == "http://x"
        assert args.token == "tok"
        assert args.concurrency == 2
        assert args.timeout_seconds == 30
        assert args.output_path == "o.json"
        assert args.resume_path == "r.json"
        assert args.config_file == "c.yaml"
        assert args.category == "cat"
        assert args.difficulty == "easy"
        assert args.question_type == "mcq"
        assert args.tags == "t1,t2"
        assert args.ids == "id1"
        assert args.free_form_strategy == "llm_judge"
        assert args.judge_endpoint == "http://je"
        assert args.judge_model == "jm"
        assert args.judge_token == "jt"
        assert args.judge_pass_threshold == 0.8
        assert args.embedding_model == "em"
        assert args.similarity_pass_threshold == 0.9

    def test_no_args_defaults_to_none(self):
        parser = EvalConfig.build_arg_parser()
        args = parser.parse_args([])
        assert args.endpoint is None
        assert args.token is None
        assert args.concurrency is None
        assert args.timeout_seconds is None
        assert args.output_path is None
        assert args.resume_path is None
        assert args.config_file is None
        assert args.category is None
        assert args.difficulty is None
        assert args.question_type is None
        assert args.tags is None
        assert args.ids is None
        assert args.free_form_strategy is None
        assert args.judge_endpoint is None
        assert args.judge_model is None
        assert args.judge_token is None
        assert args.judge_pass_threshold is None
        assert args.embedding_model is None
        assert args.similarity_pass_threshold is None

    def test_free_form_strategy_choices(self):
        parser = EvalConfig.build_arg_parser()
        # Invalid choice should cause SystemExit.
        with pytest.raises(SystemExit):
            parser.parse_args(["--free-form-strategy", "invalid_strategy"])


# ===================================================================
# Integration: round-trip tests
# ===================================================================


class TestIntegration:
    def test_json_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._clear_env(monkeypatch)
        data = {
            "endpoint": "http://integration:9999",
            "token": "int-tok",
            "concurrency": 3,
            "timeout_seconds": 45,
            "output_path": "int_out.json",
            "resume_path": "int_resume.json",
            "filters": {
                "category": "storage",
                "difficulty": "hard",
                "question_type": "free_form",
                "tags": ["alpha", "beta"],
                "ids": ["q100"],
            },
            "free_form": {
                "strategy": "semantic_similarity",
                "judge_endpoint": "",
                "judge_model": "",
                "judge_token": "",
                "judge_pass_threshold": 0.7,
                "embedding_model": "all-MiniLM",
                "similarity_pass_threshold": 0.85,
            },
        }
        p = tmp_path / "full.json"
        p.write_text(json.dumps(data))

        cfg = EvalConfig.from_all_sources(config_file=str(p))
        assert cfg.endpoint == "http://integration:9999"
        assert cfg.token == "int-tok"
        assert cfg.concurrency == 3
        assert cfg.timeout_seconds == 45
        assert cfg.output_path == "int_out.json"
        assert cfg.resume_path == "int_resume.json"
        assert cfg.filters.category == "storage"
        assert cfg.filters.difficulty == "hard"
        assert cfg.filters.question_type == "free_form"
        assert cfg.filters.tags == ["alpha", "beta"]
        assert cfg.filters.ids == ["q100"]
        assert cfg.free_form.strategy == "semantic_similarity"
        assert cfg.free_form.embedding_model == "all-MiniLM"
        assert cfg.free_form.similarity_pass_threshold == 0.85

    @staticmethod
    def _clear_env(monkeypatch):
        for var in (
            "LIGHTSPEED_ENDPOINT",
            "LIGHTSPEED_TOKEN",
            "LIGHTSPEED_CONCURRENCY",
            "LIGHTSPEED_TIMEOUT",
            "LIGHTSPEED_OUTPUT_PATH",
            "LIGHTSPEED_RESUME_PATH",
            "LIGHTSPEED_FREE_FORM_STRATEGY",
            "LIGHTSPEED_JUDGE_ENDPOINT",
            "LIGHTSPEED_JUDGE_MODEL",
            "LIGHTSPEED_JUDGE_TOKEN",
            "LIGHTSPEED_EMBEDDING_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)
