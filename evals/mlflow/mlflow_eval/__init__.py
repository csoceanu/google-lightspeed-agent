"""MLflow integration for the Lightspeed Agent evaluation dataset.

Provides custom scorers, dataset loading, and judge configuration for
``mlflow.genai.evaluate()``. Single source of truth for evaluation
questions is the ``cases/`` directory in AEH format.

Quick start::

    from mlflow_eval import load_cases, AnswerCorrectness, ToolMatch, BehaviorCoverage

    dataset = load_cases()  # loads 7 curated cases from cases/

    mlflow.genai.evaluate(
        data=dataset,
        predict_fn=call_agent,
        scorers=[AnswerCorrectness(), ToolMatch(), BehaviorCoverage()],
    )
"""

from mlflow_eval.dataset import load_cases, load_dataset
from mlflow_eval.scorers import (
    AnswerCorrectness,
    BehaviorCoverage,
    DomainCorrectnessJudge,
    ErrorHandlingGuidelines,
    ErrorHandlingJudge,
    ResponseReceived,
    SafetyGuidelines,
    SafetyJudge,
    ToolCallCorrectness,
    ToolMatch,
)
from mlflow_eval.a2a_client import a2a_predict_fn
from mlflow_eval.mock_insights_api import start_mock_api
from mlflow_eval.vertex_judge import start_vertex_judge

__all__ = [
    "load_cases",
    "load_dataset",
    "AnswerCorrectness",
    "ToolMatch",
    "BehaviorCoverage",
    "ToolCallCorrectness",
    "ResponseReceived",
    "SafetyGuidelines",
    "ErrorHandlingGuidelines",
    "SafetyJudge",
    "ErrorHandlingJudge",
    "DomainCorrectnessJudge",
    "a2a_predict_fn",
    "start_mock_api",
    "start_vertex_judge",
]
