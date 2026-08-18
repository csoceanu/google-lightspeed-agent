"""MLflow integration for the Lightspeed Agent evaluation dataset.

Provides custom scorers, dataset loading, and judge configuration for
``mlflow.genai.evaluate()``. Single source of truth for evaluation
questions is the ``cases/`` directory in AEH format.

Quick start::

    from mlflow_eval import load_cases, AnswerCorrectness, ToolCallCorrectness

    dataset = load_cases()  # loads 8 curated cases from cases/

    mlflow.genai.evaluate(
        data=dataset,
        predict_fn=call_agent,
        scorers=[AnswerCorrectness(), ToolCallCorrectness(agent_experiment_name="...")],
    )
"""

from mlflow_eval.dataset import load_cases, load_dataset
from mlflow_eval.scorers import (
    AnswerCorrectness,
    DomainCorrectnessJudge,
    ErrorHandlingGuidelines,
    ErrorHandlingJudge,
    ResponseReceived,
    SafetyGuidelines,
    SafetyJudge,
    ToolCallCorrectness,
)
from mlflow_eval.a2a_client import a2a_predict_fn
from mlflow_eval.mock_insights_api import start_mock_api

__all__ = [
    "load_cases",
    "load_dataset",
    "AnswerCorrectness",
    "ToolCallCorrectness",
    "ResponseReceived",
    "SafetyGuidelines",
    "ErrorHandlingGuidelines",
    "SafetyJudge",
    "ErrorHandlingJudge",
    "DomainCorrectnessJudge",
    "a2a_predict_fn",
    "start_mock_api",
]
