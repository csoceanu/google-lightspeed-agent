#!/usr/bin/env python3
"""
Provision MLflow with the Lightspeed evaluation resources.

Registers all custom judges and the evaluation dataset in the MLflow
tracking server. Run this once per MLflow instance (or after a reset)
to populate the Judges, Datasets, and Scorers tabs in the UI.

Usage:
    # Against local sqlite (default):
    python -m mlflow_eval.setup

    # Against a remote MLflow server:
    python -m mlflow_eval.setup --tracking-uri http://mlflow.example.com:5000

    # With custom experiment name:
    python -m mlflow_eval.setup --experiment lightspeed-eval-prod
"""

import argparse
import logging

logger = logging.getLogger("mlflow_eval.setup")


def setup_judges(experiment_name: str) -> list[str]:
    """Register all custom judges. Returns list of registered names."""
    from mlflow_eval.scorers import (
        DomainCorrectnessJudge,
        ErrorHandlingJudge,
        SafetyJudge,
    )

    registered = []
    for factory in [SafetyJudge, ErrorHandlingJudge, DomainCorrectnessJudge]:
        judge = factory()
        try:
            judge.register(name=judge.name, experiment_id=None)
            registered.append(judge.name)
            logger.info("Registered judge: %s", judge.name)
        except Exception as exc:
            if "already exists" in str(exc).lower():
                registered.append(judge.name)
                logger.info("Judge already registered: %s", judge.name)
            else:
                logger.warning("Failed to register %s: %s", judge.name, exc)

    return registered


def setup_dataset(
    name: str = "lightspeed-eval",
    dataset_path: str | None = None,
) -> str:
    """Create and populate the MLflow dataset. Returns dataset ID."""
    import mlflow
    from mlflow_eval.dataset import load_dataset

    existing = mlflow.genai.search_datasets()
    for ds in existing:
        if ds.name == name:
            logger.info("Dataset '%s' already exists (%s)", name, ds.dataset_id)
            records = load_dataset(path=dataset_path)
            ds.merge_records(records)
            logger.info("Merged %d records into existing dataset", len(records))
            return ds.dataset_id

    ds = mlflow.genai.create_dataset(
        name=name,
        tags={
            "source": "eval_dataset.json",
            "framework": "lightspeed-eval",
            "version": "1.0",
        },
    )
    records = load_dataset(path=dataset_path)
    ds.merge_records(records)
    logger.info("Created dataset '%s' with %d records (%s)", name, len(records), ds.dataset_id)
    return ds.dataset_id


def setup_all(
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "lightspeed-eval",
    dataset_name: str = "lightspeed-eval",
    dataset_path: str | None = None,
) -> dict:
    """Provision everything in MLflow. Returns summary dict."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    result = {
        "tracking_uri": tracking_uri,
        "experiment": experiment_name,
    }

    result["judges"] = setup_judges(experiment_name)
    result["dataset_id"] = setup_dataset(dataset_name, dataset_path)

    registered = mlflow.genai.list_scorers()
    result["registered_scorers"] = [s.name for s in registered]

    datasets = mlflow.genai.search_datasets()
    result["registered_datasets"] = [d.name for d in datasets]

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Provision MLflow with Lightspeed evaluation resources"
    )
    parser.add_argument(
        "--tracking-uri", default="sqlite:///mlflow.db",
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    parser.add_argument(
        "--experiment", default="lightspeed-eval",
        help="MLflow experiment name (default: lightspeed-eval)",
    )
    parser.add_argument(
        "--dataset-name", default="lightspeed-eval",
        help="Name for the registered dataset (default: lightspeed-eval)",
    )
    parser.add_argument(
        "--dataset-path", default=None,
        help="Path to eval_dataset.json (default: auto-detected)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Provisioning MLflow with Lightspeed evaluation resources...")
    print(f"  Tracking URI: {args.tracking_uri}")
    print(f"  Experiment:   {args.experiment}")
    print()

    result = setup_all(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
    )

    print("Done.")
    print(f"  Judges:   {result['judges']}")
    print(f"  Scorers:  {result['registered_scorers']}")
    print(f"  Datasets: {result['registered_datasets']}")
    print(f"  Dataset ID: {result['dataset_id']}")


if __name__ == "__main__":
    main()
