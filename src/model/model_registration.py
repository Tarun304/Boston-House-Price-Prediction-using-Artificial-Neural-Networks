import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.keras

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MLFLOW_TRACKING_URI

# Logging configuration
logger = logging.getLogger("model_registration")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_experiment_info(file_path: str) -> dict:
    """Load the experiment info from a JSON file."""
    try:
        with open(file_path, "r") as file:
            experiment_info = json.load(file)
        logger.debug("Experiment info loaded from %s", file_path)
        return experiment_info
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logger.error(
            "Unexpected error occurred while loading the experiment info: %s", e
        )
        raise


def register_model(model_name: str, experiment_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        run_id = experiment_info["run_id"]
        model_uri = f"runs:/{run_id}/model"

        logger.info(f"Registering model from run: {run_id}")

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        logger.info(
            f"Model {model_name} version {model_version.version} registered successfully"
        )

        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=False,
        )

        logger.info(
            f"Model {model_name} version {model_version.version} transitioned to Staging"
        )

        # Add description with metrics
        if "metrics" in experiment_info:
            metrics = experiment_info["metrics"]
            description = f"MSE: {metrics.get('mse', 'N/A'):.4f}, MAE: {metrics.get('mae', 'N/A'):.4f}, R²: {metrics.get('r2_score', 'N/A'):.4f}"
            client.update_model_version(
                name=model_name, version=model_version.version, description=description
            )
            logger.info(
                f"Added description with metrics to model version {model_version.version}"
            )

        return model_version

    except Exception as e:
        logger.error("Error during model registration: %s", e)
        raise


def main():
    try:
        experiment_info_path = "reports/experiment_info.json"
        experiment_info = load_experiment_info(experiment_info_path)

        model_name = "boston-house-price-model"
        model_version = register_model(model_name, experiment_info)

        print(f"\n Model Registration Complete!")
        print(f"   Model Name: {model_name}")
        print(f"   Version: {model_version.version}")
        print(f"   Stage: Staging")
        print(f"   Run ID: {experiment_info['run_id']}")

    except Exception as e:
        logger.error("Failed to complete the model registration process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
