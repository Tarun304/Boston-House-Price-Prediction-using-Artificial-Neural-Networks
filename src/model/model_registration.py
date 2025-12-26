import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient

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
    """Register the model to MLflow Model Registry with aliases and tags."""
    try:
        run_id = experiment_info["run_id"]
        model_path = experiment_info["model_path"]
        model_uri = f"runs:/{run_id}/{model_path}"

        logger.info(f"Registering model from URI: {model_uri}")

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        logger.info(
            f"Model {model_name} version {model_version.version} registered successfully"
        )

        # Use MLflow Client for alias management and tags
        client = MlflowClient()

        # Set alias "challenger" (staging for testing)
        client.set_registered_model_alias(
            name=model_name,
            alias="challenger",
            version=model_version.version,
        )

        logger.info(
            f"Model version {model_version.version} assigned alias 'challenger'"
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

        # Add tags for better organization
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="deployment_status",
            value="staging",
        )

        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="framework",
            value="tensorflow-keras",
        )

        logger.info(
            f"Successfully registered and configured model version {model_version.version}"
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
        print(f"   Alias: challenger (staging - ready for testing)")
        print(f"   Run ID: {experiment_info['run_id']}")
        print(f"\n   Access via: models:/{model_name}@challenger")
        print(f"\n To promote to production after validation:")
        print(
            f"   client.set_registered_model_alias('{model_name}', 'champion', {model_version.version})"
        )

    except Exception as e:
        logger.error("Failed to complete the model registration process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
