"""Model Registration to MLflow Model Registry"""

import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.keras
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# Load environment variables
load_dotenv()

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


class ModelRegistrar:
    """Handle model registration to MLflow Model Registry"""

    def __init__(self, experiment_info_path: str = "reports/experiment_info.json"):
        """Initialize ModelRegistrar"""
        self.experiment_info_path = experiment_info_path
        self.model_name = "boston-house-price-model"
        self.experiment_info = None
        self.model_version = None

    def load_experiment_info(self) -> dict:
        """Load the experiment info from a JSON file"""
        try:
            with open(self.experiment_info_path, "r") as file:
                self.experiment_info = json.load(file)
            logger.debug("Experiment info loaded from %s", self.experiment_info_path)
            return self.experiment_info
        except FileNotFoundError:
            logger.error("File not found: %s", self.experiment_info_path)
            raise
        except Exception as e:
            logger.error("Unexpected error while loading experiment info: %s", e)
            raise

    def register_model(self):
        """Register the model to MLflow Model Registry with aliases and tags"""
        try:
            run_id = self.experiment_info["run_id"]
            model_path = self.experiment_info["model_path"]
            model_uri = f"runs:/{run_id}/{model_path}"

            logger.info(f"Registering model from URI: {model_uri}")

            # Register the model
            self.model_version = mlflow.register_model(model_uri, self.model_name)

            logger.info(
                f"Model {self.model_name} version {self.model_version.version} registered successfully"
            )

            # Use MLflow Client for alias management and tags
            client = MlflowClient()

            # Set alias "challenger" (staging for testing)
            client.set_registered_model_alias(
                name=self.model_name,
                alias="challenger",
                version=self.model_version.version,
            )

            logger.info(
                f"Model version {self.model_version.version} assigned alias 'challenger'"
            )

            # Add description with metrics
            if "metrics" in self.experiment_info:
                metrics = self.experiment_info["metrics"]
                description = f"MSE: {metrics.get('mse', 'N/A'):.4f}, MAE: {metrics.get('mae', 'N/A'):.4f}, R2: {metrics.get('r2_score', 'N/A'):.4f}"
                client.update_model_version(
                    name=self.model_name,
                    version=self.model_version.version,
                    description=description,
                )
                logger.info(
                    f"Added description with metrics to model version {self.model_version.version}"
                )

            # Add tags for better organization
            client.set_model_version_tag(
                name=self.model_name,
                version=self.model_version.version,
                key="deployment_status",
                value="staging",
            )

            client.set_model_version_tag(
                name=self.model_name,
                version=self.model_version.version,
                key="framework",
                value="tensorflow-keras",
            )

            logger.info(
                f"Successfully registered and configured model version {self.model_version.version}"
            )

            return self.model_version

        except Exception as e:
            logger.error("Error during model registration: %s", e)
            raise

    def run(self):
        """Execute model registration pipeline"""
        try:
            self.load_experiment_info()
            self.register_model()

            print("=" * 70)
            print("MODEL REGISTRATION COMPLETE")
            print("=" * 70)
            print(f"  Model Name: {self.model_name}")
            print(f"  Version: {self.model_version.version}")
            print(f"  Alias: challenger (staging - ready for testing)")
            print(f"  Run ID: {self.experiment_info['run_id']}")
            print(f"  Access via: models:/{self.model_name}@challenger")
            print("=" * 70)

        except Exception as e:
            logger.error("Failed to complete model registration: %s", e)
            print(f"ERROR: {e}")


if __name__ == "__main__":
    registrar = ModelRegistrar()
    registrar.run()
