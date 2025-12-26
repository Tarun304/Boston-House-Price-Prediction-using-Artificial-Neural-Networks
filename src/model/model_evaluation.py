import json
import logging
import os
import shutil
import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MLFLOW_TRACKING_URI

# Logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class ModelEvaluator:
    def __init__(self, params_path: str = "params.yaml"):
        """Initialize ModelEvaluator with parameters."""
        self.params = self.load_params(params_path)
        self.config = self.params["model_evaluation"]
        self.build_config = self.params["model_building"]
        self.model = None
        self.metrics = {}

    def load_params(self, params_path: str) -> dict:
        """Load parameters from YAML file."""
        try:
            with open(params_path, "r") as file:
                params = yaml.safe_load(file)
            logger.debug("Parameters retrieved from %s", params_path)
            return params
        except Exception as e:
            logger.error("Failed to load params: %s", e)
            raise

    def load_model(self) -> None:
        """Load trained model."""
        try:
            model_path = self.config["model_path"]
            self.model = keras.models.load_model(model_path)
            logger.info("Model loaded from %s", model_path)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    def load_test_data(self) -> tuple:
        """Load test data."""
        try:
            X_test = pd.read_csv(self.config["X_test_file"])
            y_test = pd.read_csv(self.config["y_test_file"])

            logger.info(
                "Test data loaded. X_test: %s, y_test: %s", X_test.shape, y_test.shape
            )
            return X_test, y_test
        except Exception as e:
            logger.error("Failed to load test data: %s", e)
            raise

    def make_predictions(self, X_test) -> object:
        """Make predictions on test data."""
        try:
            y_pred = self.model.predict(X_test, verbose=0)
            logger.info("Predictions generated for %d samples", len(y_pred))
            return y_pred
        except Exception as e:
            logger.error("Failed to make predictions: %s", e)
            raise

    def calculate_metrics(self, y_test, y_pred) -> dict:
        """Calculate evaluation metrics."""
        try:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics = {"mse": mse, "mae": mae, "r2_score": r2}

            logger.info("Metrics calculated:")
            logger.info("  Mean Squared Error (MSE): %.4f", mse)
            logger.info("  Mean Absolute Error (MAE): %.4f", mae)
            logger.info("  R-squared (R²): %.4f", r2)

            return metrics
        except Exception as e:
            logger.error("Failed to calculate metrics: %s", e)
            raise

    def save_metrics(self, metrics: dict) -> None:
        """Save metrics to JSON file."""
        try:
            metrics_file = self.config["metrics_file"]
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)

            logger.info("Metrics saved to %s", metrics_file)
        except Exception as e:
            logger.error("Failed to save metrics: %s", e)
            raise

    def run(self):
        """Execute the model evaluation pipeline."""
        try:
            logger.info("Starting model evaluation...")

            with mlflow.start_run(run_name="model_evaluation"):
                # Log model building parameters
                mlflow.log_param("units", self.build_config["units"])
                mlflow.log_param("num_layers", self.build_config["num_layers"])
                mlflow.log_param("learning_rate", self.build_config["learning_rate"])
                mlflow.log_param("epochs", self.build_config["epochs"])
                mlflow.log_param("batch_size", self.build_config["batch_size"])
                mlflow.log_param("patience", self.build_config["patience"])

                # Load model and data
                self.load_model()
                X_test, y_test = self.load_test_data()
                y_pred = self.make_predictions(X_test)
                self.metrics = self.calculate_metrics(y_test, y_pred)

                # MANUAL MODEL LOGGING
                logger.info("Creating MLflow model structure manually...")

                temp_dir = "temp_mlflow_model"
                model_dir = os.path.join(temp_dir, "model")
                data_dir = os.path.join(model_dir, "data")
                os.makedirs(data_dir, exist_ok=True)

                # Save Keras model in data folder
                model_file = os.path.join(data_dir, "model.keras")
                self.model.save(model_file)

                run_id = mlflow.active_run().info.run_id

                # Create MLmodel file
                mlmodel_content = f"""artifact_path: model
flavors:
  keras:
    data: data
    keras_backend: tensorflow
    keras_version: 3.13.0
    save_exported_model: false
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.keras
    python_version: 3.13.0
mlflow_version: {mlflow.__version__}
model_size_bytes: {os.path.getsize(model_file)}
run_id: {run_id}
utc_time_created: '{pd.Timestamp.utcnow()}'
"""
                with open(os.path.join(model_dir, "MLmodel"), "w") as f:
                    f.write(mlmodel_content)

                # Create conda.yaml
                conda_yaml = """channels:
- conda-forge
dependencies:
- python=3.13.0
- pip
- pip:
  - mlflow
  - tensorflow
  - keras
"""
                with open(os.path.join(model_dir, "conda.yaml"), "w") as f:
                    f.write(conda_yaml)

                # Create python_env.yaml
                python_env = """python: 3.13.0
build_dependencies:
- pip
dependencies:
- -r requirements.txt
"""
                with open(os.path.join(model_dir, "python_env.yaml"), "w") as f:
                    f.write(python_env)

                # Create requirements.txt
                requirements = f"""mlflow=={mlflow.__version__}
tensorflow
keras
"""
                with open(os.path.join(model_dir, "requirements.txt"), "w") as f:
                    f.write(requirements)

                # Log entire model directory
                mlflow.log_artifacts(model_dir, artifact_path="model")

                # Clean up
                shutil.rmtree(temp_dir)
                logger.info("Model logged to MLflow successfully")

                # Log evaluation metrics
                mlflow.log_metric("test_mse", self.metrics["mse"])
                mlflow.log_metric("test_mae", self.metrics["mae"])
                mlflow.log_metric("test_r2_score", self.metrics["r2_score"])

                # Save and log metrics file
                self.save_metrics(self.metrics)
                mlflow.log_artifact(self.config["metrics_file"])

                # Save experiment info
                experiment_info = {
                    "run_id": run_id,
                    "model_path": "model",
                    "metrics": self.metrics,
                }

                info_file = "reports/experiment_info.json"
                os.makedirs("reports", exist_ok=True)
                with open(info_file, "w") as f:
                    json.dump(experiment_info, f, indent=4)

                mlflow.log_artifact(info_file)

                log_file = "model_evaluation_errors.log"
                if os.path.exists(log_file):
                    mlflow.log_artifact(log_file)

                logger.info("Model evaluation completed successfully")
                logger.info("MLflow run ID: %s", run_id)

                return self.metrics

        except Exception as e:
            logger.error("Model evaluation failed: %s", e)
            raise


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run()
