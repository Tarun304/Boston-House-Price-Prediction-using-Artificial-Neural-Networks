import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.keras
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

            # Get active MLflow run or start new one
            active_run = mlflow.active_run()
            if active_run is None:
                mlflow.start_run(run_name="model_evaluation")

            try:
                # Load model
                self.load_model()

                # Load test data
                X_test, y_test = self.load_test_data()

                # Make predictions
                y_pred = self.make_predictions(X_test)

                # Calculate metrics
                self.metrics = self.calculate_metrics(y_test, y_pred)

                # Log metrics to MLflow
                mlflow.log_metric("test_mse", self.metrics["mse"])
                mlflow.log_metric("test_mae", self.metrics["mae"])
                mlflow.log_metric("test_r2_score", self.metrics["r2_score"])

                # Save metrics to file
                self.save_metrics(self.metrics)

                # Log metrics file as artifact
                mlflow.log_artifact(self.config["metrics_file"])

                # Save and log experiment info
                run_id = mlflow.active_run().info.run_id
                experiment_info = {
                    "run_id": run_id,
                    "model_path": self.config["model_path"],
                    "metrics": self.metrics,
                }

                info_file = "reports/experiment_info.json"
                os.makedirs("reports", exist_ok=True)
                with open(info_file, "w") as f:
                    json.dump(experiment_info, f, indent=4)

                mlflow.log_artifact(info_file)

                # Log any error logs if they exist
                log_file = "model_evaluation_errors.log"
                if os.path.exists(log_file):
                    mlflow.log_artifact(log_file)

                logger.info("Model evaluation completed successfully")
                logger.info("MLflow run ID: %s", run_id)

                return self.metrics

            finally:
                if active_run is None:
                    mlflow.end_run()

        except Exception as e:
            logger.error("Model evaluation failed: %s", e)
            raise


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run()
