import logging

import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

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

    def run(self):
        """Execute the model evaluation pipeline."""
        try:
            logger.info("Starting model evaluation...")

            # Load model
            self.load_model()

            # Load test data
            X_test, y_test = self.load_test_data()

            # Make predictions
            y_pred = self.make_predictions(X_test)

            # Calculate metrics
            self.metrics = self.calculate_metrics(y_test, y_pred)

            logger.info("Model evaluation completed successfully")
            return self.metrics
        except Exception as e:
            logger.error("Model evaluation failed: %s", e)
            raise


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run()
