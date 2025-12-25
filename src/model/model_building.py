import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.keras
import pandas as pd
import yaml
from tensorflow import keras

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MLFLOW_TRACKING_URI

# Logging configuration
logger = logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class ModelBuilder:
    def __init__(self, params_path: str = "params.yaml"):
        """Initialize ModelBuilder with parameters."""
        self.params = self.load_params(params_path)
        self.config = self.params["model_building"]
        self.model = None
        self.history = None

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

    def load_processed_data(self) -> tuple:
        """Load processed training and validation data."""
        try:
            X_train = pd.read_csv(self.config["X_train_file"])
            y_train = pd.read_csv(self.config["y_train_file"])
            X_test = pd.read_csv(self.config["X_test_file"])
            y_test = pd.read_csv(self.config["y_test_file"])

            logger.info(
                "Processed data loaded. X_train: %s, y_train: %s",
                X_train.shape,
                y_train.shape,
            )
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error("Failed to load processed data: %s", e)
            raise

    def build_model(self, input_dim: int) -> keras.models.Sequential:
        """Build neural network model with configured architecture."""
        try:
            units = self.config["units"]
            num_layers = self.config["num_layers"]
            learning_rate = self.config["learning_rate"]

            model = keras.models.Sequential()

            # Input layer and first hidden layer
            model.add(
                keras.layers.Dense(units=units, input_dim=input_dim, activation="relu")
            )

            # Additional hidden layers
            for _ in range(num_layers - 1):
                model.add(keras.layers.Dense(units=units, activation="relu"))

            # Output layer
            model.add(keras.layers.Dense(1))

            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss="mean_squared_error",
                metrics=["mae"],
            )

            logger.info(
                "Model built: units=%d, layers=%d, lr=%f",
                units,
                num_layers,
                learning_rate,
            )
            logger.debug("Model summary:\n%s", model.summary())

            return model
        except Exception as e:
            logger.error("Failed to build model: %s", e)
            raise

    def train_model(self, X_train, y_train, X_test, y_test) -> None:
        """Train the model with early stopping and model checkpointing."""
        try:
            # Start MLflow run
            with mlflow.start_run(run_name="model_training"):

                # Log ALL parameters
                mlflow.log_param("units", self.config["units"])
                mlflow.log_param("num_layers", self.config["num_layers"])
                mlflow.log_param("learning_rate", self.config["learning_rate"])
                mlflow.log_param("epochs", self.config["epochs"])
                mlflow.log_param("batch_size", self.config["batch_size"])
                mlflow.log_param("patience", self.config["patience"])
                mlflow.log_param("input_dim", X_train.shape[1])
                mlflow.log_param("train_samples", X_train.shape[0])
                mlflow.log_param("test_samples", X_test.shape[0])

                epochs = self.config["epochs"]
                batch_size = self.config["batch_size"]
                patience = self.config["patience"]

                model_path = os.path.join(
                    self.config["model_output_path"], self.config["model_name"]
                )
                os.makedirs(self.config["model_output_path"], exist_ok=True)

                # Callbacks
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                    mode="auto",
                )

                checkpoint = keras.callbacks.ModelCheckpoint(
                    model_path, save_best_only=True, monitor="val_loss", mode="min"
                )

                logger.info("Starting model training...")

                # Train model
                self.history = self.model.fit(
                    X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, checkpoint],
                    verbose=2,
                )

                # Log metrics to MLflow
                final_train_loss = self.history.history["loss"][-1]
                final_val_loss = self.history.history["val_loss"][-1]
                final_train_mae = self.history.history["mae"][-1]
                final_val_mae = self.history.history["val_mae"][-1]

                mlflow.log_metric("train_loss", final_train_loss)
                mlflow.log_metric("val_loss", final_val_loss)
                mlflow.log_metric("train_mae", final_train_mae)
                mlflow.log_metric("val_mae", final_val_mae)

                # Log model to MLflow
                mlflow.keras.log_model(self.model, "model")

                # Save and log model info
                run_id = mlflow.active_run().info.run_id
                model_info = {
                    "run_id": run_id,
                    "model_path": model_path,
                    "framework": "keras",
                }

                info_file = os.path.join(
                    self.config["model_output_path"], "training_info.json"
                )
                with open(info_file, "w") as f:
                    json.dump(model_info, f, indent=4)

                mlflow.log_artifact(info_file)

                logger.info("Model training completed")
                logger.info("Best model saved to %s", model_path)
                logger.info("MLflow run ID: %s", run_id)

        except Exception as e:
            logger.error("Failed to train model: %s", e)
            raise

    def run(self):
        """Execute the model building pipeline."""
        try:
            logger.info("Starting model building...")

            # Load data
            X_train, y_train, X_test, y_test = self.load_processed_data()

            # Build model
            input_dim = X_train.shape[1]
            self.model = self.build_model(input_dim)

            # Train model
            self.train_model(X_train, y_train, X_test, y_test)

            logger.info("Model building completed successfully")
        except Exception as e:
            logger.error("Model building failed: %s", e)
            raise


if __name__ == "__main__":
    builder = ModelBuilder()
    builder.run()
