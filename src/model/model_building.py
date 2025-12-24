import logging
import os

import pandas as pd
import yaml
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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

    def build_model(self, input_dim: int) -> Sequential:
        """Build neural network model with configured architecture."""
        try:
            units = self.config["units"]
            num_layers = self.config["num_layers"]
            learning_rate = self.config["learning_rate"]

            model = Sequential()

            # Input layer and first hidden layer
            model.add(Dense(units=units, input_dim=input_dim, activation="relu"))

            # Additional hidden layers
            for _ in range(num_layers - 1):
                model.add(Dense(units=units, activation="relu"))

            # Output layer
            model.add(Dense(1))

            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
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
            epochs = self.config["epochs"]
            batch_size = self.config["batch_size"]
            patience = self.config["patience"]

            model_path = os.path.join(
                self.config["model_output_path"], self.config["model_name"]
            )
            os.makedirs(self.config["model_output_path"], exist_ok=True)

            # Callbacks
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                mode="auto",
            )

            checkpoint = ModelCheckpoint(
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

            logger.info("Model training completed")
            logger.info("Best model saved to %s", model_path)

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
