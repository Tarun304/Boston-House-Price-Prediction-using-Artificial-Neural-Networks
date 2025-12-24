import logging
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Logging configuration
logger = logging.getLogger("feature_engineering")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class FeatureEngineering:
    def __init__(self, params_path: str = "params.yaml"):
        """Initialize FeatureEngineering with parameters."""
        self.params = self.load_params(params_path)
        self.config = self.params["feature_engineering"]
        self.scaler = StandardScaler()
        self.selected_features = None

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

    def load_interim_data(self) -> tuple:
        """Load interim train and test data."""
        try:
            train_path = self.config["interim_train_file"]
            test_path = self.config["interim_test_file"]

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(
                "Interim data loaded. Train: %s, Test: %s",
                train_df.shape,
                test_df.shape,
            )
            return train_df, test_df
        except Exception as e:
            logger.error("Failed to load interim data: %s", e)
            raise

    def separate_features_target(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple:
        """Separate features (X) and target (y)."""
        try:
            X_train = train_df.drop("MEDV", axis=1)
            y_train = train_df["MEDV"]

            X_test = test_df.drop("MEDV", axis=1)
            y_test = test_df["MEDV"]

            logger.info("Features and target separated. X_train: %s", X_train.shape)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error("Failed to separate features and target: %s", e)
            raise

    def normalize_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """Normalize features using StandardScaler (fit on train only)."""
        try:
            # Fit scaler on train data only
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )

            # Transform test data using train scaler
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )

            logger.info("Feature normalization completed (fitted on train only)")
            return X_train_scaled, X_test_scaled
        except Exception as e:
            logger.error("Failed to normalize features: %s", e)
            raise

    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series) -> list:
        """Select top N features using Random Forest feature importance."""
        try:
            n_features = self.config["n_features"]
            random_state = self.config["random_state"]

            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
            rf.fit(X_train, y_train)

            # Get feature importances
            feature_importances = pd.Series(
                rf.feature_importances_, index=X_train.columns
            )
            sorted_features = feature_importances.sort_values(ascending=False)

            # Select top N features
            selected_features = sorted_features.index[:n_features].tolist()

            logger.info("Selected top %d features: %s", n_features, selected_features)
            logger.debug(
                "Feature importance scores:\n%s", sorted_features.head(n_features)
            )

            return selected_features
        except Exception as e:
            logger.error("Failed to select features: %s", e)
            raise

    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """Save processed features and targets."""
        try:
            output_path = self.config["processed_data_path"]
            os.makedirs(output_path, exist_ok=True)

            X_train.to_csv(os.path.join(output_path, "X_train.csv"), index=False)
            X_test.to_csv(os.path.join(output_path, "X_test.csv"), index=False)
            y_train.to_csv(os.path.join(output_path, "y_train.csv"), index=False)
            y_test.to_csv(os.path.join(output_path, "y_test.csv"), index=False)

            logger.info("Processed data saved to %s", output_path)
        except Exception as e:
            logger.error("Failed to save processed data: %s", e)
            raise

    def save_artifacts(self) -> None:
        """Save scaler and selected features for later use."""
        try:
            models_path = "models"
            os.makedirs(models_path, exist_ok=True)

            # Save scaler
            scaler_path = os.path.join(models_path, "scaler.pkl")
            joblib.dump(self.scaler, scaler_path)

            # Save selected features
            features_path = os.path.join(models_path, "selected_features.pkl")
            joblib.dump(self.selected_features, features_path)

            logger.info("Artifacts saved: scaler.pkl and selected_features.pkl")
        except Exception as e:
            logger.error("Failed to save artifacts: %s", e)
            raise

    def run(self):
        """Execute the feature engineering pipeline."""
        try:
            logger.info("Starting feature engineering...")

            # Load interim data
            train_df, test_df = self.load_interim_data()

            # Separate features and target
            X_train, X_test, y_train, y_test = self.separate_features_target(
                train_df, test_df
            )

            # Normalize features (fit on train only)
            X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)

            # Feature selection (on train data)
            self.selected_features = self.select_features(X_train_scaled, y_train)

            # Keep only selected features
            X_train_final = X_train_scaled[self.selected_features]
            X_test_final = X_test_scaled[self.selected_features]

            # Save processed data
            self.save_processed_data(X_train_final, X_test_final, y_train, y_test)

            # Save artifacts
            self.save_artifacts()

            logger.info("Feature engineering completed successfully")
        except Exception as e:
            logger.error("Feature engineering failed: %s", e)
            raise


if __name__ == "__main__":
    feature_eng = FeatureEngineering()
    feature_eng.run()
