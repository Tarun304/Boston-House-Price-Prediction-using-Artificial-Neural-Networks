"""Model prediction logic - Load everything from MLflow/DagsHub"""

import json
import logging
import tempfile
from typing import Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml

from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handles model loading and predictions - All artifacts from MLflow"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.winsorization_bounds = None
        self.params = None
        self.model_alias = None
        self.run_id = None

    def load_artifacts(self) -> bool:
        """Load all artifacts from MLflow/DagsHub"""
        try:
            logger.info(" Loading artifacts from MLflow/DagsHub...")

            # Configure MLflow
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            logger.info(f"MLflow URI: {settings.MLFLOW_TRACKING_URI}")

            # Load model from registry and get run_id
            self._load_model_from_registry()

            # Load preprocessing artifacts from the same MLflow run
            self._load_preprocessing_artifacts()

            # Load params.yaml (local config file)
            self._load_params()

            logger.info(f"✅ All artifacts loaded successfully!")
            logger.info(f"   Model: {self.model_alias}")
            logger.info(f"   Run ID: {self.run_id}")
            logger.info(f"   Selected features: {self.selected_features}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load artifacts: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _load_model_from_registry(self):
        """Load model from MLflow Model Registry - production first, then staging"""
        client = mlflow.MlflowClient()

        # Try production model (champion)
        try:
            model_uri = f"models:/{settings.MODEL_NAME}@champion"
            logger.info(f"🔍 Attempting to load PRODUCTION model: {model_uri}")

            # Get model version details
            model_version = client.get_model_version_by_alias(
                settings.MODEL_NAME, "champion"
            )
            self.run_id = model_version.run_id

            # Load model
            self.model = mlflow.keras.load_model(model_uri)
            self.model_alias = "champion"
            logger.info(f"✅ Loaded PRODUCTION model (run: {self.run_id})")
            return

        except Exception as e:
            logger.warning(f"⚠️ No production model found: {str(e)[:150]}")

        # Fallback to staging model (challenger)
        try:
            model_uri = f"models:/{settings.MODEL_NAME}@challenger"
            logger.info(f"🔍 Attempting to load STAGING model: {model_uri}")

            # Get model version details
            model_version = client.get_model_version_by_alias(
                settings.MODEL_NAME, "challenger"
            )
            self.run_id = model_version.run_id

            # Load model
            self.model = mlflow.keras.load_model(model_uri)
            self.model_alias = "challenger"
            logger.info(f"✅ Loaded STAGING model (run: {self.run_id})")
            return

        except Exception as e:
            logger.error(f"❌ No staging model found: {str(e)[:150]}")
            raise Exception(
                f"No model found in registry with 'champion' or 'challenger' alias"
            )

    def _load_preprocessing_artifacts(self):
        """Download and load preprocessing artifacts from MLflow run"""
        try:
            client = mlflow.MlflowClient()

            # Create temp directory for downloads
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Downloading artifacts to temp dir: {temp_dir}")

            # Download scaler.pkl
            logger.info("Downloading scaler.pkl...")
            scaler_path = client.download_artifacts(self.run_id, "scaler.pkl", temp_dir)
            self.scaler = joblib.load(scaler_path)
            logger.info("✅ Scaler loaded from MLflow")

            # Download selected_features.pkl
            logger.info("Downloading selected_features.pkl...")
            features_path = client.download_artifacts(
                self.run_id, "selected_features.pkl", temp_dir
            )
            self.selected_features = joblib.load(features_path)
            logger.info(f"✅ Features loaded from MLflow: {self.selected_features}")

            # Download winsorization_bounds.json
            logger.info("Downloading winsorization_bounds.json...")
            bounds_path = client.download_artifacts(
                self.run_id, "winsorization_bounds.json", temp_dir
            )
            with open(bounds_path, "r") as f:
                self.winsorization_bounds = json.load(f)
            logger.info(
                f"✅ Winsorization bounds loaded from MLflow ({len(self.winsorization_bounds)} features)"
            )

        except Exception as e:
            logger.error(f"Failed to load preprocessing artifacts from MLflow: {e}")
            raise

    def _load_params(self):
        """Load params.yaml (local configuration file)"""
        try:
            with open(settings.PARAMS_PATH, "r") as f:
                self.params = yaml.safe_load(f)
            logger.info("✅ params.yaml loaded")
        except Exception as e:
            logger.error(f"Failed to load params.yaml: {e}")
            raise

    def preprocess_input(self, features: Dict) -> np.ndarray:
        """
        Preprocess input EXACTLY as done during training:
        1. Convert to DataFrame (13 features)
        2. Drop CHAS column (12 features)
        3. Apply winsorization using saved bounds
        4. Normalize ALL 12 features using saved scaler (MUST match column order!)
        5. Select top 5 features using saved feature list
        """
        try:
            # Step 1: Convert to DataFrame
            df = pd.DataFrame([features])
            logger.debug(f"Step 1 - Input: {df.shape}")

            # Step 2: Drop CHAS column (same as training)
            columns_to_drop = self.params["data_preprocessing"]["drop_columns"]
            df = df.drop(columns=columns_to_drop, errors="ignore")
            logger.debug(f"Step 2 - After dropping {columns_to_drop}: {df.shape}")

            # Step 3: Apply winsorization (using saved bounds from training)
            for col in df.columns:
                if col in self.winsorization_bounds:
                    bounds = self.winsorization_bounds[col]
                    df[col] = df[col].clip(lower=bounds["lower"], upper=bounds["upper"])
            logger.debug(f"Step 3 - After winsorization: {df.shape}")

            # Step 4a: Reorder columns to match scaler's expected order
            scaler_feature_order = self.scaler.feature_names_in_
            df = df[scaler_feature_order]
            logger.debug(f"Step 4a - Reordered to match scaler: {list(df.columns)}")

            # Step 4b: Normalize ALL 12 features (using saved scaler)
            df_scaled = pd.DataFrame(
                self.scaler.transform(df), columns=df.columns, index=df.index
            )
            logger.debug(f"Step 4b - After scaling: {df_scaled.shape}")

            # Step 5: Select only the 5 features selected during training
            df_final = df_scaled[self.selected_features]
            logger.debug(f"Step 5 - Final features: {list(df_final.columns)}")

            return df_final.values

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    def predict(self, features: Dict) -> float:
        """Make prediction using loaded model"""
        try:
            # Preprocess input
            X = self.preprocess_input(features)

            # Make prediction
            prediction = self.model.predict(X, verbose=0)
            predicted_price = float(prediction[0][0])

            logger.info(
                f"✅ Prediction: ${predicted_price:.2f}k (Model: {self.model_alias})"
            )

            return predicted_price

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def get_artifacts_status(self) -> dict:
        """Get status of all loaded artifacts"""
        return {
            "model": self.model is not None,
            "scaler": self.scaler is not None,
            "selected_features": self.selected_features is not None,
            "winsorization_bounds": self.winsorization_bounds is not None,
            "params": self.params is not None,
        }


# Global predictor instance
predictor = ModelPredictor()
