"""Utility functions for Streamlit app"""

import json
import logging
import tempfile
from typing import Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from keras.models import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalPredictor:
    """Load model and artifacts from MLflow for Streamlit (same as API)"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.winsorization_bounds = None
        self.params = None
        self.model_alias = None
        self.run_id = None
        self.mlflow_uri = "https://dagshub.com/tkbehera304/Boston-House-Price-Prediction-using-Artificial-Neural-Networks.mlflow"

    def load_artifacts(self) -> bool:
        """Load all artifacts from MLflow"""
        try:
            logger.info("Loading artifacts from MLflow...")

            # Configure MLflow
            mlflow.set_tracking_uri(self.mlflow_uri)

            # Load model from registry
            self._load_model_from_registry()

            # Load preprocessing artifacts
            self._load_preprocessing_artifacts()

            # Load params
            self._load_params()

            logger.info(f"✅ All artifacts loaded! Model: {self.model_alias}")
            return True

        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            return False

    def _load_model_from_registry(self):
        """Load model from MLflow Registry"""
        client = mlflow.MlflowClient()

        # Try production
        try:
            model_uri = "models:/boston-house-price-model@champion"
            model_version = client.get_model_version_by_alias(
                "boston-house-price-model", "champion"
            )
            self.run_id = model_version.run_id
            self.model = mlflow.keras.load_model(model_uri)
            self.model_alias = "champion"
            return
        except:
            pass

        # Fallback to staging
        try:
            model_uri = "models:/boston-house-price-model@challenger"
            model_version = client.get_model_version_by_alias(
                "boston-house-price-model", "challenger"
            )
            self.run_id = model_version.run_id
            self.model = mlflow.keras.load_model(model_uri)
            self.model_alias = "challenger"
            return
        except Exception as e:
            raise Exception(f"No model in registry: {e}")

    def _load_preprocessing_artifacts(self):
        """Download preprocessing artifacts from MLflow"""
        client = mlflow.MlflowClient()
        temp_dir = tempfile.mkdtemp()

        # Download scaler
        scaler_path = client.download_artifacts(self.run_id, "scaler.pkl", temp_dir)
        self.scaler = joblib.load(scaler_path)

        # Download selected features
        features_path = client.download_artifacts(
            self.run_id, "selected_features.pkl", temp_dir
        )
        self.selected_features = joblib.load(features_path)

        # Download winsorization bounds
        bounds_path = client.download_artifacts(
            self.run_id, "winsorization_bounds.json", temp_dir
        )
        with open(bounds_path, "r") as f:
            self.winsorization_bounds = json.load(f)

    def _load_params(self):
        """Load params.yaml"""
        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)

    def preprocess_input(self, features: Dict) -> np.ndarray:
        """Preprocess input (same as API)"""
        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Drop CHAS
        columns_to_drop = self.params["data_preprocessing"]["drop_columns"]
        df = df.drop(columns=columns_to_drop, errors="ignore")

        # Winsorize
        for col in df.columns:
            if col in self.winsorization_bounds:
                bounds = self.winsorization_bounds[col]
                df[col] = df[col].clip(lower=bounds["lower"], upper=bounds["upper"])

        # Reorder to match scaler
        scaler_feature_order = self.scaler.feature_names_in_
        df = df[scaler_feature_order]

        # Scale
        df_scaled = pd.DataFrame(self.scaler.transform(df), columns=df.columns)

        # Select features
        df_final = df_scaled[self.selected_features]

        return df_final.values

    def predict(self, features: Dict) -> float:
        """Make prediction"""
        X = self.preprocess_input(features)
        prediction = self.model.predict(X, verbose=0)
        predicted_price_thousands = float(prediction[0][0])
        return predicted_price_thousands * 1000  # Convert to dollars


def load_metrics():
    """Load model metrics"""
    try:
        with open("reports/metrics.json", "r") as f:
            return json.load(f)
    except:
        return None


# Feature information for UI
FEATURE_INFO = {
    "CRIM": {
        "name": "Crime Rate",
        "desc": "Per capita crime rate by town",
        "range": (0.0, 100.0),
        "default": 0.00632,
    },
    "ZN": {
        "name": "Residential Land",
        "desc": "% land zoned for large lots",
        "range": (0.0, 100.0),
        "default": 18.0,
    },
    "INDUS": {
        "name": "Non-Retail Business",
        "desc": "% non-retail business acres",
        "range": (0.0, 30.0),
        "default": 2.31,
    },
    "CHAS": {
        "name": "Charles River",
        "desc": "Bounds river (1=Yes, 0=No)",
        "range": (0, 1),
        "default": 0,
    },
    "NOX": {
        "name": "Nitric Oxide",
        "desc": "NOx concentration (ppm)",
        "range": (0.0, 1.0),
        "default": 0.538,
    },
    "RM": {
        "name": "Rooms",
        "desc": "Avg rooms per dwelling",
        "range": (3.0, 9.0),
        "default": 6.575,
    },
    "AGE": {
        "name": "Age",
        "desc": "% units built before 1940",
        "range": (0.0, 100.0),
        "default": 65.2,
    },
    "DIS": {
        "name": "Distance to Employment",
        "desc": "Distance to job centers",
        "range": (1.0, 12.0),
        "default": 4.09,
    },
    "RAD": {
        "name": "Highway Access",
        "desc": "Highway accessibility index",
        "range": (1, 24),
        "default": 1,
    },
    "TAX": {
        "name": "Property Tax",
        "desc": "Tax rate per $10,000",
        "range": (100.0, 800.0),
        "default": 296.0,
    },
    "PTRATIO": {
        "name": "Pupil-Teacher Ratio",
        "desc": "Students per teacher",
        "range": (12.0, 22.0),
        "default": 15.3,
    },
    "B": {
        "name": "Black Population",
        "desc": "1000(Bk - 0.63)^2",
        "range": (0.0, 400.0),
        "default": 396.90,
    },
    "LSTAT": {
        "name": "Lower Status %",
        "desc": "% lower status population",
        "range": (1.0, 40.0),
        "default": 4.98,
    },
}
