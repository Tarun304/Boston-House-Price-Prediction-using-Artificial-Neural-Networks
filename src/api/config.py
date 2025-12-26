import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """API Configuration Settings"""

    # API Information
    APP_NAME: str = "Boston House Price Prediction API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "MLOps-powered API for predicting Boston house prices using ANN"

    # MLflow/DagsHub Configuration
    MODEL_NAME: str = "boston-house-price-model"
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/tkbehera304/Boston-House-Price-Prediction-using-Artificial-Neural-Networks.mlflow",
    )

    # Local configuration file
    PARAMS_PATH: str = "params.yaml"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()
