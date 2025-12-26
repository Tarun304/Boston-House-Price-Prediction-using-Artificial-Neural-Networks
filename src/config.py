import os

import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get credentials
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# Set MLflow environment variables
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USERNAME}/Boston-House-Price-Prediction-using-Artificial-Neural-Networks.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set experiment name
mlflow.set_experiment("boston-house-price-dvc-pipeline")

print(f"MLflow configured: {MLFLOW_TRACKING_URI}")
