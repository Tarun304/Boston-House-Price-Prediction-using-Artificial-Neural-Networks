"""Test FastAPI application components"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def test_import_app():
    """Test that FastAPI app can be imported"""
    try:
        from src.api.app import app

        assert app is not None
    except Exception as e:
        assert False, f"Failed to import app: {e}"


def test_pydantic_schemas():
    """Test Pydantic schema validation"""
    from src.api.schemas import HealthResponse, HouseFeatures, PredictionResponse

    # Test HouseFeatures with example data
    test_data = {
        "AGE": 65.2,
        "B": 396.90,
        "CHAS": 0,
        "CRIM": 0.00632,
        "DIS": 4.0900,
        "INDUS": 2.31,
        "LSTAT": 4.98,
        "NOX": 0.538,
        "PTRATIO": 15.3,
        "RAD": 1,
        "RM": 6.575,
        "TAX": 296.0,
        "ZN": 18.0,
    }

    features = HouseFeatures(**test_data)
    assert features.CRIM == 0.00632
    assert features.RM == 6.575

    # Test PredictionResponse
    response = PredictionResponse(predicted_price=25890.04)
    assert response.predicted_price == 25890.04

    # Test HealthResponse
    health = HealthResponse(status="healthy", model_loaded=True)
    assert health.status == "healthy"


def test_api_config():
    """Test API configuration"""
    from src.api.config import settings

    assert settings.APP_NAME is not None
    assert settings.MODEL_NAME == "boston-house-price-model"
    assert settings.MLFLOW_TRACKING_URI is not None
    assert settings.PARAMS_PATH == "params.yaml"


def test_predictor_class():
    """Test ModelPredictor class can be instantiated"""
    from src.api.predict import ModelPredictor

    predictor = ModelPredictor()
    assert predictor.model is None  # Not loaded in tests
    assert hasattr(predictor, "load_artifacts")
    assert hasattr(predictor, "predict")
