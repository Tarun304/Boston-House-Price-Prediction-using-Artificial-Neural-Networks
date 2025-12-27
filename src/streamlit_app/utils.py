"""Utility functions for Streamlit app - Calls FastAPI for predictions"""

import os
from typing import Dict

import requests

# Get API URL from environment variable (for Docker) or use localhost (for local dev)
API_URL = os.getenv("API_URL", "http://localhost:8000")


def predict_via_api(features: Dict) -> float:
    """
    Call FastAPI to make prediction

    Args:
        features: Dictionary with house features

    Returns:
        Predicted price in dollars
    """
    try:
        response = requests.post(f"{API_URL}/predict", json=features, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["predicted_price"]

    except requests.exceptions.ConnectionError:
        raise Exception(
            f"Cannot connect to API at {API_URL}. Make sure FastAPI is running!"
        )
    except requests.exceptions.Timeout:
        raise Exception("API request timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"API error: {e.response.text}")
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")


def check_api_health() -> bool:
    """Check if API is accessible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


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
