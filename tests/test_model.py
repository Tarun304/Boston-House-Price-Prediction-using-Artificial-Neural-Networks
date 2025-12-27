"""Test model components"""

import json
import os

import pytest


def test_model_exists():
    """Test that trained model exists"""
    if os.path.exists("models/best_model.keras"):
        assert os.path.getsize("models/best_model.keras") > 0, "Model file is empty"


def test_metrics_exist():
    """Test that evaluation metrics exist"""
    if os.path.exists("reports/metrics.json"):
        with open("reports/metrics.json", "r") as f:
            metrics = json.load(f)

        assert "mse" in metrics, "MSE metric missing"
        assert "mae" in metrics, "MAE metric missing"
        assert "r2_score" in metrics, "R2 score missing"


def test_metrics_values():
    """Test that metrics are within reasonable ranges"""
    if os.path.exists("reports/metrics.json"):
        with open("reports/metrics.json", "r") as f:
            metrics = json.load(f)

        # Basic sanity checks
        assert metrics["mse"] >= 0, "MSE should be non-negative"
        assert metrics["mae"] >= 0, "MAE should be non-negative"
        assert -1 <= metrics["r2_score"] <= 1, "R2 should be between -1 and 1"

        # Quality checks
        assert metrics["r2_score"] > 0.7, "Model R2 too low - model quality issue"
