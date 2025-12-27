"""Test data pipeline components"""

import os

import pandas as pd
import pytest


def test_raw_data_exists():
    """Test that raw data file exists"""
    assert os.path.exists("data/raw/boston.csv"), "Raw data not found"


def test_params_yaml_exists():
    """Test that params.yaml exists"""
    assert os.path.exists("params.yaml"), "params.yaml not found"


def test_params_yaml_structure():
    """Test params.yaml has required sections"""
    import yaml

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    assert "data_preprocessing" in params
    assert "feature_engineering" in params
    assert "model_building" in params
    assert "model_evaluation" in params


def test_interim_data_after_preprocessing():
    """Test interim data is created after preprocessing"""
    if os.path.exists("data/interim/train.csv"):
        train_df = pd.read_csv("data/interim/train.csv")
        test_df = pd.read_csv("data/interim/test.csv")

        assert train_df.shape[0] > 0, "Training data is empty"
        assert test_df.shape[0] > 0, "Test data is empty"
        assert "MEDV" in train_df.columns, "Target column missing"


def test_processed_data_after_feature_engineering():
    """Test processed data is created after feature engineering"""
    if os.path.exists("data/processed/X_train.csv"):
        X_train = pd.read_csv("data/processed/X_train.csv")
        y_train = pd.read_csv("data/processed/y_train.csv")

        assert X_train.shape[0] > 0, "X_train is empty"
        assert y_train.shape[0] > 0, "y_train is empty"
        assert X_train.shape[0] == y_train.shape[0], "X and y dimensions don't match"


def test_artifacts_exist():
    """Test that model artifacts exist"""
    if os.path.exists("models"):
        artifacts = ["scaler.pkl", "selected_features.pkl", "winsorization_bounds.json"]

        for artifact in artifacts:
            artifact_path = os.path.join("models", artifact)
            if os.path.exists(artifact_path):
                assert os.path.getsize(artifact_path) > 0, f"{artifact} is empty"
