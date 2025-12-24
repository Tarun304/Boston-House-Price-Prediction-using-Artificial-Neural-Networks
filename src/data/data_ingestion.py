import logging
import os

import pandas as pd
import yaml
from sklearn.datasets import fetch_openml

# Logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class DataIngestion:
    def __init__(self, params_path: str = "params.yaml"):
        """Initialize DataIngestion with parameters."""
        self.params = self.load_params(params_path)

    def load_params(self, params_path: str) -> dict:
        """Load parameters from YAML file."""
        try:
            with open(params_path, "r") as file:
                params = yaml.safe_load(file)
            logger.debug("Parameters retrieved from %s", params_path)
            return params
        except FileNotFoundError:
            logger.error("File not found: %s", params_path)
            raise
        except yaml.YAMLError as e:
            logger.error("YAML error: %s", e)
            raise

    def fetch_data(self) -> pd.DataFrame:
        """Fetch Boston Housing dataset from OpenML."""
        try:
            logger.info("Fetching Boston Housing dataset from OpenML...")
            boston = fetch_openml(name="boston", version=1, as_frame=True)

            # Combine features and target
            data = pd.DataFrame(boston.data, columns=boston.feature_names)
            target = pd.Series(boston.target, name="MEDV")
            df = pd.concat([data, target], axis=1)

            logger.info("Dataset fetched successfully. Shape: %s", df.shape)
            return df
        except Exception as e:
            logger.error("Failed to fetch data: %s", e)
            raise

    def save_raw_data(self, df: pd.DataFrame, output_path: str = "data/raw") -> None:
        """Save raw data to CSV."""
        try:
            os.makedirs(output_path, exist_ok=True)
            file_path = os.path.join(output_path, "boston.csv")
            df.to_csv(file_path, index=False)
            logger.info("Raw data saved to %s", file_path)
        except Exception as e:
            logger.error("Failed to save raw data: %s", e)
            raise

    def run(self):
        """Execute the data ingestion pipeline."""
        try:
            logger.info("Starting data ingestion...")
            df = self.fetch_data()
            self.save_raw_data(df)
            logger.info("Data ingestion completed successfully")
        except Exception as e:
            logger.error("Data ingestion failed: %s", e)
            raise


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()
