import logging
import os

import numpy as np
import pandas as pd
import yaml
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split

# Logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class DataPreprocessing:
    def __init__(self, params_path: str = "params.yaml"):
        """Initialize DataPreprocessing with parameters."""
        self.params = self.load_params(params_path)
        self.config = self.params["data_preprocessing"]

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

    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV."""
        try:
            input_path = self.config["raw_data_file"]
            df = pd.read_csv(input_path)
            logger.info("Data loaded from %s. Shape: %s", input_path, df.shape)
            return df
        except Exception as e:
            logger.error("Failed to load data: %s", e)
            raise

    def convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numerical columns to numeric type and drop NaN."""
        try:
            num_feat = [
                "CRIM",
                "ZN",
                "INDUS",
                "CHAS",
                "NOX",
                "RM",
                "AGE",
                "DIS",
                "RAD",
                "TAX",
                "PTRATIO",
                "B",
                "LSTAT",
            ]

            df[num_feat] = df[num_feat].apply(pd.to_numeric, errors="coerce")
            df.dropna(subset=num_feat, inplace=True)

            logger.info(
                "Converted columns to numeric. Shape after dropping NaN: %s", df.shape
            )
            return df
        except Exception as e:
            logger.error("Failed to convert to numeric: %s", e)
            raise

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop specified columns from params."""
        try:
            columns = self.config["drop_columns"]
            df.drop(columns=columns, axis=1, inplace=True)
            logger.info("Dropped columns: %s", columns)
            return df
        except Exception as e:
            logger.error("Failed to drop columns: %s", e)
            raise

    def treat_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Winsorization to treat outliers."""
        try:
            limits = self.config["winsorize_limits"]
            cols = list(df.columns)
            for col in cols:
                if col in df.select_dtypes(include=np.number).columns:
                    df[col] = winsorize(df[col], limits=limits, inclusive=(True, True))

            logger.info(
                "Outlier treatment completed using Winsorization with limits %s", limits
            )
            return df
        except Exception as e:
            logger.error("Failed to treat outliers: %s", e)
            raise

    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train and test sets."""
        try:
            test_size = self.config["test_size"]
            random_state = self.config["random_state"]

            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state
            )

            logger.info(
                "Data split completed. Train shape: %s, Test shape: %s",
                train_df.shape,
                test_df.shape,
            )
            return train_df, test_df
        except Exception as e:
            logger.error("Failed to split data: %s", e)
            raise

    def save_interim_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Save interim train and test data (before normalization)."""
        try:
            output_path = self.config["interim_data_path"]
            os.makedirs(output_path, exist_ok=True)

            train_path = os.path.join(output_path, "train.csv")
            test_path = os.path.join(output_path, "test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info("Interim data saved to %s", output_path)
        except Exception as e:
            logger.error("Failed to save interim data: %s", e)
            raise

    def run(self):
        """Execute the data preprocessing pipeline."""
        try:
            logger.info("Starting data preprocessing...")

            # Load raw data
            df = self.load_data()

            # Preprocessing steps
            df = self.convert_to_numeric(df)
            df = self.drop_columns(df)
            df = self.treat_outliers(df)

            # Split into train/test BEFORE normalization
            train_df, test_df = self.split_data(df)

            # Save interim data
            self.save_interim_data(train_df, test_df)

            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error("Data preprocessing failed: %s", e)
            raise


if __name__ == "__main__":
    preprocessing = DataPreprocessing()
    preprocessing.run()
