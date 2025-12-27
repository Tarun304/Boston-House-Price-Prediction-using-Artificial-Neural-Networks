"""
Promote model from staging (challenger) to production (champion)
Compare with production baseline or auto-promote if no production exists
"""

import json
import logging
import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# Load environment variables
load_dotenv()

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from src.config import MLFLOW_TRACKING_URI

# Logging configuration
logger = logging.getLogger("model_promotion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_promotion_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class ModelPromoter:
    """Handle model promotion from staging to production"""

    def __init__(self):
        """Initialize ModelPromoter with configuration"""
        self.model_name = "boston-house-price-model"
        self.min_r2_score = 0.70  # Minimum acceptable R2 for any production model
        self.improvement_threshold = (
            0.02  # Require 2% improvement to replace production
        )
        self.mlflow_uri = MLFLOW_TRACKING_URI
        self.client = None
        self.new_metrics = None
        self.prod_metrics = None
        self.old_champion_version = None

    def load_metrics(self) -> dict:
        """Load current model metrics from evaluation"""
        try:
            with open("reports/metrics.json", "r") as f:
                metrics = json.load(f)
            logger.info("Loaded metrics from reports/metrics.json")
            return metrics
        except FileNotFoundError:
            logger.error("Metrics file not found at reports/metrics.json")
            return None

    def get_production_metrics(self) -> dict:
        """Get metrics from current production model"""
        try:
            # Get production model version
            champion = self.client.get_model_version_by_alias(
                self.model_name, "champion"
            )
            run_id = champion.run_id
            self.old_champion_version = champion.version

            # Get run metrics
            run = self.client.get_run(run_id)
            metrics = run.data.metrics

            prod_metrics = {
                "r2_score": metrics.get("test_r2_score", metrics.get("r2_score", 0)),
                "mse": metrics.get("test_mse", metrics.get("mse", float("inf"))),
                "mae": metrics.get("test_mae", metrics.get("mae", float("inf"))),
                "version": champion.version,
            }

            logger.info(f"Production model found: version {champion.version}")
            return prod_metrics

        except Exception as e:
            logger.info(f"No production model found: {e}")
            self.old_champion_version = None
            return None

    def should_promote(self) -> tuple:
        """
        Decision logic for promotion

        Case 1: No production model -> Auto-promote if meets minimum threshold
        Case 2: Production exists -> Promote if new model is better by improvement threshold
        """
        new_r2 = self.new_metrics["r2_score"]

        # Case 1: No production model
        if self.prod_metrics is None:
            if new_r2 >= self.min_r2_score:
                logger.info("DECISION: No production model exists")
                logger.info(
                    f"New model R2 ({new_r2:.4f}) meets minimum threshold ({self.min_r2_score})"
                )
                return True, "first_production"
            else:
                logger.warning(
                    f"DECISION: New model R2 ({new_r2:.4f}) below minimum threshold ({self.min_r2_score})"
                )
                return False, "below_minimum"

        # Case 2: Production model exists - compare performance
        prod_r2 = self.prod_metrics["r2_score"]
        improvement = new_r2 - prod_r2

        logger.info("=" * 70)
        logger.info("MODEL COMPARISON:")
        logger.info(
            f"  Production R2: {prod_r2:.4f} (version {self.prod_metrics['version']})"
        )
        logger.info(f"  New Model R2:  {new_r2:.4f}")
        logger.info(f"  Improvement:   {improvement:+.4f} ({improvement*100:+.2f}%)")
        logger.info("=" * 70)

        if improvement >= self.improvement_threshold:
            logger.info(
                f"DECISION: New model is better by {improvement*100:.2f}% (threshold: {self.improvement_threshold*100:.0f}%)"
            )
            return True, "better_than_production"
        else:
            logger.warning(
                f"DECISION: Improvement ({improvement*100:.2f}%) below threshold ({self.improvement_threshold*100:.0f}%)"
            )
            return False, "not_better_enough"

    def retire_old_champion(self):
        """Remove champion alias from old production model and mark as retired"""
        if self.old_champion_version is not None:
            try:
                logger.info(
                    f"Retiring old production model (version {self.old_champion_version})"
                )

                # Remove champion alias from old version
                self.client.delete_registered_model_alias(self.model_name, "champion")

                # Mark as retired
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=self.old_champion_version,
                    key="deployment_status",
                    value="retired",
                )

                logger.info(f"Version {self.old_champion_version} marked as retired")

            except Exception as e:
                logger.warning(f"Failed to retire old champion: {e}")

    def promote_to_production(self) -> bool:
        """Execute model promotion logic"""
        try:
            # Load new model metrics
            self.new_metrics = self.load_metrics()
            if self.new_metrics is None:
                return False

            logger.info("=" * 70)
            logger.info("NEW MODEL METRICS:")
            logger.info(f"  R2 Score: {self.new_metrics['r2_score']:.4f}")
            logger.info(f"  MSE:      {self.new_metrics['mse']:.4f}")
            logger.info(f"  MAE:      {self.new_metrics['mae']:.4f}")
            logger.info("=" * 70)

            # Connect to MLflow
            mlflow.set_tracking_uri(self.mlflow_uri)
            self.client = MlflowClient()

            # Get production model metrics (if exists)
            self.prod_metrics = self.get_production_metrics()

            # Decision: Should we promote?
            should_promote_flag, reason = self.should_promote()

            if not should_promote_flag:
                logger.warning(f"Model NOT promoted: {reason}")
                return False

            # Get challenger version
            challenger = self.client.get_model_version_by_alias(
                self.model_name, "challenger"
            )
            version_number = challenger.version

            logger.info(f"Promoting model version {version_number} to production...")

            # IMPORTANT: Retire old champion first
            self.retire_old_champion()

            # Promote new champion
            self.client.set_registered_model_alias(
                self.model_name, "champion", version_number
            )

            # Update deployment status tag
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version_number,
                key="deployment_status",
                value="production",
            )

            logger.info(
                f"SUCCESS: Model v{version_number} is now CHAMPION (production)"
            )
            logger.info(f"Promotion reason: {reason}")

            return True

        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            raise

    def run(self):
        """Main execution method"""
        logger.info("=" * 70)
        logger.info("MODEL PROMOTION PROCESS STARTED")
        logger.info("=" * 70)

        success = self.promote_to_production()

        logger.info("=" * 70)
        if success:
            logger.info("RESULT: MODEL PROMOTED TO PRODUCTION")
        else:
            logger.info("RESULT: MODEL REMAINS IN STAGING")
        logger.info("=" * 70)

        return success


if __name__ == "__main__":
    promoter = ModelPromoter()
    promoter.run()
