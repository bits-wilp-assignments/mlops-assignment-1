import argparse
import mlflow
from mlflow.tracking import MlflowClient

# Import centralized logging configuration
from src.common.logging_config import get_logger
from src.training.config.training_config import (
    MLFLOW_TRACKING_URI, VALIDATION_METRIC, PROD_READY_TAG, PASSED_TAG,
    PRODUCTION_ALIAS, STAGE_TAG_KEY, VALIDATION_TAG_KEY,
    REGISTERED_MODEL_NAME
)

# Initialize logger for this module
logger = get_logger(__name__)


## ----------------- Validation Functions ----------------- ##
def get_latest_model_version(client: MlflowClient, model_name: str):
    """Get the latest version of a registered model."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            logger.warning(f"No versions found for model '{model_name}'")
            return None

        # Sort by version number (descending) to get the latest
        latest_version = max(versions, key=lambda v: int(v.version))
        return latest_version
    except Exception as e:
        logger.error(f"Error getting latest model version: {e}")
        return None


def get_previous_model_version(client: MlflowClient, model_name: str, current_version: str):
    """Get the previous version of a registered model (before the current version)."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if len(versions) <= 1:
            logger.info(f"Only one version exists for model '{model_name}' - no previous version to compare")
            return None

        # Filter out current version and get the latest among remaining
        previous_versions = [v for v in versions if v.version != current_version]
        if not previous_versions:
            return None

        previous_version = max(previous_versions, key=lambda v: int(v.version))
        return previous_version
    except Exception as e:
        logger.error(f"Error getting previous model version: {e}")
        return None


def get_model_f1_score(client: MlflowClient, model_version):
    """Extract F1 score from a model version's run metrics."""
    try:
        run_id = model_version.run_id
        run = client.get_run(run_id)

        # Try to get F1 score from parent run metrics first
        f1_score = run.data.metrics.get(VALIDATION_METRIC)

        if f1_score is not None:
            logger.info(f"Model version {model_version.version}: F1 Score = {f1_score:.4f} (from parent run)")
            return f1_score

        # If not found in parent run, search nested runs for evaluation metrics
        logger.info(f"F1 score not found in parent run, searching nested runs...")
        experiment_id = run.info.experiment_id

        # Search for nested runs with this parent
        nested_runs = client.search_runs(
            [experiment_id],
            filter_string=f'tags.mlflow.parentRunId="{run_id}"'
        )

        # Look for evaluation runs that contain test_f1_score
        best_f1 = None
        for nested_run in nested_runs:
            if VALIDATION_METRIC in nested_run.data.metrics:
                nested_f1 = nested_run.data.metrics[VALIDATION_METRIC]
                logger.info(f"Found F1 score in nested run '{nested_run.info.run_name}': {nested_f1:.4f}")
                # Take the maximum F1 score if multiple evaluation runs exist
                if best_f1 is None or nested_f1 > best_f1:
                    best_f1 = nested_f1

        if best_f1 is not None:
            logger.info(f"Model version {model_version.version}: Best F1 Score = {best_f1:.4f} (from nested runs)")
            return best_f1

        logger.warning(f"F1 score not found for run {run_id}, version {model_version.version}")
        return None

    except Exception as e:
        logger.error(f"Error getting F1 score for model version {model_version.version}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error getting F1 score for model version {model_version.version}: {e}")
        return None


def add_prod_tag(client: MlflowClient, model_name: str, version: str):
    """Add @prod-ready and @passed tags to a model version and set 'production' alias."""
    try:
        client.set_model_version_tag(model_name, version, STAGE_TAG_KEY, PROD_READY_TAG)
        logger.info(f"✓ Added '{PROD_READY_TAG}' tag to model '{model_name}' version {version}")

        client.set_model_version_tag(model_name, version, VALIDATION_TAG_KEY, PASSED_TAG)
        logger.info(f"✓ Added '{PASSED_TAG}' tag to model '{model_name}' version {version}")

        # Set 'production' alias for this version
        client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, version)
        logger.info(f"✓ Set '{PRODUCTION_ALIAS}' alias to model '{model_name}' version {version}")
        return True
    except Exception as e:
        logger.error(f"Error adding tags or setting alias: {e}")
        return False


def remove_production_alias_from_previous(client: MlflowClient, model_name: str, current_version: str):
    """Remove 'production' alias from previous versions (tags remain unchanged)."""
    try:
        # Get registered model to check current aliases
        model = client.get_registered_model(model_name)

        # Check if production alias exists and points to a different version
        if hasattr(model, 'aliases') and PRODUCTION_ALIAS in model.aliases:
            current_alias_version = model.aliases[PRODUCTION_ALIAS]
            if current_alias_version != current_version:
                # Delete the old alias (it will be reassigned to new version)
                client.delete_registered_model_alias(model_name, PRODUCTION_ALIAS)
                logger.info(f"Removed '{PRODUCTION_ALIAS}' alias from version {current_alias_version}")
    except Exception as e:
        logger.warning(f"Error removing production alias from previous version: {e}")


def validate_model_and_promote(model_name: str):
    """
    Validate the latest model version by comparing F1 score with previous version.
    Add @prod-ready tag and 'production' alias if F1 score is higher or equal, or if it's the first version.
    """
    # Initialize MLflow client
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    logger.info("="*60)
    logger.info("Starting Model Validation")
    logger.info("="*60)

    # Get latest model version
    latest_version = get_latest_model_version(client, model_name)
    if not latest_version:
        logger.error(f"No model versions found for '{model_name}'. Cannot validate.")
        return False

    logger.info(f"\nLatest model version: {latest_version.version}")
    logger.info(f"Run ID: {latest_version.run_id}")

    # Get F1 score of latest version
    latest_f1 = get_model_f1_score(client, latest_version)
    if latest_f1 is None:
        logger.error("Cannot proceed with validation - F1 score not found for latest version")
        return False

    # Get previous model version
    previous_version = get_previous_model_version(client, model_name, latest_version.version)

    # Case 1: First version - No validation needed, just add @prod-ready tag and 'production' alias
    if previous_version is None:
        logger.info("\n" + "="*60)
        logger.info("FIRST VERSION DETECTED - No previous version to compare")
        logger.info("="*60)
        logger.info(f"Adding '@prod-ready' tag to version {latest_version.version} (F1 Score: {latest_f1:.4f})")

        success = add_prod_tag(client, model_name, latest_version.version)
        if success:
            logger.info("\n✓ Model validation complete - First version promoted to @prod-ready")
        return success

    # Case 2: Compare with previous version
    logger.info(f"\nPrevious model version: {previous_version.version}")
    previous_f1 = get_model_f1_score(client, previous_version)

    if previous_f1 is None:
        logger.warning("Cannot get F1 score for previous version - proceeding with promotion")
        success = add_prod_tag(client, model_name, latest_version.version)
        return success

    # Compare F1 scores
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    logger.info(f"Previous version {previous_version.version}: F1 Score = {previous_f1:.4f}")
    logger.info(f"Latest version {latest_version.version}:   F1 Score = {latest_f1:.4f}")
    logger.info(f"Difference: {latest_f1 - previous_f1:+.4f}")

    # Validation logic: promote if F1 score is >= previous
    if latest_f1 >= previous_f1:
        logger.info("\n✓ VALIDATION PASSED - New model F1 score is >= previous model")
        logger.info(f"Promoting version {latest_version.version} to @prod-ready")

        # Remove 'production' alias from previous version (keeping @prod-ready tag)
        remove_production_alias_from_previous(client, model_name, latest_version.version)

        # Add @prod-ready and @passed tags, plus 'production' alias to new version
        success = add_prod_tag(client, model_name, latest_version.version)
        if success:
            logger.info("\n✓ Model validation complete - New version promoted to @prod-ready")
        return success
    else:
        logger.warning("\n✗ VALIDATION FAILED - New model F1 score is lower than previous model")
        logger.warning(f"Version {latest_version.version} will NOT be promoted to @prod-ready")
        logger.info(f"Current @prod-ready version remains: {previous_version.version}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate model and promote to production")
    parser.add_argument("--model_name", type=str, default=REGISTERED_MODEL_NAME,
                       help="Name of the registered model in MLflow")

    args = parser.parse_args()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Run validation
    success = validate_model_and_promote(args.model_name)

    if success:
        logger.info("\n" + "="*60)
        logger.info("Model validation and promotion completed successfully!")
        logger.info("="*60)
    else:
        logger.error("\n" + "="*60)
        logger.error("Model validation failed or promotion was not performed")
        logger.error("="*60)


if __name__ == "__main__":
    main()
