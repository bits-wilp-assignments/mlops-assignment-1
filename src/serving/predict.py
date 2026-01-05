import pandas as pd
import mlflow
import mlflow.sklearn

# Import centralized logging configuration
from src.common.logging_config import get_logger
from src.serving.config.serving_config import MLFLOW_TRACKING_URI

# Initialize logger for this module
logger = get_logger(__name__)


class Predictor:
    """
    Simple predictor class for making predictions using MLflow pipelines.
    """

    def __init__(self, pipeline_uri: str, tracking_uri: str = None):
        """
        Initialize predictor with MLflow pipeline URI.

        Args:
            pipeline_uri: MLflow pipeline URI (e.g., 'runs:/<run_id>/model' or 'models:/model_name/version')
            tracking_uri: Optional MLflow tracking URI (defaults to MLFLOW_TRACKING_URI from config)
        """
        self.pipeline_uri = pipeline_uri
        self.pipeline = None
        self.tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
        self._load_pipeline()

    def _load_pipeline(self):
        """Load pipeline (preprocessing + model) from MLflow."""
        try:
            logger.info(f"Loading pipeline from MLflow URI: {self.pipeline_uri}")
            self.pipeline = mlflow.sklearn.load_model(self.pipeline_uri)
            logger.info("Pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Error loading pipeline: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on raw input data (preprocessing is handled by the pipeline).

        Args:
            X: Raw input features as DataFrame (will be preprocessed by pipeline)

        Returns:
            DataFrame with predictions, confidence, and probabilities
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not loaded")

        logger.info(f"Making predictions for {len(X)} samples...")

        # Make predictions (pipeline handles preprocessing + prediction)
        predictions = self.pipeline.predict(X)

        # Create results dataframe
        results = pd.DataFrame({
            'prediction': predictions
        })

        # Add probabilities and confidence if model supports it
        try:
            probabilities = self.pipeline.predict_proba(X)
            # Add confidence as max probability
            results['confidence'] = probabilities.max(axis=1)
            # Add individual class probabilities
            for i in range(probabilities.shape[1]):
                results[f'probability_class_{i}'] = probabilities[:, i]
        except AttributeError:
            logger.warning("Model does not support predict_proba")
            results['confidence'] = None

        logger.info("Predictions complete.")
        return results
