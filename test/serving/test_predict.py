import pytest
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
import sys
import os

# Add parent directory to path to import src as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.serving.predict import Predictor


@pytest.fixture
def sample_data():
    """Create sample input data."""
    return pd.DataFrame({
        'feature_1': [1, 2, 3, 4, 5],
        'feature_2': [5, 4, 3, 2, 1],
        'feature_3': [2, 3, 4, 5, 6]
    })


@pytest.fixture
def mlflow_pipeline_uri(tmp_path):
    """Create and log a pipeline to MLflow, return its URI."""
    # Set tracking URI to temp directory
    tracking_uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test_experiment")
    
    # Create simple training data (raw)
    X_train = pd.DataFrame({
        'feature_1': [1, 2, 3, 4, 5],
        'feature_2': [5, 4, 3, 2, 1],
        'feature_3': [2, 3, 4, 5, 6]
    })
    y_train = [0, 1, 0, 1, 0]
    
    # Import here to avoid circular imports
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    # Train and log pipeline (preprocessing + model)
    with mlflow.start_run() as run:
        # Build simple pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        mlflow.sklearn.log_model(pipeline, "model")
        run_id = run.info.run_id
    
    pipeline_uri = f"runs:/{run_id}/model"
    return pipeline_uri, tracking_uri


def test_predictor_initialization(mlflow_pipeline_uri):
    """Test Predictor initialization."""
    pipeline_uri, tracking_uri = mlflow_pipeline_uri
    
    predictor = Predictor(pipeline_uri=pipeline_uri, tracking_uri=tracking_uri)
    
    assert predictor.pipeline is not None
    assert predictor.pipeline_uri == pipeline_uri


def test_predict_returns_dataframe(mlflow_pipeline_uri, sample_data):
    """Test that predict returns a DataFrame."""
    pipeline_uri, tracking_uri = mlflow_pipeline_uri
    
    predictor = Predictor(pipeline_uri=pipeline_uri, tracking_uri=tracking_uri)
    results = predictor.predict(sample_data)
    
    assert isinstance(results, pd.DataFrame)
    assert 'prediction' in results.columns
    assert len(results) == len(sample_data)


def test_predict_with_probabilities(mlflow_pipeline_uri, sample_data):
    """Test that predict includes probabilities."""
    pipeline_uri, tracking_uri = mlflow_pipeline_uri
    
    predictor = Predictor(pipeline_uri=pipeline_uri, tracking_uri=tracking_uri)
    results = predictor.predict(sample_data)
    
    # Check for probability columns
    prob_columns = [col for col in results.columns if col.startswith('probability_class_')]
    assert len(prob_columns) > 0


def test_predict_shape(mlflow_pipeline_uri, sample_data):
    """Test that prediction output has correct shape."""
    pipeline_uri, tracking_uri = mlflow_pipeline_uri
    
    predictor = Predictor(pipeline_uri=pipeline_uri, tracking_uri=tracking_uri)
    results = predictor.predict(sample_data)
    
    assert results.shape[0] == sample_data.shape[0]
    assert 'prediction' in results.columns


def test_predict_with_confidence(mlflow_pipeline_uri, sample_data):
    """Test that predict includes confidence scores."""
    pipeline_uri, tracking_uri = mlflow_pipeline_uri
    
    predictor = Predictor(pipeline_uri=pipeline_uri, tracking_uri=tracking_uri)
    results = predictor.predict(sample_data)
    
    assert 'confidence' in results.columns
    # Confidence should be between 0 and 1
    assert (results['confidence'] >= 0).all()
    assert (results['confidence'] <= 1).all()
