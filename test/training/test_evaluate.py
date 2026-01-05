import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add parent directory to the path to import src as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.training.evaluate import (
    load_test_data,
    load_pipeline,
    evaluate_pipeline,
    plot_confusion_matrix,
    plot_roc_curve
)


## ----------------- Fixtures ----------------- ##
@pytest.fixture
def sample_test_data():
    """Create sample test data."""
    np.random.seed(42)
    X_test = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50),
        'feature3': np.random.rand(50)
    })
    y_test = np.random.randint(0, 2, 50)
    return X_test, y_test


@pytest.fixture
def temp_test_data(tmp_path, sample_test_data):
    """Create temporary CSV files for raw test data."""
    X_test, y_test = sample_test_data

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Save as raw data (pipelines expect raw data)
    X_test.to_csv(data_dir / "X_test_raw.csv", index=False)
    pd.DataFrame(y_test, columns=['target']).to_csv(data_dir / "y_test.csv", index=False)

    return str(data_dir)


@pytest.fixture
def trained_pipeline(sample_test_data):
    """Create and train a simple pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_test, y_test = sample_test_data
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=200))
    ])
    pipeline.fit(X_test, y_test)  # Training on test data for simplicity
    return pipeline


@pytest.fixture
def temp_pipeline_file(tmp_path, trained_pipeline):
    """Save trained pipeline to temporary file."""
    pipeline_path = tmp_path / "test_pipeline.pkl"
    joblib.dump(trained_pipeline, pipeline_path)
    return str(pipeline_path)


## ----------------- Test Functions ----------------- ##
class TestLoadTestData:
    """Test cases for load_test_data function."""

    def test_load_test_data_raw(self, temp_test_data):
        """Test that raw test data loads correctly."""
        X_test, y_test = load_test_data(temp_test_data, use_raw=True)

        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, np.ndarray)
        assert len(X_test) == 50
        assert len(y_test) == 50
        assert X_test.shape[1] == 3  # 3 features

    def test_load_test_data_invalid_path(self):
        """Test that loading from invalid path raises error."""
        with pytest.raises(FileNotFoundError):
            load_test_data("/invalid/path", use_raw=True)


class TestLoadPipeline:
    """Test cases for load_pipeline function."""

    def test_load_pipeline(self, temp_pipeline_file):
        """Test that pipeline loads correctly."""
        pipeline = load_pipeline(temp_pipeline_file)

        assert pipeline is not None
        assert hasattr(pipeline, 'predict')
        assert hasattr(pipeline, 'predict_proba')

    def test_load_pipeline_invalid_path(self):
        """Test that loading from invalid path raises error."""
        with pytest.raises(FileNotFoundError):
            load_pipeline("/invalid/path/pipeline.pkl")


class TestEvaluatePipeline:
    """Test cases for evaluate_pipeline function."""

    def test_evaluate_pipeline_returns_metrics_and_cm(self, trained_pipeline, sample_test_data):
        """Test that evaluate_pipeline returns metrics, confusion matrix, and probabilities."""
        X_test, y_test = sample_test_data
        metrics, cm, y_pred_proba = evaluate_pipeline(trained_pipeline, X_test, y_test, "Test Pipeline")

        assert isinstance(metrics, dict)
        assert isinstance(cm, np.ndarray)
        assert isinstance(y_pred_proba, np.ndarray)

    def test_evaluate_pipeline_metrics_structure(self, trained_pipeline, sample_test_data):
        """Test that metrics dictionary has expected keys."""
        X_test, y_test = sample_test_data
        metrics, _, _ = evaluate_pipeline(trained_pipeline, X_test, y_test, "Test Pipeline")

        expected_keys = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1_score', 'test_roc_auc']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (float, np.floating))
            assert 0.0 <= metrics[key] <= 1.0

    def test_evaluate_pipeline_confusion_matrix_shape(self, trained_pipeline, sample_test_data):
        """Test that confusion matrix has correct shape."""
        X_test, y_test = sample_test_data
        _, cm, _ = evaluate_pipeline(trained_pipeline, X_test, y_test, "Test Pipeline")

        n_classes = len(np.unique(y_test))
        assert cm.shape == (n_classes, n_classes)

    def test_evaluate_pipeline_with_random_forest(self, sample_test_data):
        """Test evaluation with Random Forest pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_test, y_test = sample_test_data
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        pipeline.fit(X_test, y_test)

        metrics, cm, y_pred_proba = evaluate_pipeline(pipeline, X_test, y_test, "Random Forest")

        assert isinstance(metrics, dict)
        assert isinstance(cm, np.ndarray)
        assert isinstance(y_pred_proba, np.ndarray)
        assert 'test_roc_auc' in metrics


class TestPlotConfusionMatrix:
    """Test cases for plot_confusion_matrix function."""

    def test_plot_confusion_matrix_creates_file(self, sample_test_data):
        """Test that confusion matrix plot is created."""
        _, y_test = sample_test_data
        y_pred = y_test.copy()  # Perfect prediction for testing
        cm = confusion_matrix(y_test, y_pred)

        # Create test plots directory in project
        test_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(test_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Change to plots directory for plot saving
        original_dir = os.getcwd()
        os.chdir(plots_dir)

        try:
            plot_path = plot_confusion_matrix(cm, "Test Model", output_dir=".")

            assert os.path.exists(plot_path)
            assert plot_path.endswith('.png')
            assert 'confusion_matrix' in plot_path
        finally:
            os.chdir(original_dir)

    def test_plot_confusion_matrix_filename_format(self, sample_test_data):
        """Test that plot filename is correctly formatted."""
        _, y_test = sample_test_data
        cm = confusion_matrix(y_test, y_test)

        # Create test plots directory in project
        test_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(test_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Change to plots directory
        original_dir = os.getcwd()
        os.chdir(plots_dir)

        try:
            plot_path = plot_confusion_matrix(cm, "Logistic Regression", output_dir=".")

            assert 'logistic_regression' in plot_path.lower()
            assert plot_path.endswith('.png')
        finally:
            os.chdir(original_dir)


class TestPlotROCCurve:
    """Test cases for plot_roc_curve function."""

    def test_plot_roc_curve_creates_file(self, trained_pipeline, sample_test_data):
        """Test that ROC curve plot is created."""
        X_test, y_test = sample_test_data
        y_pred_proba = trained_pipeline.predict_proba(X_test)

        # Create test plots directory in project
        test_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(test_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Change to plots directory for plot saving
        original_dir = os.getcwd()
        os.chdir(plots_dir)

        try:
            plot_path = plot_roc_curve(y_test, y_pred_proba, "Test Model", output_dir=".")

            assert os.path.exists(plot_path)
            assert plot_path.endswith('.png')
            assert 'roc_curve' in plot_path
        finally:
            os.chdir(original_dir)

    def test_plot_roc_curve_filename_format(self, trained_pipeline, sample_test_data):
        """Test that ROC curve filename is correctly formatted."""
        X_test, y_test = sample_test_data
        y_pred_proba = trained_pipeline.predict_proba(X_test)

        # Create test plots directory in project
        test_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(test_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Change to plots directory
        original_dir = os.getcwd()
        os.chdir(plots_dir)

        try:
            plot_path = plot_roc_curve(y_test, y_pred_proba, "Random Forest", output_dir=".")

            assert 'random_forest' in plot_path.lower()
            assert plot_path.endswith('.png')
        finally:
            os.chdir(original_dir)
