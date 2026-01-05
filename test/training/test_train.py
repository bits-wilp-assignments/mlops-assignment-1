import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
import os

# Add parent directory to the path to import src as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.training.train import (
    load_raw_training_data,
    get_param_grids,
    train_pipeline_with_gridsearch,
    build_model_pipeline
)
from src.training.preprocess import build_preprocessing_pipeline


## ----------------- Fixtures ----------------- ##
@pytest.fixture
def sample_training_data():
    """Create sample training data matching heart disease schema."""
    np.random.seed(42)
    X_train = pd.DataFrame({
        'age': np.random.randint(30, 80, 100),
        'sex': np.random.randint(0, 2, 100),
        'cp': np.random.randint(1, 5, 100),
        'trestbps': np.random.randint(90, 200, 100),
        'chol': np.random.randint(120, 400, 100),
        'fbs': np.random.randint(0, 2, 100),
        'restecg': np.random.randint(0, 3, 100),
        'thalach': np.random.randint(70, 200, 100),
        'exang': np.random.randint(0, 2, 100),
        'oldpeak': np.random.rand(100) * 5,
        'slope': np.random.randint(1, 4, 100),
        'ca': np.random.randint(0, 4, 100),
        'thal': np.random.randint(3, 8, 100)
    })
    y_train = np.random.randint(0, 2, 100)
    return X_train, y_train


@pytest.fixture
def temp_training_data(tmp_path, sample_training_data):
    """Create temporary CSV files for raw training data."""
    X_train, y_train = sample_training_data

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Save as raw data (pipelines expect raw data)
    X_train.to_csv(data_dir / "X_train_raw.csv", index=False)
    pd.DataFrame(y_train, columns=['num']).to_csv(data_dir / "y_train.csv", index=False)

    return str(data_dir)


## ----------------- Test Functions ----------------- ##
class TestLoadRawTrainingData:
    """Test cases for load_raw_training_data function."""

    def test_load_raw_training_data(self, temp_training_data):
        """Test that raw training data loads correctly."""
        X_train, y_train = load_raw_training_data(temp_training_data)

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, np.ndarray)
        assert len(X_train) == 100
        assert len(y_train) == 100
        assert X_train.shape[1] == 13  # 13 features in heart disease dataset

    def test_load_raw_training_data_invalid_path(self):
        """Test that loading from invalid path raises error."""
        with pytest.raises(FileNotFoundError):
            load_raw_training_data("/invalid/path")


class TestGetParamGrids:
    """Test cases for get_param_grids function."""

    def test_get_param_grids_returns_two_dicts(self):
        """Test that get_param_grids returns two dictionaries."""
        logistic_params, rf_params = get_param_grids()

        assert isinstance(logistic_params, dict)
        assert isinstance(rf_params, dict)

    def test_logistic_params_structure(self):
        """Test that logistic regression params have expected keys."""
        logistic_params, _ = get_param_grids()

        assert 'C' in logistic_params
        assert 'solver' in logistic_params
        assert 'max_iter' in logistic_params
        assert len(logistic_params['C']) > 0

    def test_rf_params_structure(self):
        """Test that random forest params have expected keys."""
        _, rf_params = get_param_grids()

        assert 'n_estimators' in rf_params
        assert 'max_depth' in rf_params
        assert 'min_samples_split' in rf_params
        assert len(rf_params['n_estimators']) > 0


class TestTrainPipelineWithGridsearch:
    """Test cases for train_pipeline_with_gridsearch function."""

    def test_train_logistic_regression_pipeline(self, sample_training_data):
        """Test training logistic regression pipeline with grid search."""
        X_train, y_train = sample_training_data

        # Build preprocessing pipeline
        preprocessor = build_preprocessing_pipeline(
            use_scaling=True
        )

        # Build model pipeline
        model = LogisticRegression(random_state=42)
        pipeline = build_model_pipeline(preprocessor, model)
        param_grid = {'C': [0.1, 1.0], 'max_iter': [100]}

        best_pipeline, best_params, best_score = train_pipeline_with_gridsearch(
            pipeline, param_grid, X_train, y_train, "Logistic Regression"
        )

        assert best_pipeline is not None
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)
        assert 0.0 <= best_score <= 1.0
        assert hasattr(best_pipeline, 'predict')

    def test_train_random_forest_pipeline(self, sample_training_data):
        """Test training random forest pipeline with grid search."""
        X_train, y_train = sample_training_data

        # Build preprocessing pipeline
        preprocessor = build_preprocessing_pipeline(
            use_scaling=True
        )

        # Build model pipeline
        model = RandomForestClassifier(random_state=42)
        pipeline = build_model_pipeline(preprocessor, model)
        param_grid = {'n_estimators': [10, 50], 'max_depth': [3, 5]}

        best_pipeline, best_params, best_score = train_pipeline_with_gridsearch(
            pipeline, param_grid, X_train, y_train, "Random Forest"
        )

        assert best_pipeline is not None
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)
        assert 0.0 <= best_score <= 1.0
        assert hasattr(best_pipeline, 'predict')

    def test_best_params_in_param_grid(self, sample_training_data):
        """Test that best params are from the provided param grid."""
        X_train, y_train = sample_training_data

        # Build preprocessing pipeline
        preprocessor = build_preprocessing_pipeline(
            use_scaling=True
        )

        model = LogisticRegression(random_state=42)
        pipeline = build_model_pipeline(preprocessor, model)
        param_grid = {'C': [0.1, 1.0], 'max_iter': [100]}

        _, best_params, _ = train_pipeline_with_gridsearch(
            pipeline, param_grid, X_train, y_train, "Test Model"
        )

        # Parameters are prefixed with 'classifier__' in pipeline
        assert best_params['classifier__C'] in param_grid['C']
        assert best_params['classifier__max_iter'] in param_grid['max_iter']
