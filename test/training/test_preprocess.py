import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import sys
import os

# Add parent directory to the path to import src as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.training.preprocess import (
    OutlierClipper,
    build_preprocessing_pipeline
)
from src.training.split import (
    load_data,
    split_features_target,
    split_train_test
)


## ----------------- Fixtures ----------------- ##
@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50, 55, 60, 100, 28],  # 100 is an outlier
        'income': [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 35000],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })


@pytest.fixture
def sample_dataframe_with_nulls():
    """Create a sample dataframe with null values."""
    return pd.DataFrame({
        'age': [25, 30, np.nan, 40, 45],
        'income': [30000, np.nan, 50000, 60000, 70000],
        'category': ['A', 'B', None, 'B', 'A'],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def numerical_data():
    """Create numerical data for outlier testing."""
    return np.array([[1, 2], [3, 4], [5, 6], [7, 8], [100, 200]])


@pytest.fixture
def temp_csv(tmp_path, sample_dataframe):
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def heart_disease_sample_dataframe():
    """Create a sample dataframe matching the heart disease schema with 14 columns."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': [63, 67, 67, 37, 41, 56, 62, 57, 63, 53],
        'sex': [1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
        'cp': [1, 4, 4, 3, 2, 2, 4, 4, 4, 4],
        'trestbps': [145, 160, 120, 130, 130, 120, 140, 120, 130, 140],
        'chol': [233, 286, 229, 250, 204, 236, 268, 354, 254, 203],
        'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        'restecg': [2, 2, 2, 0, 2, 0, 2, 0, 2, 2],
        'thalach': [150, 108, 129, 187, 172, 178, 160, 163, 147, 155],
        'exang': [0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
        'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4, 0.8, 3.6, 0.6, 1.4, 3.1],
        'slope': [3, 2, 2, 3, 1, 1, 3, 1, 2, 3],
        'ca': [0, 3, 2, 0, 0, 0, 2, 0, 1, 0],
        'thal': [6, 3, 7, 3, 3, 3, 3, 3, 7, 7],
        'num': [0, 2, 1, 0, 0, 0, 3, 0, 2, 1]
    })


@pytest.fixture
def temp_csv_heart_disease(tmp_path, heart_disease_sample_dataframe):
    """Create a temporary CSV file with heart disease schema data."""
    csv_path = tmp_path / "test_data.csv"
    heart_disease_sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def heart_disease_dataframe_with_nulls():
    """Create a sample dataframe with null values matching heart disease schema."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': [63, np.nan, 67, 37, 41],
        'sex': [1, 1, 1, 1, 0],
        'cp': [1, 4, 4, 3, 2],
        'trestbps': [145, 160, np.nan, 130, 130],
        'chol': [233, 286, 229, 250, 204],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [2, 2, 2, 0, 2],
        'thalach': [150, 108, 129, 187, 172],
        'exang': [0, 1, 1, 0, 0],
        'oldpeak': [2.3, 1.5, np.nan, 3.5, 1.4],
        'slope': [3, 2, 2, 3, 1],
        'ca': [0, 3, 2, 0, 0],
        'thal': [6, 3, 7, 3, 3],
        'num': [0, 1, 1, 0, 0]
    })


## ----------------- OutlierClipper Tests ----------------- ##
class TestOutlierClipper:
    """Test cases for the OutlierClipper class."""

    def test_outlier_clipper_initialization(self):
        """Test that OutlierClipper initializes correctly."""
        clipper = OutlierClipper()
        assert clipper.lower_bound_ is None
        assert clipper.upper_bound_ is None

    def test_outlier_clipper_fit(self, numerical_data):
        """Test that OutlierClipper calculates bounds correctly."""
        clipper = OutlierClipper()
        clipper.fit(numerical_data)

        assert clipper.lower_bound_ is not None
        assert clipper.upper_bound_ is not None
        assert len(clipper.lower_bound_) == 2  # Two columns
        assert len(clipper.upper_bound_) == 2

    def test_outlier_clipper_transform_without_fit(self, numerical_data):
        """Test that transform raises error when called before fit."""
        clipper = OutlierClipper()
        with pytest.raises(ValueError, match="OutlierClipper must be fitted before transform"):
            clipper.transform(numerical_data)

    def test_outlier_clipper_transform(self, numerical_data):
        """Test that OutlierClipper clips outliers correctly."""
        clipper = OutlierClipper()
        clipper.fit(numerical_data)
        transformed = clipper.transform(numerical_data)

        # Check that extreme values are clipped
        assert transformed[4, 0] < numerical_data[4, 0]  # 100 should be clipped
        assert transformed[4, 1] < numerical_data[4, 1]  # 200 should be clipped

        # Check that normal values remain unchanged
        np.testing.assert_array_equal(transformed[:4], numerical_data[:4])

    def test_outlier_clipper_with_dataframe(self, sample_dataframe):
        """Test that OutlierClipper preserves DataFrame structure."""
        clipper = OutlierClipper()
        age_income = sample_dataframe[['age', 'income']]

        clipper.fit(age_income)
        transformed = clipper.transform(age_income)

        # Check that output is still a DataFrame
        assert isinstance(transformed, pd.DataFrame)
        assert list(transformed.columns) == ['age', 'income']
        assert len(transformed) == len(age_income)

    def test_outlier_clipper_with_numpy_array(self, numerical_data):
        """Test that OutlierClipper works with numpy arrays."""
        clipper = OutlierClipper()
        clipper.fit(numerical_data)
        transformed = clipper.transform(numerical_data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == numerical_data.shape


## ----------------- load_data Tests ----------------- ##
class TestLoadData:
    """Test cases for the load_data function."""

    def test_load_data_success(self, temp_csv_heart_disease, heart_disease_sample_dataframe):
        """Test that load_data successfully loads a CSV file."""
        df = load_data(temp_csv_heart_disease)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(heart_disease_sample_dataframe)
        assert list(df.columns) == list(heart_disease_sample_dataframe.columns)

    def test_load_data_file_not_found(self):
        """Test that load_data raises error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_data("non_existent_file.csv")

    def test_load_data_preserves_data_types(self, temp_csv_heart_disease):
        """Test that load_data preserves correct data types."""
        df = load_data(temp_csv_heart_disease)

        assert df['age'].dtype in [np.int64, np.float64]
        assert df['trestbps'].dtype in [np.int64, np.float64]
        assert df['sex'].dtype in [np.int64, np.float64]
        assert df['num'].dtype in [np.int64, np.float64]


## ----------------- split_features_target Tests ----------------- ##
class TestSplitFeaturesTarget:
    """Test cases for the split_features_target function."""

    def test_split_features_target_basic(self, sample_dataframe):
        """Test basic splitting of features and target."""
        X, y = split_features_target(sample_dataframe, 'target')

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert 'target' not in X.columns
        assert len(X) == len(y)
        assert len(X) == len(sample_dataframe)

    def test_split_features_target_columns(self, sample_dataframe):
        """Test that X contains all columns except target."""
        X, y = split_features_target(sample_dataframe, 'target')

        expected_columns = [col for col in sample_dataframe.columns if col != 'target']
        assert list(X.columns) == expected_columns

    def test_split_features_target_values(self, sample_dataframe):
        """Test that y contains the correct target values."""
        X, y = split_features_target(sample_dataframe, 'target')

        pd.testing.assert_series_equal(y, sample_dataframe['target'], check_names=False)

    def test_split_features_target_invalid_column(self, sample_dataframe):
        """Test error handling for invalid target column."""
        with pytest.raises(KeyError):
            split_features_target(sample_dataframe, 'non_existent_column')


## ----------------- split_train_test Tests ----------------- ##
class TestSplitTrainTest:
    """Test cases for the split_train_test function."""

    def test_split_train_test_basic(self, sample_dataframe):
        """Test basic train-test split."""
        X, y = split_features_target(sample_dataframe, 'target')
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_split_train_test_size(self, sample_dataframe):
        """Test that test_size parameter works correctly."""
        X, y = split_features_target(sample_dataframe, 'target')
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.3, random_state=42)

        expected_test_size = int(len(X) * 0.3)
        assert len(X_test) == expected_test_size or len(X_test) == expected_test_size + 1

    def test_split_train_test_stratified(self, sample_dataframe):
        """Test stratified sampling."""
        X, y = split_features_target(sample_dataframe, 'target')
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, test_size=0.2, stratified_sample=True, random_state=42
        )

        # Check that class distribution is maintained
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        overall_ratio = y.mean()

        # Ratios should be similar (within tolerance)
        assert abs(train_ratio - overall_ratio) < 0.2
        assert abs(test_ratio - overall_ratio) < 0.2

    def test_split_train_test_reproducibility(self, sample_dataframe):
        """Test that random_state ensures reproducibility."""
        X, y = split_features_target(sample_dataframe, 'target')

        X_train1, X_test1, y_train1, y_test1 = split_train_test(X, y, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = split_train_test(X, y, random_state=42)

        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)

    def test_split_train_test_non_stratified(self, sample_dataframe):
        """Test non-stratified sampling."""
        X, y = split_features_target(sample_dataframe, 'target')
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, test_size=0.2, stratified_sample=False, random_state=42
        )

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)


## ----------------- build_preprocessing_pipeline Tests ----------------- ##
class TestBuildPreprocessingPipeline:
    """Test cases for the build_preprocessing_pipeline function."""

    def test_build_pipeline_basic(self):
        """Test basic pipeline construction."""
        pipeline = build_preprocessing_pipeline()

        assert isinstance(pipeline, ColumnTransformer)
        assert len(pipeline.transformers) == 2

    def test_build_pipeline_with_scaling(self, heart_disease_sample_dataframe):
        """Test pipeline with scaling enabled."""
        X, y = split_features_target(heart_disease_sample_dataframe, 'num')

        pipeline = build_preprocessing_pipeline(
            use_scaling=True
        )

        # Fit and transform
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed is not None
        assert X_transformed.shape[0] == len(X)

    def test_build_pipeline_without_scaling(self, heart_disease_sample_dataframe):
        """Test pipeline without scaling."""
        X, y = split_features_target(heart_disease_sample_dataframe, 'num')

        pipeline = build_preprocessing_pipeline(
            use_scaling=False
        )

        X_transformed = pipeline.fit_transform(X)
        assert X_transformed is not None
        assert X_transformed.shape[0] == len(X)

    def test_build_pipeline_with_pca(self, heart_disease_sample_dataframe):
        """Test pipeline with PCA enabled."""
        X, y = split_features_target(heart_disease_sample_dataframe, 'num')

        pipeline = build_preprocessing_pipeline(
            use_pca=True,
            n_components=1
        )

        X_transformed = pipeline.fit_transform(X)
        assert X_transformed is not None
        # With PCA(n_components=1), numerical features are reduced to 1 dimension
        # Plus one-hot encoded categorical features

    def test_build_pipeline_handles_missing_values(self, heart_disease_dataframe_with_nulls):
        """Test that pipeline handles missing values correctly."""
        X, y = split_features_target(heart_disease_dataframe_with_nulls, 'num')

        pipeline = build_preprocessing_pipeline()

        # Should not raise error with missing values
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed is not None
        # Check no NaN values in output
        assert not np.isnan(X_transformed).any()

    def test_build_pipeline_empty_categorical(self, heart_disease_sample_dataframe):
        """Test pipeline with no categorical features."""
        X, y = split_features_target(heart_disease_sample_dataframe, 'num')

        pipeline = build_preprocessing_pipeline()

        X_transformed = pipeline.fit_transform(X)
        assert X_transformed is not None

    def test_build_pipeline_empty_numerical(self, heart_disease_sample_dataframe):
        """Test pipeline with no numerical features."""
        X, y = split_features_target(heart_disease_sample_dataframe, 'num')

        pipeline = build_preprocessing_pipeline()

        X_transformed = pipeline.fit_transform(X)
        assert X_transformed is not None


## ----------------- Integration Tests ----------------- ##
class TestIntegration:
    """Integration tests for the complete preprocessing workflow."""

    def test_full_preprocessing_workflow(self, heart_disease_sample_dataframe):
        """Test the complete preprocessing workflow."""
        # Split features and target
        X, y = split_features_target(heart_disease_sample_dataframe, 'num')

        # Split train-test (use non-stratified for small test dataset)
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42, stratified_sample=False)

        # Build pipeline
        pipeline = build_preprocessing_pipeline(
            use_scaling=True
        )

        # Fit on training data
        X_train_transformed = pipeline.fit_transform(X_train)

        # Transform test data
        X_test_transformed = pipeline.transform(X_test)

        assert X_train_transformed is not None
        assert X_test_transformed is not None
        assert X_train_transformed.shape[0] == len(X_train)
        assert X_test_transformed.shape[0] == len(X_test)
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]

    def test_workflow_with_pca(self, heart_disease_sample_dataframe):
        """Test preprocessing workflow with PCA."""
        X, y = split_features_target(heart_disease_sample_dataframe, 'num')
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42, stratified_sample=False)

        pipeline = build_preprocessing_pipeline(
            use_scaling=True,
            use_pca=True,
            n_components=1
        )

        X_train_transformed = pipeline.fit_transform(X_train)
        X_test_transformed = pipeline.transform(X_test)

        assert X_train_transformed is not None
        assert X_test_transformed is not None
