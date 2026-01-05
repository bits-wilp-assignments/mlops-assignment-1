import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch, call

# Add parent directory to the path to import src as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.training.validate import (
    get_latest_model_version,
    get_previous_model_version,
    get_model_f1_score,
    add_prod_tag,
    remove_production_alias_from_previous,
    validate_model_and_promote
)


## ----------------- Fixtures ----------------- ##
@pytest.fixture
def mock_mlflow_client():
    """Create a mock MLflow client."""
    return Mock()


@pytest.fixture
def mock_model_version():
    """Create a mock model version."""
    version = Mock()
    version.version = "1"
    version.run_id = "run_123"
    version.name = "test_model"
    version.tags = {}
    return version


@pytest.fixture
def mock_model_versions():
    """Create multiple mock model versions."""
    versions = []
    for i in range(1, 4):
        version = Mock()
        version.version = str(i)
        version.run_id = f"run_{i}00"
        version.name = "test_model"
        version.tags = {}
        versions.append(version)
    return versions


@pytest.fixture
def mock_run_with_f1():
    """Create a mock MLflow run with F1 score."""
    run = Mock()
    run.data.metrics = {'test_f1_score': 0.85}
    run.info.experiment_id = "exp_1"
    run.info.run_id = "run_123"
    return run


@pytest.fixture
def mock_run_without_f1():
    """Create a mock MLflow run without F1 score."""
    run = Mock()
    run.data.metrics = {}
    run.info.experiment_id = "exp_1"
    run.info.run_id = "run_123"
    return run


## ----------------- Test Functions ----------------- ##
class TestGetLatestModelVersion:
    """Test cases for get_latest_model_version function."""

    def test_get_latest_model_version_success(self, mock_mlflow_client, mock_model_versions):
        """Test getting latest model version successfully."""
        mock_mlflow_client.search_model_versions.return_value = mock_model_versions
        
        latest = get_latest_model_version(mock_mlflow_client, "test_model")
        
        assert latest is not None
        assert latest.version == "3"  # Should return version 3 (highest)
        mock_mlflow_client.search_model_versions.assert_called_once_with("name='test_model'")

    def test_get_latest_model_version_no_versions(self, mock_mlflow_client):
        """Test when no versions exist."""
        mock_mlflow_client.search_model_versions.return_value = []
        
        latest = get_latest_model_version(mock_mlflow_client, "test_model")
        
        assert latest is None

    def test_get_latest_model_version_single_version(self, mock_mlflow_client, mock_model_version):
        """Test with single version."""
        mock_mlflow_client.search_model_versions.return_value = [mock_model_version]
        
        latest = get_latest_model_version(mock_mlflow_client, "test_model")
        
        assert latest is not None
        assert latest.version == "1"

    def test_get_latest_model_version_exception(self, mock_mlflow_client):
        """Test exception handling."""
        mock_mlflow_client.search_model_versions.side_effect = Exception("API Error")
        
        latest = get_latest_model_version(mock_mlflow_client, "test_model")
        
        assert latest is None


class TestGetPreviousModelVersion:
    """Test cases for get_previous_model_version function."""

    def test_get_previous_model_version_success(self, mock_mlflow_client, mock_model_versions):
        """Test getting previous model version successfully."""
        mock_mlflow_client.search_model_versions.return_value = mock_model_versions
        
        previous = get_previous_model_version(mock_mlflow_client, "test_model", "3")
        
        assert previous is not None
        assert previous.version == "2"  # Should return version 2 (latest before 3)

    def test_get_previous_model_version_only_one_version(self, mock_mlflow_client, mock_model_version):
        """Test when only one version exists."""
        mock_mlflow_client.search_model_versions.return_value = [mock_model_version]
        
        previous = get_previous_model_version(mock_mlflow_client, "test_model", "1")
        
        assert previous is None

    def test_get_previous_model_version_no_previous(self, mock_mlflow_client, mock_model_versions):
        """Test when current version is the only one."""
        # Return only version 1
        mock_mlflow_client.search_model_versions.return_value = [mock_model_versions[0]]
        
        previous = get_previous_model_version(mock_mlflow_client, "test_model", "1")
        
        assert previous is None

    def test_get_previous_model_version_exception(self, mock_mlflow_client):
        """Test exception handling."""
        mock_mlflow_client.search_model_versions.side_effect = Exception("API Error")
        
        previous = get_previous_model_version(mock_mlflow_client, "test_model", "2")
        
        assert previous is None


class TestGetModelF1Score:
    """Test cases for get_model_f1_score function."""

    def test_get_model_f1_score_from_parent_run(self, mock_mlflow_client, mock_model_version, mock_run_with_f1):
        """Test getting F1 score from parent run."""
        mock_mlflow_client.get_run.return_value = mock_run_with_f1
        
        f1_score = get_model_f1_score(mock_mlflow_client, mock_model_version)
        
        assert f1_score == 0.85
        mock_mlflow_client.get_run.assert_called_once_with("run_123")

    def test_get_model_f1_score_from_nested_run(self, mock_mlflow_client, mock_model_version, mock_run_without_f1):
        """Test getting F1 score from nested run when not in parent."""
        mock_mlflow_client.get_run.return_value = mock_run_without_f1
        
        # Create mock nested run with F1 score
        nested_run = Mock()
        nested_run.data.metrics = {'test_f1_score': 0.90}
        nested_run.info.run_name = "evaluation_run"
        mock_mlflow_client.search_runs.return_value = [nested_run]
        
        f1_score = get_model_f1_score(mock_mlflow_client, mock_model_version)
        
        assert f1_score == 0.90
        mock_mlflow_client.search_runs.assert_called_once()

    def test_get_model_f1_score_multiple_nested_runs(self, mock_mlflow_client, mock_model_version, mock_run_without_f1):
        """Test selecting best F1 score from multiple nested runs."""
        mock_mlflow_client.get_run.return_value = mock_run_without_f1
        
        # Create multiple nested runs with different F1 scores
        nested_run1 = Mock()
        nested_run1.data.metrics = {'test_f1_score': 0.85}
        nested_run1.info.run_name = "eval_1"
        
        nested_run2 = Mock()
        nested_run2.data.metrics = {'test_f1_score': 0.92}
        nested_run2.info.run_name = "eval_2"
        
        mock_mlflow_client.search_runs.return_value = [nested_run1, nested_run2]
        
        f1_score = get_model_f1_score(mock_mlflow_client, mock_model_version)
        
        assert f1_score == 0.92  # Should return the maximum

    def test_get_model_f1_score_not_found(self, mock_mlflow_client, mock_model_version, mock_run_without_f1):
        """Test when F1 score is not found anywhere."""
        mock_mlflow_client.get_run.return_value = mock_run_without_f1
        mock_mlflow_client.search_runs.return_value = []
        
        f1_score = get_model_f1_score(mock_mlflow_client, mock_model_version)
        
        assert f1_score is None

    def test_get_model_f1_score_exception(self, mock_mlflow_client, mock_model_version):
        """Test exception handling."""
        mock_mlflow_client.get_run.side_effect = Exception("API Error")
        
        f1_score = get_model_f1_score(mock_mlflow_client, mock_model_version)
        
        assert f1_score is None


class TestAddProdTag:
    """Test cases for add_prod_tag function."""

    def test_add_prod_tag_success(self, mock_mlflow_client):
        """Test successfully adding @prod-ready, @passed tags and 'production' alias."""
        result = add_prod_tag(mock_mlflow_client, "test_model", "1")
        
        assert result is True
        # Check that both tags are set
        assert mock_mlflow_client.set_model_version_tag.call_count == 2
        calls = [
            call("test_model", "1", "stage", "@prod-ready"),
            call("test_model", "1", "validation", "@passed")
        ]
        mock_mlflow_client.set_model_version_tag.assert_has_calls(calls)
        # Check alias is set
        mock_mlflow_client.set_registered_model_alias.assert_called_once_with(
            "test_model", "production", "1"
        )

    def test_add_prod_tag_exception(self, mock_mlflow_client):
        """Test exception handling when adding tag."""
        mock_mlflow_client.set_model_version_tag.side_effect = Exception("Tag Error")
        
        result = add_prod_tag(mock_mlflow_client, "test_model", "1")
        
        assert result is False


class TestRemoveProductionAliasFromPrevious:
    """Test cases for remove_production_alias_from_previous function."""

    def test_remove_production_alias_from_previous_version(self, mock_mlflow_client):
        """Test removing 'production' alias from previous version."""
        # Create mock model with production alias pointing to version 2
        mock_model = Mock()
        mock_model.aliases = {"production": "2"}
        mock_mlflow_client.get_registered_model.return_value = mock_model
        
        # Call with current version 3
        remove_production_alias_from_previous(mock_mlflow_client, "test_model", "3")
        
        # Should delete the old alias
        mock_mlflow_client.delete_registered_model_alias.assert_called_once_with(
            "test_model", "production"
        )

    def test_remove_production_alias_no_previous_alias(self, mock_mlflow_client):
        """Test when no previous 'production' alias exists."""
        # Create mock model with no aliases
        mock_model = Mock()
        mock_model.aliases = {}
        mock_mlflow_client.get_registered_model.return_value = mock_model
        
        remove_production_alias_from_previous(mock_mlflow_client, "test_model", "2")
        
        # Should not call delete_registered_model_alias
        mock_mlflow_client.delete_registered_model_alias.assert_not_called()

    def test_remove_production_alias_exception(self, mock_mlflow_client):
        """Test exception handling (should not raise)."""
        mock_mlflow_client.get_registered_model.side_effect = Exception("API Error")
        
        # Should not raise exception
        remove_production_alias_from_previous(mock_mlflow_client, "test_model", "2")


class TestValidateModelAndPromote:
    """Test cases for validate_model_and_promote function."""

    @patch('src.training.validate.MlflowClient')
    @patch('src.training.validate.get_latest_model_version')
    @patch('src.training.validate.get_model_f1_score')
    @patch('src.training.validate.add_prod_tag')
    def test_validate_first_version(self, mock_add_tag, mock_get_f1, mock_get_latest, mock_client_class):
        """Test validation when it's the first version (no previous version)."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        latest_version = Mock()
        latest_version.version = "1"
        latest_version.run_id = "run_1"
        mock_get_latest.return_value = latest_version
        
        mock_get_f1.return_value = 0.85
        mock_add_tag.return_value = True
        
        # Mock get_previous_model_version to return None (first version)
        with patch('src.training.validate.get_previous_model_version', return_value=None):
            result = validate_model_and_promote("test_model")
        
        assert result is True
        mock_add_tag.assert_called_once_with(mock_client, "test_model", "1")

    @patch('src.training.validate.MlflowClient')
    @patch('src.training.validate.get_latest_model_version')
    @patch('src.training.validate.get_previous_model_version')
    @patch('src.training.validate.get_model_f1_score')
    @patch('src.training.validate.add_prod_tag')
    @patch('src.training.validate.remove_production_alias_from_previous')
    def test_validate_promotion_higher_f1(self, mock_remove_alias, mock_add_tag, mock_get_f1, 
                                         mock_get_previous, mock_get_latest, mock_client_class):
        """Test promotion when new version has higher F1 score."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        latest_version = Mock()
        latest_version.version = "2"
        latest_version.run_id = "run_2"
        mock_get_latest.return_value = latest_version
        
        previous_version = Mock()
        previous_version.version = "1"
        previous_version.run_id = "run_1"
        mock_get_previous.return_value = previous_version
        
        # New version has higher F1
        mock_get_f1.side_effect = [0.90, 0.85]  # latest, then previous
        mock_add_tag.return_value = True
        
        result = validate_model_and_promote("test_model")
        
        assert result is True
        mock_remove_alias.assert_called_once_with(mock_client, "test_model", "2")
        mock_add_tag.assert_called_once_with(mock_client, "test_model", "2")

    @patch('src.training.validate.MlflowClient')
    @patch('src.training.validate.get_latest_model_version')
    @patch('src.training.validate.get_previous_model_version')
    @patch('src.training.validate.get_model_f1_score')
    @patch('src.training.validate.add_prod_tag')
    @patch('src.training.validate.remove_production_alias_from_previous')
    def test_validate_promotion_equal_f1(self, mock_remove_alias, mock_add_tag, mock_get_f1, 
                                        mock_get_previous, mock_get_latest, mock_client_class):
        """Test promotion when new version has equal F1 score."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        latest_version = Mock()
        latest_version.version = "2"
        mock_get_latest.return_value = latest_version
        
        previous_version = Mock()
        previous_version.version = "1"
        mock_get_previous.return_value = previous_version
        
        # Equal F1 scores
        mock_get_f1.side_effect = [0.85, 0.85]
        mock_add_tag.return_value = True
        
        result = validate_model_and_promote("test_model")
        
        assert result is True
        mock_remove_alias.assert_called_once_with(mock_client, "test_model", "2")
        mock_add_tag.assert_called_once_with(mock_client, "test_model", "2")

    @patch('src.training.validate.MlflowClient')
    @patch('src.training.validate.get_latest_model_version')
    @patch('src.training.validate.get_previous_model_version')
    @patch('src.training.validate.get_model_f1_score')
    def test_validate_no_promotion_lower_f1(self, mock_get_f1, mock_get_previous, 
                                           mock_get_latest, mock_client_class):
        """Test no promotion when new version has lower F1 score."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        latest_version = Mock()
        latest_version.version = "2"
        mock_get_latest.return_value = latest_version
        
        previous_version = Mock()
        previous_version.version = "1"
        mock_get_previous.return_value = previous_version
        
        # New version has lower F1
        mock_get_f1.side_effect = [0.80, 0.90]  # latest, then previous
        
        result = validate_model_and_promote("test_model")
        
        assert result is False

    @patch('src.training.validate.MlflowClient')
    @patch('src.training.validate.get_latest_model_version')
    def test_validate_no_model_versions(self, mock_get_latest, mock_client_class):
        """Test when no model versions exist."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_get_latest.return_value = None
        
        result = validate_model_and_promote("test_model")
        
        assert result is False

    @patch('src.training.validate.MlflowClient')
    @patch('src.training.validate.get_latest_model_version')
    @patch('src.training.validate.get_model_f1_score')
    def test_validate_no_f1_score(self, mock_get_f1, mock_get_latest, mock_client_class):
        """Test when F1 score cannot be retrieved."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        latest_version = Mock()
        latest_version.version = "1"
        mock_get_latest.return_value = latest_version
        
        mock_get_f1.return_value = None
        
        result = validate_model_and_promote("test_model")
        
        assert result is False

    @patch('src.training.validate.MlflowClient')
    @patch('src.training.validate.get_latest_model_version')
    @patch('src.training.validate.get_previous_model_version')
    @patch('src.training.validate.get_model_f1_score')
    @patch('src.training.validate.add_prod_tag')
    def test_validate_no_previous_f1_score(self, mock_add_tag, mock_get_f1, mock_get_previous, 
                                          mock_get_latest, mock_client_class):
        """Test when previous version F1 score cannot be retrieved."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        latest_version = Mock()
        latest_version.version = "2"
        mock_get_latest.return_value = latest_version
        
        previous_version = Mock()
        previous_version.version = "1"
        mock_get_previous.return_value = previous_version
        
        # Latest has F1, previous doesn't
        mock_get_f1.side_effect = [0.85, None]
        mock_add_tag.return_value = True
        
        result = validate_model_and_promote("test_model")
        
        # Should still promote (warning case)
        assert result is True
