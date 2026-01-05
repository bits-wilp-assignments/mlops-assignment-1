import pytest
import sys
import os
import json
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

# Add parent directory to the path to import src as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.serving.app import app, initialize_predictor


## ----------------- Fixtures ----------------- ##
@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_predictor():
    """Create a mock predictor."""
    predictor = Mock()
    predictor.pipeline_uri = "models:/test_model@production"
    predictor.tracking_uri = "http://localhost:5050"
    return predictor


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request data."""
    return {
        "instances": [
            {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        ]
    }


@pytest.fixture
def sample_prediction_response():
    """Sample prediction response from predictor."""
    return pd.DataFrame({
        'prediction': [1],
        'confidence': [0.85],
        'probability_class_0': [0.15],
        'probability_class_1': [0.85]
    })


## ----------------- Test Health Check Endpoint ----------------- ##
class TestHealthCheck:
    """Test cases for /health endpoint."""

    def test_health_check_without_predictor(self, client):
        """Test health check when predictor is not initialized."""
        import src.serving.app as app_module
        app_module.predictor = None

        response = client.get('/health')

        assert response.status_code == 503
        data = json.loads(response.data)
        assert data['status'] == 'unhealthy'
        assert 'Model not loaded' in data['message']

    def test_health_check_with_predictor(self, client, mock_predictor):
        """Test health check when predictor is initialized."""
        import src.serving.app as app_module
        app_module.predictor = mock_predictor

        response = client.get('/health')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'Service is running' in data['message']


## ----------------- Test Predict Endpoint ----------------- ##
class TestPredict:
    """Test cases for /predict endpoint."""

    def test_predict_without_predictor(self, client, sample_prediction_request):
        """Test prediction when predictor is not initialized."""
        import src.serving.app as app_module
        app_module.predictor = None

        response = client.post('/predict',
                             data=json.dumps(sample_prediction_request),
                             content_type='application/json')

        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Model not loaded' in data['error']

    def test_predict_no_data(self, client, mock_predictor):
        """Test prediction with no data."""
        import src.serving.app as app_module
        app_module.predictor = mock_predictor

        response = client.post('/predict',
                             data='',
                             content_type='application/json')

        # Flask returns 500 when it can't parse the JSON
        # The app catches this in the general exception handler
        assert response.status_code in [400, 500]
        data = json.loads(response.data)
        assert 'error' in data

    def test_predict_empty_instances(self, client, mock_predictor):
        """Test prediction with empty instances list."""
        import src.serving.app as app_module
        app_module.predictor = mock_predictor

        response = client.post('/predict',
                             data=json.dumps({'instances': []}),
                             content_type='application/json')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'empty' in data['error'].lower()

    def test_predict_invalid_instances_type(self, client, mock_predictor):
        """Test prediction with non-list instances."""
        import src.serving.app as app_module
        app_module.predictor = mock_predictor

        response = client.post('/predict',
                             data=json.dumps({'instances': 'not a list'}),
                             content_type='application/json')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'must be a list' in data['error']

    def test_predict_success_with_instances(self, client, mock_predictor,
                                           sample_prediction_request,
                                           sample_prediction_response):
        """Test successful prediction with instances key."""
        import src.serving.app as app_module
        app_module.predictor = mock_predictor
        mock_predictor.predict.return_value = sample_prediction_response

        response = client.post('/predict',
                             data=json.dumps(sample_prediction_request),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'predictions' in data
        assert len(data['predictions']) == 1
        assert data['predictions'][0]['prediction'] == 1
        assert data['predictions'][0]['confidence'] == 0.85
        assert 'probabilities' in data['predictions'][0]

    def test_predict_success_without_instances_key(self, client, mock_predictor,
                                                   sample_prediction_response):
        """Test successful prediction without instances key."""
        import src.serving.app as app_module
        app_module.predictor = mock_predictor
        mock_predictor.predict.return_value = sample_prediction_response

        # Send data without 'instances' key
        request_data = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1
        }

        response = client.post('/predict',
                             data=json.dumps(request_data),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'predictions' in data

    def test_predict_multiple_instances(self, client, mock_predictor):
        """Test prediction with multiple instances."""
        import src.serving.app as app_module
        app_module.predictor = mock_predictor

        # Create response for multiple predictions
        multi_response = pd.DataFrame({
            'prediction': [0, 1, 1],
            'confidence': [0.92, 0.78, 0.88]
        })
        mock_predictor.predict.return_value = multi_response

        request_data = {
            "instances": [
                {"age": 50, "sex": 0, "cp": 0, "trestbps": 120, "chol": 200,
                 "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
                 "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 2},
                {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                 "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                 "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1},
                {"age": 70, "sex": 1, "cp": 2, "trestbps": 160, "chol": 280,
                 "fbs": 1, "restecg": 1, "thalach": 130, "exang": 1,
                 "oldpeak": 3.0, "slope": 2, "ca": 2, "thal": 3}
            ]
        }

        response = client.post('/predict',
                             data=json.dumps(request_data),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'predictions' in data
        assert len(data['predictions']) == 3
        assert data['predictions'][0]['prediction'] == 0
        assert data['predictions'][1]['prediction'] == 1
        assert data['predictions'][2]['prediction'] == 1

    def test_predict_missing_required_field(self, client, mock_predictor):
        """Test prediction with missing required field."""
        import src.serving.app as app_module
        app_module.predictor = mock_predictor
        mock_predictor.predict.side_effect = KeyError('age')

        request_data = {
            "instances": [
                {"sex": 1, "cp": 3}  # Missing many required fields
            ]
        }

        response = client.post('/predict',
                             data=json.dumps(request_data),
                             content_type='application/json')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Missing required field' in data['error']

    def test_predict_general_exception(self, client, mock_predictor):
        """Test prediction with general exception."""
        import src.serving.app as app_module
        app_module.predictor = mock_predictor
        mock_predictor.predict.side_effect = Exception('Unexpected error')

        request_data = {
            "instances": [
                {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                 "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                 "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}
            ]
        }

        response = client.post('/predict',
                             data=json.dumps(request_data),
                             content_type='application/json')

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Prediction failed' in data['error']


## ----------------- Test Model Info Endpoint ----------------- ##
class TestModelInfo:
    """Test cases for /model-info endpoint."""

    def test_model_info_without_predictor(self, client):
        """Test model info when predictor is not initialized."""
        import src.serving.app as app_module
        app_module.predictor = None

        response = client.get('/model-info')

        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Model not loaded' in data['error']

    def test_model_info_with_predictor(self, client, mock_predictor):
        """Test model info when predictor is initialized."""
        import src.serving.app as app_module
        app_module.predictor = mock_predictor

        response = client.get('/model-info')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['model_uri'] == "models:/test_model@production"
        assert data['tracking_uri'] == "http://localhost:5050"
        assert data['status'] == 'loaded'


## ----------------- Test Initialize Predictor ----------------- ##
class TestInitializePredictor:
    """Test cases for initialize_predictor function."""

    @patch('src.serving.app.Predictor')
    def test_initialize_predictor_success(self, mock_predictor_class):
        """Test successful predictor initialization."""
        mock_instance = Mock()
        mock_predictor_class.return_value = mock_instance

        initialize_predictor("models:/test_model@production")

        mock_predictor_class.assert_called_once()
        import src.serving.app as app_module
        assert app_module.predictor is not None

    @patch('src.serving.app.Predictor')
    def test_initialize_predictor_failure(self, mock_predictor_class):
        """Test predictor initialization failure."""
        mock_predictor_class.side_effect = Exception("Failed to load model")

        with pytest.raises(Exception) as exc_info:
            initialize_predictor("models:/invalid_model")

        assert "Failed to load model" in str(exc_info.value)
