import sys
from pathlib import Path

# Add src directory to Python path for imports
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from flask import Flask, request, jsonify
import pandas as pd
from flasgger import Swagger
import yaml
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import generate_latest, REGISTRY

# Use absolute imports from src
from src.serving.predict import Predictor
from src.common.logging_config import get_logger
from src.serving.config.serving_config import MLFLOW_TRACKING_URI, MODEL_URI, PORT

# Initialize logger
logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Add custom metrics
metrics.info('app_info', 'Application info', version='1.0.0', model='heart_disease_best_model')


# Load Swagger specification from external YAML file
swagger_spec_path = Path(__file__).parent / "config" / "spec" / "swagger_specs.yaml"
with open(swagger_spec_path, 'r') as f:
    swagger_template = yaml.safe_load(f)

# Update host dynamically
swagger_template['host'] = f"localhost:{PORT}"

# Initialize Swagger with external specification
swagger = Swagger(app, template=swagger_template)

# Global predictor instance
predictor = None


def initialize_predictor(model_uri: str = MODEL_URI):
    """
    Initialize the predictor with a model from MLflow.

    Args:
        model_uri: MLflow model URI (e.g., 'models:/heart_disease_model/Production')
                   Defaults to MODEL_URI from config
    """
    global predictor

    try:
        logger.info(f"Initializing predictor with model URI: {model_uri}")
        predictor = Predictor(pipeline_uri=model_uri, tracking_uri=MLFLOW_TRACKING_URI)
        logger.info("Predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the service is running."""
    if predictor is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded'
        }), 503

    return jsonify({
        'status': 'healthy',
        'message': 'Service is running'
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint that takes JSON input and returns predictions with confidence."""
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded. Please initialize the predictor first.'
        }), 503

    try:
        # Parse request JSON
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No data provided'
            }), 400

        # Extract instances from request
        instances = data.get('instances', None)

        if instances is None:
            # If no 'instances' key, assume the entire payload is a single instance
            instances = [data]

        if not isinstance(instances, list):
            return jsonify({
                'error': 'instances must be a list'
            }), 400

        if len(instances) == 0:
            return jsonify({
                'error': 'instances list is empty'
            }), 400

        # Convert to DataFrame
        input_df = pd.DataFrame(instances)

        logger.info(f"Received prediction request with {len(input_df)} instances")

        # Make predictions
        results = predictor.predict(input_df)

        # Format response
        predictions = []
        for idx in range(len(results)):
            pred_dict = {
                'prediction': int(results.iloc[idx]['prediction'])
            }

            if 'confidence' in results.columns and results.iloc[idx]['confidence'] is not None:
                pred_dict['confidence'] = float(results.iloc[idx]['confidence'])

            # Add individual class probabilities if available
            prob_cols = [col for col in results.columns if col.startswith('probability_class_')]
            if prob_cols:
                pred_dict['probabilities'] = {
                    f'class_{i}': float(results.iloc[idx][f'probability_class_{i}'])
                    for i in range(len(prob_cols))
                }

            predictions.append(pred_dict)

        logger.info(f"Successfully generated {len(predictions)} predictions")

        return jsonify({
            'predictions': predictions
        }), 200

    except KeyError as e:
        logger.error(f"Missing required field: {str(e)}")
        return jsonify({
            'error': f'Missing required field: {str(e)}'
        }), 400

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint to get information about the loaded model."""
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503

    return jsonify({
        'model_uri': predictor.pipeline_uri,
        'tracking_uri': predictor.tracking_uri,
        'status': 'loaded'
    }), 200

@app.route('/metrics')
def metrics_endpoint():
    return generate_latest(REGISTRY)

if __name__ == '__main__':
    # Initialize predictor on startup
    # MODEL_URI can be overridden via environment variable
    try:
        initialize_predictor()
        logger.info(f"Flask app will run on port {PORT}")
    except Exception as e:
        logger.warning(f"Failed to initialize predictor: {str(e)}")
        logger.warning("App will start but predictions will not work until model is loaded.")

    # Run Flask app
    app.run(host='0.0.0.0', port=PORT, debug=False)
