# Heart Disease Prediction - MLOps Project

A complete MLOps pipeline for heart disease prediction using machine learning. This project includes data preprocessing, model training with MLflow tracking, model evaluation, and deployment-ready model serving with Flask API.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup Guide](#setup-guide)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Training Pipeline](#2-training-pipeline)
  - [3. Model Serving](#3-model-serving)
- [API Usage](#api-usage)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Monitoring](#monitoring)

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting heart disease based on patient health metrics. It uses the Cleveland Heart Disease dataset and trains both Logistic Regression and Random Forest models with hyperparameter tuning.

**Key Technologies:**
- **ML Frameworks:** scikit-learn, pandas, numpy
- **Experiment Tracking:** MLflow
- **API Framework:** Flask
- **API Documentation:** Swagger/Flasgger
- **Monitoring:** Prometheus
- **Containerization:** Docker
- **Orchestration:** Kubernetes with Helm
- **Testing:** pytest

## Features

- Automated ML pipeline with data splitting, training, and evaluation
- MLflow experiment tracking and model registry
- GridSearchCV for hyperparameter optimization
- RESTful API with Swagger documentation
- Prometheus metrics for monitoring
- Health checks and model information endpoints
- Docker containerization
- Kubernetes deployment with Helm charts
- Horizontal Pod Autoscaling (HPA)
- Comprehensive unit tests

## Project Structure

```
mlops-assignment-1/
├── src/
│   ├── training/              # Training pipeline modules
│   │   ├── split.py           # Data splitting
│   │   ├── train.py           # Model training
│   │   └── evaluate.py        # Model evaluation
│   ├── serving/               # Model serving modules
│   │   ├── app.py             # Flask application
│   │   ├── predict.py         # Prediction logic
│   │   └── config/            # Configuration files
│   └── common/                # Shared utilities
├── data/                      # Data directory
│   └── processed/             # Processed datasets
├── models/                    # Saved models
├── mlruns/                    # MLflow tracking data
├── helm-chart/                # Kubernetes Helm charts
├── test/                      # Unit tests
├── notebooks/                 # Jupyter notebooks for EDA
├── requirements-train.txt     # Training dependencies
├── requirements-serving.txt   # Serving dependencies
├── run_pipeline.sh            # Training pipeline script
├── prediction_client.py       # Example API client
└── Dockerfile                 # Docker configuration
```

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Docker (for containerization)
- Kubernetes cluster (for K8s deployment)
- Helm 3.x (for K8s deployment)

## Setup Guide

### 1. Environment Setup

#### Clone the Repository

```bash
git clone <repository-url>
cd mlops-assignment-1
```

#### Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

#### Install Dependencies

For training:
```bash
pip install -r requirements-train.txt
```

For serving (if training separately):
```bash
pip install -r requirements-serving.txt
```

### 2. Training Pipeline

The training pipeline consists of three main steps: data splitting, model training, and evaluation.

#### Quick Start - Run Complete Pipeline

```bash
# Make the script executable
chmod +x run_pipeline.sh

# Run the complete pipeline
./run_pipeline.sh
```

#### Step-by-Step Execution

Alternatively, run each step individually:

**Step 1: Data Splitting**
```bash
python -m src.training.split \
    --data_path data/processed.cleveland.data \
    --output_path data/processed \
    --target_column num
```

**Step 2: Model Training**
```bash
python -m src.training.train \
    --data_path data/processed \
    --model_output_path models \
    --use_scaling
```

**Step 3: Model Evaluation**
```bash
python -m src.training.evaluate \
    --data_path data/processed \
    --lr_model_path models/logistic_regression_pipeline.pkl \
    --rf_model_path models/random_forest_pipeline.pkl \
    --output_best_model models/best_pipeline.pkl
```

#### MLflow Tracking

View experiment results in MLflow UI:

```bash
mlflow ui --port 5000
```

Then open your browser to http://localhost:5000

### 3. Model Serving

#### Local Serving

**Start the Flask API server:**

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Start the server
python -m src.serving.app
```

The API will be available at http://localhost:5555

**Access Swagger Documentation:**

Open http://localhost:5555/apidocs in your browser for interactive API documentation.

#### Configuration

Update serving configuration in [src/serving/config/serving_config.py](src/serving/config/serving_config.py):

- `MODEL_URI`: Path or MLflow model URI
- `MLFLOW_TRACKING_URI`: MLflow tracking server
- `PORT`: API server port (default: 5555)

## API Usage

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model-info` | GET | Model information |
| `/predict` | POST | Single prediction |
| `/predict-batch` | POST | Batch prediction |
| `/metrics` | GET | Prometheus metrics |

### Example Usage with curl

**Health Check:**
```bash
curl http://localhost:5555/health
```

**Single Prediction:**
```bash
curl -X POST http://localhost:5555/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:5555/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, 
        "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
      },
      {
        "age": 37, "sex": 1, "cp": 2, "trestbps": 130,
        "chol": 250, "fbs": 0, "restecg": 1, "thalach": 187,
        "exang": 0, "oldpeak": 3.5, "slope": 0, "ca": 0, "thal": 2
      }
    ]
  }'
```

### Python Client

Use the provided example client:

```bash
python prediction_client.py
```

This script demonstrates:
- Health checks
- Model information retrieval
- Single predictions
- Batch predictions

## Testing

Run the test suite:

```bash
# Run all tests
pytest test/

# Run with coverage
pytest test/ --cov=src --cov-report=html

# Run specific test module
pytest test/training/test_train.py
pytest test/serving/test_app.py
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t heart-disease-api:latest .
```

### Run Docker Container

```bash
docker run -p 5555:5555 \
  -e MODEL_URI="models:/heart_disease_best_model@production" \
  -e MLFLOW_TRACKING_URI="http://your-mlflow-server" \
  heart-disease-api:latest
```

### Push to Docker Registry

```bash
# Tag the image
docker tag heart-disease-api:latest <your-registry>/heart-disease-api:latest

# Push to registry
docker push <your-registry>/heart-disease-api:latest
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (Minikube, EKS, GKE, AKS, etc.)
- kubectl configured
- Helm 3.x installed

### Deploy with Helm

**1. Update values in [helm-chart/values.yaml](helm-chart/values.yaml):**

```yaml
image:
  repository: <your-registry>/heart-disease-api
  tag: "latest"

env:
  - name: MODEL_URI
    value: "models:/heart_disease_best_model@production"
  - name: MLFLOW_TRACKING_URI
    value: "http://your-mlflow-server"
```

**2. Install the Helm chart:**

```bash
# Install
helm install heart-disease-api ./helm-chart

# Or upgrade if already installed
helm upgrade heart-disease-api ./helm-chart
```

**3. Check deployment status:**

```bash
kubectl get pods
kubectl get services
kubectl get hpa
```

**4. Access the service:**

```bash
# Port forward for local access
kubectl port-forward service/heart-disease-api 5555:5555

# Or access via NodePort
kubectl get service heart-disease-api
# Access at http://<node-ip>:30555
```

### Uninstall

```bash
helm uninstall heart-disease-api
```

## Monitoring

### Prometheus Metrics

The application exposes Prometheus metrics at `/metrics`:

```bash
curl http://localhost:5555/metrics
```

**Available metrics:**
- `flask_http_request_duration_seconds` - Request duration
- `flask_http_request_total` - Total requests
- `flask_http_request_exceptions_total` - Failed requests
- Custom application metrics

### Health Monitoring

```bash
# Health check
curl http://localhost:5555/health

# Model information
curl http://localhost:5555/model-info
```

## Model Features

The model expects the following 13 features:

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | int |
| sex | Sex (1 = male; 0 = female) | int |
| cp | Chest pain type (0-3) | int |
| trestbps | Resting blood pressure (mm Hg) | int |
| chol | Serum cholesterol (mg/dl) | int |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) | int |
| restecg | Resting ECG results (0-2) | int |
| thalach | Maximum heart rate achieved | int |
| exang | Exercise induced angina (1 = yes; 0 = no) | int |
| oldpeak | ST depression induced by exercise | float |
| slope | Slope of peak exercise ST segment (0-2) | int |
| ca | Number of major vessels colored by fluoroscopy (0-3) | int |
| thal | Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect) | int |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is part of the MLOps course assignment.

## Additional Resources

- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
- [Helm Documentation](https://helm.sh/docs/)

---

**Note:** Ensure you have the raw dataset file `data/processed.cleveland.data` before running the training pipeline. Update the MLflow tracking URI in the configuration files according to your setup.
