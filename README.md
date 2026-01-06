# Heart Disease Prediction - MLOps Project

End-to-end MLOps pipeline for heart disease prediction using scikit-learn, MLflow, and Flask API with Swagger documentation.

## Prerequisites

- Python 3.10+
- pip

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install training dependencies
pip install -r requirements-train.txt
```

### 2. Run Training Pipeline

Execute each step of the ML pipeline:

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

**Step 4: Model Validation and Promotion**

```bash
python -m src.training.validate \
    --model_name heart_disease_best_model
```

**View MLflow Results:**

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 3. Start Serving API

```bash
# Install serving dependencies
pip install -r requirements-serving.txt

# Start Flask API server
python -m src.serving.app
```

The API runs on `http://localhost:5555`

### 4. Access Swagger Documentation

Open `http://localhost:5555/apidocs` in your browser for interactive API documentation.

## API Testing

### Available Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | GET | Health check |
| `/model-info` | GET | Model information |
| `/predict` | POST | Single or batch prediction |
| `/metrics` | GET | Prometheus metrics |
| `/apidocs` | GET | Swagger UI |

### Test with curl

**Health Check:**

```bash
curl http://localhost:5555/health
```

**Single Prediction:**

```bash
curl -X POST http://localhost:5555/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

**Batch Prediction:**

```bash
curl -X POST http://localhost:5555/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, 
       "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, 
       "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1},
      {"age": 37, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250,
       "fbs": 0, "restecg": 1, "thalach": 187, "exang": 0,
       "oldpeak": 3.5, "slope": 0, "ca": 0, "thal": 2}
    ]
  }'
```

### Python Client Example

Use the provided test script to test all endpoints:

```bash
python test_prediction.py
```

This script demonstrates:
- Health check (`/health`)
- Model information (`/model-info`)
- Single prediction
- Batch prediction with multiple patients
- Direct prediction without instances wrapper

## Running Tests

```bash
# Run all tests
pytest test/

# Run with coverage
pytest test/ --cov=src --cov-report=html

# Test specific modules
pytest test/training/test_train.py
pytest test/serving/test_app.py
```

## Model Features

The model expects 13 input features:

- `age`: Age in years
- `sex`: 1=male, 0=female
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (1=true, 0=false)
- `restecg`: Resting ECG results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina (1=yes, 0=no)
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of peak exercise ST segment (0-2)
- `ca`: Number of major vessels (0-3)
- `thal`: Thalassemia (0=normal, 1=fixed defect, 2=reversible defect)

## Docker Deployment

**Build Docker Image:**

```bash
docker build -t heart-disease-api:latest .
```

**Run Container:**

```bash
docker run -p 5555:5555 heart-disease-api:latest
```

**Push to Docker Registry:**

```bash
# Tag the image
docker tag heart-disease-api:latest your-registry.com/your-namespace/heart-disease-api:latest

# Push to registry
docker push your-registry.com/your-namespace/heart-disease-api:latest
```

## Kubernetes Deployment with Helm

```bash
# Install with Helm
helm install heart-disease-api ./helm-chart

# Check status
kubectl get pods
kubectl get svc

# Port forward to access locally
kubectl port-forward service/heart-disease-api 5555:5555

# Upgrade deployment
helm upgrade heart-disease-api ./helm-chart

# Uninstall
helm uninstall heart-disease-api
```

**Update [helm-chart/values.yaml](helm-chart/values.yaml) to configure:**
- Docker image repository and tag
- Environment variables (MODEL_URI, MLFLOW_TRACKING_URI)
- Resource limits and HPA settings

## Project Structure

```
├── src/
│   ├── training/          # Training pipeline (split, train, evaluate)
│   ├── serving/           # Flask API and prediction logic
│   └── common/            # Shared utilities
├── data/processed/        # Train/test datasets
├── models/                # Saved models
├── mlruns/                # MLflow experiment tracking
├── test/                  # Unit tests
├── helm-chart/            # Kubernetes deployment
└── notebooks/             # EDA and analysis notebooks
```
