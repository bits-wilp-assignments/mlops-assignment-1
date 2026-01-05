"""
Training configuration file for MLOps project.
Contains centralized settings for training and evaluation.
"""
import os
from src.common.base_config import MLFLOW_TRACKING_URI

# MLflow Configuration
MLFLOW_EXPERIMENT_NAME = os.environ.get('MLFLOW_EXPERIMENT_NAME', "heart_disease_prediction")


# Data Schema Configuration
TARGET_COLUMN = "num"
HEADER_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                   "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
NUMERICAL_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Data Splitting Configuration
TEST_SIZE = 0.2
STRATIFIED_SAMPLING = True
RANDOM_STATE = 42

# Model Training Configuration
# Logistic Regression Hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 20, 50],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 500]
}

# Random Forest Hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# GridSearch Configuration
GRIDSEARCH_CV_FOLDS = 5
GRIDSEARCH_SCORING = 'accuracy'
GRIDSEARCH_N_JOBS = -1
GRIDSEARCH_VERBOSE = 1

# Model Evaluation Configuration
METRICS_AVERAGE = 'weighted'
PLOT_OUTPUT_DIR = "plots"
PLOT_FIGURE_SIZE = (8, 6)
PLOT_DPI = 300
BEST_MODEL_SELECTION_METRIC = 'test_accuracy'

# Model Registry Configuration
REGISTERED_MODEL_NAME = "heart_disease_best_model"
MODEL_ARTIFACT_PATH = "best_model"

# Model Validation Configuration
VALIDATION_METRIC = "test_f1_score"
PROD_READY_TAG = "@prod-ready"
PASSED_TAG = "@passed"
PRODUCTION_ALIAS = "production"
STAGE_TAG_KEY = "stage"
VALIDATION_TAG_KEY = "validation"
