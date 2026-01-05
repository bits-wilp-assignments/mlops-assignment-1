"""
Base configuration file for MLOps project.
Contains common settings shared across training and inference.
"""
import os

# MLflow Configuration - Shared across all modules
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', "http://localhost:5050")
