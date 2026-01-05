"""
Inference configuration file for MLOps project.
Contains centralized settings for model serving and inference.
"""
import os
from src.common.base_config import MLFLOW_TRACKING_URI

# Model Serving Configuration
MODEL_URI = os.environ.get('MODEL_URI', 'models:/heart_disease_best_model@production')
PORT = int(os.environ.get('PORT', 5555))
