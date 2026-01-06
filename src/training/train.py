import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib

# Import centralized logging configuration
from src.common.logging_config import get_logger
from src.training.config.training_config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    RANDOM_STATE, LOGISTIC_REGRESSION_PARAMS, RANDOM_FOREST_PARAMS,
    GRIDSEARCH_CV_FOLDS, GRIDSEARCH_SCORING, GRIDSEARCH_N_JOBS, GRIDSEARCH_VERBOSE
)
from src.training.preprocess import build_preprocessing_pipeline

# Initialize logger for this module
logger = get_logger(__name__)


## ----------------- Training Functions ----------------- ##
def load_raw_training_data(data_path: str):
    """Load raw training data (before preprocessing)."""
    # Load the original data split (not preprocessed)
    X_train = pd.read_csv(f"{data_path}/X_train_raw.csv")
    y_train = pd.read_csv(f"{data_path}/y_train.csv").values.ravel()
    return X_train, y_train


def get_param_grids():
    """Define parameter grids for models."""
    return LOGISTIC_REGRESSION_PARAMS, RANDOM_FOREST_PARAMS


def build_model_pipeline(preprocessor, model):
    """Build a pipeline containing preprocessing + model."""
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    return pipeline


def train_pipeline_with_gridsearch(pipeline, param_grid, X_train, y_train, model_name: str):
    """Train pipeline (preprocessing + model) using GridSearchCV and return best estimator."""
    logger.info(f"Training {model_name} pipeline with GridSearchCV...")

    # Prefix parameter names with 'classifier__' for the pipeline
    pipeline_param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=pipeline_param_grid,
        cv=GRIDSEARCH_CV_FOLDS,
        scoring=GRIDSEARCH_SCORING,
        n_jobs=GRIDSEARCH_N_JOBS,
        verbose=GRIDSEARCH_VERBOSE
    )

    grid_search.fit(X_train, y_train)
    logger.info(f"{model_name} - Best params: {grid_search.best_params_}")
    logger.info(f"{model_name} - Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def log_experiment_to_mlflow(model_name, pipeline, params, cv_score):
    """Log pipeline training experiment to MLflow as nested run."""
    with mlflow.start_run(run_name=f"{model_name}_Training", nested=True):
        # Log parameters
        mlflow.log_params(params)

        # Log CV score
        mlflow.log_metric("cv_score", cv_score)

        # Log pipeline (preprocessing + model)
        mlflow.sklearn.log_model(pipeline, model_name)

        logger.info(f"Logged {model_name} pipeline to MLflow - CV Score: {cv_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train models with GridSearchCV and MLflow tracking")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw data folder")
    parser.add_argument("--experiment_name", type=str, default=MLFLOW_EXPERIMENT_NAME, help="MLflow experiment name")
    parser.add_argument("--model_output_path", type=str, default="models", help="Path to save trained pipelines")
    parser.add_argument("--use_scaling", action='store_true', default=True, help="Whether to apply scaling")
    parser.add_argument("--use_pca", action='store_true', default=False, help="Whether to apply PCA")
    parser.add_argument("--n_components", type=int, default=None, help="Number of PCA components")

    args = parser.parse_args()

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment_name)

    # Create the directory if it doesn't exist
    if args.model_output_path and not os.path.exists(args.model_output_path):
        os.makedirs(args.model_output_path, exist_ok=True)

    # Start parent pipeline run
    with mlflow.start_run(run_name="Heart_Disease_Pipeline") as parent_run:
        parent_run_id = parent_run.info.run_id
        logger.info(f"Started pipeline run: {parent_run_id}")

        # Load training data (raw, not preprocessed)
        logger.info("Loading raw training data...")
        X_train, y_train = load_raw_training_data(args.data_path)

        # Build preprocessing pipeline
        logger.info("Building preprocessing pipeline...")
        preprocessor = build_preprocessing_pipeline(
            use_scaling=args.use_scaling,
            use_pca=args.use_pca,
            n_components=args.n_components
        )

        # Get parameter grids
        logistic_params, rf_params = get_param_grids()

        # Train Logistic Regression Pipeline
        lr_model = LogisticRegression(random_state=RANDOM_STATE)
        lr_pipeline = build_model_pipeline(preprocessor, lr_model)
        best_lr_pipeline, best_lr_params, best_lr_cv_score = train_pipeline_with_gridsearch(
            lr_pipeline, logistic_params, X_train, y_train, "Logistic Regression"
        )

        # Log to MLflow
        log_experiment_to_mlflow("Logistic_Regression", best_lr_pipeline, best_lr_params, best_lr_cv_score)

        # Save Logistic Regression pipeline
        lr_pipeline_path = f"{args.model_output_path}/logistic_regression_pipeline.pkl"
        # Get the directory path from your lr_pipeline_path variable

        joblib.dump(best_lr_pipeline, lr_pipeline_path)
        logger.info(f"Logistic Regression pipeline saved to {lr_pipeline_path}")

        # Train Random Forest Pipeline
        rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
        rf_pipeline = build_model_pipeline(preprocessor, rf_model)
        best_rf_pipeline, best_rf_params, best_rf_cv_score = train_pipeline_with_gridsearch(
            rf_pipeline, rf_params, X_train, y_train, "Random Forest"
        )

        # Log to MLflow
        log_experiment_to_mlflow("Random_Forest", best_rf_pipeline, best_rf_params, best_rf_cv_score)

        # Save Random Forest pipeline
        rf_pipeline_path = f"{args.model_output_path}/random_forest_pipeline.pkl"
        joblib.dump(best_rf_pipeline, rf_pipeline_path)
        logger.info(f"Random Forest pipeline saved to {rf_pipeline_path}")

        # Save parent run ID for evaluation phase
        os.makedirs(args.model_output_path, exist_ok=True)
        run_id_path = f"{args.model_output_path}/pipeline_run_id.txt"
        with open(run_id_path, 'w') as f:
            f.write(parent_run_id)
        logger.info(f"Pipeline run ID saved to {run_id_path}")

        logger.info("Training complete. Pipelines (preprocessing + model) saved. Use evaluate.py to evaluate pipelines on test data.")


if __name__ == "__main__":
    main()
