import argparse
import pandas as pd
import joblib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Import centralized logging configuration
from src.common.logging_config import get_logger
from src.training.config.training_config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    METRICS_AVERAGE, PLOT_OUTPUT_DIR, PLOT_FIGURE_SIZE, PLOT_DPI,
    BEST_MODEL_SELECTION_METRIC, REGISTERED_MODEL_NAME, MODEL_ARTIFACT_PATH
)

# Initialize logger for this module
logger = get_logger(__name__)


## ----------------- Evaluation Functions ----------------- ##
def load_test_data(data_path: str, use_raw: bool = True):
    """Load test data (raw or preprocessed)."""
    if use_raw:
        # Load raw data for pipeline evaluation
        X_test = pd.read_csv(f"{data_path}/X_test_raw.csv")
    else:
        # Load preprocessed data for model-only evaluation
        X_test = pd.read_csv(f"{data_path}/X_test.csv")
    y_test = pd.read_csv(f"{data_path}/y_test.csv").values.ravel()
    return X_test, y_test


def load_pipeline(pipeline_path: str):
    """Load trained pipeline from file."""
    return joblib.load(pipeline_path)


def evaluate_pipeline(pipeline, X_test, y_test, model_name: str):
    """Evaluate pipeline (preprocessing + model) and return metrics, predictions, and probabilities."""
    logger.info(f"Evaluating {model_name} pipeline...")

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    # Calculate ROC AUC (handle binary and multiclass)
    if len(set(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average=METRICS_AVERAGE)

    metrics = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, average=METRICS_AVERAGE),
        'test_recall': recall_score(y_test, y_pred, average=METRICS_AVERAGE),
        'test_f1_score': f1_score(y_test, y_pred, average=METRICS_AVERAGE),
        'test_roc_auc': roc_auc
    }

    logger.info(f"{model_name} Results:")
    logger.info(f"  Accuracy:  {metrics['test_accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['test_precision']:.4f}")
    logger.info(f"  Recall:    {metrics['test_recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['test_f1_score']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['test_roc_auc']:.4f}")

    # Print detailed classification report
    logger.info(f"\n{model_name} Classification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm, y_pred_proba


def plot_confusion_matrix(cm, model_name: str, output_dir: str = PLOT_OUTPUT_DIR):
    """Plot confusion matrix and save as file."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=PLOT_FIGURE_SIZE)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                annot_kws={'size': 14}, linewidths=0.5, linecolor='gray')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    # Save plot
    plot_path = f"{output_dir}/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {plot_path}")

    return plot_path


def plot_roc_curve(y_test, y_pred_proba, model_name: str, output_dir: str = PLOT_OUTPUT_DIR):
    """Plot ROC curve and save as file."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=PLOT_FIGURE_SIZE)

    # Binary classification (most common case)
    n_classes = y_pred_proba.shape[1]

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        # For multiclass, use macro average
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average=METRICS_AVERAGE)

    # Plot ROC curve
    plt.plot(fpr, tpr, linewidth=2, label=f'Model (AUC = {roc_auc:.3f})', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_path = f"{output_dir}/roc_curve_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved to {plot_path}")

    return plot_path


def log_evaluation_to_mlflow(model_name: str, metrics: dict, cm, y_test, y_pred_proba, output_dir: str = PLOT_OUTPUT_DIR):
    """Log evaluation metrics, confusion matrix, and ROC curve to MLflow as nested run."""
    with mlflow.start_run(run_name=f"{model_name}_Evaluation", nested=True):
        # Log test metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Plot and log confusion matrix
        cm_plot_path = plot_confusion_matrix(cm, model_name, output_dir)
        mlflow.log_artifact(cm_plot_path)

        # Plot and log ROC curve
        roc_plot_path = plot_roc_curve(y_test, y_pred_proba, model_name, output_dir)
        mlflow.log_artifact(roc_plot_path)

        logger.info(f"Logged {model_name} evaluation metrics, confusion matrix, and ROC curve to MLflow")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data folder (with raw data)")
    parser.add_argument("--lr_model_path", type=str, required=True, help="Path to Logistic Regression pipeline")
    parser.add_argument("--rf_model_path", type=str, required=True, help="Path to Random Forest pipeline")
    parser.add_argument("--experiment_name", type=str, default=MLFLOW_EXPERIMENT_NAME, help="MLflow experiment name")
    parser.add_argument("--output_best_model", type=str, default="models/best_pipeline.pkl", help="Path to save best pipeline")
    parser.add_argument("--run_id_file", type=str, default="models/pipeline_run_id.txt", help="Path to pipeline run ID file")
    parser.add_argument("--model_output_path", type=str, default="models", help="Path to save model outputs and plots")

    args = parser.parse_args()

    # Create plots directory
    import os
    plots_dir = f"{args.model_output_path}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Evaluation plots will be saved to {plots_dir}")

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment_name)

    # Load parent pipeline run ID
    if os.path.exists(args.run_id_file):
        with open(args.run_id_file, 'r') as f:
            parent_run_id = f.read().strip()
        logger.info(f"Continuing pipeline run: {parent_run_id}")
    else:
        parent_run_id = None
        logger.warning(f"Pipeline run ID file not found: {args.run_id_file}")

    # Continue parent pipeline run or create new one
    with mlflow.start_run(run_id=parent_run_id) if parent_run_id else mlflow.start_run(run_name="Heart_Disease_Pipeline"):
        # Load raw test data (pipelines handle preprocessing)
        logger.info("Loading raw test data...")
        X_test, y_test = load_test_data(args.data_path, use_raw=True)

        # Load and evaluate Logistic Regression pipeline
        lr_pipeline = load_pipeline(args.lr_model_path)
        lr_metrics, lr_cm, lr_proba = evaluate_pipeline(lr_pipeline, X_test, y_test, "Logistic Regression")
        log_evaluation_to_mlflow("Logistic_Regression", lr_metrics, lr_cm, y_test, lr_proba, plots_dir)

        # Load and evaluate Random Forest pipeline
        rf_pipeline = load_pipeline(args.rf_model_path)
        rf_metrics, rf_cm, rf_proba = evaluate_pipeline(rf_pipeline, X_test, y_test, "Random Forest")
        log_evaluation_to_mlflow("Random_Forest", rf_metrics, rf_cm, y_test, rf_proba, plots_dir)

        # Select best pipeline based on configured metric
        if lr_metrics[BEST_MODEL_SELECTION_METRIC] > rf_metrics[BEST_MODEL_SELECTION_METRIC]:
            best_pipeline = lr_pipeline
            best_model_name = "Logistic Regression"
            best_accuracy = lr_metrics[BEST_MODEL_SELECTION_METRIC]
        else:
            best_pipeline = rf_pipeline
            best_model_name = "Random Forest"
            best_accuracy = rf_metrics[BEST_MODEL_SELECTION_METRIC]

        # Save best pipeline
        joblib.dump(best_pipeline, args.output_best_model)
        logger.info(f"\nBest model: {best_model_name} with {BEST_MODEL_SELECTION_METRIC}: {best_accuracy:.4f}")
        logger.info(f"Best pipeline saved to {args.output_best_model}")

        # Register best model to MLFlow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{MODEL_ARTIFACT_PATH}"

        # Log the best model to the current run
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path=MODEL_ARTIFACT_PATH,
            registered_model_name=REGISTERED_MODEL_NAME
        )

        logger.info(f"Best model registered to MLFlow Model Registry as '{REGISTERED_MODEL_NAME}'")
        logger.info(f"Model URI: {model_uri}")

        # Log best model metadata
        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_metric("best_model_accuracy", best_accuracy)


if __name__ == "__main__":
    main()
