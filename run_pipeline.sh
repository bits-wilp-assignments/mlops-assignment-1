#!/bin/bash
# Heart Disease Prediction - Complete Pipeline Execution Script
# This script runs the entire ML pipeline step by step

# Set Python executable and PYTHONPATH
PYTHON_BIN="./venv/bin/python"
export PYTHONPATH="./src"

echo "=========================================="
echo "Heart Disease Prediction - ML Pipeline"
echo "=========================================="

# Step 1: Split data (loads raw data directly)
echo ""
echo "Step 1: Loading and splitting data into train and test sets..."
$PYTHON_BIN -m src.training.split \
    --data_path data/processed.cleveland.data \
    --output_path data/processed \
    --target_column num
if [ $? -eq 0 ]; then
    echo "✓ Data splitting completed successfully"
else
    echo "✗ Data splitting failed"
    exit 1
fi

# Step 2: Train models
echo ""
echo "Step 2: Training models with GridSearchCV..."
$PYTHON_BIN -m src.training.train \
    --data_path data/processed \
    --model_output_path models \
    --use_scaling
if [ $? -eq 0 ]; then
    echo "✓ Model training completed successfully"
else
    echo "✗ Model training failed"
    exit 1
fi

# Step 3: Evaluate models
echo ""
echo "Step 3: Evaluating models on test set..."
$PYTHON_BIN -m src.training.evaluate \
    --data_path data/processed \
    --lr_model_path models/logistic_regression_pipeline.pkl \
    --rf_model_path models/random_forest_pipeline.pkl \
    --output_best_model models/best_pipeline.pkl
if [ $? -eq 0 ]; then
    echo "✓ Model evaluation completed successfully"
else
    echo "✗ Model evaluation failed"
    exit 1
fi

# Step 4: Validate and promote model
echo ""
echo "Step 4: Validating model and promoting to production..."
$PYTHON_BIN -m src.training.validate \
    --model_name heart_disease_best_model
if [ $? -eq 0 ]; then
    echo "✓ Model validation and promotion completed successfully"
else
    echo "✗ Model validation failed (model not promoted)"
    # Don't exit - validation failure shouldn't stop the pipeline
fi

echo ""
echo "=========================================="
echo "Pipeline Execution Complete!"
echo "=========================================="
echo ""
echo "Generated Files:"
echo "  - Data splits: data/processed/X_train_raw.csv, X_test_raw.csv, y_train.csv, y_test.csv"
echo "  - Trained models: models/logistic_regression_pipeline.pkl, models/random_forest_pipeline.pkl"
echo "  - Best model: models/best_pipeline.pkl"
echo "  - Evaluation plots: confusion_matrix_*.png, roc_curve_*.png"
echo ""
echo "View MLflow UI at: http://localhost:5050"
echo ""
