import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

# Import centralized logging configuration
from src.common.logging_config import get_logger
from src.training.config.training_config import (
    TEST_SIZE, STRATIFIED_SAMPLING, RANDOM_STATE, TARGET_COLUMN, HEADER_COLUMNS
)

# Initialize logger for this module
logger = get_logger(__name__)

## ----------------- Data Splitting Functions ----------------- ##
def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV.
    If the data has no headers, it will add column names and convert target to binary.
    """
    # Try to read with headers first
    df = pd.read_csv(path)
    
    # Check if columns match our expected header columns (case-insensitive)
    cols_lower = [str(c).lower() for c in df.columns]
    expected_lower = [c.lower() for c in HEADER_COLUMNS]
    
    if cols_lower == expected_lower:
        # Has proper headers, return as is
        logger.info("Loaded data with existing headers")
        return df
    
    # No proper headers - reload with header=None and assign column names
    df = pd.read_csv(path, header=None)
    df.columns = HEADER_COLUMNS
    
    # Convert target column to binary (0 = no disease, 1-4 = disease present)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if x > 0 else 0)
    logger.info("Loaded raw data and added column names")
    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into X and y.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    stratified_sample: bool = STRATIFIED_SAMPLING,
    random_state: int = RANDOM_STATE
) -> tuple:
    """
    Split data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y if stratified_sample else None, random_state=random_state)


def main():
    parser = argparse.ArgumentParser(description="Preprocess given dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed data")
    parser.add_argument("--target_column", type=str, required=True, help="Name of the target column")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE, help="Proportion of test set")

    args = parser.parse_args()
    data_path = args.data_path
    target_column = args.target_column
    test_size = args.test_size
    # Load data
    df = load_data(data_path)

    # Split features and target
    X, y = split_features_target(df, target_column)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=test_size, stratified_sample=STRATIFIED_SAMPLING, random_state=RANDOM_STATE)
    logger.info("Train-test split complete.")

    # Save raw data splits (preprocessing will be done in pipeline during training)
    X_train.to_csv(f"{args.output_path}/X_train_raw.csv", index=False)
    X_test.to_csv(f"{args.output_path}/X_test_raw.csv", index=False)
    y_train.to_csv(f"{args.output_path}/y_train.csv", index=False)
    y_test.to_csv(f"{args.output_path}/y_test.csv", index=False)

    logger.info(f"Raw data splits saved to {args.output_path}")


if __name__ == "__main__":
    main()
