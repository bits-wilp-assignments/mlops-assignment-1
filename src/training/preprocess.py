import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Import centralized logging configuration
from src.training.config.training_config import NUMERICAL_COLS, CATEGORICAL_COLS
from src.common.logging_config import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


## ----------------- Custom Transformers ----------------- ##
class OutlierClipper:
    """
    Custom transformer to clip outliers using IQR method.
    Handles multiple columns by computing bounds per column during fit.
    """

    def __init__(self):
        self.lower_bound_ = None
        self.upper_bound_ = None

    def fit(self, X, y=None):
        """
        Calculate and store the lower and upper bounds for each column.
        """
        X_array = np.array(X) if not isinstance(X, np.ndarray) else X

        Q1 = np.percentile(X_array, 25, axis=0)
        Q3 = np.percentile(X_array, 75, axis=0)
        IQR = Q3 - Q1

        self.lower_bound_ = Q1 - 1.5 * IQR
        self.upper_bound_ = Q3 + 1.5 * IQR

        return self

    def transform(self, X):
        """
        Apply the stored bounds to clip outliers in each column.
        """
        if self.lower_bound_ is None or self.upper_bound_ is None:
            raise ValueError("OutlierClipper must be fitted before transform.")

        X_array = np.array(X) if not isinstance(X, np.ndarray) else X
        X_clipped = np.clip(X_array, self.lower_bound_, self.upper_bound_)

        # Preserve pandas DataFrame structure if input was a DataFrame
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_clipped, columns=X.columns, index=X.index)

        return X_clipped


## ----------------- Preprocessing Functions ----------------- ##
def build_preprocessing_pipeline(
    use_scaling: bool = True,
    use_pca: bool = False,
    n_components: int = None
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for numerical and categorical features.
    """

    # Numerical pipeline
    # Impute missing values, clip outliers, and optionally scale
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="mean")),
        ("clipper", OutlierClipper())
    ]

    if use_scaling:
        numeric_steps.append(("scaler", StandardScaler()))

    # Apply PCA to numerical features if specified
    if use_pca:
        numeric_steps.append(("pca", PCA(n_components=n_components)))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    # Categorical pipeline
    # Impute missing values and one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine numerical and categorical pipelines
    preprocessor = ColumnTransformer(transformers=[
        ("numerical", numeric_pipeline, NUMERICAL_COLS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLS)
    ])

    return preprocessor