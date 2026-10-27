"""
Preprocessing Pipeline Module for Telecom Customer Churn Prediction.
Builds robust ColumnTransformers and handles train/test splitting without data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """
    Creates a Scikit-learn ColumnTransformer that scales numeric features
    and one-hot encodes categorical features with robust handling of missing values
    and unknown categories.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def prepare_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the data into stratified training and testing sets.
    Preserves customer IDs for reporting while excluding them from predictive features.
    
    Returns:
        (X_train, X_test, y_train, y_test, id_train, id_test)
    """
    df_clean = df.copy()

    if TARGET_COLUMN not in df_clean.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    # Extract ID series if present
    if ID_COLUMN in df_clean.columns:
        ids = df_clean[ID_COLUMN]
    else:
        ids = pd.Series([f"CUST-{i:05d}" for i in range(len(df_clean))], index=df_clean.index)

    # Extract target series safely mapping strings like Yes/No
    target_series = df_clean[TARGET_COLUMN]
    if target_series.dtype == object or str(target_series.dtype).startswith("str"):
        y = target_series.astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0, "True": 1, "False": 0})
        y = y.fillna(0).astype(int)
    else:
        y = target_series.astype(int)
    X = df_clean[ALL_FEATURE_COLUMNS].copy()

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, id_train, id_test


def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer) -> list[str]:
    """
    Returns the transformed feature names from a fitted preprocessor.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out fails
        num_names = list(NUMERIC_FEATURES)
        cat_names = []
        if hasattr(preprocessor.named_transformers_["cat"].named_steps["ohe"], "get_feature_names_out"):
            cat_names = list(
                preprocessor.named_transformers_["cat"]
                .named_steps["ohe"]
                .get_feature_names_out(CATEGORICAL_FEATURES)
            )
        return num_names + cat_names
