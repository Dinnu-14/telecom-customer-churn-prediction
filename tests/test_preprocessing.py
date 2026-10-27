"""
Unit tests for data loader and preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np

from src.data_loader import clean_raw_data, generate_synthetic_telco_data
from src.preprocessing import (
    create_preprocessor,
    prepare_train_test_split,
    ALL_FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def test_clean_raw_data_total_charges_spaces():
    """Verify that whitespace in TotalCharges is converted to valid float."""
    raw_df = pd.DataFrame({
        "customerID": ["CUST-1", "CUST-2"],
        "tenure": [0, 5],
        "MonthlyCharges": [29.85, 50.0],
        "TotalCharges": [" ", "250.0"],
        "Churn": ["No", "Yes"],
    })
    cleaned = clean_raw_data(raw_df)
    assert not cleaned["TotalCharges"].isna().any()
    assert cleaned["TotalCharges"].iloc[0] == 29.85 or cleaned["TotalCharges"].iloc[0] == 0.0
    assert cleaned["TotalCharges"].iloc[1] == 250.0
    assert cleaned["Churn"].iloc[0] == 0
    assert cleaned["Churn"].iloc[1] == 1


def test_synthetic_data_generation():
    """Verify that fallback synthetic dataset matches expected schema."""
    df_synth = generate_synthetic_telco_data(num_samples=100, random_state=123)
    assert len(df_synth) == 100
    for col in ALL_FEATURE_COLUMNS:
        assert col in df_synth.columns
    assert "customerID" in df_synth.columns
    assert "Churn" in df_synth.columns
    assert df_synth["customerID"].str.startswith("SYNTH-").all()


def test_prepare_train_test_split_stratification():
    """Verify train-test split stratifies target and retains customer IDs."""
    df_synth = generate_synthetic_telco_data(num_samples=200, random_state=42)
    cleaned = clean_raw_data(df_synth)
    
    X_train, X_test, y_train, y_test, id_train, id_test = prepare_train_test_split(
        cleaned, test_size=0.2, random_state=42
    )

    assert len(X_train) == 160
    assert len(X_test) == 40
    assert len(id_train) == 160
    assert len(id_test) == 40
    
    # Check that customerID is not inside X
    assert "customerID" not in X_train.columns
    assert "Churn" not in X_train.columns

    # Check stratification: churn rates should be reasonably close
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.08


def test_preprocessor_transformation():
    """Verify ColumnTransformer handles scaling and encoding properly."""
    df_synth = generate_synthetic_telco_data(num_samples=150, random_state=42)
    cleaned = clean_raw_data(df_synth)
    X = cleaned[ALL_FEATURE_COLUMNS]

    preprocessor = create_preprocessor()
    preprocessor.fit(X)
    X_trans = preprocessor.transform(X)

    assert X_trans.shape[0] == 150
    # One-hot encoding increases columns count beyond raw features
    assert X_trans.shape[1] > len(ALL_FEATURE_COLUMNS)
    assert not np.isnan(X_trans).any()
