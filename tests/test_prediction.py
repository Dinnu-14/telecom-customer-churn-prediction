"""
Unit tests for model loading and prediction workflows (single and batch).
"""

import pytest
import os
import pandas as pd
from src.predict import (
    load_model,
    classify_risk,
    predict_single_customer,
    predict_batch,
    sanitize_input_row,
)


def test_classify_risk_thresholds():
    """Verify that probability mapping properly assigns risk categories."""
    assert classify_risk(0.15)[0] == "Low"
    assert classify_risk(0.29)[0] == "Low"
    assert classify_risk(0.30)[0] == "Medium"
    assert classify_risk(0.55)[0] == "Medium"
    assert classify_risk(0.60)[0] == "High"
    assert classify_risk(0.92)[0] == "High"

    # Custom thresholds
    assert classify_risk(0.25, low_threshold=0.20, high_threshold=0.50)[0] == "Medium"
    assert classify_risk(0.51, low_threshold=0.20, high_threshold=0.50)[0] == "High"


def test_sanitize_input_row():
    """Verify that input dictionaries are cleaned into valid DataFrames."""
    sample_input = {
        "tenure": "24",
        "MonthlyCharges": "65.5",
        "TotalCharges": "1572.0",
        "Contract": "One year",
    }
    df = sanitize_input_row(sample_input)
    assert len(df) == 1
    assert df["tenure"].iloc[0] == 24
    assert df["MonthlyCharges"].iloc[0] == 65.5
    assert df["TotalCharges"].iloc[0] == 1572.0
    assert df["Contract"].iloc[0] == "One year"
    # Fallback default check
    assert df["InternetService"].iloc[0] == "Fiber optic"


def test_model_loaded_and_predict_single():
    """Verify that saved stacked model runs single-customer prediction."""
    model = load_model()
    assert model is not None

    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 179.0,
    }

    result = predict_single_customer(customer, model=model)
    assert "churn_prediction" in result
    assert result["churn_prediction"] in [0, 1]
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_category"] in ["Low", "Medium", "High"]
    assert "retention_recommendation" in result
    assert result["retention_recommendation"]["primary_action"] != ""


def test_predict_batch():
    """Verify batch prediction processes batch DataFrame cleanly."""
    model = load_model()
    test_batch = pd.DataFrame([
        {
            "customerID": "BATCH-001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 55.0,
            "TotalCharges": 3300.0,
        },
        {
            "customerID": "BATCH-002",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0,
            "TotalCharges": 95.0,
        },
    ])

    results = predict_batch(test_batch, model=model)
    assert len(results) == 2
    assert "CustomerID" in results.columns
    assert "Churn_Probability" in results.columns
    assert "Predicted_Class" in results.columns
    assert "Risk_Category" in results.columns
    assert "Recommended_Action" in results.columns

    # High risk customer 2 should have higher probability than loyal customer 1
    prob_loyal = results.loc[results["CustomerID"] == "BATCH-001", "Churn_Probability"].iloc[0]
    prob_risky = results.loc[results["CustomerID"] == "BATCH-002", "Churn_Probability"].iloc[0]
    assert prob_risky > prob_loyal
