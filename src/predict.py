"""
Prediction Module for Telecom Customer Churn Prediction System.
Handles single-customer prediction, batch CSV processing, risk classification,
and integration with the retention recommendation engine.
"""

import os
import sys
import logging
from typing import Any, Union
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import ALL_FEATURE_COLUMNS, ID_COLUMN
from src.retention import generate_retention_recommendation

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "stacked_churn_model.joblib")

_CACHED_MODEL = None


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads and caches the trained Stacked Pipeline from disk.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please execute 'python src/train_model.py' to train and persist the model first."
        )

    logger.info(f"Loading stacked model pipeline from {model_path}...")
    _CACHED_MODEL = joblib.load(model_path)
    return _CACHED_MODEL


def classify_risk(
    prob: float,
    low_threshold: float = 0.30,
    high_threshold: float = 0.60,
) -> tuple[str, str]:
    """
    Maps probability to Risk Category and CSS/UI badge color code.
    Configurable via low_threshold and high_threshold.
    
    Returns:
        (risk_category, color_hex)
    """
    if prob < low_threshold:
        return "Low", "#28a745"  # Green
    elif prob < high_threshold:
        return "Medium", "#fd7e14"  # Amber/Orange
    else:
        return "High", "#dc3545"  # Red


def sanitize_input_row(customer_data: dict[str, Any]) -> pd.DataFrame:
    """
    Validates and formats a single customer input dictionary into a 1-row DataFrame
    matching expected feature columns and data types.
    """
    data = dict(customer_data)

    # Defaults for essential fields if omitted
    defaults = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }

    cleaned = {}
    for col in ALL_FEATURE_COLUMNS:
        val = data.get(col, defaults.get(col))
        if col in ["tenure"]:
            cleaned[col] = [int(val) if val is not None else 0]
        elif col in ["MonthlyCharges", "TotalCharges"]:
            cleaned[col] = [float(val) if val is not None else 0.0]
        elif col in ["SeniorCitizen"]:
            cleaned[col] = [int(val) if val is not None else 0]
        else:
            cleaned[col] = [str(val).strip() if val is not None else str(defaults.get(col))]

    return pd.DataFrame(cleaned)


def predict_single_customer(
    customer_dict: dict[str, Any],
    decision_threshold: float = 0.50,
    low_risk_threshold: float = 0.30,
    high_risk_threshold: float = 0.60,
    model=None,
) -> dict[str, Any]:
    """
    Predicts churn probability and class for an individual customer profile,
    computes risk tier, and prescribes personalized retention offers.
    """
    if model is None:
        model = load_model()

    input_df = sanitize_input_row(customer_dict)

    # Predict probability from the Stacked Classifier pipeline
    prob = float(model.predict_proba(input_df)[0, 1])
    is_churn = int(prob >= decision_threshold)
    risk_category, color_hex = classify_risk(prob, low_risk_threshold, high_risk_threshold)

    # Generate personalized retention recommendation
    retention_rec = generate_retention_recommendation(risk_category, prob, customer_dict)

    # Natural language summary explanation
    explanation = (
        f"The subscriber has an estimated churn probability of {prob * 100:.1f}%, "
        f"placing them in the {risk_category} Risk category. "
    )
    if risk_category == "High":
        explanation += (
            "Key churn indicators include their short contract commitment, payment channel, "
            "and absence of value-added support services."
        )
    elif risk_category == "Medium":
        explanation += (
            "The customer displays moderate stability but may be susceptible to competitive "
            "operator promotions if service satisfaction drops."
        )
    else:
        explanation += (
            "The subscriber exhibits strong retention attributes, including long-term plan alignment "
            "or steady multi-service tenure."
        )

    return {
        "churn_prediction": is_churn,
        "churn_label": "Likely to Churn" if is_churn == 1 else "Likely to Stay",
        "churn_probability": round(prob, 4),
        "churn_probability_pct": round(prob * 100, 2),
        "risk_category": risk_category,
        "risk_color": color_hex,
        "decision_threshold": decision_threshold,
        "explanation": explanation,
        "retention_recommendation": retention_rec,
    }


def predict_batch(
    data_source: Union[pd.DataFrame, str],
    decision_threshold: float = 0.50,
    low_risk_threshold: float = 0.30,
    high_risk_threshold: float = 0.60,
    model=None,
) -> pd.DataFrame:
    """
    Executes high-throughput batch prediction on a customer dataset or uploaded CSV.
    Validates required columns, cleans values, predicts probabilities and risk tiers,
    and returns an enriched DataFrame suitable for presentation and CSV export.
    """
    if model is None:
        model = load_model()

    if isinstance(data_source, str):
        if not os.path.exists(data_source):
            raise FileNotFoundError(f"Batch file not found: {data_source}")
        df = pd.read_csv(data_source)
    else:
        df = data_source.copy()

    # Preserve customer ID or synthesize if absent
    if ID_COLUMN in df.columns:
        customer_ids = df[ID_COLUMN].astype(str)
    else:
        customer_ids = pd.Series([f"CUST-{i+1:05d}" for i in range(len(df))], index=df.index)

    # Validate missing feature columns
    missing_cols = [c for c in ALL_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning(f"Batch data missing columns: {missing_cols}. Imputing defaults.")
        # Impute defaults for missing columns
        dummy_row = sanitize_input_row({})
        for col in missing_cols:
            df[col] = dummy_row[col].iloc[0]

    # Pre-clean numeric and text types
    df_clean = df.copy()
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"].astype(str).str.strip(), errors="coerce"
        )
        if "MonthlyCharges" in df_clean.columns and "tenure" in df_clean.columns:
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(
                df_clean["MonthlyCharges"] * df_clean["tenure"].clip(lower=1)
            )
        else:
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    if "tenure" in df_clean.columns:
        df_clean["tenure"] = pd.to_numeric(df_clean["tenure"], errors="coerce").fillna(0).astype(int)
    if "MonthlyCharges" in df_clean.columns:
        df_clean["MonthlyCharges"] = pd.to_numeric(df_clean["MonthlyCharges"], errors="coerce").fillna(0.0)

    # Slice strictly the feature columns
    X = df_clean[ALL_FEATURE_COLUMNS].copy()

    # Predict via pipeline
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= decision_threshold).astype(int)

    risk_categories = []
    recommended_actions = []

    for i, prob in enumerate(probabilities):
        cat, _ = classify_risk(prob, low_risk_threshold, high_risk_threshold)
        risk_categories.append(cat)
        
        row_dict = df_clean.iloc[i].to_dict()
        rec = generate_retention_recommendation(cat, prob, row_dict)
        recommended_actions.append(rec["primary_action"])

    results_df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Churn_Probability": np.round(probabilities, 4),
        "Churn_Probability_Pct": np.round(probabilities * 100, 2),
        "Predicted_Class": np.where(predictions == 1, "Churn", "Stay"),
        "Risk_Category": risk_categories,
        "Recommended_Action": recommended_actions,
        "Contract": df_clean["Contract"],
        "Tenure": df_clean["tenure"],
        "MonthlyCharges": df_clean["MonthlyCharges"],
        "InternetService": df_clean["InternetService"],
    })

    return results_df
