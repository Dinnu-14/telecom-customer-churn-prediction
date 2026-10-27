"""
Unit tests for the rule-based customer retention recommendation engine.
"""

import pytest
from src.retention import generate_retention_recommendation


def test_retention_high_risk_contract_and_support():
    """Verify high risk month-to-month and no tech support triggers appropriate retention offers."""
    customer = {
        "tenure": 3,
        "MonthlyCharges": 85.0,
        "Contract": "Month-to-month",
        "TechSupport": "No",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }
    rec = generate_retention_recommendation("High", 0.78, customer)

    assert rec["priority"] == "Critical"
    assert "Immediate Senior Account Intervention" in rec["primary_action"]
    assert any("contract transition" in s.lower() for s in rec["action_details"])
    assert any("technical support" in s.lower() for s in rec["action_details"])
    assert any("autopay" in s.lower() for s in rec["action_details"])
    assert len(rec["targeted_factors"]) >= 3


def test_retention_low_risk_routine():
    """Verify low risk profile yields standard loyalty appreciation."""
    customer = {
        "tenure": 60,
        "MonthlyCharges": 35.0,
        "Contract": "Two year",
        "TechSupport": "Yes",
        "InternetService": "DSL",
        "PaymentMethod": "Credit card (automatic)",
    }
    rec = generate_retention_recommendation("Low", 0.12, customer)

    assert rec["priority"] == "Standard"
    assert "Routine Service Monitoring" in rec["primary_action"]
    assert any("loyalty" in s.lower() or "referral" in s.lower() for s in rec["action_details"])


def test_retention_medium_risk():
    """Verify medium risk profile yields proactive account optimization."""
    customer = {
        "tenure": 18,
        "MonthlyCharges": 65.0,
        "Contract": "Month-to-month",
        "TechSupport": "No",
        "InternetService": "DSL",
        "PaymentMethod": "Mailed check",
    }
    rec = generate_retention_recommendation("Medium", 0.42, customer)

    assert rec["priority"] == "Moderate"
    assert "Account Optimization" in rec["primary_action"]
    assert len(rec["action_details"]) > 0
