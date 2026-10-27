"""
Customer Retention Recommendation Engine for Telecom Operator Subscribers.
Translates predicted churn probability, risk category, and key customer attributes
into actionable, transparent, rule-based business retention strategies.
"""

from typing import Any


def generate_retention_recommendation(
    risk_category: str,
    churn_prob: float,
    customer_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluates customer profile and churn risk to prescribe targeted retention actions.
    
    Parameters:
        risk_category: 'High', 'Medium', or 'Low'
        churn_prob: float estimated probability (0.0 to 1.0)
        customer_data: dictionary containing customer attributes (tenure, MonthlyCharges,
                       Contract, TechSupport, InternetService, PaymentMethod, etc.)
                       
    Returns:
        dict containing:
            - primary_action: Short descriptive action title
            - action_details: Detailed business justification and offer details
            - priority: 'Critical', 'Urgent', 'Moderate', or 'Standard'
            - targeted_factors: List of features triggering the recommendation
            - disclaimer: Standard business note indicating suggestions are not guarantees
    """
    tenure = float(customer_data.get("tenure", 0))
    monthly_charges = float(customer_data.get("MonthlyCharges", 0.0))
    contract = str(customer_data.get("Contract", "Month-to-month"))
    tech_support = str(customer_data.get("TechSupport", "No"))
    internet_service = str(customer_data.get("InternetService", "Fiber optic"))
    payment_method = str(customer_data.get("PaymentMethod", "Electronic check"))
    online_security = str(customer_data.get("OnlineSecurity", "No"))

    triggered_factors = []
    strategies = []
    priority = "Standard"

    if risk_category == "High":
        priority = "Critical" if churn_prob >= 0.75 else "Urgent"

        # Rule 1: Month-to-month contract vulnerability
        if contract == "Month-to-month":
            triggered_factors.append("Month-to-month contract (low lock-in)")
            strategies.append(
                "Present an exclusive 12-month or 24-month contract transition offer with "
                "price-lock guarantee and a 10% annual loyalty billing credit."
            )

        # Rule 2: High monthly spend churn sensitivity
        if monthly_charges >= 70.0:
            triggered_factors.append(f"High monthly billing (${monthly_charges:.2f}/mo)")
            strategies.append(
                "Evaluate custom loyalty bundling (e.g. streaming discount or bundled "
                "family plan) to reduce perceived expense while maintaining APRU."
            )

        # Rule 3: Early tenure onboarding fragility
        if tenure <= 12:
            triggered_factors.append(f"Early customer lifecycle stage ({int(tenure)} months tenure)")
            strategies.append(
                "Assign a dedicated onboarding customer success agent for a courtesy "
                "service satisfaction check-in to resolve first-year onboarding frictions."
            )

        # Rule 4: Absence of technical support on complex internet service
        if tech_support == "No" and internet_service in ["Fiber optic", "DSL"]:
            triggered_factors.append(f"{internet_service} connection without Tech Support")
            strategies.append(
                "Provision a complimentary 3-month Premium Technical Support trial and "
                "automated line diagnostics to resolve bandwidth/latency grievances."
            )

        # Rule 5: Electronic check payment friction
        if "Electronic check" in payment_method:
            triggered_factors.append("Payment via Electronic Check (high churn association)")
            strategies.append(
                "Encourage automatic credit card or bank debit setup with a one-time "
                "$10 billing credit for switching to recurring autopay."
            )

        # Rule 6: Missing online security
        if online_security == "No" and internet_service != "No":
            triggered_factors.append("Lacks online security package")
            strategies.append(
                "Offer complimentary cybersecurity & antivirus suite to elevate perceived service value."
            )

        if not strategies:
            strategies.append(
                "Dispatch an urgent senior customer relationship specialist to conduct "
                "an account health review and offer tailored contract relief."
            )

        primary_action = "Immediate Senior Account Intervention & Contract Incentive"

    elif risk_category == "Medium":
        priority = "Moderate"

        if contract == "Month-to-month":
            triggered_factors.append("Month-to-month flexible plan")
            strategies.append(
                "Share educational comparisons showcasing long-term savings available on annual plans."
            )

        if tech_support == "No":
            triggered_factors.append("No active tech support subscription")
            strategies.append(
                "Send helpful digital self-service guides and customer support hotline contacts."
            )

        if monthly_charges >= 60.0:
            triggered_factors.append(f"Above-average billing (${monthly_charges:.2f})")
            strategies.append(
                "Perform an account optimization review to ensure subscriber is utilizing all included features."
            )

        if not strategies:
            strategies.append(
                "Schedule a quarterly customer satisfaction survey with a small loyalty reward coupon."
            )

        primary_action = "Proactive Account Optimization & Mid-Tier Engagement"

    else:  # Low Risk
        priority = "Standard"
        triggered_factors.append(f"Stable relationship ({int(tenure)} months tenure, {contract} plan)")
        strategies.append(
            "Maintain standard service excellence, deliver automated quarterly newsletters, "
            "and invite to participate in brand referral reward programs."
        )
        primary_action = "Routine Service Monitoring & Loyalty Appreciation"

    return {
        "primary_action": primary_action,
        "action_details": strategies,
        "priority": priority,
        "targeted_factors": triggered_factors,
        "disclaimer": (
            "Suggested retention actions are rule-based advisory guidelines designed for operational "
            "support. They do not guarantee customer retention or substitute for direct customer dialogue."
        ),
    }
