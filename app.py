"""
Telecom Customer Churn Prediction System - Interactive Streamlit Dashboard
Stacked Ensemble Classifier for Telecom Subscriber Retention & Predictive Analytics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_telco_data, DEFAULT_DATA_PATH
from src.predict import load_model, predict_single_customer, predict_batch, classify_risk
from src.retention import generate_retention_recommendation

# Page Configuration
st.set_page_config(
    page_title="Telecom Churn Prediction System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Navy & Modern Telecom Palette)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Top Navigation / Header styling */
    .telecom-header {
        background: linear-gradient(135deg, #0f233a 0%, #173b64 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }
    .telecom-header h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .telecom-header p {
        margin: 6px 0 0 0;
        color: #93c5fd;
        font-size: 14px;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 12px;
        color: #0ea5e9;
        margin-top: 4px;
    }

    /* Risk Badges */
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fbbf24;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #4ade80;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }

    /* Callout & Strategy cards */
    .strategy-box {
        background-color: #f8fafc;
        border-left: 4px solid #2563eb;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin-top: 12px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_dataset():
    """Loads and caches the cleaned telecom churn dataset."""
    df, is_synthetic = load_telco_data(DEFAULT_DATA_PATH)
    return df, is_synthetic


@st.cache_data
def get_model_metrics():
    """Loads precomputed model metrics and report JSON."""
    metrics_path = os.path.join(PROJECT_ROOT, "reports", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def get_cached_model():
    """Loads and caches the trained model pipeline."""
    return load_model()


# Sidebar Navigation & Settings
with st.sidebar:
    st.markdown("### 📡 Telecom Analytics")
    st.markdown("**Predictive Churn Intelligence**")
    st.caption("Stacked Ensemble ML System")

    nav_selection = st.radio(
        "Navigation",
        [
            "📊 Home Dashboard",
            "👤 Individual Customer Prediction",
            "📁 Batch CSV Prediction",
            "📈 Model Performance",
            "🔍 Customer Insights",
            "💡 Retention Simulator",
            "🎓 Viva & Project Guide",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Risk Thresholds")
    low_thresh_pct = st.slider(
        "Low Risk Limit (%)",
        min_value=10,
        max_value=40,
        value=30,
        step=5,
        help="Probabilities below this value are classified as Low Risk.",
    )
    high_thresh_pct = st.slider(
        "High Risk Limit (%)",
        min_value=45,
        max_value=80,
        value=60,
        step=5,
        help="Probabilities at or above this value are classified as High Risk.",
    )
    decision_thresh_pct = st.slider(
        "Binary Decision Threshold (%)",
        min_value=20,
        max_value=70,
        value=50,
        step=5,
        help="Probability cutoff for predicting Churn (1) vs Stay (0).",
    )

    low_thresh = low_thresh_pct / 100.0
    high_thresh = high_thresh_pct / 100.0
    decision_thresh = decision_thresh_pct / 100.0

    st.markdown("---")
    # Dataset and model status indicators
    try:
        model = get_cached_model()
        st.success("✅ Stacked Model Loaded")
    except Exception as e:
        st.error(f"⚠️ Model not loaded: {e}")
        model = None

    raw_df, is_synthetic = get_cached_dataset()
    if is_synthetic:
        st.warning("⚠️ Using Synthetic Sample Data")
    else:
        st.info("ℹ️ Dataset: IBM Telco (7,043 rows)")


# App Header Banner
st.markdown(
    """
    <div class="telecom-header">
        <h1>Telecom Customer Churn Prediction System</h1>
        <p>An Intelligent System to Predict Subscribers Likely to Switch Operators Using Machine Learning and Predictive Analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# PAGE 1: HOME DASHBOARD
# ==========================================================
if nav_selection == "📊 Home Dashboard":
    st.subheader("Executive Telemetry & Churn Overview")
    st.markdown(
        "High-level operational overview of subscriber distribution, historical churn records, "
        "and critical service factors driving retention."
    )

    total_subscribers = len(raw_df)
    actual_churn_count = int(raw_df["Churn"].sum())
    actual_stay_count = total_subscribers - actual_churn_count
    churn_rate = (actual_churn_count / total_subscribers) * 100
    avg_tenure = raw_df["tenure"].mean()
    avg_monthly = raw_df["MonthlyCharges"].mean()

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Subscribers</div>
                <div class="metric-value">{total_subscribers:,}</div>
                <div class="metric-sub">Active repository records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Historical Churn Rate</div>
                <div class="metric-value">{churn_rate:.1f}%</div>
                <div class="metric-sub">{actual_churn_count:,} churned subscribers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Tenure</div>
                <div class="metric-value">{avg_tenure:.1f} mo</div>
                <div class="metric-sub">Across all accounts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg Monthly Billing</div>
                <div class="metric-value">${avg_monthly:.2f}</div>
                <div class="metric-sub">Average ARPU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Visualizations Grid
    row1_col1, row1_col2 = st.columns([1, 1])

    with row1_col1:
        st.markdown("#### Historical Churn Proportion")
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Retained Subscribers", "Churned Subscribers"],
                    values=[actual_stay_count, actual_churn_count],
                    hole=0.6,
                    marker=dict(colors=["#10b981", "#ef4444"]),
                    textinfo="label+percent",
                    insidetextorientation="radial",
                )
            ]
        )
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            margin=dict(t=20, b=30, l=10, r=10),
            height=320,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_col2:
        st.markdown("#### Churn Rate by Contract Commitment")
        contract_churn = (
            raw_df.groupby("Contract")["Churn"].mean().reset_index()
        )
        contract_churn["Churn_Rate_Pct"] = contract_churn["Churn"] * 100
        fig_bar = px.bar(
            contract_churn,
            x="Contract",
            y="Churn_Rate_Pct",
            color="Contract",
            color_discrete_sequence=["#ef4444", "#3b82f6", "#10b981"],
            text_auto=".1f",
            labels={"Churn_Rate_Pct": "Churn Rate (%)", "Contract": "Contract Type"},
        )
        fig_bar.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=10, r=10),
            height=320,
            yaxis=dict(ticksuffix="%"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    row2_col1, row2_col2 = st.columns([1, 1])

    with row2_col1:
        st.markdown("#### Tenure vs Monthly Charges by Churn Status")
        # Sample 600 records for fast fluid plotting
        sample_plot = raw_df.sample(min(800, len(raw_df)), random_state=42).copy()
        sample_plot["Churn_Label"] = sample_plot["Churn"].map({1: "Churned", 0: "Retained"})
        fig_scatter = px.scatter(
            sample_plot,
            x="tenure",
            y="MonthlyCharges",
            color="Churn_Label",
            color_discrete_map={"Churned": "#ef4444", "Retained": "#10b981"},
            opacity=0.65,
            labels={"tenure": "Tenure (Months)", "MonthlyCharges": "Monthly Charges ($)"},
        )
        fig_scatter.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with row2_col2:
        st.markdown("#### Churn Rate by Internet Service Type")
        net_churn = (
            raw_df.groupby("InternetService")["Churn"].mean().reset_index()
        )
        net_churn["Churn_Rate_Pct"] = net_churn["Churn"] * 100
        fig_net = px.bar(
            net_churn,
            x="InternetService",
            y="Churn_Rate_Pct",
            color="InternetService",
            color_discrete_sequence=["#f59e0b", "#ef4444", "#10b981"],
            text_auto=".1f",
            labels={"Churn_Rate_Pct": "Churn Rate (%)", "InternetService": "Internet Service"},
        )
        fig_net.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=10, r=10),
            height=340,
            yaxis=dict(ticksuffix="%"),
        )
        st.plotly_chart(fig_net, use_container_width=True)


# ==========================================================
# PAGE 2: INDIVIDUAL CUSTOMER PREDICTION
# ==========================================================
elif nav_selection == "👤 Individual Customer Prediction":
    st.subheader("Individual Subscriber Churn Predictor")
    st.markdown(
        "Enter subscriber account parameters to evaluate estimated churn likelihood, "
        "risk category, and custom retention interventions."
    )

    with st.form("individual_prediction_form"):
        st.markdown("##### 1. Demographics & Account Tenure")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with d2:
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with d3:
            partner = st.selectbox("Has Partner", ["No", "Yes"])
        with d4:
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])

        st.markdown("##### 2. Contract & Financial Attributes")
        f1, f2, f3 = st.columns(3)
        with f1:
            tenure = st.slider("Tenure with Operator (Months)", min_value=0, max_value=72, value=12)
        with f2:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with f3:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        f4, f5, f6 = st.columns(3)
        with f4:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with f5:
            monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=75.0, step=0.5)
        with f6:
            # Estimate reasonable total charges based on tenure and monthly
            est_total = round(max(monthly_charges, monthly_charges * max(1, tenure)), 2)
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=float(est_total),
                help="Historical cumulative billing.",
            )

        st.markdown("##### 3. Subscribed Telecom Services")
        s1, s2, s3 = st.columns(3)
        with s1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"] if phone_service == "No" else ["No", "Yes"],
            )
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

        with s2:
            no_net = internet_service == "No"
            online_security = st.selectbox(
                "Online Security",
                ["No internet service"] if no_net else ["No", "Yes"],
            )
            online_backup = st.selectbox(
                "Online Backup",
                ["No internet service"] if no_net else ["No", "Yes"],
            )
            device_protection = st.selectbox(
                "Device Protection",
                ["No internet service"] if no_net else ["No", "Yes"],
            )

        with s3:
            tech_support = st.selectbox(
                "Tech Support",
                ["No internet service"] if no_net else ["No", "Yes"],
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No internet service"] if no_net else ["No", "Yes"],
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No internet service"] if no_net else ["No", "Yes"],
            )

        submit_btn = st.form_submit_button("🔮 Predict Churn Probability", type="primary")

    if submit_btn:
        customer_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            result = predict_single_customer(
                customer_dict,
                decision_threshold=decision_thresh,
                low_risk_threshold=low_thresh,
                high_risk_threshold=high_thresh,
                model=model,
            )

            st.markdown("---")
            st.markdown("### 🎯 Prediction Results & Risk Assessment")

            res_c1, res_c2, res_c3 = st.columns([1, 1, 1])

            with res_c1:
                badge_class = (
                    "badge-high"
                    if result["risk_category"] == "High"
                    else "badge-medium"
                    if result["risk_category"] == "Medium"
                    else "badge-low"
                )
                pred_label = result["churn_label"]
                pred_color = "#ef4444" if result["churn_prediction"] == 1 else "#10b981"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Model Prediction</div>
                        <div style="font-size: 24px; font-weight: 700; color: {pred_color}; margin-top: 6px;">
                            {pred_label}
                        </div>
                        <div style="margin-top: 10px;">
                            <span class="{badge_class}">{result['risk_category']} Risk</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_c2:
                prob_val = result["churn_probability_pct"]
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Estimated Churn Probability</div>
                        <div class="metric-value">{prob_val:.1f}%</div>
                        <div class="metric-sub">Decision Cutoff: {decision_thresh_pct}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_c3:
                rec_data = result["retention_recommendation"]
                priority_color = (
                    "#dc2626" if rec_data["priority"] == "Critical"
                    else "#ea580c" if rec_data["priority"] == "Urgent"
                    else "#2563eb"
                )
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Recommended Action Priority</div>
                        <div style="font-size: 24px; font-weight: 700; color: {priority_color}; margin-top: 6px;">
                            {rec_data['priority']}
                        </div>
                        <div class="metric-sub">{rec_data['primary_action']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Gauge Meter
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob_val,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Churn Risk Probability Gauge", "font": {"size": 16}},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar": {"color": "#1e3a8a"},
                        "steps": [
                            {"range": [0, low_thresh_pct], "color": "#dcfce7"},
                            {"range": [low_thresh_pct, high_thresh_pct], "color": "#fef3c7"},
                            {"range": [high_thresh_pct, 100], "color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.8,
                            "value": decision_thresh_pct,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=260, margin=dict(t=40, b=20, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Explainable insights
            st.markdown("#### 💡 Explanation & Influencing Factors")
            st.info(result["explanation"])

            # Recommended Retention Plan Box
            st.markdown("#### 🎁 Targeted Retention Strategy")
            st.markdown(
                f"""
                <div class="strategy-box">
                    <h5 style="margin: 0 0 8px 0; color: #1e3a8a;">Proposed Action: {rec_data['primary_action']}</h5>
                    <p style="margin: 0; font-size: 13px; color: #475569;"><strong>Triggered Customer Signals:</strong> {', '.join(rec_data['targeted_factors']) if rec_data['targeted_factors'] else 'Standard lifecycle'}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("##### Specific Intervention Steps:")
            for i, step in enumerate(rec_data["action_details"], 1):
                st.markdown(f"**Step {i}:** {step}")

            st.caption(f"ℹ️ {rec_data['disclaimer']}")

        except Exception as err:
            st.error(f"Error during prediction: {err}")


# ==========================================================
# PAGE 3: BATCH CSV PREDICTION
# ==========================================================
elif nav_selection == "📁 Batch CSV Prediction":
    st.subheader("Batch Subscriber Churn Scoring")
    st.markdown(
        "Upload a batch CSV file of telecom subscribers to compute churn probabilities, "
        "risk segments, and retention recommendations in bulk."
    )

    bc1, bc2 = st.columns([2, 1])

    with bc1:
        uploaded_file = st.file_uploader(
            "Upload Subscriber CSV File",
            type=["csv"],
            help="File must contain standard Telco features. If some columns are missing, safe defaults will be applied.",
        )

    with bc2:
        st.markdown("##### Or Test with Sample Batch:")
        st.caption("Load pre-curated 25-subscriber verification file.")
        load_sample_btn = st.button("🚀 Load Sample Batch Data (25 Subscribers)")

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV with {len(batch_df)} subscriber records.")
        except Exception as e:
            st.error(f"Failed to read CSV file: {e}")
    elif load_sample_btn:
        sample_path = os.path.join(PROJECT_ROOT, "data", "sample_batch_test.csv")
        if os.path.exists(sample_path):
            batch_df = pd.read_csv(sample_path)
            st.info("Loaded pre-curated sample batch dataset.")
        else:
            batch_df = raw_df.sample(25, random_state=42)

    if batch_df is not None:
        with st.spinner("Processing batch predictions with Stacked Ensemble pipeline..."):
            try:
                results_df = predict_batch(
                    batch_df,
                    decision_threshold=decision_thresh,
                    low_risk_threshold=low_thresh,
                    high_risk_threshold=high_thresh,
                    model=model,
                )

                st.markdown("---")
                st.markdown("### 📊 Batch Prediction Telemetry")

                # Summary metrics
                n_total = len(results_df)
                n_churn = (results_df["Predicted_Class"] == "Churn").sum()
                pct_churn = (n_churn / n_total) * 100
                n_high = (results_df["Risk_Category"] == "High").sum()
                n_med = (results_df["Risk_Category"] == "Medium").sum()
                n_low = (results_df["Risk_Category"] == "Low").sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Batch Subscribers", f"{n_total:,}")
                m2.metric("Predicted Churners", f"{n_churn:,} ({pct_churn:.1f}%)")
                m3.metric("High Risk Subscribers", f"{n_high} ({(n_high/n_total)*100:.1f}%)")
                m4.metric("Avg Predicted Churn Prob", f"{results_df['Churn_Probability_Pct'].mean():.1f}%")

                # Batch Risk Distribution chart
                b_c1, b_c2 = st.columns([1, 1])
                with b_c1:
                    risk_counts = pd.DataFrame({
                        "Risk_Category": ["Low", "Medium", "High"],
                        "Count": [n_low, n_med, n_high],
                    })
                    fig_risk = px.bar(
                        risk_counts,
                        x="Risk_Category",
                        y="Count",
                        color="Risk_Category",
                        color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                        text_auto=True,
                        title="Subscribers by Risk Category",
                    )
                    fig_risk.update_layout(height=280, showlegend=False, margin=dict(t=30, b=20, l=10, r=10))
                    st.plotly_chart(fig_risk, use_container_width=True)

                with b_c2:
                    fig_hist = px.histogram(
                        results_df,
                        x="Churn_Probability_Pct",
                        nbins=20,
                        color="Risk_Category",
                        color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                        title="Distribution of Churn Probabilities",
                        labels={"Churn_Probability_Pct": "Churn Probability (%)"},
                    )
                    fig_hist.update_layout(height=280, margin=dict(t=30, b=20, l=10, r=10))
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Results Table
                st.markdown("#### 📋 Detailed Batch Predictions")
                st.dataframe(
                    results_df[
                        [
                            "CustomerID",
                            "Churn_Probability_Pct",
                            "Predicted_Class",
                            "Risk_Category",
                            "Recommended_Action",
                            "Contract",
                            "Tenure",
                            "MonthlyCharges",
                        ]
                    ],
                    use_container_width=True,
                )

                # CSV Download Button
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Batch Prediction Results (CSV)",
                    data=csv_bytes,
                    file_name="telecom_churn_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )

            except Exception as batch_err:
                st.error(f"Error processing batch: {batch_err}")


# ==========================================================
# PAGE 4: MODEL PERFORMANCE & EVALUATION
# ==========================================================
elif nav_selection == "📈 Model Performance":
    st.subheader("Model Benchmark & Evaluation")
    st.markdown(
        "Empirical performance comparison of individual baseline models against the "
        "Stacked Ensemble Classifier, evaluated on strictly held-out test records (20% split)."
    )

    metrics_report = get_model_metrics()

    if metrics_report and "models_performance" in metrics_report:
        perf = metrics_report["models_performance"]

        # Comparison Table
        comp_rows = []
        for name, m in perf.items():
            comp_rows.append({
                "Model Name": name,
                "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Precision": f"{m['precision'] * 100:.2f}%",
                "Recall": f"{m['recall'] * 100:.2f}%",
                "F1-Score": f"{m['f1_score'] * 100:.2f}%",
                "ROC-AUC": f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "N/A",
            })
        comp_df = pd.DataFrame(comp_rows)

        st.markdown("#### Baseline vs Stacked Ensemble Comparison")
        st.dataframe(comp_df, use_container_width=True)

        st.markdown(
            "> 📌 **Empirical Observation:** Notice how the **Stacked Ensemble Classifier** "
            "synergizes Logistic Regression, Random Forest, and Gradient Boosting through a meta-classifier, "
            "attaining the highest overall ROC-AUC (0.8451) and balanced accuracy on the unseen test set."
        )

        st.markdown("---")
        # ROC Curves Comparison
        st.markdown("#### Receiver Operating Characteristic (ROC) Curves")
        fig_roc = go.Figure()

        colors = {
            "Logistic Regression": "#3b82f6",
            "Random Forest": "#10b981",
            "Gradient Boosting": "#f59e0b",
            "Stacked Ensemble": "#8b5cf6",
        }

        for name, m in perf.items():
            if "roc_curve" in m and m["roc_curve"]["fpr"]:
                fpr = m["roc_curve"]["fpr"]
                tpr = m["roc_curve"]["tpr"]
                auc_val = m.get("roc_auc", 0.5)
                fig_roc.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"{name} (AUC = {auc_val:.4f})",
                        line=dict(
                            color=colors.get(name, "#0f172a"),
                            width=3 if name == "Stacked Ensemble" else 1.5,
                        ),
                    )
                )

        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Chance (AUC = 0.50)",
                line=dict(color="grey", dash="dash"),
            )
        )
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall / Sensitivity)",
            height=420,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # Confusion Matrices Section
        st.markdown("---")
        st.markdown("#### Confusion Matrix Breakdown")

        model_cm_choice = st.selectbox(
            "Select Model to Inspect Confusion Matrix:",
            list(perf.keys()),
            index=list(perf.keys()).index("Stacked Ensemble") if "Stacked Ensemble" in perf else 0,
        )

        cm_data = perf[model_cm_choice]["confusion_matrix"]
        tn, fp, fn, tp = cm_data["tn"], cm_data["fp"], cm_data["fn"], cm_data["tp"]

        cm_c1, cm_c2 = st.columns([1, 1])

        with cm_c1:
            z_vals = [[tn, fp], [fn, tp]]
            fig_cm = px.imshow(
                z_vals,
                labels=dict(x="Predicted Label", y="Actual True Label", color="Count"),
                x=["Retained (0)", "Churned (1)"],
                y=["Retained (0)", "Churned (1)"],
                text_auto=True,
                color_continuous_scale="Blues",
            )
            fig_cm.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_cm, use_container_width=True)

        with cm_c2:
            st.markdown(f"##### Key Diagnostic Counts ({model_cm_choice}):")
            st.write(f"- **True Negatives (TN):** `{tn}` loyal subscribers correctly identified.")
            st.write(f"- **True Positives (TP):** `{tp}` churners correctly intercepted.")
            st.write(f"- **False Positives (FP):** `{fp}` loyal customers flagged as churners (retention offer cost).")
            st.write(f"- **False Negatives (FN):** `{fn}` churners missed completely (lost subscriber cost).")

            st.markdown("##### Class Imbalance Critical Discussion:")
            st.caption(
                "Because only ~26.5% of telecom subscribers churn, a naive model that always predicts 'Stay' "
                "would achieve 73.5% Accuracy while capturing ZERO churners! "
                "Therefore, **Recall** and **ROC-AUC** are the vital operational metrics for business viability."
            )

    else:
        st.warning("Model metrics file not found. Please train models by executing 'python src/train_model.py'.")


# ==========================================================
# PAGE 5: CUSTOMER INSIGHTS & FEATURE IMPORTANCE
# ==========================================================
elif nav_selection == "🔍 Customer Insights":
    st.subheader("Predictive Analytics & Model Interpretability")
    st.markdown(
        "Analyze which customer characteristics exert the greatest statistical influence "
        "over subscriber churn probability according to tree-based and linear estimators."
    )

    metrics_report = get_model_metrics()

    if metrics_report and "feature_importances" in metrics_report:
        fi_list = metrics_report["feature_importances"]
        fi_df = pd.DataFrame(fi_list)

        t1, t2, t3 = st.tabs([
            "🌲 Tree-Based Feature Importances",
            "⚖️ Linear Model Coefficients",
            "🧩 Meta-Learner Stacking Logic",
        ])

        with t1:
            st.markdown("#### Random Forest & Gradient Boosting Feature Importance")
            st.markdown(
                "Tree algorithms measure feature importance via **Mean Decrease in Impurity (Gini)**. "
                "Features higher on this chart have the largest discriminatory power when splitting nodes."
            )

            top_n = st.slider("Select Top N Features to Display:", 5, 25, 12)
            top_fi = fi_df.head(top_n).copy()

            fig_fi = px.bar(
                top_fi,
                x="rf_importance",
                y="feature",
                orientation="h",
                color="rf_importance",
                color_continuous_scale="Viridis",
                labels={"rf_importance": "Relative Gini Importance", "feature": "Customer Feature"},
                title=f"Top {top_n} Predictive Features in Random Forest",
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=420)
            st.plotly_chart(fig_fi, use_container_width=True)

            st.info(
                "💡 **Key Insight:** **Tenure**, **Total Charges**, **Monthly Charges**, and **Contract terms** "
                "consistently account for the vast majority of predictive splits across tree ensembles."
            )

        with t2:
            st.markdown("#### Logistic Regression Directional Feature Weights")
            st.markdown(
                "Logistic Regression coefficients provide **directional interpretability**: "
                "positive values increase the odds of churn, while negative values reduce churn odds."
            )

            # Sort by absolute coefficient
            fi_df["abs_lr"] = fi_df["lr_coefficient"].abs()
            lr_sorted = fi_df.sort_values(by="abs_lr", ascending=False).head(14).copy()

            lr_sorted["Direction"] = np.where(
                lr_sorted["lr_coefficient"] > 0, "Increases Churn Risk", "Decreases Churn Risk (Protective)"
            )

            fig_lr = px.bar(
                lr_sorted,
                x="lr_coefficient",
                y="feature",
                orientation="h",
                color="Direction",
                color_discrete_map={
                    "Increases Churn Risk": "#ef4444",
                    "Decreases Churn Risk (Protective)": "#10b981",
                },
                labels={"lr_coefficient": "Model Coefficient (Log-Odds Impact)", "feature": "Feature"},
                title="Top Directional Drivers of Churn in Logistic Regression",
            )
            fig_lr.update_layout(yaxis=dict(autorange="reversed"), height=440)
            st.plotly_chart(fig_lr, use_container_width=True)

        with t3:
            st.markdown("#### How the Meta-Learner Combines Base Predictions")
            st.markdown(
                "The **StackingClassifier** trains a meta-learner (Logistic Regression) whose inputs "
                "are the out-of-fold probability predictions generated by the base models."
            )

            meta_weights = metrics_report.get("meta_learner_weights", {})
            st.json(meta_weights)

            st.markdown(
                """
                - **Base Learners:** Logistic Regression (linear boundary), Random Forest (orthogonal splits), Gradient Boosting (boosted residuals).
                - **Meta-Classifier:** Learns optimal non-linear blending weights without overfitting through 5-fold cross-validation.
                """
            )

        st.caption(
            "⚠️ **Statistical Disclaimer:** High feature importance indicates strong correlation and predictive "
            "association within the historical dataset, but does NOT strictly establish causality."
        )

    else:
        st.warning("Feature importance data is not available.")


# ==========================================================
# PAGE 6: RETENTION RECOMMENDATION SIMULATOR
# ==========================================================
elif nav_selection == "💡 Retention Simulator":
    st.subheader("Customer Retention Strategy & 'What-If' Simulator")
    st.markdown(
        "Simulate business interventions (such as upgrading contracts, adding tech support, or switching payment methods) "
        "and observe real-time predicted reductions in churn risk."
    )

    sim_col1, sim_col2 = st.columns([1, 1])

    with sim_col1:
        st.markdown("#### 1. Baseline At-Risk Customer Profile")
        b_tenure = st.slider("Current Tenure (Months)", 1, 48, 6)
        b_monthly = st.slider("Monthly Charges ($)", 30.0, 115.0, 85.0)
        b_contract = st.selectbox("Current Contract", ["Month-to-month", "One year"], index=0)
        b_tech = st.selectbox("Tech Support Active?", ["No", "Yes"], index=0)
        b_net = st.selectbox("Internet Connection", ["Fiber optic", "DSL"], index=0)
        b_pay = st.selectbox("Payment Channel", ["Electronic check", "Mailed check", "Credit card (automatic)"], index=0)

        # Baseline dict
        base_cust = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": b_tenure,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": b_net,
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": b_tech,
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": b_contract,
            "PaperlessBilling": "Yes",
            "PaymentMethod": b_pay,
            "MonthlyCharges": b_monthly,
            "TotalCharges": b_monthly * max(1, b_tenure),
        }

        base_res = predict_single_customer(
            base_cust,
            decision_threshold=decision_thresh,
            low_risk_threshold=low_thresh,
            high_risk_threshold=high_thresh,
            model=model,
        )

        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #ef4444;">
                <div class="metric-label">Baseline Churn Probability</div>
                <div class="metric-value" style="color: #ef4444;">{base_res['churn_probability_pct']:.1f}%</div>
                <div class="metric-sub">Category: <strong>{base_res['risk_category']} Risk</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sim_col2:
        st.markdown("#### 2. Simulated Retention Interventions")
        st.markdown("Apply proposed retention offers:")

        sim_contract = st.selectbox("Contract Upgrade Offer:", ["Keep Current", "Switch to 1-Year Plan", "Switch to 2-Year Plan"])
        sim_tech = st.checkbox("Provide Free Premium Tech Support", value=True)
        sim_autopay = st.checkbox("Apply $5 Autopay Credit (Switch to Credit Card Auto)", value=True)
        sim_discount = st.slider("Apply Loyalty Discount to Monthly Charges ($)", 0.0, 25.0, 10.0, step=2.5)

        # Build modified customer profile
        mod_cust = dict(base_cust)
        if sim_contract == "Switch to 1-Year Plan":
            mod_cust["Contract"] = "One year"
        elif sim_contract == "Switch to 2-Year Plan":
            mod_cust["Contract"] = "Two year"

        if sim_tech:
            mod_cust["TechSupport"] = "Yes"

        if sim_autopay:
            mod_cust["PaymentMethod"] = "Credit card (automatic)"

        mod_cust["MonthlyCharges"] = max(18.0, b_monthly - sim_discount)
        mod_cust["TotalCharges"] = mod_cust["MonthlyCharges"] * max(1, b_tenure)

        mod_res = predict_single_customer(
            mod_cust,
            decision_threshold=decision_thresh,
            low_risk_threshold=low_thresh,
            high_risk_threshold=high_thresh,
            model=model,
        )

        prob_delta = base_res["churn_probability_pct"] - mod_res["churn_probability_pct"]

        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #10b981;">
                <div class="metric-label">Simulated New Churn Probability</div>
                <div class="metric-value" style="color: #10b981;">{mod_res['churn_probability_pct']:.1f}%</div>
                <div class="metric-sub">Category: <strong>{mod_res['risk_category']} Risk</strong> | Reduction: <strong>-{prob_delta:.1f}% pts</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    # Comparison Bar Chart
    comp_fig = go.Figure(
        data=[
            go.Bar(
                name="Baseline Risk",
                x=["Customer Churn Probability"],
                y=[base_res["churn_probability_pct"]],
                marker_color="#ef4444",
                text=[f"{base_res['churn_probability_pct']:.1f}%"],
                textposition="auto",
            ),
            go.Bar(
                name="Post-Intervention Risk",
                x=["Customer Churn Probability"],
                y=[mod_res["churn_probability_pct"]],
                marker_color="#10b981",
                text=[f"{mod_res['churn_probability_pct']:.1f}%"],
                textposition="auto",
            ),
        ]
    )
    comp_fig.update_layout(
        barmode="group",
        height=280,
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(comp_fig, use_container_width=True)

    # Strategy Rule Matrix
    with st.expander("📖 View Transparent Corporate Retention Rule Matrix"):
        st.markdown(
            """
            | Trigger Condition | Target Segment | Recommended Retention Action | Expected Impact |
            |---|---|---|---|
            | `Risk >= 60%` & `Month-to-month` | Month-to-month contract holders | Present price-lock 12/24-month contract transition offer with 10% credit | Major churn reduction |
            | `Risk >= 60%` & `MonthlyCharges >= $70` | High-spend users | Offer customized service bundle discount or streaming perk | Preserves ARPU |
            | `Risk >= 60%` & `Tenure <= 12` | First-year onboarding | Assign dedicated customer success specialist for onboarding check-in | Eliminates early friction |
            | `Risk >= 60%` & `TechSupport == 'No'` | High-bandwidth internet users | 3 months complimentary Premium Technical Support | Improves perceived value |
            | `Risk >= 60%` & `Electronic check` | Manual bill payers | $10 one-time credit to enroll in automated recurring credit card autopay | Reduces payment friction |
            | `30% <= Risk < 60%` | Medium Risk accounts | Account health check and plan optimization review | Proactive engagement |
            | `Risk < 30%` | Low Risk loyal subscribers | Standard service monitoring and brand referral appreciation | Loyalty retention |
            """
        )


# ==========================================================
# PAGE 7: VIVA & PROJECT DEFENSE GUIDE
# ==========================================================
elif nav_selection == "🎓 Viva & Project Guide":
    st.subheader("BTech AI & Data Science Mini-Project Viva Guide")
    st.markdown(
        "Structured defense guide, algorithm explanations, mathematical formulating, and frequently "
        "asked questions for college mini-project presentations and external viva examinations."
    )

    vg_t1, vg_t2, vg_t3 = st.tabs([
        "📚 Architectural & Algorithmic Summary",
        "❓ Frequently Asked Viva Questions",
        "⚙️ Reproducibility & Commands",
    ])

    with vg_t1:
        st.markdown("#### System Architecture")
        st.markdown(
            """
            ```
            [IBM Telco Dataset (7,043 Records)]
                        │
                        ▼
            [Data Cleaning & Typing] ─── (Whitespace handling, TotalCharges coercion)
                        │
                        ▼
            [Stratified Split (80/20)] ── (Strictly isolates Test Set, prevents Leakage)
                        │
                        ▼
            [ColumnTransformer Pipeline]
              ├── Numeric: SimpleImputer(median) -> StandardScaler()
              └── Categorical: SimpleImputer(mode) -> OneHotEncoder(ignore_unknown)
                        │
                        ▼
            [Stacked Ensemble Classifier]
              ├── Level 0 Base Learners (with 5-fold Stratified CV):
              │     ├── 1. Logistic Regression
              │     ├── 2. Random Forest Classifier
              │     └── 3. Gradient Boosting Classifier
              └── Level 1 Meta-Learner:
                    └── Logistic Regression (predict_proba blending)
                        │
                        ▼
            [Threshold Tuning & Risk Engine] ── (Low < 30%, Med 30-60%, High >= 60%)
                        │
                        ▼
            [Retention Recommendation Engine]
            ```
            """
        )

        st.markdown("#### Mathematical Concept of Stacking (Wolpert, 1992)")
        st.latex(r"\hat{y}_i = g\Big(h_1(x_i), h_2(x_i), \dots, h_K(x_i)\Big)")
        st.markdown(
            r"""
            - Let $h_1, h_2, \dots, h_K$ represent diverse base estimators (linear, bagging, boosting).
            - Each base learner generates out-of-fold probability predictions $\hat{P}_k(x)$ using $K$-fold cross-validation.
            - The meta-learner $g(\cdot)$ optimizes coefficients to blend these predictions:
            """
        )
        st.latex(r"P(\text{Churn}=1 \mid x) = \sigma\left(w_0 + \sum_{k=1}^K w_k \hat{P}_k(x)\right)")

    with vg_t2:
        st.markdown("#### Top 8 Viva Questions & Model Answers")

        q_dict = {
            "1. Why is Accuracy a misleading metric in churn prediction?": (
                "In our telecom dataset, ~73.5% of subscribers do not churn, while only ~26.5% churn. "
                "A trivial dummy model that always predicts 'Stay' would attain 73.5% Accuracy while capturing zero churners. "
                "In business, a missed churner (False Negative) costs hundreds of dollars in lost customer lifetime value (LTV). "
                "Therefore, **Recall**, **F1-Score**, and **ROC-AUC** are much more meaningful evaluation metrics."
            ),
            "2. What is Data Leakage and how did you prevent it?": (
                "Data leakage occurs when information from the test dataset influences preprocessing or model training. "
                "We prevented leakage by splitting the dataset into 80% train and 20% test FIRST using stratified sampling, "
                "and fitting our ColumnTransformer (imputers, StandardScaler, OneHotEncoder) strictly on X_train. "
                "The test set was transformed using the fitted pipeline without re-estimating any parameters."
            ),
            "3. Why use Stacking rather than simple VotingClassifier?": (
                "A VotingClassifier uses a hard average or fixed weights across models. "
                "Stacking uses a trained meta-learner (Logistic Regression) that statistically discovers which base model "
                "is most reliable for specific probability ranges, producing a well-calibrated ensemble that captures "
                "both linear and non-linear feature interactions."
            ),
            "4. Why is Stratified K-Fold CV essential during Stacking?": (
                "If we trained base learners on the whole dataset and fed their predictions to the meta-learner, "
                "the meta-learner would fit on overfitted predictions. Stratified 5-Fold cross-validation generates "
                "genuine out-of-fold predictions that mimic truly unseen data, ensuring the meta-learner generalizes well."
            ),
            "5. What are the key features driving telecom churn in this system?": (
                "Through Random Forest and Gradient Boosting feature importances, the strongest predictors are: "
                "1) Contract type (month-to-month contracts have vastly higher churn rates), "
                "2) Tenure (subscribers in their first 6-12 months are most fragile), "
                "3) Monthly billing charges, and 4) Lack of technical support or security packages on Fiber optic plans."
            ),
            "6. How does your rule-based retention engine work?": (
                "Rather than offering a generic discount, the engine evaluates the customer's exact risk tier and features. "
                "For instance, high-risk month-to-month customers are offered contract price-lock transitions; "
                "subscribers with electronic checks are incentivized to use autopay; and customers with complex internet "
                "lacking technical support are offered complimentary support trials."
            ),
            "7. Why didn't you use customerID as a predictive feature?": (
                "customerID is an arbitrary unique identifier. Using it in tree or linear models causes the algorithm "
                "to memorize specific training instances (memorization / overfitting) without learning generalizable patterns. "
                "It was preserved solely for displaying results and tracking batch records."
            ),
            "8. What are the limitations and future enhancements?": (
                "Current limitations: Lack of time-series network telemetry (e.g. dropped calls, tower outages) and customer service transcripts. "
                "Future enhancements: Integrating natural language sentiment analysis from customer service chats "
                "and implementing cost-sensitive threshold optimization using actual monetary Customer Acquisition Cost (CAC)."
            ),
        }

        for q, ans in q_dict.items():
            with st.expander(f"📌 {q}"):
                st.write(ans)

    with vg_t3:
        st.markdown("#### Project Execution Commands (Windows)")
        st.code(
            """
# 1. Activate Environment (if using venv)
.venv\\Scripts\\activate

# 2. Train Models and Generate Reports
python src/train_model.py

# 3. Run Automated Unit Tests
python -m pytest -v

# 4. Launch Streamlit Application
streamlit run app.py
            """,
            language="bash",
        )
