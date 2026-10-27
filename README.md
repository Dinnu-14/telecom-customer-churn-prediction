# An Intelligent System to Predict Subscribers Likely to Switch Operators Using Machine Learning and Predictive Analytics

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Ensemble Method](https://img.shields.io/badge/Ensemble-Stacked%20Classifier-green.svg)](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization)
[![Tests](https://img.shields.io/badge/Unit%20Tests-11%20Passed-brightgreen.svg)]()

---

## 📌 Project Abstract
Customer churn presents one of the most critical revenue risks in the telecommunications sector, where subscriber acquisition costs often exceed retention costs by five to twenty-five times. This project presents an end-to-end, student-friendly, and mathematically grounded **Intelligent Telecom Customer Churn Prediction System**. Built with the canonical **IBM Telco Customer Churn dataset** (7,043 subscriber records), the system trains three diverse baseline models—**Logistic Regression**, **Random Forest**, and **Gradient Boosting**—and synthesizes their predictive capabilities through a genuine **Stacked Ensemble Classifier (`StackingClassifier`)** with 5-fold Stratified Cross-Validation.

In addition to estimating churn probabilities and categorizing subscribers into **Low, Medium, and High Risk** categories, the platform features a transparent, rule-based **Customer Retention Recommendation Engine** and an interactive, modern **Streamlit Web Dashboard** featuring single-customer simulation, batch CSV scoring with export, model comparison, feature interpretability, and a dedicated BTech viva defense guide.

---

## 🎯 Problem Statement & Motivation
Telecommunications providers operate in saturated, highly competitive markets characterized by low switching friction and commoditized pricing. Traditional customer retention workflows suffer from two primary shortcomings:
1. **Reactive Interventions:** Retention outreach occurs only after a subscriber submits a disconnection or port-out request.
2. **Untargeted Retention Cost:** Blanket discount programs erode Average Revenue Per User (ARPU) without addressing the root cause of subscriber dissatisfaction.

This project addresses these challenges by developing a **proactive predictive intelligence pipeline** that identifies attrition risk before cancellation occurs and prescribes personalized, feature-driven retention interventions.

---

## 🚀 Objectives
- Predict subscriber churn with high discrimination (ROC-AUC > 0.84 on unseen data).
- Prevent data leakage by isolating preprocessing parameters strictly within training splits.
- Assemble a genuine Stacked Ensemble Classifier using stratified out-of-fold cross-validation.
- Classify customer churn probabilities into configurable risk tiers (Low, Medium, High).
- Map predictive signals into actionable, transparent business retention strategies.
- Deliver an interactive web dashboard for real-time inference, batch scoring, and model explanation.

---

## ⚖️ Existing System vs. Proposed System

| Dimension | Existing System | Proposed Intelligent System |
|---|---|---|
| **Approach** | Reactive (post-cancellation or port-out notice) | Proactive predictive risk scoring |
| **Model Type** | Simple heuristic rules or single isolated classifier | Genuine Stacked Ensemble Classifier (LR + RF + GB -> Meta LR) |
| **Cross-Validation** | Standard train/test split without meta-stacking | 5-Fold Stratified Cross-Validation with out-of-fold blending |
| **Data Leakage Defense** | Ad-hoc scaling/encoding on full dataset | Scikit-learn `Pipeline` & `ColumnTransformer` fitted strictly on $X_{train}$ |
| **Operational Guidance** | Generic blanket discounts for all callers | Transparent, rule-based targeted retention strategies |
| **Interface** | Static terminal scripts or spreadsheets | Modern, interactive multi-page Streamlit Dashboard with Plotly charts |

---

## 🏗️ System Architecture

```
                      ┌────────────────────────────────────────┐
                      │    IBM Telco Customer Churn Dataset    │
                      │         (7,043 Customer Rows)          │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │       Data Cleaning & Validation       │
                      │  - TotalCharges whitespace coercion    │
                      │  - Duplicate elimination               │
                      │  - Type casting & Churn target mapping │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │   Stratified Train-Test Split (80/20)  │
                      │  - Preserves customerID for tracking   │
                      │  - Eliminates target / data leakage    │
                      └──────────┬──────────────────┬──────────┘
                                 │                  │
                      X_train, y_train           X_test, y_test (Held-Out)
                                 │                  │
                                 ▼                  │
                      ┌────────────────────────┐    │
                      │   ColumnTransformer    │    │
                      │ - Imputation           │    │
                      │ - StandardScaler       │    │
                      │ - OneHotEncoder        │    │
                      └──────────┬─────────────┘    │
                                 │                  │
                                 ▼                  │
         ┌──────────────────────────────────────┐   │
         │   Stacked Ensemble Classifier        │   │
         │                                      │   │
         │  ┌────────────────────────────────┐  │   │
         │  │ Level 0 Base Learners:         │  │   │
         │  │ 1. Logistic Regression         │  │   │
         │  │ 2. Random Forest Classifier    │  │   │
         │  │ 3. Gradient Boosting Classifier│  │   │
         │  └───────────────┬────────────────┘  │   │
         │                  │ 5-Fold Stratified │   │
         │                  ▼ Out-of-Fold Probs │   │
         │  ┌────────────────────────────────┐  │   │
         │  │ Level 1 Meta-Learner:          │  │   │
         │  │ - Logistic Regression Blending │  │   │
         │  └───────────────┬────────────────┘  │   │
         └──────────────────┼───────────────────┘   │
                            │                       │
                            ▼                       ▼
         ┌─────────────────────────────────────────────────────┐
         │     Held-Out Model Evaluation & Metric Tracking     │
         │  - Accuracy, Precision, Recall, F1-Score, ROC-AUC   │
         │  - Confusion Matrix & Multi-Model ROC Curves        │
         └──────────────────────────┬──────────────────────────┘
                                    │
                                    ▼
         ┌─────────────────────────────────────────────────────┐
         │         Production Prediction & Retention Core      │
         │  - Risk Tiering: Low (<30%), Medium (30-60%), High  │
         │  - Feature-Driven Rule-Based Retention Engine       │
         └──────────────────────────┬──────────────────────────┘
                                    │
                                    ▼
         ┌─────────────────────────────────────────────────────┐
         │           Streamlit Interactive Dashboard           │
         │  Home | Single Prediction | Batch CSV | Insights    │
         └─────────────────────────────────────────────────────┘
```

---

## 📊 Dataset & Feature Dictionary

The dataset utilized is the standard **IBM Telco Customer Churn Dataset** stored locally at `data/telco_churn.csv`.

| Feature | Category | Description | Domain Impact |
|---|---|---|---|
| `customerID` | Identifier | Unique subscriber ID string | **Excluded from predictive features** to prevent memorization leakage. |
| `gender` | Demographic | Male / Female | Baseline demographic distribution. |
| `SeniorCitizen` | Demographic | 0 (No) / 1 (Yes) | Seniors often exhibit different technology adoption rates. |
| `Partner` | Demographic | Yes / No | Household stability indicator. |
| `Dependents` | Demographic | Yes / No | Family subscription indicator. |
| `tenure` | Account | Months with operator (0 to 72) | **Critical feature:** early months show highest churn vulnerability. |
| `Contract` | Account | Month-to-month, One year, Two year | **Top predictor:** Month-to-month plans experience over 42% churn. |
| `PaperlessBilling` | Billing | Yes / No | Electronic invoicing indicator. |
| `PaymentMethod` | Billing | Electronic check, Mailed check, Bank transfer, Credit card | Electronic check users show statistically elevated churn. |
| `MonthlyCharges` | Financial | Monthly recurring bill amount ($) | High spenders are sensitive to competitor pricing. |
| `TotalCharges` | Financial | Cumulative spend over tenure ($) | Blank spaces for tenure=0 are safely imputed. |
| `PhoneService` | Service | Yes / No | Voice subscription. |
| `MultipleLines` | Service | Yes, No, No phone service | Multi-line account status. |
| `InternetService` | Service | Fiber optic, DSL, No | High-speed fiber accounts have distinct retention profiles. |
| `OnlineSecurity` | Add-on | Yes, No, No internet service | High value-added service reducing churn propensity. |
| `OnlineBackup` | Add-on | Yes, No, No internet service | Value-added cloud storage. |
| `DeviceProtection`| Add-on | Yes, No, No internet service | Hardware insurance coverage. |
| `TechSupport` | Add-on | Yes, No, No internet service | **Major retention driver:** absence correlates with dissatisfaction. |
| `StreamingTV` | Add-on | Yes, No, No internet service | Entertainment package bundle. |
| `StreamingMovies`| Add-on | Yes, No, No internet service | Movie streaming package bundle. |
| `Churn` | Target | Yes (1) / No (0) | Binary classification ground truth (~26.5% positive). |

---

## 🧠 Machine Learning & Stacking Methodology

### 1. Mathematical Formulation of Stacking (Wolpert, 1992)
Stacked Generalization combines multiple heterogeneous estimators $h_1, h_2, \dots, h_K$ into a two-level architecture:

1. **Level-0 (Base Estimators):**
   - $h_1(x)$: **Logistic Regression** (captures linear decision boundaries and log-odds).
   - $h_2(x)$: **Random Forest Classifier** (captures orthogonal feature thresholds and interactions via bagging).
   - $h_3(x)$: **Gradient Boosting Classifier** (sequentially minimizes pseudo-residuals via gradient descent).

2. **Cross-Validation Blending:**
   To prevent meta-overfitting, base learners are trained via $K$-fold stratified cross-validation ($K=5$). For each fold $k$, models trained on $K-1$ folds generate out-of-fold class probability predictions $\hat{P}_j(x_i)$ for the held-out fold:
   $$\mathbf{z}_i = \Big[\hat{P}_1(x_i), \hat{P}_2(x_i), \hat{P}_3(x_i)\Big]$$

3. **Level-1 (Meta-Estimator):**
   A final meta-classifier $g(\mathbf{z})$ (Logistic Regression with $L_2$ regularization) learns optimal weighting coefficients $w_1, w_2, w_3$ and intercept $w_0$:
   $$P(\text{Churn}=1 \mid x) = \sigma\left(w_0 + \sum_{k=1}^K w_k \hat{P}_k(x)\right) = \frac{1}{1 + e^{-(w_0 + \mathbf{w}^T \mathbf{z})}}$$

---

## 📈 Empirical Model Evaluation (Held-Out Test Set)

Models were evaluated on a strictly held-out 20% test split (1,409 unseen subscribers) with stratified class representation:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 80.48% | 65.62% | 55.61% | 0.6020 | 0.8422 |
| **Random Forest** | 79.99% | 66.43% | 49.73% | 0.5688 | 0.8427 |
| **Gradient Boosting** | 79.70% | 64.97% | 51.07% | 0.5719 | 0.8392 |
| **Stacked Ensemble Classifier** | **80.62%** | **66.89%** | **53.48%** | **0.5944** | **0.8451** |

> 📌 **Key Findings for Presentation:**
> - The **Stacked Ensemble Classifier** achieves the highest **ROC-AUC (0.8451)** and highest **Accuracy (80.62%)**, demonstrating superior ranking ability across all decision thresholds.
> - **Accuracy alone is deceptive:** In a dataset with 73.5% non-churners, a naive dummy model predicting "Stay" would achieve 73.5% accuracy with 0% recall.
> - **Precision vs. Recall Trade-Off:** In telecommunications, missing a churner (False Negative) loses hundreds of dollars in Customer Lifetime Value (LTV). Lowering the decision threshold to **0.26 - 0.35** increases Recall to capture over 75% of churners while maintaining cost-effective targeted retention offers.

---

## 🎁 Rule-Based Customer Retention Engine

The system features a transparent business logic engine that evaluates the subscriber's predicted risk and individual account attributes:

| Condition / Signal | Target Profile | Prescribed Intervention Action | Strategic Rationale |
|---|---|---|---|
| **High Risk + Month-to-Month** | Flexible plan subscribers | 12-Month or 24-Month contract transition offer with price lock and 10% credit | Mitigates switching vulnerability. |
| **High Risk + MonthlyCharges >= $70** | High ARPU subscribers | Customized loyalty bundle discount (e.g., free streaming or family add-on) | Defends high-revenue accounts. |
| **High Risk + Tenure <= 12 months** | First-year subscribers | Dedicated onboarding customer success specialist check-in call | Resolves early lifecycle friction. |
| **High Risk + No Tech Support** | Fiber / DSL internet users | Complimentary 3-month Premium Tech Support trial and diagnostic test | Solves bandwidth latency issues. |
| **High Risk + Electronic Check** | Manual bill payers | $10 one-time credit to enroll in automated recurring credit card autopay | Reduces recurring payment friction. |
| **Medium Risk (30% to 60%)** | Borderline subscribers | Proactive account health optimization and feature utilization review | Prevents escalation to high risk. |
| **Low Risk (< 30%)** | Stable loyal subscribers | Quarterly satisfaction monitoring and brand referral reward incentives | Reinforces long-term loyalty. |

---

## 📁 Project Folder Structure

```
telecom_churn_prediction/
│
├── app.py                     # Interactive Streamlit Web Application
├── requirements.txt           # Project package dependencies
├── README.md                  # Comprehensive documentation and viva guide
├── .gitignore                 # Git ignore for virtual environments and cache
│
├── data/
│   ├── README.md              # Dataset schema notes and provenance
│   ├── telco_churn.csv        # Genuine IBM Telco Customer Churn dataset (7,043 rows)
│   └── sample_batch_test.csv  # 25-subscriber sample batch for verification
│
├── models/
│   ├── stacked_churn_model.joblib  # Full inference pipeline (preprocessor + stacking)
│   ├── baseline_models.joblib      # Dictionary of trained baseline models
│   └── preprocessor.joblib         # Fitted Scikit-learn ColumnTransformer
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_loader.py         # Data loading, validation, and synthetic fallback
│   ├── preprocessing.py       # ColumnTransformer, encoders, and stratified split
│   ├── train_model.py         # Model training, 5-fold CV stacking, and persistence
│   ├── evaluate_model.py      # Multi-metric evaluation and ROC/PR curve generators
│   ├── predict.py             # Single-customer and batch CSV prediction pipelines
│   └── retention.py           # Rule-based customer retention recommendation engine
│
├── tests/
│   ├── __init__.py            # Test suite package initialization
│   ├── test_preprocessing.py  # Unit tests for data cleaning and transformations
│   ├── test_prediction.py     # Unit tests for single and batch inference
│   └── test_retention.py      # Unit tests for rule-based retention engine
│
└── reports/
    └── model_metrics.json     # Complete serialized metrics, ROC coords, and feature weights
```

---

## 💻 Installation & Setup Instructions (Windows)

Follow these step-by-step instructions in Windows **Command Prompt (`cmd`)** or **PowerShell**:

### Step 1: Open Terminal & Navigate to Project Directory
```cmd
cd /d "D:\mini project"
```

### Step 2: Create a Virtual Environment (Recommended)
```cmd
python -m venv venv
```

### Step 3: Activate the Virtual Environment
- **Command Prompt (`cmd`):**
  ```cmd
  venv\Scripts\activate
  ```
- **PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If execution policies block scripts in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 4: Install Dependencies
```cmd
python -m pip install -r requirements.txt
```

### Step 5: Train and Evaluate Models
Execute the training script to load the dataset, fit baseline and stacked ensemble models, compute test metrics, and serialize artifacts:
```cmd
python src/train_model.py
```

### Step 6: Run Automated Unit Tests
Verify all 11 automated unit tests across data cleaning, preprocessing, predictions, and retention logic:
```cmd
python -m pytest -v
```

### Step 7: Launch the Interactive Streamlit Web Application
```cmd
python -m streamlit run app.py
```
Open your browser and navigate to:
**`http://localhost:8501`**

---

## 🎓 Viva Voce Questions & Model Answers

### Q1: What is a Stacked Ensemble Classifier and how does it work?
> **Answer:** Stacking is an ensemble learning technique that combines multiple heterogeneous base classifiers (such as Logistic Regression, Random Forest, and Gradient Boosting). Instead of simple majority voting, Stacking trains a Level-1 meta-learner (Logistic Regression) whose input features are the out-of-fold probability predictions produced by the base models. The meta-learner discovers which base model performs best under different feature conditions, generating a superior combined prediction.

### Q2: What is Data Leakage, and how did you prevent it?
> **Answer:** Data leakage occurs when information from outside the training dataset influences model parameters, resulting in overly optimistic evaluation. We prevented data leakage by:
> 1. Performing an 80/20 stratified split **before** fitting any transformers.
> 2. Encapsulating numeric scalers (`StandardScaler`) and categorical encoders (`OneHotEncoder`) inside a Scikit-learn `ColumnTransformer`.
> 3. Calling `.fit()` strictly on the training partition $X_{train}$ and applying `.transform()` to the test set without re-estimating means, variances, or category mappings.
> 4. Generating meta-learner training inputs strictly via 5-fold out-of-fold cross-validation.

### Q3: Why didn't you use `customerID` as a feature?
> **Answer:** `customerID` is an arbitrary high-cardinality unique identifier. If included, tree models or linear models would memorize individual rows, causing severe overfitting and zero generalization to new customers. We preserved `customerID` purely for displaying results in batch predictions and reports.

### Q4: Why is Accuracy misleading in this problem?
> **Answer:** In our Telco dataset, ~73.5% of customers do not churn (negative class) and only ~26.5% churn (positive class). A naive classifier that predicts "No Churn" for every customer would achieve 73.5% Accuracy while capturing zero churners. Therefore, **Recall** (percentage of actual churners identified), **Precision** (avoiding false alarms), **F1-Score**, and **ROC-AUC** are the scientifically valid metrics.

### Q5: What features have the highest impact on churn?
> **Answer:** According to Random Forest and Gradient Boosting feature importances and Logistic Regression log-odds coefficients:
> 1. **Contract Type:** Month-to-month contracts exhibit significantly higher churn rates compared to one- or two-year contracts.
> 2. **Tenure:** Customers in their first 6–12 months have the highest attrition risk.
> 3. **Monthly Charges:** High monthly billing correlates with price sensitivity.
> 4. **Tech Support & Online Security:** Customers without technical support on Fiber optic connections have higher churn rates.

---

## 🔮 Limitations & Future Enhancements
- **Network Telemetry Integration:** Incorporate real-time Quality of Service (QoS) telemetry such as dropped call rates, packet loss, and cellular tower handover statistics.
- **NLP Sentiment Mining:** Analyze customer support chat logs and call transcript sentiments using Transformer models.
- **Cost-Sensitive Threshold Optimization:** Incorporate explicit dollar Customer Acquisition Cost (CAC) and Retention Offer Cost (ROC) parameters to dynamically compute the profit-maximizing threshold.

---

## 📜 License & Acknowledgments
- Dataset provided by **IBM Sample Datasets**.
- Implemented for academic demonstration and viva defense in BTech Artificial Intelligence & Data Science.
