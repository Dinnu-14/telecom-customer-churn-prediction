# Telecom Customer Churn Dataset

## Dataset Source
This directory contains the primary dataset for predicting subscriber churn.
The primary dataset used is the widely recognized **IBM Telco Customer Churn Dataset**.

- **File name**: `telco_churn.csv`
- **Total records**: 7,043 customers
- **Total features**: 21 columns (20 predictors + 1 binary target `Churn`)

## Schema Information

| Column Name | Type | Description |
|---|---|---|
| `customerID` | String | Unique identification string for each subscriber |
| `gender` | Categorical | Whether customer is Male or Female |
| `SeniorCitizen` | Binary (0/1) | Whether customer is a senior citizen (1) or not (0) |
| `Partner` | Categorical | Whether the customer has a partner (Yes, No) |
| `Dependents` | Categorical | Whether customer has dependents (Yes, No) |
| `tenure` | Numeric (Integer) | Number of months the customer has stayed with the company |
| `PhoneService` | Categorical | Whether customer has phone service (Yes, No) |
| `MultipleLines` | Categorical | Whether customer has multiple lines (Yes, No, No phone service) |
| `InternetService` | Categorical | Customer’s internet service provider (DSL, Fiber optic, No) |
| `OnlineSecurity` | Categorical | Customer has online security (Yes, No, No internet service) |
| `OnlineBackup` | Categorical | Customer has online backup (Yes, No, No internet service) |
| `DeviceProtection` | Categorical | Customer has device protection (Yes, No, No internet service) |
| `TechSupport` | Categorical | Customer has tech support (Yes, No, No internet service) |
| `StreamingTV` | Categorical | Customer has streaming TV (Yes, No, No internet service) |
| `StreamingMovies` | Categorical | Customer has streaming movies (Yes, No, No internet service) |
| `Contract` | Categorical | Contract term (Month-to-month, One year, Two year) |
| `PaperlessBilling` | Categorical | Whether customer has paperless billing (Yes, No) |
| `PaymentMethod` | Categorical | Electronic check, Mailed check, Bank transfer (auto), Credit card (auto) |
| `MonthlyCharges` | Numeric (Float) | The amount charged to customer monthly |
| `TotalCharges` | Numeric (Float) | Total amount charged to customer over tenure |
| `Churn` | Binary (Yes/No) | Whether customer churned (Yes: 1, No: 0) |

## Data Cleaning Notes
1. **TotalCharges Whitespace Handling**: In the raw dataset, 11 customer records with `tenure == 0` have blank spaces (`" "`) in `TotalCharges`. The loader automatically converts these to `NaN` and imputes them with `0.0` or median value.
2. **CustomerID Exclusion**: `customerID` is retained for report tracking and batch predictions, but is strictly excluded from machine learning model features to avoid memorization/leakage.
3. **Class Imbalance**: Around 26.5% of customers in the dataset churned (`Yes`), while 73.5% stayed (`No`). This is accounted for using Stratified K-Fold validation and stratified train-test splits.
