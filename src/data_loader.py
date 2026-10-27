"""
Data Loader Module for Telecom Customer Churn Prediction System.
Handles loading, validation, downloading, and synthetic fallback generation.
"""

import os
import urllib.request
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Canonical IBM Telco Dataset URL
IBM_TELCO_DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "telco_churn.csv"
)

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]

FEATURE_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in ["customerID", "Churn"]]


def download_dataset(target_path: str = DEFAULT_DATA_PATH) -> bool:
    """
    Downloads the genuine IBM Telco Customer Churn dataset from GitHub repository.
    """
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        logger.info(f"Attempting to download dataset from {IBM_TELCO_DATASET_URL}...")
        urllib.request.urlretrieve(IBM_TELCO_DATASET_URL, target_path)
        logger.info(f"Dataset successfully downloaded to {target_path}")
        return True
    except Exception as e:
        logger.warning(f"Could not download dataset from remote URL: {e}")
        return False


def generate_synthetic_telco_data(num_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a reproducible, realistically distributed synthetic Telecom Churn dataset.
    Explicitly labeled as synthetic for demonstration and fallback purposes.
    """
    logger.warning("Generating clearly labeled SYNTHETIC dataset as fallback. Not using real-world data.")
    rng = np.random.default_rng(random_state)

    customer_ids = [f"SYNTH-{10000 + i}" for i in range(num_samples)]
    genders = rng.choice(["Male", "Female"], size=num_samples)
    senior = rng.choice([0, 1], size=num_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=num_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=num_samples, p=[0.30, 0.70])
    
    # Tenure skewed towards shorter and longer contracts
    tenure = rng.integers(1, 73, size=num_samples)
    
    phone = rng.choice(["Yes", "No"], size=num_samples, p=[0.90, 0.10])
    multiple_lines = []
    for p in phone:
        if p == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(rng.choice(["Yes", "No"], p=[0.42, 0.58]))
            
    internet = rng.choice(["Fiber optic", "DSL", "No"], size=num_samples, p=[0.44, 0.34, 0.22])
    
    online_sec = []
    online_bak = []
    dev_prot = []
    tech_sup = []
    stream_tv = []
    stream_mov = []
    
    for net in internet:
        if net == "No":
            online_sec.append("No internet service")
            online_bak.append("No internet service")
            dev_prot.append("No internet service")
            tech_sup.append("No internet service")
            stream_tv.append("No internet service")
            stream_mov.append("No internet service")
        else:
            online_sec.append(rng.choice(["Yes", "No"], p=[0.35, 0.65]))
            online_bak.append(rng.choice(["Yes", "No"], p=[0.40, 0.60]))
            dev_prot.append(rng.choice(["Yes", "No"], p=[0.40, 0.60]))
            tech_sup.append(rng.choice(["Yes", "No"], p=[0.35, 0.65]))
            stream_tv.append(rng.choice(["Yes", "No"], p=[0.45, 0.55]))
            stream_mov.append(rng.choice(["Yes", "No"], p=[0.45, 0.55]))
            
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=num_samples, p=[0.55, 0.21, 0.24])
    paperless = rng.choice(["Yes", "No"], size=num_samples, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=num_samples,
        p=[0.34, 0.23, 0.22, 0.21]
    )
    
    # Calculate realistic monthly and total charges
    monthly_charges = []
    for net, p in zip(internet, phone):
        base = 20.0
        if p == "Yes":
            base += 10.0
        if net == "DSL":
            base += 30.0
        elif net == "Fiber optic":
            base += 50.0
        noise = rng.uniform(-5.0, 15.0)
        monthly_charges.append(round(max(18.25, base + noise), 2))
        
    monthly_charges = np.array(monthly_charges)
    total_charges = [round(m * t + rng.uniform(-10.0, 10.0), 2) for m, t in zip(monthly_charges, tenure)]
    total_charges = [max(m, tc) for m, tc in zip(monthly_charges, total_charges)]
    
    # Calculate churn probability with logical realistic domain weights
    churn = []
    for i in range(num_samples):
        score = 0.0
        if contract[i] == "Month-to-month":
            score += 0.85
        elif contract[i] == "Two year":
            score -= 1.1
            
        if internet[i] == "Fiber optic":
            score += 0.5
        elif internet[i] == "No":
            score -= 0.6
            
        if tech_sup[i] == "No":
            score += 0.4
        if online_sec[i] == "No":
            score += 0.35
            
        if payment[i] == "Electronic check":
            score += 0.45
            
        score += (monthly_charges[i] - 65.0) * 0.015
        score -= (tenure[i] - 30.0) * 0.03
        
        prob = 1.0 / (1.0 + np.exp(-score))
        churn.append("Yes" if rng.uniform(0, 1) < prob else "No")
        
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bak,
        "DeviceProtection": dev_prot,
        "TechSupport": tech_sup,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    })
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw telecom data:
    - Removes whitespace from string columns
    - Handles TotalCharges numeric conversion (empty strings to NaN, imputed)
    - Removes duplicate rows
    - Validates target Churn values
    """
    df = df.copy()
    
    # Drop pure duplicates
    df = df.drop_duplicates()
    
    # TotalCharges contains blank spaces ' ' in raw IBM dataset when tenure == 0
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # For new customers with tenure == 0 or missing TotalCharges, impute with 0.0 or MonthlyCharges
        if "MonthlyCharges" in df.columns:
            df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"].clip(lower=1))
        else:
            df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
            
    # Ensure numeric columns are strictly numeric
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
        
    # Map Churn column if present
    if "Churn" in df.columns:
        if df["Churn"].dtype != int and df["Churn"].dtype != np.int64:
            df["Churn"] = (
                df["Churn"]
                .astype(str)
                .str.strip()
                .map({"Yes": 1, "No": 0, "1": 1, "0": 0, "True": 1, "False": 0})
                .fillna(0)
                .astype(int)
            )
            
    return df


def load_telco_data(filepath: str = DEFAULT_DATA_PATH, allow_download: bool = True) -> tuple[pd.DataFrame, bool]:
    """
    Loads telecom customer churn dataset.
    Returns:
        (DataFrame, is_synthetic_flag)
    """
    is_synthetic = False
    
    if not os.path.exists(filepath):
        if allow_download:
            logger.info(f"File {filepath} not found. Attempting to download IBM Telco dataset...")
            success = download_dataset(filepath)
            if not success:
                logger.warning("Download failed. Generating synthetic dataset fallback...")
                df_synth = generate_synthetic_telco_data()
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                df_synth.to_csv(filepath, index=False)
                is_synthetic = True
        else:
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")
            
    df = pd.read_csv(filepath)
    df_clean = clean_raw_data(df)
    
    # Check if dataset is synthetic based on ID format
    if "customerID" in df_clean.columns and df_clean["customerID"].str.startswith("SYNTH-").any():
        is_synthetic = True
        
    logger.info(f"Loaded dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns. Synthetic: {is_synthetic}")
    return df_clean, is_synthetic


if __name__ == "__main__":
    df, synth = load_telco_data()
    print("Dataset shape:", df.shape)
    print("Is synthetic:", synth)
    print("Columns:", df.columns.tolist())
    print("Churn distribution:\n", df["Churn"].value_counts(normalize=True))
