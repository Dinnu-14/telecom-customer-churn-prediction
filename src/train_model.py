"""
Model Training Module for Telecom Customer Churn Prediction System.
Trains baseline classifiers (Logistic Regression, Random Forest, Gradient Boosting)
and a Scikit-learn StackingClassifier (meta-learner: Logistic Regression) with 5-fold CV.
Saves models, pipelines, and evaluation metrics to disk.
"""

import os
import sys
import json
import logging

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.data_loader import load_telco_data, DEFAULT_DATA_PATH
from src.preprocessing import (
    create_preprocessor,
    prepare_train_test_split,
    get_feature_names_after_preprocessing,
    ALL_FEATURE_COLUMNS,
)
from src.evaluate_model import evaluate_classifier_performance, find_optimal_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def train_and_evaluate_models(data_path: str = DEFAULT_DATA_PATH) -> dict:
    """
    Orchestrates end-to-end model training, stacking ensemble assembly,
    rigorous held-out testing, feature importance extraction, and persistence.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load Data
    logger.info("Loading telecom churn dataset...")
    df, is_synthetic = load_telco_data(data_path)

    # 2. Train-Test Split (Strict Stratified Split, 80% train, 20% test)
    logger.info("Performing stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test, id_train, id_test = prepare_train_test_split(
        df, test_size=0.2, random_state=42
    )
    logger.info(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Fit Preprocessing Pipeline ONLY on training data
    logger.info("Fitting ColumnTransformer on X_train to prevent data leakage...")
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)
    feature_names = get_feature_names_after_preprocessing(preprocessor)

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 4. Instantiate Baseline Models
    logger.info("Initializing baseline models...")
    baseline_estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
        ),
    }

    metrics_results = {}
    trained_baselines = {}

    # Train and evaluate each baseline model
    for name, model in baseline_estimators.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_proc, y_train)
        trained_baselines[name] = model

        # Test set evaluation
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else None
        
        eval_dict = evaluate_classifier_performance(y_test.values, y_pred, y_prob, model_name=name)
        metrics_results[name] = eval_dict
        logger.info(
            f"--> {name} Results: Acc={eval_dict['accuracy']:.4f}, "
            f"F1={eval_dict['f1_score']:.4f}, Recall={eval_dict['recall']:.4f}, "
            f"ROC-AUC={eval_dict['roc_auc']:.4f}"
        )

    # 5. Build Genuine Stacked Ensemble Classifier
    logger.info("Assembling StackingClassifier with 5-fold cross-validation...")
    # Base estimators for stacking
    base_learners = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=10, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)),
    ]
    # Meta-learner is Logistic Regression combining base predictions
    meta_learner = LogisticRegression(random_state=42, C=1.0)

    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    logger.info("Training Stacked Ensemble Classifier (generating out-of-fold predictions)...")
    stacking_clf.fit(X_train_proc, y_train)

    # Evaluate Stacked Ensemble on held-out test data
    stack_pred = stacking_clf.predict(X_test_proc)
    stack_prob = stacking_clf.predict_proba(X_test_proc)[:, 1]
    stack_eval = evaluate_classifier_performance(
        y_test.values, stack_pred, stack_prob, model_name="Stacked Ensemble"
    )
    metrics_results["Stacked Ensemble"] = stack_eval
    logger.info(
        f"--> Stacked Ensemble Results: Acc={stack_eval['accuracy']:.4f}, "
        f"F1={stack_eval['f1_score']:.4f}, Recall={stack_eval['recall']:.4f}, "
        f"ROC-AUC={stack_eval['roc_auc']:.4f}"
    )

    # 6. Find optimal probability threshold on training/validation for high-recall operational use
    opt_thresh_f1, best_f1 = find_optimal_threshold(y_train.values, stacking_clf.predict_proba(X_train_proc)[:, 1], "f1")
    logger.info(f"Optimal decision threshold on training data (maximizing F1): {opt_thresh_f1} (Score: {best_f1})")

    # 7. Extract Feature Importances & Model Interpretability
    logger.info("Computing feature importances and interpretability data...")
    # Random Forest importances
    rf_importances = trained_baselines["Random Forest"].feature_importances_
    # Gradient Boosting importances
    gb_importances = trained_baselines["Gradient Boosting"].feature_importances_
    # Logistic Regression coefficients
    lr_coefficients = trained_baselines["Logistic Regression"].coef_[0]

    # Meta-learner weights for stacking base models
    meta_weights = {}
    if hasattr(stacking_clf.final_estimator_, "coef_"):
        # Since predict_proba outputs 2 probabilities per base learner (or 1 for binary), check shape:
        coefs = stacking_clf.final_estimator_.coef_[0]
        meta_weights = {
            "coefficients": [round(float(c), 4) for c in coefs],
            "intercept": round(float(stacking_clf.final_estimator_.intercept_[0]), 4),
        }

    # Format top features dataframe
    feature_importance_data = []
    for i, fname in enumerate(feature_names):
        feature_importance_data.append({
            "feature": fname,
            "rf_importance": round(float(rf_importances[i]), 5),
            "gb_importance": round(float(gb_importances[i]), 5),
            "lr_coefficient": round(float(lr_coefficients[i]), 5),
        })

    # Sort by average tree importance
    feature_importance_data = sorted(
        feature_importance_data,
        key=lambda x: (x["rf_importance"] + x["gb_importance"]) / 2,
        reverse=True,
    )

    # 8. Create Full Production Inference Pipeline
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("stacking", stacking_clf),
        ]
    )

    # 9. Save Models and Metadata
    model_path = os.path.join(MODELS_DIR, "stacked_churn_model.joblib")
    baselines_path = os.path.join(MODELS_DIR, "baseline_models.joblib")
    preproc_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    metrics_path = os.path.join(REPORTS_DIR, "model_metrics.json")

    logger.info(f"Saving full stacked pipeline to {model_path}...")
    joblib.dump(full_pipeline, model_path)

    logger.info(f"Saving baseline models to {baselines_path}...")
    joblib.dump(trained_baselines, baselines_path)

    logger.info(f"Saving preprocessor to {preproc_path}...")
    joblib.dump(preprocessor, preproc_path)

    # Structure final report JSON
    final_report = {
        "dataset_summary": {
            "total_records": int(len(df)),
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "features_count": len(ALL_FEATURE_COLUMNS),
            "transformed_features_count": len(feature_names),
            "churn_rate_total": round(float(df["Churn"].mean()), 4),
            "churn_rate_train": round(float(y_train.mean()), 4),
            "churn_rate_test": round(float(y_test.mean()), 4),
            "is_synthetic": is_synthetic,
        },
        "optimal_threshold": {
            "threshold": opt_thresh_f1,
            "metric": "f1_score",
            "score_on_train": best_f1,
            "default_threshold": 0.50,
        },
        "models_performance": metrics_results,
        "feature_importances": feature_importance_data,
        "meta_learner_weights": meta_weights,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info(f"Model evaluation report written to {metrics_path}")
    return final_report


if __name__ == "__main__":
    report = train_and_evaluate_models()
    print("\n--- MODEL COMPARISON SUMMARY (TEST SET) ---")
    print(f"{'Model Name':<22} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9}")
    print("-" * 75)
    for name, m in report["models_performance"].items():
        print(f"{name:<22} | {m['accuracy']:<9.4f} | {m['precision']:<9.4f} | {m['recall']:<9.4f} | {m['f1_score']:<9.4f} | {m['roc_auc']:<9.4f}")
