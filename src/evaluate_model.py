"""
Evaluation Module for Telecom Customer Churn Prediction Models.
Calculates comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
confusion matrices, and ROC curves to properly assess class-imbalanced churn predictions.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
)


def evaluate_classifier_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    model_name: str = "Classifier",
) -> dict:
    """
    Computes standard evaluation metrics for churn classification.
    
    Returns a dictionary of scalar metrics, confusion matrix counts,
    and curve points suitable for JSON serialization and Plotly visualizations.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = None
    fpr_list, tpr_list = [], []
    pr_auc = None
    prec_curve, rec_curve = [], []

    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fpr_list = [round(float(x), 4) for x in fpr]
            tpr_list = [round(float(x), 4) for x in tpr]

            p_vals, r_vals, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(r_vals, p_vals))
            prec_curve = [round(float(x), 4) for x in p_vals]
            rec_curve = [round(float(x), 4) for x in r_vals]
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred)
    # cm format: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": cm.tolist(),
        },
        "classification_report": report,
        "roc_curve": {
            "fpr": fpr_list,
            "tpr": tpr_list,
        },
        "pr_curve": {
            "precision": prec_curve,
            "recall": rec_curve,
        },
    }


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
) -> tuple[float, float]:
    """
    Finds the optimal probability decision threshold on validation data.
    Supported metrics: 'f1', 'balanced_accuracy'.
    Returns (best_threshold, best_score).
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_score = -1.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, preds, zero_division=0)
        else:
            score = accuracy_score(y_true, preds)

        if score > best_score:
            best_score = score
            best_thresh = round(float(t), 3)

    return best_thresh, round(float(best_score), 4)
