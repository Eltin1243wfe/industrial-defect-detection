"""
Metric computations, pulled into their own module so evaluate.py doesn't
turn into a wall of sklearn calls. Nothing here is reinventing the wheel,
just wrapping sklearn with the specific inputs/outputs I need across the
different model types (classifier gives logits, autoencoder/patchcore give
continuous anomaly scores that need a threshold).
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_curve,
)


def compute_classification_metrics(y_true, y_score, threshold=0.5):
    """
    y_true: 0/1 array, 1 = defective
    y_score: continuous anomaly score or defect-class probability, higher = more defective
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "auroc": roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    return metrics


def find_best_threshold(y_true, y_score):
    """
    sweeps thresholds along the ROC curve and picks the one that maximizes
    youden's J statistic (tpr - fpr). Simple, works well enough for this use
    case - could swap in a cost-weighted version if false negatives are more
    expensive than false positives in a given factory setting (usually are,
    honestly, might be worth revisiting).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def get_confusion_matrix(y_true, y_score, threshold=0.5):
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def per_defect_type_breakdown(defect_types, y_true, y_score, threshold=0.5):
    """
    how well are we catching each specific defect type (scratch, dent, crack,
    etc)? Aggregate accuracy hides this - a model can look great overall while
    completely missing one rare-but-important defect category.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    breakdown = {}
    for dt in sorted(set(defect_types)):
        idx = [i for i, d in enumerate(defect_types) if d == dt]
        if not idx:
            continue
        correct = sum(1 for i in idx if y_pred[i] == y_true[i])
        breakdown[dt] = {
            "n_samples": len(idx),
            "accuracy": correct / len(idx),
            "mean_score": float(np.mean([y_score[i] for i in idx])),
        }
    return breakdown
