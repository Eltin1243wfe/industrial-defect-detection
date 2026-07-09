import numpy as np

from src.evaluation.metrics import (
    compute_classification_metrics, find_best_threshold,
    get_confusion_matrix, per_defect_type_breakdown,
)


def test_perfect_separation_gives_perfect_metrics():
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.05, 0.2, 0.9, 0.95, 0.85]
    metrics = compute_classification_metrics(y_true, y_score, threshold=0.5)

    assert metrics["auroc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_best_threshold_lands_between_the_two_clusters():
    y_true = [0] * 20 + [1] * 20
    y_score = list(np.random.normal(0.2, 0.05, 20)) + list(np.random.normal(0.8, 0.05, 20))
    thresh = find_best_threshold(y_true, y_score)
    assert 0.2 < thresh < 0.8


def test_confusion_matrix_shape():
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.2, 0.8]
    cm = get_confusion_matrix(y_true, y_score, threshold=0.5)
    assert cm.shape == (2, 2)
    assert cm.sum() == 4


def test_per_defect_type_breakdown_covers_all_types():
    defect_types = ["good", "good", "scratch", "scratch", "dent"]
    y_true = [0, 0, 1, 1, 1]
    y_score = [0.1, 0.15, 0.8, 0.75, 0.6]
    breakdown = per_defect_type_breakdown(defect_types, y_true, y_score, threshold=0.5)

    assert set(breakdown.keys()) == {"good", "scratch", "dent"}
    assert breakdown["scratch"]["n_samples"] == 2
