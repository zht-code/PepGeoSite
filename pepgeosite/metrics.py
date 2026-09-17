from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, matthews_corrcoef, precision_recall_fscore_support,
    roc_auc_score,
)


def best_mcc_threshold(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    candidates = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, 501)))
    scores = [matthews_corrcoef(labels, probabilities >= threshold) for threshold in candidates]
    index = int(np.argmax(scores))
    return float(candidates[index]), float(scores[index])


def compute_metrics(labels, probabilities, threshold=None):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if threshold is None:
        threshold, _ = best_mcc_threshold(labels, probabilities)
    predictions = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "auroc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else float("nan"),
        "auprc": float(average_precision_score(labels, probabilities)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "threshold": float(threshold),
    }

