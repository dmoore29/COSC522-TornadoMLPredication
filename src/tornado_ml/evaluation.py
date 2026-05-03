from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsEvaluator:
    def evaluate(
        self,
        model_name: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> dict:
        labels = [0, 1]
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        metrics = {
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": {
                "true_negative": int(matrix[0, 0]),
                "false_positive": int(matrix[0, 1]),
                "false_negative": int(matrix[1, 0]),
                "true_positive": int(matrix[1, 1]),
            },
        }
        if y_prob is not None and pd.Series(y_true).nunique() == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        return metrics

    @staticmethod
    def find_optimal_threshold(
        y_true: pd.Series,
        y_prob: np.ndarray,
        beta: float = 2.0,
        min_recall: float = 0.80,
    ) -> float:
        """Find the threshold maximizing F-beta on the given set.

        beta > 1 weights recall more than precision (beta=2 means recall counts
        twice as much). min_recall enforces a floor so the threshold never drops
        below a practically useful detection rate.
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        # precisions/recalls have one extra element (for threshold=0); align with thresholds
        p, r = precisions[:-1], recalls[:-1]
        denom = beta**2 * p + r
        fbeta = np.where(denom == 0, 0.0, (1 + beta**2) * p * r / denom)
        # Mask out thresholds where recall falls below the minimum acceptable rate
        fbeta = np.where(r >= min_recall, fbeta, 0.0)
        best_idx = int(np.argmax(fbeta))
        return float(thresholds[best_idx])
