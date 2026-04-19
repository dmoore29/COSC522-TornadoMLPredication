from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
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
