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


def predictions_from_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    return (y_prob >= threshold).astype(int)

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
    
    def evaluate_thresholds(
        self,
        model_name: str,
        y_true: pd.Series,
        y_prob: np.ndarray,
        thresholds: list[float],
    ) -> pd.DataFrame:
        rows = []

        for threshold in thresholds:
            predictions = predictions_from_threshold(y_prob, threshold)
            metrics = self.evaluate(model_name, y_true, predictions, y_prob)
            matrix = metrics["confusion_matrix"]

            rows.append(
                {
                    "model_name": model_name,
                    "threshold": threshold,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics.get("roc_auc"),
                    "true_negative": matrix["true_negative"],
                    "false_positive": matrix["false_positive"],
                    "false_negative": matrix["false_negative"],
                    "true_positive": matrix["true_positive"],
                }
            )

        return pd.DataFrame(rows)

    def select_threshold(
        self,
        threshold_metrics: pd.DataFrame,
        model_name: str,
    ) -> float:
        model_rows = threshold_metrics[threshold_metrics["model_name"] == model_name].copy()

        if model_rows.empty:
            return 0.5

        # Prefer F1, but avoid selecting thresholds that never predict a positive.
        candidates = model_rows[model_rows["true_positive"] > 0]
        if candidates.empty:
            return 0.5

        best_row = candidates.sort_values(
            by=["f1", "recall", "precision"],
            ascending=False,
        ).iloc[0]

        return float(best_row["threshold"])