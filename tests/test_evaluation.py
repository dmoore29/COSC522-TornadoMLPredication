from __future__ import annotations

import numpy as np
import pandas as pd

from tornado_ml.evaluation import MetricsEvaluator


def test_evaluator_returns_required_metrics_and_confusion_matrix() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])

    metrics = MetricsEvaluator().evaluate("Example", y_true, y_pred)

    assert metrics["model_name"] == "Example"
    assert set(["accuracy", "precision", "recall", "f1", "confusion_matrix"]).issubset(metrics)
    assert metrics["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }
