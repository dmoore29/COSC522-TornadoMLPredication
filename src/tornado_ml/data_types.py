from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    metrics: dict
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    selected_features: list[str]
    val_metrics: Optional[dict] = None
    val_predictions: Optional[np.ndarray] = None
    val_probabilities: Optional[np.ndarray] = None
