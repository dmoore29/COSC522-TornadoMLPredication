from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from tornado_ml.config import ProjectConfig


class LogisticRegressionModel:
    def __init__(self, config: ProjectConfig):
        self.estimator = LogisticRegression(**config.logistic_regression_params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.estimator.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]


class RandomForestModel:
    def __init__(self, config: ProjectConfig):
        self.estimator = RandomForestClassifier(**config.random_forest_params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.estimator.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]
