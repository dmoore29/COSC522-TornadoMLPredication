from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from tornado_ml.config import ProjectConfig


class FeaturePreprocessor:
    def __init__(self, config: ProjectConfig, scale: bool):
        self.config = config
        self.scale = scale
        self.selected_features_: list[str] = []
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler() if scale else None

    def fit(self, X_train: pd.DataFrame) -> FeaturePreprocessor:
        missingness = X_train.isna().mean()
        self.selected_features_ = missingness[
            missingness <= self.config.missingness_threshold
        ].index.tolist()
        if not self.selected_features_:
            raise ValueError("No features remain after missingness filtering.")

        train_selected = X_train.loc[:, self.selected_features_]
        imputed = self.imputer.fit_transform(train_selected)
        if self.scaler is not None:
            self.scaler.fit(imputed)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        selected = X.loc[:, self.selected_features_]
        imputed = self.imputer.transform(selected)
        values = self.scaler.transform(imputed) if self.scaler is not None else imputed
        return pd.DataFrame(values, columns=self.selected_features_, index=X.index)

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)
