from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from tornado_ml.config import ProjectConfig


class FeatureEngineer:
    """Adds meteorologically meaningful interaction features before preprocessing.

    All derived features use only raw inputs so they can be computed before imputation.
    NaN propagates naturally — imputer handles it downstream.
    """

    def __init__(self) -> None:
        self._slp_mean: float | None = None

    def fit(self, X_train: pd.DataFrame) -> FeatureEngineer:
        if "SLP" in X_train.columns:
            self._slp_mean = float(X_train["SLP"].mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Low dewpoint depression = high humidity; a key tornado precursor
        if "TEMP" in X.columns and "DEWP" in X.columns:
            X["DEWP_DEPRESSION"] = X["TEMP"] - X["DEWP"]
        # High diurnal temperature range = atmospheric instability
        if "MAX" in X.columns and "MIN" in X.columns:
            X["TEMP_RANGE"] = X["MAX"] - X["MIN"]
        # Sudden gust spike relative to sustained wind = wind shear indicator
        # Clipped to [0, 10] — physically, gusts rarely exceed 3–4x sustained wind;
        # without clipping, near-zero MXSPD produces extreme values that overflow StandardScaler
        if "GUST" in X.columns and "MXSPD" in X.columns:
            X["GUST_RATIO"] = (X["GUST"] / (X["MXSPD"] + 1e-6)).clip(upper=10.0)
        # Negative pressure anomaly = storm center proximity
        if "SLP" in X.columns and self._slp_mean is not None:
            X["PRESSURE_DEFICIT"] = self._slp_mean - X["SLP"]
        return X

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)


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
