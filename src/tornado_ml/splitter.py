from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from tornado_ml.config import ProjectConfig
from tornado_ml.data_types import DataSplit


class DatasetSplitter:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def split(self, df: pd.DataFrame, target_column: str) -> DataSplit:
        X = df.drop(columns=[target_column])
        y = df[target_column].astype(int)

        stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            train_size=self.config.train_ratio,
            random_state=self.config.random_seed,
            stratify=stratify,
        )

        relative_val_ratio = self.config.val_ratio / (
            self.config.val_ratio + self.config.test_ratio
        )
        temp_stratify = (
            y_temp if y_temp.nunique() == 2 and y_temp.value_counts().min() >= 2 else None
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            train_size=relative_val_ratio,
            random_state=self.config.random_seed,
            stratify=temp_stratify,
        )

        return DataSplit(X_train, X_val, X_test, y_train, y_val, y_test)
