from __future__ import annotations

import pandas as pd

from tornado_ml.config import ProjectConfig
from tornado_ml.preprocessing import FeaturePreprocessor


def test_preprocessor_drops_high_missingness_feature_and_imputes() -> None:
    X_train = pd.DataFrame(
        {
            "TEMP": [70.0, None, 75.0, 80.0],
            "SPARSE": [None, None, None, 1.0],
        }
    )
    X_test = pd.DataFrame({"TEMP": [None], "SPARSE": [5.0]})
    config = ProjectConfig(
        raw_data_paths=["unused.csv"],
        candidate_features=["TEMP", "SPARSE"],
        missingness_threshold=0.50,
    )
    preprocessor = FeaturePreprocessor(config, scale=False)

    transformed_train = preprocessor.fit_transform(X_train)
    transformed_test = preprocessor.transform(X_test)

    assert preprocessor.selected_features_ == ["TEMP"]
    assert transformed_train.isna().sum().sum() == 0
    assert transformed_test.iloc[0]["TEMP"] == 75.0
