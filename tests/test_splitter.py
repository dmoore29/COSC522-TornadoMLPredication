from __future__ import annotations

import pandas as pd

from tornado_ml.config import ProjectConfig
from tornado_ml.splitter import DatasetSplitter


def test_splitter_creates_expected_train_val_test_sizes() -> None:
    df = pd.DataFrame(
        {
            "TEMP": range(100),
            "TORNADO_LABEL": [0, 1] * 50,
        }
    )
    config = ProjectConfig(raw_data_paths=["unused.csv"], candidate_features=["TEMP"])

    split = DatasetSplitter(config).split(df, "TORNADO_LABEL")

    assert len(split.X_train) == 70
    assert len(split.X_val) == 15
    assert len(split.X_test) == 15
    assert split.y_train.nunique() == 2
