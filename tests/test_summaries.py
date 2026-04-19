from __future__ import annotations

import pandas as pd

from tornado_ml.config import ProjectConfig
from tornado_ml.splitter import DatasetSplitter
from tornado_ml.summaries import build_dataset_summary, build_missingness_table, build_split_summary


def test_summary_helpers_return_report_ready_tables() -> None:
    df = pd.DataFrame(
        {
            "TEMP": [70.0, None, 75.0, 80.0],
            "DEWP": [60.0, 61.0, 62.0, None],
            "TORNADO_LABEL": [0, 0, 1, 1],
        }
    )

    dataset_summary = build_dataset_summary(
        df,
        "TORNADO_LABEL",
        ["TEMP", "DEWP"],
        date_min="2024-01-01",
        date_max="2024-01-04",
    )
    missingness = build_missingness_table(df, "TORNADO_LABEL")

    assert dataset_summary["n_rows"] == 4
    assert dataset_summary["positive_count"] == 2
    assert dataset_summary["positive_rate"] == 0.5
    assert set(missingness.columns) == {"feature", "missing_count", "missing_percent"}


def test_split_summary_reports_each_split() -> None:
    df = pd.DataFrame({"TEMP": range(100), "TORNADO_LABEL": [0, 1] * 50})
    config = ProjectConfig(raw_data_paths=["unused.csv"], candidate_features=["TEMP"])
    split = DatasetSplitter(config).split(df, "TORNADO_LABEL")

    split_summary = build_split_summary(split)

    assert split_summary["split"].tolist() == ["train", "validation", "test"]
    assert split_summary["n_rows"].tolist() == [70, 15, 15]
