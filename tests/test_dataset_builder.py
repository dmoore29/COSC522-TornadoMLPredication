from __future__ import annotations

import pandas as pd

from tornado_ml.config import ProjectConfig
from tornado_ml.dataset_builder import GsodDatasetBuilder


def make_config() -> ProjectConfig:
    return ProjectConfig(
        raw_data_paths=["unused.csv"],
        candidate_features=["TEMP", "LATITUDE", "LONGITUDE"],
        forbidden_features=["FRSHTT", "DATE", "STATION", "NAME", "source_file"],
    )


def test_build_target_left_pads_frshtt_digit() -> None:
    df = pd.DataFrame(
        {
            "DATE": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "FRSHTT": [1, 100001, 10000],
            "LATITUDE": [35.0, 35.0, 35.0],
            "LONGITUDE": [-97.0, -97.0, -97.0],
            "TEMP": [40.0, 45.0, 50.0],
        }
    )
    builder = GsodDatasetBuilder(make_config())

    labeled = builder._build_target(df)

    assert labeled["TORNADO_LABEL"].tolist() == [1, 1, 0]


def test_filter_mainland_us_uses_configured_coordinate_bounds() -> None:
    df = pd.DataFrame(
        {
            "LATITUDE": [35.0, 60.0, 21.0],
            "LONGITUDE": [-97.0, -149.0, -157.0],
        }
    )
    builder = GsodDatasetBuilder(make_config())

    filtered = builder._filter_mainland_us(df)

    assert len(filtered) == 1
    assert filtered.iloc[0]["LATITUDE"] == 35.0


def test_select_model_columns_excludes_forbidden_features() -> None:
    df = pd.DataFrame(
        {
            "TEMP": [70.0],
            "FRSHTT": [1],
            "DATE": ["2024-01-01"],
            "TORNADO_LABEL": [1],
            "LATITUDE": [35.0],
            "LONGITUDE": [-97.0],
        }
    )
    builder = GsodDatasetBuilder(make_config())

    selected = builder._select_model_columns(df)

    assert selected.columns.tolist() == ["TEMP", "LATITUDE", "LONGITUDE", "TORNADO_LABEL"]
