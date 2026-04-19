from __future__ import annotations

import pandas as pd

from tornado_ml.artifact_manager import ArtifactManager
from tornado_ml.config import ProjectConfig
from tornado_ml.dataset_builder import GsodDatasetBuilder
from tornado_ml.summaries import build_dataset_summary, build_missingness_table


def test_inspection_outputs_can_be_written_without_training(tmp_path) -> None:
    raw_path = tmp_path / "synthetic_gsod.csv"
    pd.DataFrame(
        {
            "DATE": ["2024-01-01", "2024-01-02"],
            "FRSHTT": [0, 1],
            "LATITUDE": [35.0, 35.0],
            "LONGITUDE": [-97.0, -97.0],
            "TEMP": [50.0, None],
        }
    ).to_csv(raw_path, index=False)
    config = ProjectConfig(
        raw_data_paths=[str(raw_path)],
        candidate_features=["TEMP", "LATITUDE", "LONGITUDE"],
        forbidden_features=["FRSHTT", "DATE", "STATION", "NAME", "source_file"],
        output_dir=str(tmp_path / "outputs"),
        save_processed_data=True,
        processed_data_path=str(tmp_path / "data/processed/modeling_dataset.csv"),
    )

    builder = GsodDatasetBuilder(config)
    df = builder.build()
    features = df.drop(columns=[config.target_column]).columns.tolist()
    artifacts = ArtifactManager(config.output_dir)
    artifacts.save_json(
        build_dataset_summary(
            df,
            config.target_column,
            features,
            builder.date_min_,
            builder.date_max_,
        ),
        "dataset_summary.json",
    )
    artifacts.save_dataframe(
        build_missingness_table(df, config.target_column),
        "missingness_summary.csv",
    )
    artifacts.save_dataframe_to_path(df, config.processed_data_path)

    assert (tmp_path / "outputs/metrics/dataset_summary.json").exists()
    assert (tmp_path / "outputs/tables/missingness_summary.csv").exists()
    assert (tmp_path / "data/processed/modeling_dataset.csv").exists()
