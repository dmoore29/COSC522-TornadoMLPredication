from __future__ import annotations

import argparse
import json

from tornado_ml.artifact_manager import ArtifactManager
from tornado_ml.config import ProjectConfig
from tornado_ml.dataset_builder import GsodDatasetBuilder
from tornado_ml.summaries import build_dataset_summary, build_missingness_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect configured GSOD data without training models."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    config.validate()

    builder = GsodDatasetBuilder(config)
    df = builder.build()
    features = df.drop(columns=[config.target_column]).columns.tolist()
    dataset_summary = build_dataset_summary(
        df,
        config.target_column,
        features,
        builder.date_min_,
        builder.date_max_,
    )
    missingness = build_missingness_table(df, config.target_column)

    artifacts = ArtifactManager(config.output_dir)
    artifacts.save_json(dataset_summary, "dataset_summary.json")
    artifacts.save_dataframe(missingness, "missingness_summary.csv")
    if config.save_processed_data:
        artifacts.save_dataframe_to_path(df, config.processed_data_path)

    print(json.dumps(dataset_summary, indent=2))


if __name__ == "__main__":
    main()
