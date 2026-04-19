from __future__ import annotations

from tornado_ml.config import ProjectConfig


def test_default_config_loads_and_validates() -> None:
    config = ProjectConfig.from_yaml("configs/default.yaml")

    config.validate()

    assert config.target_column == "TORNADO_LABEL"
    assert config.output_dir == "outputs"
    assert config.processed_data_path == "data/processed/modeling_dataset.csv"
