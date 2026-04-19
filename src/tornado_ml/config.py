from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass(frozen=True)
class MainlandUsBounds:
    min_latitude: float = 24.0
    max_latitude: float = 49.5
    min_longitude: float = -125.0
    max_longitude: float = -66.5


@dataclass(frozen=True)
class ProjectConfig:
    raw_data_paths: list[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    target_column: str = "TORNADO_LABEL"
    candidate_features: list[str] = field(default_factory=list)
    forbidden_features: list[str] = field(default_factory=list)
    placeholder_values: list[float] = field(default_factory=lambda: [9999.9, 999.9, 99.99, -999.9])
    missingness_threshold: float = 0.95
    output_dir: str = "outputs"
    save_processed_data: bool = False
    processed_data_path: str = "data/processed/modeling_dataset.csv"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 522
    mainland_us_bounds: MainlandUsBounds = field(default_factory=MainlandUsBounds)
    logistic_regression_params: dict[str, Any] = field(default_factory=dict)
    random_forest_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> ProjectConfig:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        bounds_data = data.pop("mainland_us_bounds", {}) or {}
        data["mainland_us_bounds"] = MainlandUsBounds(**bounds_data)
        return cls(**data)

    def validate(self) -> None:
        split_total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(split_total - 1.0) > 1e-9:
            raise ValueError(f"Split ratios must sum to 1.0, got {split_total:.4f}.")
        if not self.raw_data_paths:
            raise ValueError("At least one raw data path or glob must be configured.")
        if not self.candidate_features:
            raise ValueError("At least one candidate feature must be configured.")
