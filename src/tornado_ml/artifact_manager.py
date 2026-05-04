from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import joblib
import pandas as pd


class ArtifactManager:
    def __init__(self, output_dir: Union[str, Path] = "outputs"):
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "metrics"
        self.models_dir = self.output_dir / "models"
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"
        
        for directory in [self.metrics_dir, self.models_dir, self.tables_dir, self.figures_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def save_json(self, data: dict[str, Any], filename: str) -> Path:
        path = self.metrics_dir / filename
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        return path

    def save_model(self, model: object, filename: str) -> Path:
        path = self.models_dir / filename
        joblib.dump(model, path)
        return path

    def save_dataframe(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.tables_dir / filename
        df.to_csv(path, index=False)
        return path

    def save_dataframe_to_path(self, df: pd.DataFrame, path: Union[str, Path]) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path
