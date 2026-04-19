from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tornado_ml.config import ProjectConfig


class GsodDatasetBuilder:
    """Build the final station-day modeling table from raw GSOD CSV files."""

    REQUIRED_COLUMNS = {"LATITUDE", "LONGITUDE"}

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.date_min_: Optional[str] = None
        self.date_max_: Optional[str] = None

    def build(self) -> pd.DataFrame:
        df = self._load_raw_files()
        self._validate_required_columns(df)
        df = self._parse_dates(df)
        df = self._filter_mainland_us(df)
        df = self._filter_date_range(df)
        df = self._build_target(df)
        df = self._normalize_missing_values(df)
        self._record_date_range(df)
        df = self._select_model_columns(df)
        return df.reset_index(drop=True)

    def _load_raw_files(self) -> pd.DataFrame:
        files: list[str] = []
        for pattern in self.config.raw_data_paths:
            matches = glob(pattern)
            files.extend(matches if matches else [pattern])

        existing_files = [Path(file) for file in files if Path(file).exists()]
        if not existing_files:
            raise FileNotFoundError(f"No raw GSOD files found for {self.config.raw_data_paths}.")

        frames = [pd.read_csv(file, low_memory=False) for file in existing_files]
        return pd.concat(frames, ignore_index=True)

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS.difference(df.columns)
        if "DATE" not in df.columns:
            missing.add("DATE")
        if "FRSHTT" not in df.columns and self.config.target_column not in df.columns:
            missing.add(f"FRSHTT or {self.config.target_column}")
        if missing:
            raise ValueError(f"Raw data is missing required columns: {sorted(missing)}")

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        parsed = df.copy()
        parsed["DATE"] = pd.to_datetime(parsed["DATE"], errors="coerce")
        return parsed.dropna(subset=["DATE"])

    def _filter_mainland_us(self, df: pd.DataFrame) -> pd.DataFrame:
        bounds = self.config.mainland_us_bounds
        lat = pd.to_numeric(df["LATITUDE"], errors="coerce")
        lon = pd.to_numeric(df["LONGITUDE"], errors="coerce")
        mask = (
            lat.between(bounds.min_latitude, bounds.max_latitude)
            & lon.between(bounds.min_longitude, bounds.max_longitude)
        )
        return df.loc[mask].copy()

    def _filter_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered = df
        if self.config.start_date:
            filtered = filtered[filtered["DATE"] >= pd.Timestamp(self.config.start_date)]
        if self.config.end_date:
            filtered = filtered[filtered["DATE"] <= pd.Timestamp(self.config.end_date)]
        return filtered.copy()

    def _build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.target_column in df.columns:
            labeled = df.copy()
            labeled[self.config.target_column] = labeled[self.config.target_column].astype(int)
            return labeled

        labeled = df.copy()
        frshtt = labeled["FRSHTT"].fillna(0).astype(str).str.replace(r"\.0$", "", regex=True)
        sixth_digit = frshtt.str.zfill(6).str[5]
        labeled[self.config.target_column] = np.where(sixth_digit == "1", 1, 0).astype(int)
        return labeled

    def _normalize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        for feature in self.config.candidate_features:
            if feature in normalized.columns:
                normalized[feature] = pd.to_numeric(normalized[feature], errors="coerce")
                normalized[feature] = normalized[feature].replace(
                    self.config.placeholder_values,
                    np.nan,
                )
        return normalized

    def _record_date_range(self, df: pd.DataFrame) -> None:
        if df.empty:
            self.date_min_ = None
            self.date_max_ = None
            return
        self.date_min_ = df["DATE"].min().date().isoformat()
        self.date_max_ = df["DATE"].max().date().isoformat()

    def _select_model_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        available_features = [
            feature
            for feature in self.config.candidate_features
            if feature in df.columns and feature not in self.config.forbidden_features
        ]
        columns = available_features + [self.config.target_column]
        return df.loc[:, columns].copy()
