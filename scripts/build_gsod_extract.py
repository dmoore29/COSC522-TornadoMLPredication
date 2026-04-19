from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from tornado_ml.config import MainlandUsBounds, ProjectConfig

DEFAULT_INPUT_ROOT = Path("/Users/davidmoore/Downloads/gsod_recent/extracted")
DEFAULT_OUTPUT_PATH = Path("data/raw/gsod_2021_2025_mainland_us.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mainland U.S. GSOD extract from downloaded yearly station CSVs."
    )
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_INPUT_ROOT),
        help="Root folder containing extracted GSOD year folders.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2021", "2022", "2023", "2024", "2025"],
        help="Year folders to scan.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Combined output CSV path.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Project config used for feature and bound settings.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for smoke-testing the extractor.",
    )
    return parser.parse_args()


def iter_csv_files(input_root: Path, years: list[str], max_files: Optional[int]) -> list[Path]:
    files: list[Path] = []
    for year in years:
        year_dir = input_root / year
        files.extend(sorted(year_dir.glob("*.csv")))
        if max_files is not None and len(files) >= max_files:
            return files[:max_files]
    return files


def filter_mainland_us(df: pd.DataFrame, bounds: MainlandUsBounds) -> pd.DataFrame:
    lat = pd.to_numeric(df["LATITUDE"], errors="coerce")
    lon = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    mask = (
        lat.between(bounds.min_latitude, bounds.max_latitude)
        & lon.between(bounds.min_longitude, bounds.max_longitude)
    )
    return df.loc[mask].copy()


def build_target(df: pd.DataFrame, target_column: str) -> pd.Series:
    frshtt = df["FRSHTT"].fillna(0).astype(str).str.replace(r"\.0$", "", regex=True)
    return (frshtt.str.zfill(6).str[5] == "1").astype(int).rename(target_column)


def main() -> None:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    input_root = Path(args.input_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    required_columns = [
        "STATION",
        "DATE",
        "LATITUDE",
        "LONGITUDE",
        "ELEVATION",
        "NAME",
        "FRSHTT",
    ]
    model_columns = list(dict.fromkeys(required_columns + config.candidate_features))
    files = iter_csv_files(input_root, args.years, args.max_files)
    if not files:
        raise FileNotFoundError(f"No station CSV files found under {input_root} for {args.years}.")

    rows_written = 0
    positives = 0
    files_scanned = 0
    files_with_rows = 0
    output_exists = False
    date_min = None
    date_max = None

    for file in files:
        files_scanned += 1
        df = pd.read_csv(file, usecols=lambda column: column in model_columns, low_memory=False)
        missing_required = set(required_columns).difference(df.columns)
        if missing_required:
            raise ValueError(f"{file} is missing required columns: {sorted(missing_required)}")

        filtered = filter_mainland_us(df, config.mainland_us_bounds)
        if filtered.empty:
            continue

        filtered[config.target_column] = build_target(filtered, config.target_column)
        keep_columns = ["DATE"] + [
            column
            for column in model_columns
            if column in filtered.columns
            and column not in config.forbidden_features
            and column != "DATE"
        ]
        output_df = filtered[keep_columns + [config.target_column]]
        output_df.to_csv(output_path, mode="a", header=not output_exists, index=False)
        output_exists = True

        files_with_rows += 1
        rows_written += len(output_df)
        positives += int(output_df[config.target_column].sum())
        dates = pd.to_datetime(filtered["DATE"], errors="coerce")
        current_min = dates.min()
        current_max = dates.max()
        if pd.notna(current_min):
            date_min = current_min if date_min is None else min(date_min, current_min)
        if pd.notna(current_max):
            date_max = current_max if date_max is None else max(date_max, current_max)

    print(f"Files scanned: {files_scanned}")
    print(f"Files with mainland U.S. rows: {files_with_rows}")
    print(f"Rows written: {rows_written}")
    print(f"Positive labels: {positives}")
    print(f"Positive rate: {positives / rows_written if rows_written else 0:.8f}")
    print(f"Date min: {date_min.date().isoformat() if date_min is not None else 'NA'}")
    print(f"Date max: {date_max.date().isoformat() if date_max is not None else 'NA'}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
