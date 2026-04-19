from __future__ import annotations

from typing import Optional

import pandas as pd

from tornado_ml.data_types import DataSplit


def build_dataset_summary(
    df: pd.DataFrame,
    target_column: str,
    features: list[str],
    date_min: Optional[str] = None,
    date_max: Optional[str] = None,
) -> dict:
    counts = df[target_column].value_counts().to_dict()
    positive_count = int(counts.get(1, 0))
    negative_count = int(counts.get(0, 0))
    n_rows = int(len(df))
    summary = {
        "n_rows": n_rows,
        "n_features": int(len(features)),
        "target_column": target_column,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_count / n_rows if n_rows else 0.0,
        "features": features,
    }
    if date_min is not None:
        summary["date_min"] = date_min
    if date_max is not None:
        summary["date_max"] = date_max
    return summary


def build_missingness_table(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    feature_df = df.drop(columns=[target_column])
    rows = []
    for feature in feature_df.columns:
        missing_count = int(feature_df[feature].isna().sum())
        rows.append(
            {
                "feature": feature,
                "missing_count": missing_count,
                "missing_percent": missing_count / len(feature_df) if len(feature_df) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("missing_percent", ascending=False)


def build_split_summary(split: DataSplit) -> pd.DataFrame:
    rows = []
    for split_name, y in [
        ("train", split.y_train),
        ("validation", split.y_val),
        ("test", split.y_test),
    ]:
        counts = y.value_counts().to_dict()
        positive_count = int(counts.get(1, 0))
        negative_count = int(counts.get(0, 0))
        n_rows = int(len(y))
        rows.append(
            {
                "split": split_name,
                "n_rows": n_rows,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_rate": positive_count / n_rows if n_rows else 0.0,
            }
        )
    return pd.DataFrame(rows)
