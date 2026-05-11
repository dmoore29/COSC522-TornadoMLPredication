from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    metrics: dict
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    selected_features: list[str]
    val_metrics: Optional[dict] = None
    val_predictions: Optional[np.ndarray] = None
    val_probabilities: Optional[np.ndarray] = None


class GsodDatasetBuilder:
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
                    self.config.placeholder_values, np.nan
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


class DatasetSplitter:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def split(self, df: pd.DataFrame, target_column: str) -> DataSplit:
        X = df.drop(columns=[target_column])
        y = df[target_column].astype(int)

        stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=self.config.train_ratio,
            random_state=self.config.random_seed,
            stratify=stratify,
        )

        relative_val_ratio = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
        temp_stratify = (
            y_temp if y_temp.nunique() == 2 and y_temp.value_counts().min() >= 2 else None
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=relative_val_ratio,
            random_state=self.config.random_seed,
            stratify=temp_stratify,
        )

        return DataSplit(X_train, X_val, X_test, y_train, y_val, y_test)


class FeatureEngineer:
    def __init__(self) -> None:
        self._slp_mean: float | None = None

    def fit(self, X_train: pd.DataFrame) -> FeatureEngineer:
        if "SLP" in X_train.columns:
            self._slp_mean = float(X_train["SLP"].mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "TEMP" in X.columns and "DEWP" in X.columns:
            X["DEWP_DEPRESSION"] = X["TEMP"] - X["DEWP"]
        if "MAX" in X.columns and "MIN" in X.columns:
            X["TEMP_RANGE"] = X["MAX"] - X["MIN"]
        if "GUST" in X.columns and "MXSPD" in X.columns:
            X["GUST_RATIO"] = (X["GUST"] / (X["MXSPD"] + 1e-6)).clip(upper=10.0)
        if "SLP" in X.columns and self._slp_mean is not None:
            X["PRESSURE_DEFICIT"] = self._slp_mean - X["SLP"]
        return X

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)


class FeaturePreprocessor:
    def __init__(self, config: ProjectConfig, scale: bool):
        self.config = config
        self.scale = scale
        self.selected_features_: list[str] = []
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler() if scale else None

    def fit(self, X_train: pd.DataFrame) -> FeaturePreprocessor:
        missingness = X_train.isna().mean()
        self.selected_features_ = missingness[
            missingness <= self.config.missingness_threshold
        ].index.tolist()
        if not self.selected_features_:
            raise ValueError("No features remain after missingness filtering.")
        train_selected = X_train.loc[:, self.selected_features_]
        imputed = self.imputer.fit_transform(train_selected)
        if self.scaler is not None:
            self.scaler.fit(imputed)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        selected = X.loc[:, self.selected_features_]
        imputed = self.imputer.transform(selected)
        values = self.scaler.transform(imputed) if self.scaler is not None else imputed
        return pd.DataFrame(values, columns=self.selected_features_, index=X.index)

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)


class LogisticRegressionModel:
    def __init__(self, config: ProjectConfig):
        self.estimator = LogisticRegression(**config.logistic_regression_params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.estimator.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]


class RandomForestModel:
    def __init__(self, config: ProjectConfig):
        self.estimator = RandomForestClassifier(**config.random_forest_params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.estimator.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]


class MetricsEvaluator:
    def evaluate(
        self,
        model_name: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> dict:
        labels = [0, 1]
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        metrics = {
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": {
                "true_negative": int(matrix[0, 0]),
                "false_positive": int(matrix[0, 1]),
                "false_negative": int(matrix[1, 0]),
                "true_positive": int(matrix[1, 1]),
            },
        }
        if y_prob is not None and pd.Series(y_true).nunique() == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        return metrics

    @staticmethod
    def find_optimal_threshold(
        y_true: pd.Series,
        y_prob: np.ndarray,
        beta: float = 2.0,
        min_recall: float = 0.80,
    ) -> float:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        p, r = precisions[:-1], recalls[:-1]
        denom = beta**2 * p + r
        fbeta = np.where(denom == 0, 0.0, (1 + beta**2) * p * r / denom)
        fbeta = np.where(r >= min_recall, fbeta, 0.0)
        best_idx = int(np.argmax(fbeta))
        return float(thresholds[best_idx])


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
        rows.append({
            "feature": feature,
            "missing_count": missing_count,
            "missing_percent": missing_count / len(feature_df) if len(feature_df) else 0.0,
        })
    return pd.DataFrame(rows).sort_values("missing_percent", ascending=False)


def build_split_summary(split: DataSplit) -> pd.DataFrame:
    rows = []
    for split_name, y in [("train", split.y_train), ("validation", split.y_val), ("test", split.y_test)]:
        counts = y.value_counts().to_dict()
        positive_count = int(counts.get(1, 0))
        negative_count = int(counts.get(0, 0))
        n_rows = int(len(y))
        rows.append({
            "split": split_name,
            "n_rows": n_rows,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_rate": positive_count / n_rows if n_rows else 0.0,
        })
    return pd.DataFrame(rows)


class ArtifactManager:
    def __init__(self, output_dir: Union[str, Path] = "outputs"):
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "metrics"
        self.models_dir = self.output_dir / "models"
        self.tables_dir = self.output_dir / "tables"
        for directory in [self.metrics_dir, self.models_dir, self.tables_dir]:
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


def build_comparison_table(results: list[ModelResult], split: str = "test") -> pd.DataFrame:
    rows = []
    for result in results:
        source = result.metrics if split == "test" else result.val_metrics
        if source is None:
            continue
        row = {k: v for k, v in source.items() if k != "confusion_matrix"}
        row["selected_feature_count"] = len(result.selected_features)
        rows.append(row)
    return pd.DataFrame(rows)


def build_confusion_matrix_table(results: list[ModelResult], split: str) -> pd.DataFrame:
    rows = []
    for result in results:
        source = result.metrics if split == "test" else result.val_metrics
        if source is None:
            continue
        cm = source["confusion_matrix"]
        rows.append({
            "model_name": result.model_name,
            "split": split,
            "true_negative": cm["true_negative"],
            "false_positive": cm["false_positive"],
            "false_negative": cm["false_negative"],
            "true_positive": cm["true_positive"],
        })
    return pd.DataFrame(rows)


def build_threshold_table(
    results: list[ModelResult],
    split: DataSplit,
    evaluator: MetricsEvaluator,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 0.55, 0.05).tolist()]
    rows = []
    for result in results:
        if result.val_probabilities is None:
            continue
        for threshold in thresholds:
            val_pred_thresh = (result.val_probabilities >= threshold).astype(int)
            metrics = evaluator.evaluate(result.model_name, split.y_val, val_pred_thresh)
            cm = metrics["confusion_matrix"]
            rows.append({
                "model_name": result.model_name,
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "true_negative": cm["true_negative"],
                "false_positive": cm["false_positive"],
                "false_negative": cm["false_negative"],
                "true_positive": cm["true_positive"],
            })
    return pd.DataFrame(rows)


def plot_confusion_matrix(cm_row: pd.Series, title: str, path: Path) -> None:
    matrix = [
        [int(cm_row["true_negative"]),  int(cm_row["false_positive"])],
        [int(cm_row["false_negative"]), int(cm_row["true_positive"])],
    ]
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        matrix,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_figures(
    dataset_summary: dict,
    missingness: pd.DataFrame,
    val_comparison: pd.DataFrame,
    test_comparison: pd.DataFrame,
    test_cm_table: pd.DataFrame,
    lr_coef_df: pd.DataFrame,
    rf_importance_df: pd.DataFrame,
    lr_roc_df: pd.DataFrame,
    lr_roc_auc: float,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["Negative (0)", "Positive (1)"],
        [dataset_summary["negative_count"], dataset_summary["positive_count"]],
        color=["steelblue", "tomato"],
    )
    ax.set_title("Class Balance")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.3,
            f"{int(bar.get_height()):,}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(figures_dir / "class_balance.png", dpi=150)
    plt.close(fig)

    miss_plot = missingness[missingness["missing_percent"] > 0].sort_values("missing_percent", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, len(miss_plot) * 0.45)))
    ax.barh(miss_plot["feature"], miss_plot["missing_percent"] * 100, color="steelblue")
    ax.axvline(95, color="red", linestyle="--", label="95% drop threshold")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Feature Missingness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "missingness.png", dpi=150)
    plt.close(fig)

    metric_cols = ["precision", "recall", "f1", "roc_auc"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, (comp_df, title) in zip(axes, [(val_comparison, "Validation"), (test_comparison, "Test")]):
        cols = [c for c in metric_cols if c in comp_df.columns]
        plot_df = comp_df[["model_name"] + cols].set_index("model_name")
        plot_df.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(f"{title} Set Metrics")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "model_metrics_comparison.png", dpi=150)
    plt.close(fig)

    for _, row in test_cm_table.iterrows():
        slug = row["model_name"].lower().replace(" ", "_")
        plot_confusion_matrix(row, f"{row['model_name']} Confusion Matrix (Test)", figures_dir / f"confusion_matrix_{slug}.png")

    top_coef = lr_coef_df.head(15)
    colors = ["tomato" if c < 0 else "steelblue" for c in top_coef["coefficient"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_coef["feature"][::-1], top_coef["coefficient"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title("Logistic Regression - Top 15 Coefficients\n(blue = raises tornado probability, red = lowers it)")
    fig.tight_layout()
    fig.savefig(figures_dir / "logistic_regression_coefficients.png", dpi=150)
    plt.close(fig)

    top_coef_abs = lr_coef_df.nlargest(15, "abs_coefficient")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_coef_abs["feature"][::-1], top_coef_abs["abs_coefficient"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Logistic Regression - Top 15 Features")
    fig.tight_layout()
    fig.savefig(figures_dir / "logistic_regression_feature_importance.png", dpi=150)
    plt.close(fig)

    top_imp = rf_importance_df.head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_imp["feature"][::-1], top_imp["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest - Top 15 Feature Importances")
    fig.tight_layout()
    fig.savefig(figures_dir / "random_forest_feature_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier (AUC = 0.50)")
    ax.plot(lr_roc_df["fpr"], lr_roc_df["tpr"], "b-", linewidth=2.5, label=f"Logistic Regression (AUC = {lr_roc_auc:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Logistic Regression ROC Curve on Test Set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curve_logistic_regression.png", dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    config = ProjectConfig.from_yaml("configs/default.yaml")
    config.validate()

    logger.info("Building dataset")
    dataset_builder = GsodDatasetBuilder(config)
    df = dataset_builder.build()
    logger.info("Dataset: %s rows", f"{len(df):,}")

    features = df.drop(columns=[config.target_column]).columns.tolist()
    dataset_summary = build_dataset_summary(
        df, config.target_column, features,
        dataset_builder.date_min_, dataset_builder.date_max_,
    )
    missingness = build_missingness_table(df, config.target_column)

    logger.info("Splitting dataset")
    splitter = DatasetSplitter(config)
    split = splitter.split(df, config.target_column)
    split_summary = build_split_summary(split)

    evaluator = MetricsEvaluator()
    artifacts = ArtifactManager(config.output_dir)

    logger.info("Training Logistic Regression")
    lr_engineer = FeatureEngineer()
    X_train_lr_eng = lr_engineer.fit_transform(split.X_train)
    X_val_lr_eng = lr_engineer.transform(split.X_val)
    X_test_lr_eng = lr_engineer.transform(split.X_test)

    lr_preprocessor = FeaturePreprocessor(config, scale=True)
    X_train_lr = lr_preprocessor.fit_transform(X_train_lr_eng)
    X_val_lr = lr_preprocessor.transform(X_val_lr_eng)
    X_test_lr = lr_preprocessor.transform(X_test_lr_eng)

    lr_model = LogisticRegressionModel(config)
    lr_model.train(X_train_lr, split.y_train)

    lr_val_prob = lr_model.predict_proba(X_val_lr)
    optimal_threshold = evaluator.find_optimal_threshold(split.y_val, lr_val_prob)
    logger.info("LR optimal threshold: %.4f", optimal_threshold)

    lr_val_pred = (lr_val_prob >= optimal_threshold).astype(int)
    lr_val_metrics = evaluator.evaluate("Logistic Regression", split.y_val, lr_val_pred, lr_val_prob)
    lr_val_metrics["optimal_threshold"] = optimal_threshold

    lr_test_prob = lr_model.predict_proba(X_test_lr)
    lr_test_pred = (lr_test_prob >= optimal_threshold).astype(int)
    lr_test_metrics = evaluator.evaluate("Logistic Regression", split.y_test, lr_test_pred, lr_test_prob)
    lr_test_metrics["optimal_threshold"] = optimal_threshold

    lr_coef_df = pd.DataFrame({
        "feature": lr_preprocessor.selected_features_,
        "coefficient": lr_model.estimator.coef_[0],
    })
    lr_coef_df["abs_coefficient"] = lr_coef_df["coefficient"].abs()
    lr_coef_df = lr_coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    lr_fpr, lr_tpr, _ = roc_curve(split.y_test, lr_test_prob)
    lr_roc_df = pd.DataFrame({"fpr": lr_fpr, "tpr": lr_tpr})

    lr_result = ModelResult(
        model_name="Logistic Regression",
        metrics=lr_test_metrics,
        predictions=lr_test_pred,
        probabilities=lr_test_prob,
        selected_features=lr_preprocessor.selected_features_,
        val_metrics=lr_val_metrics,
        val_predictions=lr_val_pred,
        val_probabilities=lr_val_prob,
    )

    artifacts.save_model(lr_model.estimator, "logistic_regression.joblib")
    artifacts.save_dataframe(lr_coef_df, "logistic_regression_coefficients.csv")
    artifacts.save_dataframe(lr_roc_df, "logistic_regression_roc_curve.csv")

    logger.info("Training Random Forest")
    rf_preprocessor = FeaturePreprocessor(config, scale=False)
    X_train_rf = rf_preprocessor.fit_transform(split.X_train)
    X_val_rf = rf_preprocessor.transform(split.X_val)
    X_test_rf = rf_preprocessor.transform(split.X_test)

    rf_model = RandomForestModel(config)
    rf_model.train(X_train_rf, split.y_train)

    rf_val_pred = rf_model.predict(X_val_rf)
    rf_val_prob = rf_model.predict_proba(X_val_rf)
    rf_val_metrics = evaluator.evaluate("Random Forest", split.y_val, rf_val_pred, rf_val_prob)

    rf_test_pred = rf_model.predict(X_test_rf)
    rf_test_prob = rf_model.predict_proba(X_test_rf)
    rf_test_metrics = evaluator.evaluate("Random Forest", split.y_test, rf_test_pred, rf_test_prob)

    rf_importance_df = pd.DataFrame({
        "feature": rf_preprocessor.selected_features_,
        "importance": rf_model.estimator.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    rf_result = ModelResult(
        model_name="Random Forest",
        metrics=rf_test_metrics,
        predictions=rf_test_pred,
        probabilities=rf_test_prob,
        selected_features=rf_preprocessor.selected_features_,
        val_metrics=rf_val_metrics,
        val_predictions=rf_val_pred,
        val_probabilities=rf_val_prob,
    )

    artifacts.save_model(rf_model.estimator, "random_forest.joblib")
    artifacts.save_dataframe(rf_importance_df, "random_forest_feature_importance.csv")

    results = [lr_result, rf_result]

    val_comparison = build_comparison_table(results, split="val")
    test_comparison = build_comparison_table(results, split="test")
    val_cm_table = build_confusion_matrix_table(results, split="val")
    test_cm_table = build_confusion_matrix_table(results, split="test")
    threshold_table = build_threshold_table(results, split, evaluator)

    artifacts.save_dataframe(val_comparison, "validation_model_comparison.csv")
    artifacts.save_dataframe(test_comparison, "test_model_comparison.csv")
    artifacts.save_dataframe(val_cm_table, "validation_confusion_matrices.csv")
    artifacts.save_dataframe(test_cm_table, "test_confusion_matrices.csv")
    artifacts.save_dataframe(threshold_table, "threshold_metrics.csv")
    artifacts.save_dataframe(split_summary, "split_summary.csv")
    artifacts.save_dataframe(missingness, "missingness_summary.csv")
    artifacts.save_json(dataset_summary, "dataset_summary.json")
    artifacts.save_json(
        {result.model_name: result.metrics for result in results},
        "metrics.json",
    )

    logger.info("Generating figures")
    generate_figures(
        dataset_summary=dataset_summary,
        missingness=missingness,
        val_comparison=val_comparison,
        test_comparison=test_comparison,
        test_cm_table=test_cm_table,
        lr_coef_df=lr_coef_df,
        rf_importance_df=rf_importance_df,
        lr_roc_df=lr_roc_df,
        lr_roc_auc=lr_test_metrics["roc_auc"],
        figures_dir=Path(config.output_dir) / "figures",
    )

    logger.info("Done")
    print(json.dumps(
        {result.model_name: result.metrics for result in results},
        indent=2,
    ))


if __name__ == "__main__":
    main()
