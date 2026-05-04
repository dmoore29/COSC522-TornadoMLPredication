from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("outputs")
METRICS_DIR = OUTPUT_DIR / "metrics"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

FEATURE_LABELS = {
    "TEMP": "TEMP\nMean temp",
    "DEWP": "DEWP\nDew point",
    "SLP": "SLP\nSea-level pressure",
    "STP": "STP\nStation pressure",
    "VISIB": "VISIB\nVisibility",
    "WDSP": "WDSP\nAvg wind speed",
    "MXSPD": "MXSPD\nMax sustained wind",
    "GUST": "GUST\nMax gust",
    "MAX": "MAX\nDaily high temp",
    "MIN": "MIN\nDaily low temp",
    "PRCP": "PRCP\nPrecipitation",
    "ELEVATION": "ELEVATION\nStation elevation",
    "LATITUDE": "LATITUDE\nStation latitude",
    "LONGITUDE": "LONGITUDE\nStation longitude",
}


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_class_balance()
    plot_missingness()
    plot_model_metrics_comparison()
    plot_confusion_matrix("Logistic Regression", "confusion_matrix_logistic_regression.png")
    plot_confusion_matrix("Random Forest", "confusion_matrix_random_forest.png")
    plot_logistic_regression_coefficients()
    plot_random_forest_feature_importance()


def plot_class_balance() -> None:
    summary = read_json(METRICS_DIR / "dataset_summary.json")

    labels = ["Non-tornado", "Tornado"]
    counts = [summary["negative_count"], summary["positive_count"]]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, counts)
    plt.yscale("log")
    plt.title("Class Balance\nExtreme imbalance: only 217 tornado cases out of 4.49M rows")
    plt.ylabel("Count (log scale)")

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            count,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balance.png", dpi=200)
    plt.close()


def plot_missingness() -> None:
    path = TABLES_DIR / "missingness_summary.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    if "missing_rate" not in df.columns:
        return

    df = df.sort_values("missing_rate", ascending=False).head(15)

    plt.figure(figsize=(8, 5))
    plt.barh(df["column"], df["missing_rate"])
    plt.gca().invert_yaxis()
    plt.title("Missingness by Feature")
    plt.xlabel("Missing Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missingness.png", dpi=200)
    plt.close()


def plot_model_metrics_comparison() -> None:
    path = TABLES_DIR / "test_model_comparison.csv"
    if not path.exists():
        path = TABLES_DIR / "model_comparison.csv"

    df = pd.read_csv(path)

    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    available_metrics = [column for column in metric_cols if column in df.columns]

    plot_df = df.set_index("model_name")[available_metrics].T
    plot_df.index = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    plt.figure(figsize=(8, 5))
    ax = plot_df.plot(kind="bar")
    ax.set_title("Test Metric Comparison\nAccuracy is high, but recall/F1 reveal rare-event performance")
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    plt.xticks(rotation=0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_metrics_comparison.png", dpi=200)
    plt.close()


def plot_confusion_matrix(model_name: str, filename: str) -> None:
    path = TABLES_DIR / "test_confusion_matrices.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    row = df[df["model_name"] == model_name]

    if row.empty:
        return

    row = row.iloc[0]
    matrix = [
        [row["true_negative"], row["false_positive"]],
        [row["false_negative"], row["true_positive"]],
    ]

    plt.figure(figsize=(5, 4.5))
    plt.imshow(matrix)
    plt.title(f"Confusion Matrix - {model_name}\nTest set predictions")
    plt.xticks([0, 1], ["Pred 0\nNon-tornado", "Pred 1\nTornado"])
    plt.yticks([0, 1], ["Actual 0\nNon-tornado", "Actual 1\nTornado"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{int(matrix[i][j]):,}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close()


def plot_logistic_regression_coefficients() -> None:
    path = TABLES_DIR / "logistic_regression_coefficients.csv"
    if not path.exists():
        return

    df = pd.read_csv(path).head(15).copy()
    df["feature_label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    df = df.sort_values("coefficient")

    colors = ["tab:blue" if value >= 0 else "tab:red" for value in df["coefficient"]]

    plt.figure(figsize=(9, 6))
    plt.barh(df["feature_label"], df["coefficient"], color=colors)
    plt.axvline(0, linewidth=1)
    plt.title(
        "Top Logistic Regression Coefficients\n"
        "Positive values increase tornado probability; negative values decrease it"
    )
    plt.xlabel("Coefficient")

    positive_patch = plt.Rectangle((0, 0), 1, 1, color="tab:blue")
    negative_patch = plt.Rectangle((0, 0), 1, 1, color="tab:red")
    plt.legend(
        [positive_patch, negative_patch],
        ["Raises predicted tornado probability", "Lowers predicted tornado probability"],
        loc="lower right",
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "logistic_regression_coefficients.png", dpi=200)
    plt.close()


def plot_random_forest_feature_importance() -> None:
    path = TABLES_DIR / "random_forest_feature_importance.csv"
    if not path.exists():
        return

    df = pd.read_csv(path).head(15).copy()
    df["feature_label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    df = df.sort_values("importance")

    plt.figure(figsize=(9, 6))
    plt.barh(df["feature_label"], df["importance"])
    plt.title(
        "Random Forest Feature Importance\n"
        "Relative importance only; this does not show whether a feature raises or lowers risk"
    )
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "random_forest_feature_importance.png", dpi=200)
    plt.close()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()