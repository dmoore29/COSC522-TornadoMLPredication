from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TABLES = Path("outputs/tables")
METRICS = Path("outputs/metrics")
FIGURES = Path("outputs/figures")
FIGURES.mkdir(parents=True, exist_ok=True)


with open(METRICS / "dataset_summary.json") as f:
    summary = json.load(f)

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(
    ["Negative (0)", "Positive (1)"],
    [summary["negative_count"], summary["positive_count"]],
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
fig.savefig(FIGURES / "class_balance.png", dpi=150)
plt.close(fig)
print("Saved class_balance.png")


miss = pd.read_csv(TABLES / "missingness_summary.csv")
miss = miss[miss["missing_percent"] > 0].sort_values("missing_percent", ascending=True)

fig, ax = plt.subplots(figsize=(7, max(3, len(miss) * 0.45)))
ax.barh(miss["feature"], miss["missing_percent"] * 100, color="steelblue")
ax.axvline(95, color="red", linestyle="--", label="95% drop threshold")
ax.set_xlabel("Missing (%)")
ax.set_title("Feature Missingness")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "missingness.png", dpi=150)
plt.close(fig)
print("Saved missingness.png")


val_comp = pd.read_csv(TABLES / "validation_model_comparison.csv")
test_comp = pd.read_csv(TABLES / "test_model_comparison.csv")

metric_cols = ["precision", "recall", "f1", "roc_auc"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
for ax, (df, title) in zip(axes, [(val_comp, "Validation"), (test_comp, "Test")]):
    cols = [c for c in metric_cols if c in df.columns]
    plot_df = df[["model_name"] + cols].set_index("model_name")
    plot_df.plot(kind="bar", ax=ax, rot=0)
    ax.set_title(f"{title} Set Metrics")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "model_metrics_comparison.png", dpi=150)
plt.close(fig)
print("Saved model_metrics_comparison.png")


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


test_cm = pd.read_csv(TABLES / "test_confusion_matrices.csv")
for _, row in test_cm.iterrows():
    slug = row["model_name"].lower().replace(" ", "_")
    out_path = FIGURES / f"confusion_matrix_{slug}.png"
    plot_confusion_matrix(row, f"{row['model_name']} Confusion Matrix", out_path)
    print(f"Saved confusion_matrix_{slug}.png")


coef = pd.read_csv(TABLES / "logistic_regression_coefficients.csv")
top_n = coef.head(15)

colors = ["tomato" if c < 0 else "steelblue" for c in top_n["coefficient"]]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top_n["feature"][::-1], top_n["coefficient"][::-1], color=colors[::-1])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient")
ax.set_title("Logistic Regression- top 15 Coefficients\n(blue = raises tornado probability, red = lowers it)")
fig.tight_layout()
fig.savefig(FIGURES / "logistic_regression_coefficients.png", dpi=150)
plt.close(fig)
print("Saved logistic_regression_coefficients.png")


imp = pd.read_csv(TABLES / "random_forest_feature_importance.csv")
top_n = imp.head(15)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top_n["feature"][::-1], top_n["importance"][::-1], color="steelblue")
ax.set_xlabel("Importance")
ax.set_title("Random Forest — Top 15 Feature Importances")
fig.tight_layout()
fig.savefig(FIGURES / "random_forest_feature_importance.png", dpi=150)
plt.close(fig)
print("Saved random_forest_feature_importance.png")


coef = pd.read_csv(TABLES / "logistic_regression_coefficients.csv")
coef["abs_coefficient"] = coef["coefficient"].abs()
top_n = coef.nlargest(15, "abs_coefficient")

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top_n["feature"][::-1], top_n["abs_coefficient"][::-1], color="steelblue")
ax.set_xlabel("Importance")
ax.set_title("Logistic Regression- Top 15 Features")
fig.tight_layout()
fig.savefig(FIGURES / "logistic_regression_feature_importance.png", dpi=150)
plt.close(fig)
print("Saved logistic_regression_feature_importance.png")


test_comp = pd.read_csv(TABLES / "test_model_comparison.csv")
lr_test_metrics = test_comp[test_comp["model_name"] == "Logistic Regression"].iloc[0]
lr_roc_auc = lr_test_metrics.get("roc_auc", None)

roc_path = TABLES / "logistic_regression_roc_curve.csv"
if lr_roc_auc is not None and not pd.isna(lr_roc_auc) and roc_path.exists():
    roc_df = pd.read_csv(roc_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier (AUC = 0.50)")
    ax.plot(roc_df["fpr"], roc_df["tpr"], "b-", linewidth=2.5, label=f"Logistic Regression (AUC = {lr_roc_auc:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Logistic Regression ROC Curve on test set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_curve_logistic_regression.png", dpi=150)
    plt.close(fig)
    print("Saved roc_curve_logistic_regression.png")
else:
    print("Warning: ROC curve data not available for LR, skipping ROC curve figure")

print(f"\nAll figures saved to {FIGURES}/")