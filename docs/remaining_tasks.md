# Remaining Required Coding Tasks

This file tracks only the coding work required to finish the project well. Optional improvements are intentionally excluded.

## 1. Add Validation Metrics

The pipeline already creates train/validation/test splits, but model comparison currently centers on test output.

Add validation-set evaluation for both models so development decisions are not based directly on the test set.

Required outputs:

```text
outputs/tables/validation_model_comparison.csv
outputs/tables/test_model_comparison.csv
```

The validation table should be used for development decisions. The test table should be used for final reporting.

## 2. Add Threshold Analysis

This is required because the first real run showed that Random Forest predicted no positive cases at the default `0.5` threshold.

Add threshold evaluation on validation probabilities for both models.

Required output:

```text
outputs/tables/threshold_metrics.csv
```

Required columns:

```text
model_name
threshold
accuracy
precision
recall
f1
false_positive
false_negative
true_positive
true_negative
```

Use validation data for threshold selection. Apply the selected threshold once to the test set.

## 3. Export Confusion Matrix Tables

Confusion matrices are required by the project rubric.

Metrics JSON already contains confusion matrix values, but the report and presentation need cleaner table artifacts.

Required outputs:

```text
outputs/tables/validation_confusion_matrices.csv
outputs/tables/test_confusion_matrices.csv
```

Required columns:

```text
model_name
split
true_negative
false_positive
false_negative
true_positive
```

## 4. Export Model Interpretation Tables

The report should explain what the models learned and which features mattered.

Required outputs:

```text
outputs/tables/logistic_regression_coefficients.csv
outputs/tables/random_forest_feature_importance.csv
```

The Logistic Regression table should include:

```text
feature
coefficient
abs_coefficient
```

The Random Forest table should include:

```text
feature
importance
```

Sort both tables from most important to least important.

## 5. Generate Required Figures

The final report and presentation require visuals.

Add a plotting script, likely:

```text
scripts/generate_figures.py
```

Required figures:

```text
outputs/figures/class_balance.png
outputs/figures/missingness.png
outputs/figures/model_metrics_comparison.png
outputs/figures/confusion_matrix_logistic_regression.png
outputs/figures/confusion_matrix_random_forest.png
outputs/figures/logistic_regression_coefficients.png
outputs/figures/random_forest_feature_importance.png
```

Keep figures simple, readable, and report-ready.

## 6. Final Verification

Before final submission, run:

```bash
make test
make lint
make inspect
make run
```

Confirm that final required artifacts exist under:

```text
outputs/metrics/
outputs/tables/
outputs/figures/
outputs/models/
```
