# AGENTS.md

This file gives LLM coding agents stable project context. Do not use it for volatile run status, current metrics, or temporary local state.

## Project Goal

Build a machine learning pipeline that predicts whether a tornado/funnel-cloud event was recorded for a mainland U.S. weather station on a given day using NOAA Global Surface Summary of the Day (GSOD) data.

This is a short college machine learning project. Prefer simple, readable, reproducible code over heavy abstractions.

## Task Definition

The task is binary classification at the station-day level.

- One row represents one station on one calendar day.
- Input features are daily GSOD weather measurements.
- Target is `TORNADO_LABEL`.
- Positive class means a tornado/funnel-cloud event was recorded for that station-day.
- Negative class means no tornado/funnel-cloud event was recorded for that station-day.

Preferred precise wording:

```text
predict whether a tornado/funnel-cloud event was recorded for a mainland U.S. station-day
```

Avoid wording that implies real-time tornado forecasting, radar nowcasting, tornado path prediction, or independent tornado-track matching.

## Target Definition

For raw GSOD files, create `TORNADO_LABEL` from `FRSHTT`.

Required parsing rule:

1. Convert `FRSHTT` to a string.
2. Left-pad it to six digits.
3. Read the 6th character.
4. Assign `1` if the 6th character is `1`; otherwise assign `0`.

`FRSHTT` digit meanings:

```text
1 fog
2 rain/drizzle
3 snow/ice pellets
4 hail
5 thunder
6 tornado/funnel cloud
```

## Leakage Rules

Do not use `FRSHTT` as a model feature.

Do not use any feature engineered from `FRSHTT` digits as a model feature.

Forbidden model features include:

```text
FRSHTT
STATION
DATE
NAME
source_file
any target-derived feature
```

Weather event indicators such as thunder, hail, rain, fog, or snow are forbidden if they come from `FRSHTT`.

## Required Models

The project compares exactly these two required models first:

```text
Logistic Regression
Random Forest
```

Logistic Regression is the interpretable baseline.

Random Forest is the nonlinear comparison model.

Do not add more model families unless explicitly asked.

## Required Metrics

Report:

```text
accuracy
precision
recall
F1-score
confusion matrix
```

Because this is rare-event classification, emphasize:

```text
recall
F1-score
confusion matrix
```

ROC-AUC is optional if probabilities are available, but it is not the centerpiece.

## Split And Preprocessing Rules

Use one shared train/validation/test split for both models.

Default split:

```text
70% train
15% validation
15% test
```

Use stratification by the binary target if feasible.

Use a fixed random seed for reproducibility.

Fit preprocessing on the training split only. Apply fitted preprocessors to validation/test.

Current preprocessing pattern:

```text
missingness filtering
median imputation
scaling for Logistic Regression
no required scaling for Random Forest
```

## Dataset Scope

Primary data source:

```text
NOAA Global Surface Summary of the Day (GSOD)
```

Geographic scope:

```text
mainland U.S. stations only
```

The implementation may use coordinate bounds or station metadata. If filtering logic changes, document it in code and docs.

The project should use a broad enough historical window to obtain enough positive examples. Do not reintroduce the old "past 25 years" constraint.

## Coding Standards

Keep the code small and direct.

Prefer the existing project structure:

```text
src/tornado_ml/config.py
src/tornado_ml/dataset_builder.py
src/tornado_ml/splitter.py
src/tornado_ml/preprocessing.py
src/tornado_ml/models.py
src/tornado_ml/evaluation.py
src/tornado_ml/artifact_manager.py
src/tornado_ml/summaries.py
src/tornado_ml/experiment_runner.py
src/tornado_ml/main.py
scripts/build_gsod_extract.py
```

Avoid unnecessary framework patterns, abstract base classes, or generic pipeline engines.

Use `ProjectConfig` for settings instead of scattering constants.

Use `GsodDatasetBuilder` for fixed dataset construction rules.

Use `FeaturePreprocessor` only for train-fitted preprocessing.

Use `ExperimentRunner` for end-to-end orchestration.

Add tests for changes that affect:

- target parsing
- leakage prevention
- split behavior
- preprocessing
- artifact outputs
- model evaluation

Run before finishing:

```bash
pytest
ruff check .
```

## Data And Git Policy

Do not commit raw NOAA archives, extracted station files, combined CSVs, processed datasets, generated metrics, generated figures, or trained models.

Large/generated files belong in:

```text
data/raw/
data/processed/
outputs/
```

These paths are ignored except for `.gitkeep` placeholders.

## Documentation Expectations

When changing workflow behavior, update project docs under `docs/` and the README.

When changing modeling assumptions, update the relevant config and docs.

Do not put volatile metrics or temporary local state in this `AGENTS.md` file.
