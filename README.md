# Tornado ML Prediction

Machine learning project for predicting whether a tornado/funnel-cloud event was recorded for a mainland U.S. GSOD station-day.

## Project Definition

- Dataset: NOAA Global Surface Summary of the Day (GSOD)
- Unit of analysis: one weather station on one calendar day
- Target: `TORNADO_LABEL`, derived from the 6th digit of `FRSHTT` after left-padding to six digits
- Models: Logistic Regression and Random Forest
- Required metrics: accuracy, precision, recall, F1-score, and confusion matrix
- Main emphasis: recall and F1-score because the positive class is rare

The model must not use `FRSHTT` or any feature derived from the `FRSHTT` event-code digits as predictors.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run Tests

```bash
pytest
```

Or, after running `make setup`:

```bash
make test
```

## Run Experiment

Place GSOD CSV files in `data/raw/`, then update `configs/default.yaml` if needed.

Build a combined mainland U.S. extract from the downloaded NOAA archives:

```bash
make build-extract
```

Inspect the configured data without training models:

```bash
python3 -m tornado_ml.inspect_data --config configs/default.yaml
```

Or:

```bash
make inspect
```

Run the full experiment:

```bash
python3 -m tornado_ml.main --config configs/default.yaml
```

Or:

```bash
make run
```

Generated metrics, tables, and models are saved under `outputs/`.

Key report-ready outputs:

- `outputs/metrics/dataset_summary.json`
- `outputs/metrics/metrics.json`
- `outputs/tables/missingness_summary.csv`
- `outputs/tables/split_summary.csv`
- `outputs/tables/model_comparison.csv`
