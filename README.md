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
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run Tests

```bash
pytest
```

## Run Experiment

Place GSOD CSV files in `data/raw/`, then update `configs/default.yaml` if needed.

```bash
python -m tornado_ml.main --config configs/default.yaml
```

Generated metrics, tables, and models are saved under `outputs/`.
