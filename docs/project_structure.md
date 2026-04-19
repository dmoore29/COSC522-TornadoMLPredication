# Project Structure

```text
TornadoMLPrediction/
  configs/
    default.yaml
  data/
    raw/
    processed/
  docs/
  outputs/
    figures/
    metrics/
    models/
    tables/
  scripts/
    build_gsod_extract.py
  src/
    tornado_ml/
  tests/
```

## Main Folders

`configs/`

Runtime settings. The main file is `configs/default.yaml`.

`data/raw/`

Local raw or combined data files. Large files are ignored by Git.

`data/processed/`

Optional cleaned modeling datasets. Ignored by Git.

`outputs/`

Generated metrics, figures, tables, and model files. Ignored by Git except for `.gitkeep` placeholders.

`scripts/`

Standalone project scripts. `build_gsod_extract.py` creates the combined mainland U.S. modeling extract from downloaded NOAA archives.

`src/tornado_ml/`

Main Python package.

`tests/`

Pytest test suite.

## Package Modules

`config.py`

Loads and validates project configuration.

`dataset_builder.py`

Builds the modeling dataframe. It handles date parsing, mainland U.S. filtering, target creation, missing placeholder normalization, and feature selection.

`splitter.py`

Creates the shared train/validation/test split.

`preprocessing.py`

Fits train-only preprocessing. Current behavior is missingness filtering, median imputation, and optional scaling.

`models.py`

Contains Logistic Regression and Random Forest wrappers.

`evaluation.py`

Computes accuracy, precision, recall, F1-score, confusion matrix, and optional ROC-AUC.

`artifact_manager.py`

Writes JSON, CSV, and model artifacts.

`summaries.py`

Builds dataset, split, and missingness summaries for inspection/reporting.

`inspect_data.py`

CLI module for data inspection without model training.

`experiment_runner.py`

End-to-end workflow orchestration.

`main.py`

CLI entrypoint for the full experiment.

## Common Commands

```bash
make setup
make test
make lint
make build-extract
make inspect
make run
```
