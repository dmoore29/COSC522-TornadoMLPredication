# Getting Started

This guide gets a teammate from a fresh clone to a working local environment.

## 1. Open The Project

```bash
cd "/path/to/TornadoMLPrediction"
```

## 2. Create A Virtual Environment

Use Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should show `(.venv)`.

## 3. Install Dependencies

```bash
python3 -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

This installs the project plus development tools such as `pytest` and `ruff`.

## 4. Verify The Setup

```bash
make test
make lint
```

Expected result:

```text
pytest: all tests passed
ruff: all checks passed
```

## 5. Get The Data

Follow [data_workflow.md](data_workflow.md) to download, extract, and combine the GSOD archives.

After the data workflow, this file should exist:

```text
data/raw/gsod_2021_2025_mainland_us.csv
```

The data file is intentionally ignored by Git.

## 6. Inspect The Data

```bash
make inspect
```

This creates:

```text
outputs/metrics/dataset_summary.json
outputs/tables/missingness_summary.csv
```

Review these before running models.

## 7. Run The Model Pipeline

```bash
make run
```

This trains Logistic Regression and Random Forest using the configured dataset.
Progress logs print during long-running steps such as dataset building, feature preparation, model training, evaluation, and artifact saving.

Outputs are written under:

```text
outputs/
```

To reduce console output:

```bash
python3 -m tornado_ml.main --config configs/default.yaml --log-level WARNING
```

## Troubleshooting

If `python` is not found, use `python3`.

If `pip install -e ".[dev]"` fails because of an old packaging toolchain, run:

```bash
python3 -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

If `make inspect` says the raw file is missing, finish the data workflow first.

If extraction takes a few minutes, that is expected. NOAA stores the yearly archives as many station-level CSV files.
