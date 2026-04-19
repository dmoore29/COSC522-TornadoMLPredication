# Data Workflow

This project uses NOAA Global Surface Summary of the Day (GSOD) station-day CSV files.

The raw NOAA archives are large and should not be committed to Git.

## 1. Download Yearly Archives

NOAA archive page:

```text
https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/
```

For the current working extract, download:

```text
2021.tar.gz
2022.tar.gz
2023.tar.gz
2024.tar.gz
2025.tar.gz
```

Recommended local folder:

```text
~/Downloads/gsod_recent/
```

Example download command:

```bash
mkdir -p ~/Downloads/gsod_recent
curl -fL https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/2021.tar.gz -o ~/Downloads/gsod_recent/2021.tar.gz
```

Repeat for each year.

## 2. Extract The Archives

```bash
mkdir -p ~/Downloads/gsod_recent/extracted

for year in 2021 2022 2023 2024 2025; do
  mkdir -p "$HOME/Downloads/gsod_recent/extracted/$year"
  tar -xzf "$HOME/Downloads/gsod_recent/$year.tar.gz" \
    -C "$HOME/Downloads/gsod_recent/extracted/$year"
done
```

Expected structure:

```text
~/Downloads/gsod_recent/
  2021.tar.gz
  2022.tar.gz
  2023.tar.gz
  2024.tar.gz
  2025.tar.gz
  extracted/
    2021/
      station CSV files
    2022/
      station CSV files
    ...
```

The `tar` command may print `Failed to set default locale` on macOS. If files are extracted, this warning is harmless.

## 3. Build The Project Extract

From the project root:

```bash
make build-extract
```

This runs:

```bash
python scripts/build_gsod_extract.py
```

The script:

- scans the extracted yearly station CSV folders
- filters to mainland U.S. rows using coordinate bounds
- keeps only configured project columns
- creates `TORNADO_LABEL`
- writes one combined CSV for modeling

Output:

```text
data/raw/gsod_2021_2025_mainland_us.csv
```

This file is ignored by Git because it is large.

## 4. Inspect The Extract

```bash
make inspect
```

This writes:

```text
outputs/metrics/dataset_summary.json
outputs/tables/missingness_summary.csv
```

Inspection prints progress logs while loading and summarizing the configured data.

Review these files before training models.

## 5. Run The Full Experiment

```bash
make run
```

This writes model metrics, comparison tables, and trained model files under `outputs/`.

## Data Rules

Do not commit:

- NOAA archives
- extracted station CSVs
- combined raw extract CSVs
- processed datasets
- generated metrics, plots, or model files

These are ignored by `.gitignore`.

## Changing The Year Range

To use different years:

1. Download and extract the desired yearly archives.
2. Run `scripts/build_gsod_extract.py` with a custom `--years` list.

Example:

```bash
python scripts/build_gsod_extract.py --years 2020 2021 2022 2023 2024 2025
```

If the project year range changes, update:

- `configs/default.yaml` if paths change
- report Data section
- report Methods section if the change affects modeling
