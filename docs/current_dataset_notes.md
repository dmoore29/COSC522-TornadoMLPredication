# Current Dataset Notes

These notes describe the current local modeling extract. They may change if the project year range or feature list changes.

## Current Extract

Configured input:

```text
data/raw/gsod_2021_2025_mainland_us.csv
```

Source:

```text
NOAA GSOD yearly archives, 2021 through 2025
```

Scope:

```text
Mainland U.S. station-days selected by coordinate bounds
```

Current extraction summary:

```text
Rows: 4,494,957
Positive labels: 217
Negative labels: 4,494,740
Positive rate: 0.0048%
Date range: 2021-01-01 to 2025-08-28
```

## Current Model Features

```text
TEMP
DEWP
SLP
STP
VISIB
WDSP
MXSPD
GUST
MAX
MIN
PRCP
ELEVATION
LATITUDE
LONGITUDE
```

## Excluded Feature: SNDP

`SNDP` means snow depth.

It was removed from the model feature list because the current 2021-2025 mainland U.S. extract showed:

```text
97.84% missingness
```

The raw GSOD files may still contain `SNDP`, but it is not used as a predictor.

## Current Missingness Highlights

After removing `SNDP`, the largest missingness rates are:

```text
GUST   45.74%
SLP    40.31%
STP    22.80%
VISIB  21.64%
DEWP   12.38%
PRCP    8.48%
MXSPD   6.66%
WDSP    6.10%
```

The pipeline currently keeps these features and uses median imputation fit on the training split only.

## Class Imbalance

The positive class is very rare. This affects model interpretation.

Accuracy alone is not enough. Prioritize:

```text
recall
F1-score
confusion matrix
precision
```

## Regenerating These Notes

Run:

```bash
make inspect
```

Then review:

```text
outputs/metrics/dataset_summary.json
outputs/tables/missingness_summary.csv
```
