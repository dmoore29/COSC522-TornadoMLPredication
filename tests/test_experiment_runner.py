from __future__ import annotations

import pandas as pd

from tornado_ml.config import ProjectConfig
from tornado_ml.experiment_runner import ExperimentRunner


def test_experiment_runner_completes_with_synthetic_gsod_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    raw_path = tmp_path / "synthetic_gsod.csv"
    rows = []
    for index in range(40):
        is_positive = index % 4 == 0
        rows.append(
            {
                "DATE": f"2024-01-{(index % 28) + 1:02d}",
                "FRSHTT": 1 if is_positive else 0,
                "LATITUDE": 35.0,
                "LONGITUDE": -97.0,
                "TEMP": 80.0 if is_positive else 50.0,
                "DEWP": 70.0 if is_positive else 35.0,
                "SLP": 1000.0 + index,
                "GUST": 999.9 if index % 5 == 0 else 20.0 + index,
            }
        )
    pd.DataFrame(rows).to_csv(raw_path, index=False)

    config = ProjectConfig(
        raw_data_paths=[str(raw_path)],
        candidate_features=["TEMP", "DEWP", "SLP", "GUST", "LATITUDE", "LONGITUDE"],
        forbidden_features=["FRSHTT", "DATE", "STATION", "NAME", "source_file"],
        logistic_regression_params={"max_iter": 200, "class_weight": "balanced"},
        random_forest_params={
            "n_estimators": 10,
            "class_weight": "balanced",
            "random_state": 522,
            "n_jobs": 1,
        },
    )

    result = ExperimentRunner(config).run()

    assert result["dataset_summary"]["n_rows"] == 40
    assert result["dataset_summary"]["positive_count"] == 10
    assert (tmp_path / "outputs/metrics/dataset_summary.json").exists()
    assert (tmp_path / "outputs/metrics/metrics.json").exists()
    assert (tmp_path / "outputs/tables/missingness_summary.csv").exists()
    assert (tmp_path / "outputs/tables/split_summary.csv").exists()
    assert (tmp_path / "outputs/tables/model_comparison.csv").exists()
    assert (tmp_path / "outputs/models/logistic_regression.joblib").exists()
    assert (tmp_path / "outputs/models/random_forest.joblib").exists()
    assert (tmp_path / "outputs/metrics/validation_metrics.json").exists()
    assert (tmp_path / "outputs/tables/validation_model_comparison.csv").exists()
    assert (tmp_path / "outputs/tables/test_model_comparison.csv").exists()
    assert (tmp_path / "outputs/tables/threshold_metrics.csv").exists()
    assert (tmp_path / "outputs/tables/validation_confusion_matrices.csv").exists()
    assert (tmp_path / "outputs/tables/test_confusion_matrices.csv").exists()
    assert (tmp_path / "outputs/tables/logistic_regression_coefficients.csv").exists()
    assert (tmp_path / "outputs/tables/random_forest_feature_importance.csv").exists()
