from __future__ import annotations

import pandas as pd

from tornado_ml.artifact_manager import ArtifactManager
from tornado_ml.config import ProjectConfig
from tornado_ml.data_types import DataSplit, ModelResult
from tornado_ml.dataset_builder import GsodDatasetBuilder
from tornado_ml.evaluation import MetricsEvaluator
from tornado_ml.models import LogisticRegressionModel, RandomForestModel
from tornado_ml.preprocessing import FeaturePreprocessor
from tornado_ml.splitter import DatasetSplitter


class ExperimentRunner:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.dataset_builder = GsodDatasetBuilder(config)
        self.splitter = DatasetSplitter(config)
        self.evaluator = MetricsEvaluator()
        self.artifacts = ArtifactManager()

    def run(self) -> dict:
        self.config.validate()
        df = self.dataset_builder.build()
        split = self.splitter.split(df, self.config.target_column)

        results = [
            self._run_logistic_regression(split),
            self._run_random_forest(split),
        ]
        comparison = self._build_comparison_table(results)

        self.artifacts.save_dataframe(comparison, "model_comparison.csv")
        self.artifacts.save_json(
            {result.model_name: result.metrics for result in results},
            "metrics.json",
        )
        class_balance = {
            int(label): int(count)
            for label, count in df[self.config.target_column].value_counts().to_dict().items()
        }
        return {
            "n_rows": len(df),
            "class_balance": class_balance,
            "metrics": {result.model_name: result.metrics for result in results},
        }

    def _run_logistic_regression(self, split: DataSplit) -> ModelResult:
        preprocessor = FeaturePreprocessor(self.config, scale=True)
        X_train = preprocessor.fit_transform(split.X_train)
        X_test = preprocessor.transform(split.X_test)

        model = LogisticRegressionModel(self.config)
        model.train(X_train, split.y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        metrics = self.evaluator.evaluate("Logistic Regression", split.y_test, predictions, probabilities)

        self.artifacts.save_model(model.estimator, "logistic_regression.joblib")
        return ModelResult(
            "Logistic Regression",
            metrics,
            predictions,
            probabilities,
            preprocessor.selected_features_,
        )

    def _run_random_forest(self, split: DataSplit) -> ModelResult:
        preprocessor = FeaturePreprocessor(self.config, scale=False)
        X_train = preprocessor.fit_transform(split.X_train)
        X_test = preprocessor.transform(split.X_test)

        model = RandomForestModel(self.config)
        model.train(X_train, split.y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        metrics = self.evaluator.evaluate("Random Forest", split.y_test, predictions, probabilities)

        self.artifacts.save_model(model.estimator, "random_forest.joblib")
        return ModelResult(
            "Random Forest",
            metrics,
            predictions,
            probabilities,
            preprocessor.selected_features_,
        )

    def _build_comparison_table(self, results: list[ModelResult]) -> pd.DataFrame:
        rows = []
        for result in results:
            row = {
                key: value
                for key, value in result.metrics.items()
                if key not in {"confusion_matrix"}
            }
            row["selected_feature_count"] = len(result.selected_features)
            rows.append(row)
        return pd.DataFrame(rows)
