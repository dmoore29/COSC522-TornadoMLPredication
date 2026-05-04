from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tornado_ml.artifact_manager import ArtifactManager
from tornado_ml.config import ProjectConfig
from tornado_ml.data_types import DataSplit, ModelResult
from tornado_ml.dataset_builder import GsodDatasetBuilder
from tornado_ml.evaluation import MetricsEvaluator, predictions_from_threshold
from tornado_ml.models import LogisticRegressionModel, RandomForestModel
from tornado_ml.preprocessing import FeaturePreprocessor
from tornado_ml.splitter import DatasetSplitter
from tornado_ml.summaries import build_dataset_summary, build_missingness_table, build_split_summary

logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.dataset_builder = GsodDatasetBuilder(config)
        self.splitter = DatasetSplitter(config)
        self.evaluator = MetricsEvaluator()
        self.artifacts = ArtifactManager(config.output_dir)

    def run(self) -> dict:
        logger.info("Validating project configuration")
        self.config.validate()
        logger.info("Building modeling dataset")
        df = self.dataset_builder.build()
        logger.info("Built dataset with %s rows", f"{len(df):,}")
        if self.config.save_processed_data:
            logger.info("Saving processed dataset to %s", self.config.processed_data_path)
            self.artifacts.save_dataframe_to_path(df, self.config.processed_data_path)
        logger.info("Creating shared train/validation/test split")
        split = self.splitter.split(df, self.config.target_column)
        features = df.drop(columns=[self.config.target_column]).columns.tolist()
        dataset_summary = build_dataset_summary(
            df,
            self.config.target_column,
            features,
            self.dataset_builder.date_min_,
            self.dataset_builder.date_max_,
        )
        split_summary = build_split_summary(split)
        missingness = build_missingness_table(df, self.config.target_column)
        logger.info(
            "Dataset class balance: %s positives, %s negatives",
            f"{dataset_summary['positive_count']:,}",
            f"{dataset_summary['negative_count']:,}",
        )

        results = [
            self._run_logistic_regression(split),
            self._run_random_forest(split),
        ]
        
        validation_results = [result["validation"] for result in results]
        test_results = [result["test"] for result in results]
        
        validation_comparison = self._build_comparison_table(validation_results)
        test_comparison = self._build_comparison_table(test_results)
        
        validation_confusion = self._build_confusion_matrix_table(validation_results, "validation")
        test_confusion = self._build_confusion_matrix_table(test_results, "test")
        
        threshold_metrics = pd.concat(
            [result["threshold_metrics"] for result in results],
            ignore_index=True
        )
        
        logger.info("Saving comparison and summary artifacts")
        self.artifacts.save_dataframe(validation_comparison, "validation_model_comparison.csv")
        self.artifacts.save_dataframe(test_comparison, "test_model_comparison.csv")
        self.artifacts.save_dataframe(threshold_metrics, "threshold_metrics.csv")
        self.artifacts.save_dataframe(validation_confusion, "validation_confusion_matrices.csv")
        self.artifacts.save_dataframe(test_confusion, "test_confusion_matrices.csv")

        # Backward-compatible output for existing slides/report references.
        self.artifacts.save_dataframe(test_comparison, "model_comparison.csv")

        self.artifacts.save_dataframe(split_summary, "split_summary.csv")
        self.artifacts.save_dataframe(missingness, "missingness_summary.csv")
        self.artifacts.save_json(dataset_summary, "dataset_summary.json")
        self.artifacts.save_json(
            {result["test"].model_name: result["test"].metrics for result in results},
            "metrics.json",
        )
        self.artifacts.save_json(
            {result["validation"].model_name: result["validation"].metrics for result in results},
            "validation_metrics.json",
        )
        
        return {
            "dataset_summary": dataset_summary,
            "metrics": {result["test"].model_name: result["test"].metrics for result in results},
            "val_metrics": {
                result["validation"].model_name: result["validation"].metrics for result in results
            },
        }

    def _run_logistic_regression(self, split: DataSplit) -> dict[str, ModelResult | pd.DataFrame]:
        logger.info("Preparing Logistic Regression features")
        preprocessor = FeaturePreprocessor(self.config, scale=True)
        
        X_train = preprocessor.fit_transform(split.X_train)
        X_val = preprocessor.transform(split.X_val)
        X_test = preprocessor.transform(split.X_test)

        logger.info("Training Logistic Regression")
        model = LogisticRegressionModel(self.config)
        model.train(X_train, split.y_train)
        logger.info("Evaluating Logistic Regression thresholds on validation set")
        val_probabilities = model.predict_proba(X_val)
        test_probabilities = model.predict_proba(X_test)
        
        thresholds = self._threshold_grid()
        threshold_metrics = self.evaluator.evaluate_thresholds(
            "Logistic Regression",
            split.y_val,
            val_probabilities,
            thresholds,
        )
        selected_threshold = self.evaluator.select_threshold(
            threshold_metrics,
            "Logistic Regression",
        )

        logger.info(
            "Selected Logistic Regression threshold: %.3f",
            selected_threshold,
        )
        
        val_predictions = predictions_from_threshold(val_probabilities, selected_threshold)
        test_predictions = predictions_from_threshold(test_probabilities, selected_threshold)

        validation_metrics = self.evaluator.evaluate(
            "Logistic Regression",
            split.y_val,
            val_predictions,
            val_probabilities,
        )
        validation_metrics["selected_threshold"] = selected_threshold

        test_metrics = self.evaluator.evaluate(
            "Logistic Regression",
            split.y_test,
            test_predictions,
            test_probabilities,
        )
        test_metrics["selected_threshold"] = selected_threshold

        logger.info("Saving Logistic Regression model and coefficient table")
        self.artifacts.save_model(model.estimator, "logistic_regression.joblib")
        self.artifacts.save_dataframe(
            self._build_logistic_regression_coefficients(
                preprocessor.selected_features_,
                model.estimator.coef_[0],
            ),
            "logistic_regression_coefficients.csv",
        )
        
        return {
            "validation": ModelResult(
                "Logistic Regression",
                validation_metrics,
                val_predictions,
                val_probabilities,
                preprocessor.selected_features_,
            ),
            "test": ModelResult(
                "Logistic Regression",
                test_metrics,
                test_predictions,
                test_probabilities,
                preprocessor.selected_features_,
            ),
            "threshold_metrics": threshold_metrics,
        }

    def _run_random_forest(self, split: DataSplit) -> dict[str, ModelResult | pd.DataFrame]:
        logger.info("Preparing Random Forest features")
        preprocessor = FeaturePreprocessor(self.config, scale=False)
        X_train = preprocessor.fit_transform(split.X_train)
        X_val = preprocessor.transform(split.X_val)
        X_test = preprocessor.transform(split.X_test)

        logger.info("Training Random Forest")
        model = RandomForestModel(self.config)
        model.train(X_train, split.y_train)
        logger.info("Evaluating Random Forest thresholds on validation set")
        
        val_probabilities = model.predict_proba(X_val)
        test_probabilities = model.predict_proba(X_test)

        thresholds = self._threshold_grid()
        threshold_metrics = self.evaluator.evaluate_thresholds(
            "Random Forest",
            split.y_val,
            val_probabilities,
            thresholds,
        )
        selected_threshold = self.evaluator.select_threshold(
            threshold_metrics,
            "Random Forest",
        )
        
        
        
        logger.info("Selected Random Forest threshold: %.3f", selected_threshold)

        val_predictions = predictions_from_threshold(val_probabilities, selected_threshold)
        test_predictions = predictions_from_threshold(test_probabilities, selected_threshold)

        validation_metrics = self.evaluator.evaluate(
            "Random Forest",
            split.y_val,
            val_predictions,
            val_probabilities,
        )
        validation_metrics["selected_threshold"] = selected_threshold

        test_metrics = self.evaluator.evaluate(
            "Random Forest",
            split.y_test,
            test_predictions,
            test_probabilities,
        )
        test_metrics["selected_threshold"] = selected_threshold

        logger.info("Saving Random Forest model and feature importance table")
        self.artifacts.save_model(model.estimator, "random_forest.joblib")
        self.artifacts.save_dataframe(
            self._build_random_forest_feature_importance(
                preprocessor.selected_features_,
                model.estimator.feature_importances_,
            ),
            "random_forest_feature_importance.csv",
        )

        return {
            "validation": ModelResult(
                "Random Forest",
                validation_metrics,
                val_predictions,
                val_probabilities,
                preprocessor.selected_features_,
            ),
            "test": ModelResult(
                "Random Forest",
                test_metrics,
                test_predictions,
                test_probabilities,
                preprocessor.selected_features_,
            ),
            "threshold_metrics": threshold_metrics,
        }
        
    def _threshold_grid(self) -> list[float]:
        # A compact grid should be enough for project report threshold analysis.
        return [round(value, 2) for value in np.arange(0.05, 0.96, 0.05)]   

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

    def _build_confusion_matrix_table(
        self,
        results: list[ModelResult],
        split_name: str,
    ) -> pd.DataFrame:
        rows = []

        for result in results:
            matrix = result.metrics["confusion_matrix"]
            rows.append(
                {
                    "model_name": result.model_name,
                    "split": split_name,
                    "true_negative": matrix["true_negative"],
                    "false_positive": matrix["false_positive"],
                    "false_negative": matrix["false_negative"],
                    "true_positive": matrix["true_positive"],
                }
            )

        return pd.DataFrame(rows)

    def _build_logistic_regression_coefficients(
        self,
        features: list[str],
        coefficients: np.ndarray,
    ) -> pd.DataFrame:
        coefficient_table = pd.DataFrame(
            {
                "feature": features,
                "coefficient": coefficients,
            }
        )
        coefficient_table["abs_coefficient"] = coefficient_table["coefficient"].abs()

        return coefficient_table.sort_values(
            by="abs_coefficient",
            ascending=False,
        )

    def _build_random_forest_feature_importance(
        self,
        features: list[str],
        importances: np.ndarray,
    ) -> pd.DataFrame:
        importance_table = pd.DataFrame(
            {
                "feature": features,
                "importance": importances,
            }
        )

        return importance_table.sort_values(
            by="importance",
            ascending=False,
        )