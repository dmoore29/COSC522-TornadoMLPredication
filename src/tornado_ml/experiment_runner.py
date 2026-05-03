from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tornado_ml.artifact_manager import ArtifactManager
from tornado_ml.config import ProjectConfig
from tornado_ml.data_types import DataSplit, ModelResult
from tornado_ml.dataset_builder import GsodDatasetBuilder
from tornado_ml.evaluation import MetricsEvaluator
from tornado_ml.models import LogisticRegressionModel, RandomForestModel
from tornado_ml.preprocessing import FeatureEngineer, FeaturePreprocessor
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

        val_comparison = self._build_comparison_table(results, split="val")
        test_comparison = self._build_comparison_table(results, split="test")
        val_cm_table = self._build_confusion_matrix_table(results, split="val")
        test_cm_table = self._build_confusion_matrix_table(results, split="test")
        threshold_table = self._build_threshold_table(results, split)

        logger.info("Saving comparison and summary artifacts")
        self.artifacts.save_dataframe(val_comparison, "validation_model_comparison.csv")
        self.artifacts.save_dataframe(test_comparison, "test_model_comparison.csv")
        self.artifacts.save_dataframe(val_cm_table, "validation_confusion_matrices.csv")
        self.artifacts.save_dataframe(test_cm_table, "test_confusion_matrices.csv")
        self.artifacts.save_dataframe(threshold_table, "threshold_metrics.csv")
        self.artifacts.save_dataframe(split_summary, "split_summary.csv")
        self.artifacts.save_dataframe(missingness, "missingness_summary.csv")
        self.artifacts.save_json(dataset_summary, "dataset_summary.json")
        self.artifacts.save_json(
            {result.model_name: result.metrics for result in results},
            "metrics.json",
        )
        return {
            "dataset_summary": dataset_summary,
            "metrics": {result.model_name: result.metrics for result in results},
            "val_metrics": {result.model_name: result.val_metrics for result in results},
        }

    def _run_logistic_regression(self, split: DataSplit) -> ModelResult:
        logger.info("Engineering interaction features for Logistic Regression")
        engineer = FeatureEngineer()
        X_train_eng = engineer.fit_transform(split.X_train)
        X_val_eng = engineer.transform(split.X_val)
        X_test_eng = engineer.transform(split.X_test)

        logger.info("Preparing Logistic Regression features")
        preprocessor = FeaturePreprocessor(self.config, scale=True)
        X_train = preprocessor.fit_transform(X_train_eng)
        X_val = preprocessor.transform(X_val_eng)
        X_test = preprocessor.transform(X_test_eng)

        logger.info("Training Logistic Regression")
        model = LogisticRegressionModel(self.config)
        model.train(X_train, split.y_train)

        logger.info("Finding optimal decision threshold on validation set")
        val_prob = model.predict_proba(X_val)
        optimal_threshold = self.evaluator.find_optimal_threshold(split.y_val, val_prob)
        logger.info("Optimal threshold (max F1 on val): %.4f", optimal_threshold)

        val_pred = (val_prob >= optimal_threshold).astype(int)
        val_metrics = self.evaluator.evaluate("Logistic Regression", split.y_val, val_pred, val_prob)
        val_metrics["optimal_threshold"] = optimal_threshold

        logger.info("Evaluating Logistic Regression on test set")
        test_prob = model.predict_proba(X_test)
        test_pred = (test_prob >= optimal_threshold).astype(int)
        test_metrics = self.evaluator.evaluate("Logistic Regression", split.y_test, test_pred, test_prob)
        test_metrics["optimal_threshold"] = optimal_threshold

        logger.info("Exporting LR coefficients")
        coef_df = pd.DataFrame({
            "feature": preprocessor.selected_features_,
            "coefficient": model.estimator.coef_[0],
        })
        coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
        self.artifacts.save_dataframe(coef_df, "logistic_regression_coefficients.csv")

        logger.info("Saving Logistic Regression model")
        self.artifacts.save_model(model.estimator, "logistic_regression.joblib")

        return ModelResult(
            model_name="Logistic Regression",
            metrics=test_metrics,
            predictions=test_pred,
            probabilities=test_prob,
            selected_features=preprocessor.selected_features_,
            val_metrics=val_metrics,
            val_predictions=val_pred,
            val_probabilities=val_prob,
        )

    def _run_random_forest(self, split: DataSplit) -> ModelResult:
        logger.info("Preparing Random Forest features")
        preprocessor = FeaturePreprocessor(self.config, scale=False)
        X_train = preprocessor.fit_transform(split.X_train)
        X_val = preprocessor.transform(split.X_val)
        X_test = preprocessor.transform(split.X_test)

        logger.info("Training Random Forest")
        model = RandomForestModel(self.config)
        model.train(X_train, split.y_train)

        logger.info("Evaluating Random Forest on validation set")
        val_pred = model.predict(X_val)
        val_prob = model.predict_proba(X_val)
        val_metrics = self.evaluator.evaluate("Random Forest", split.y_val, val_pred, val_prob)

        logger.info("Evaluating Random Forest on test set")
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)
        test_metrics = self.evaluator.evaluate("Random Forest", split.y_test, test_pred, test_prob)

        logger.info("Exporting Random Forest feature importances")
        importance_df = pd.DataFrame({
            "feature": preprocessor.selected_features_,
            "importance": model.estimator.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        self.artifacts.save_dataframe(importance_df, "random_forest_feature_importance.csv")

        logger.info("Saving Random Forest model")
        self.artifacts.save_model(model.estimator, "random_forest.joblib")

        return ModelResult(
            model_name="Random Forest",
            metrics=test_metrics,
            predictions=test_pred,
            probabilities=test_prob,
            selected_features=preprocessor.selected_features_,
            val_metrics=val_metrics,
            val_predictions=val_pred,
            val_probabilities=val_prob,
        )

    def _build_comparison_table(self, results: list[ModelResult], split: str = "test") -> pd.DataFrame:
        rows = []
        for result in results:
            source = result.metrics if split == "test" else result.val_metrics
            if source is None:
                continue
            row = {k: v for k, v in source.items() if k != "confusion_matrix"}
            row["selected_feature_count"] = len(result.selected_features)
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_confusion_matrix_table(self, results: list[ModelResult], split: str) -> pd.DataFrame:
        rows = []
        for result in results:
            source = result.metrics if split == "test" else result.val_metrics
            if source is None:
                continue
            cm = source["confusion_matrix"]
            rows.append({
                "model_name": result.model_name,
                "split": split,
                "true_negative": cm["true_negative"],
                "false_positive": cm["false_positive"],
                "false_negative": cm["false_negative"],
                "true_positive": cm["true_positive"],
            })
        return pd.DataFrame(rows)

    def _build_threshold_table(
        self,
        results: list[ModelResult],
        split: DataSplit,
        thresholds: list[float] | None = None,
    ) -> pd.DataFrame:
        import numpy as np
        if thresholds is None:
            thresholds = [round(t, 2) for t in np.arange(0.05, 0.55, 0.05).tolist()]
        rows = []
        for result in results:
            if result.val_probabilities is None:
                continue
            for threshold in thresholds:
                val_pred_thresh = (result.val_probabilities >= threshold).astype(int)
                metrics = self.evaluator.evaluate(result.model_name, split.y_val, val_pred_thresh)
                cm = metrics["confusion_matrix"]
                rows.append({
                    "model_name": result.model_name,
                    "threshold": threshold,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "true_negative": cm["true_negative"],
                    "false_positive": cm["false_positive"],
                    "false_negative": cm["false_negative"],
                    "true_positive": cm["true_positive"],
                })
        return pd.DataFrame(rows)
