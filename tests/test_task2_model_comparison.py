"""Tests for D28-02 model comparison and D28-04 aggregation gates."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.aggregate_model_results import (
    COMMON_REQUIRED_FIELDS,
    MissingDependencyError,
    build_aggregate,
)
from src.evaluate import evaluate_classification, validate_no_leakage
from src.task2_model_comparison import (
    APPROACH_DIRECT,
    APPROACH_REGRESSION,
    COMPARISON_FIELDS,
    METHOD_ABLATION_PATH,
    MODEL_COMPARISON_PATH,
    ROOT_DIR,
    TUNING_LOG_PATH,
    load_comparison_data,
    make_comparison_record,
    model_specs,
    prepare_cv_folds,
    score_predictions_to_labels,
    select_best_record,
)


class Task2ComparisonUnitTests(unittest.TestCase):
    def test_regression_predictions_are_clipped_and_binned(self):
        labels = score_predictions_to_labels(
            [-10, 0, 60, 60.1, 70, 70.1, 85, 85.1, 100, 110]
        )
        self.assertEqual(
            labels.tolist(),
            [
                "Poor",
                "Poor",
                "Poor",
                "Average",
                "Average",
                "Good",
                "Good",
                "Excellent",
                "Excellent",
                "Excellent",
            ],
        )

    def test_six_model_families_and_sixteen_configurations(self):
        specs = model_specs()
        self.assertEqual(len(specs), 6)
        self.assertEqual(sum(len(spec["params"]) for spec in specs), 16)
        self.assertEqual(
            {spec["approach"] for spec in specs},
            {APPROACH_DIRECT, APPROACH_REGRESSION},
        )
        self.assertEqual(
            {
                approach: sum(
                    spec["approach"] == approach for spec in specs
                )
                for approach in [APPROACH_DIRECT, APPROACH_REGRESSION]
            },
            {APPROACH_DIRECT: 3, APPROACH_REGRESSION: 3},
        )

    def test_model_selection_tie_break_order(self):
        base = {
            "accuracy": 0.8,
            "macro_f1": 0.7,
            "mae": 0.3,
            "model": "B",
            "params_json": "{}",
        }
        better_f1 = dict(base, macro_f1=0.71, model="C")
        better_mae = dict(base, mae=0.2, model="D")
        better_accuracy = dict(base, accuracy=0.81, model="E")
        self.assertIs(
            select_best_record([base, better_f1, better_mae, better_accuracy]),
            better_accuracy,
        )
        self.assertIs(select_best_record([base, better_f1]), better_f1)
        self.assertIs(select_best_record([base, better_mae]), better_mae)

    def test_comparison_record_schema(self):
        metrics = {
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "balanced_accuracy": 1.0,
            "roc_auc": None,
            "mae": 0.0,
            "rmse": 0.0,
            "recall_poor": None,
            "recall_average": None,
            "recall_good": None,
            "recall_excellent": None,
        }
        record = make_comparison_record(
            context={"data_version": "data", "split_version": "split"},
            approach=APPROACH_DIRECT,
            model_family="Logistic Regression",
            model="Logistic Regression",
            params={"C": 1.0},
            split="cv_train_mean",
            n_samples=8000,
            metrics=metrics,
            runtime_sec=1.25,
        )
        self.assertEqual(list(record), COMPARISON_FIELDS)
        self.assertEqual(json.loads(record["params_json"]), {"C": 1.0})

    def test_preprocessing_is_fit_inside_each_training_fold(self):
        rows = 20
        X = pd.DataFrame(
            {
                "numeric": np.arange(rows, dtype=float),
                "category": ["A", "B"] * (rows // 2),
            }
        )
        labels = pd.Series(
            ["Poor", "Average", "Good", "Excellent"] * 5,
            dtype="string",
        )
        scores = pd.Series(np.linspace(40, 95, rows))
        folds = prepare_cv_folds(
            X,
            labels,
            scores,
            ["numeric"],
            ["category"],
        )
        self.assertEqual(len(folds), 5)
        for fold in folds:
            numeric_column = fold["X_fit"][:, 0]
            if hasattr(numeric_column, "toarray"):
                numeric_column = numeric_column.toarray()
            numeric_values = np.asarray(numeric_column).ravel()
            self.assertAlmostEqual(float(numeric_values.mean()), 0.0, places=12)


class Task2ComparisonDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = load_comparison_data()

    def test_public_split_and_person_ids(self):
        split = self.context["split"]
        merged = self.context["merged"]
        self.assertEqual(int(split.eq("train").sum()), 8000)
        self.assertEqual(int(split.eq("val").sum()), 2000)
        self.assertEqual(merged["Person_ID"].nunique(), 10000)
        train_ids = set(merged.loc[split.eq("train"), "Person_ID"])
        val_ids = set(merged.loc[split.eq("val"), "Person_ID"])
        self.assertFalse(train_ids & val_ids)

    def test_task2_features_are_leak_free(self):
        validate_no_leakage(self.context["feature_columns"], "task2")
        forbidden_examples = {
            "Health_Score",
            "Wellness_Category",
            "Fitness_Level",
            "Person_ID",
        }
        self.assertFalse(
            forbidden_examples & set(self.context["feature_columns"])
        )


class Task2ComparisonArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = [
            MODEL_COMPARISON_PATH,
            METHOD_ABLATION_PATH,
            TUNING_LOG_PATH,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Run src/task2_model_comparison.py first: " + ", ".join(missing)
            )
        cls.comparison = pd.read_csv(MODEL_COMPARISON_PATH)
        cls.ablation = pd.read_csv(METHOD_ABLATION_PATH)
        cls.tuning = pd.read_csv(TUNING_LOG_PATH)
        cls.manifest = pd.read_csv(
            ROOT_DIR / "data" / "splits" / "split_manifest.csv",
            dtype={"Person_ID": "string"},
        )

    def test_formal_output_row_counts_and_primary(self):
        self.assertEqual(len(self.tuning), 16)
        self.assertEqual(len(self.comparison), 8)
        self.assertEqual(len(self.ablation), 2)
        self.assertEqual(
            set(self.comparison["model_family"]),
            {
                "Dummy Classifier",
                "Logistic Regression",
                "LightGBM Classifier",
                "Dummy Regressor",
                "Ridge",
                "LightGBM Regressor",
            },
        )
        primary = self.comparison.loc[
            self.comparison["is_primary"].astype(str).str.lower().eq("true")
        ]
        self.assertEqual(len(primary), 1)
        self.assertEqual(primary.iloc[0]["split"], "val")

    def test_route_predictions_match_public_validation_ids(self):
        expected_ids = set(
            self.manifest.loc[
                self.manifest["split"].eq("val"), "Person_ID"
            ].astype("string")
        )
        for _, row in self.ablation.iterrows():
            predictions = pd.read_csv(
                ROOT_DIR / row["output_path"],
                dtype={"Person_ID": "string"},
            )
            self.assertEqual(len(predictions), 2000)
            self.assertEqual(predictions["Person_ID"].nunique(), 2000)
            self.assertEqual(set(predictions["Person_ID"]), expected_ids)
            self.assertEqual(
                set(predictions["Predicted_Label"]),
                {"Poor", "Average", "Good", "Excellent"},
            )

    def test_independent_metric_recalculation(self):
        for _, row in self.ablation.iterrows():
            predictions = pd.read_csv(ROOT_DIR / row["output_path"])
            metrics = evaluate_classification(
                predictions["True_Label"],
                predictions["Predicted_Label"],
                "task2",
            )
            for field in [
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "mae",
                "rmse",
            ]:
                self.assertAlmostEqual(
                    float(row[field]),
                    float(metrics[field]),
                    places=12,
                )

    def test_regression_predictions_are_within_score_range(self):
        row = self.ablation.loc[
            self.ablation["approach"].eq(APPROACH_REGRESSION)
        ].iloc[0]
        predictions = pd.read_csv(ROOT_DIR / row["output_path"])
        self.assertTrue(predictions["Predicted_Score"].notna().all())
        self.assertGreaterEqual(float(predictions["Predicted_Score"].min()), 0.0)
        self.assertLessEqual(float(predictions["Predicted_Score"].max()), 100.0)


class AggregationGateTests(unittest.TestCase):
    def _row(self, task_id, seed=20260726):
        row = {field: "" for field in COMMON_REQUIRED_FIELDS}
        row.update(
            {
                "task_id": task_id,
                "data_version": "same-data",
                "split_version": "same-split",
                "model": f"{task_id}-model",
                "params_json": "{}",
                "split": "val",
                "seed": seed,
                "n_samples": 2000,
                "accuracy": 0.8,
                "macro_f1": 0.7,
                "balanced_accuracy": 0.7,
                "runtime_sec": 1.0,
                "output_path": f"outputs/{task_id}.csv",
            }
        )
        return row

    def test_missing_dependency_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "aggregate.csv"
            sources = {
                "task1": root / "task1.csv",
                "task2": root / "task2.csv",
                "task3": root / "task3.csv",
            }
            pd.DataFrame([self._row("task1")]).to_csv(
                sources["task1"], index=False
            )
            with self.assertRaises(MissingDependencyError):
                build_aggregate(sources, output)
            self.assertFalse(output.exists())

    def test_seed_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "aggregate.csv"
            sources = {}
            for task_id in ["task1", "task2", "task3"]:
                path = root / f"{task_id}.csv"
                seed = 1 if task_id == "task3" else 20260726
                pd.DataFrame([self._row(task_id, seed=seed)]).to_csv(
                    path, index=False
                )
                sources[task_id] = path
            with self.assertRaisesRegex(ValueError, "Seed mismatch"):
                build_aggregate(sources, output)
            self.assertFalse(output.exists())

    def test_complete_three_task_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "aggregate.csv"
            sources = {}
            for task_id in ["task1", "task2", "task3"]:
                path = root / f"{task_id}.csv"
                pd.DataFrame([self._row(task_id)]).to_csv(path, index=False)
                sources[task_id] = path
            aggregate = build_aggregate(sources, output)
            self.assertTrue(output.exists())
            self.assertEqual(set(aggregate["task_id"]), set(sources))


if __name__ == "__main__":
    unittest.main()
