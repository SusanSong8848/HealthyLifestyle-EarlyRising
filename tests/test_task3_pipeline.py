import os
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from task3 import (
    EXCLUDED_FEATURES,
    add_validation_metrics,
    build_logistic_tuning_pipelines,
    build_model_pipelines,
    build_weight_ablation,
    evaluate_predictions,
    extract_feature_importance,
    lgb,
    select_best_model,
    validate_required_models,
)


class Task3PipelineTests(unittest.TestCase):
    def test_dummy_and_every_model_use_pipeline(self):
        models = build_model_pipelines(["Age"], ["Gender"])
        self.assertIn("Dummy Most Frequent", models)
        self.assertIn("Logistic Regression (Unweighted)", models)
        for model in models.values():
            self.assertIsInstance(model, Pipeline)
            self.assertEqual(
                list(model.named_steps),
                ["preprocessor", "classifier"],
            )

    def test_logistic_tuning_grid_is_small_and_pipeline_based(self):
        models = build_logistic_tuning_pipelines(["Age"], ["Gender"])
        self.assertEqual(len(models), 5)
        c_values = [
            model.named_steps["classifier"].get_params()["C"]
            for model in models.values()
        ]
        self.assertEqual(c_values, [0.1, 0.3, 1.0, 3.0, 10.0])
        for model in models.values():
            self.assertIsInstance(model, Pipeline)

    def test_required_metrics_include_zero_poor_recall(self):
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        y_pred = np.array([1, 1, 1, 1, 2, 0, 3, 2])
        metrics = evaluate_predictions(y_true, y_pred)
        self.assertAlmostEqual(metrics["Accuracy"], 0.5)
        self.assertEqual(metrics["Recall_Poor"], 0.0)
        self.assertEqual(metrics["Recall_Average"], 1.0)
        self.assertEqual(metrics["Recall_Good"], 0.5)
        self.assertEqual(metrics["Recall_Excellent"], 0.5)
        self.assertIn("Macro_F1", metrics)
        self.assertIn("Balanced_Accuracy", metrics)

    def test_selection_uses_official_accuracy_before_macro_f1(self):
        comparison = pd.DataFrame(
            [
                {
                    "Model": "Accuracy Winner",
                    "CV_Accuracy_Mean": 0.95,
                    "CV_Macro_F1_Mean": 0.40,
                    "CV_Balanced_Accuracy_Mean": 0.45,
                },
                {
                    "Model": "Macro F1 Winner",
                    "CV_Accuracy_Mean": 0.80,
                    "CV_Macro_F1_Mean": 0.70,
                    "CV_Balanced_Accuracy_Mean": 0.72,
                },
            ]
        )
        self.assertEqual(select_best_model(comparison), "Accuracy Winner")

    def test_selection_uses_macro_f1_as_accuracy_tie_breaker(self):
        comparison = pd.DataFrame(
            [
                {
                    "Model": "Lower Macro F1",
                    "CV_Accuracy_Mean": 0.90,
                    "CV_Macro_F1_Mean": 0.55,
                    "CV_Balanced_Accuracy_Mean": 0.60,
                },
                {
                    "Model": "Higher Macro F1",
                    "CV_Accuracy_Mean": 0.90,
                    "CV_Macro_F1_Mean": 0.65,
                    "CV_Balanced_Accuracy_Mean": 0.62,
                },
            ]
        )
        self.assertEqual(select_best_model(comparison), "Higher Macro F1")

    def test_model_comparison_has_one_selected_row(self):
        comparison = pd.DataFrame(
            [
                {
                    "Model": "A",
                    "CV_Accuracy_Mean": 0.80,
                    "CV_Macro_F1_Mean": 0.70,
                    "CV_Balanced_Accuracy_Mean": 0.70,
                },
                {
                    "Model": "B",
                    "CV_Accuracy_Mean": 0.75,
                    "CV_Macro_F1_Mean": 0.60,
                    "CV_Balanced_Accuracy_Mean": 0.60,
                },
            ]
        )
        validation = {
            model: {
                "Accuracy": 0.50,
                "Macro_F1": 0.40,
                "Balanced_Accuracy": 0.40,
                "Recall_Poor": 0.10,
                "Recall_Average": 0.20,
                "Recall_Good": 0.50,
                "Recall_Excellent": 0.80,
            }
            for model in ["A", "B"]
        }
        result = add_validation_metrics(comparison, validation, "A")
        self.assertEqual(int(result["Selected"].sum()), 1)
        self.assertIn("Val_Recall_Poor", result.columns)
        self.assertIn("Val_Macro_F1", result.columns)

    def test_logistic_regression_produces_original_feature_importance(self):
        x = pd.DataFrame(
            {
                "Age": [20, 21, 30, 31, 40, 41, 50, 51, 22, 32, 42, 52],
                "Gender": [
                    "F",
                    "M",
                    "F",
                    "M",
                    "F",
                    "M",
                    "F",
                    "M",
                    "M",
                    "F",
                    "M",
                    "F",
                ],
            }
        )
        y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3])
        model = build_model_pipelines(["Age"], ["Gender"])[
            "Logistic Regression (Unweighted)"
        ]
        model.fit(x, y)
        importance = extract_feature_importance(model, x, y)
        self.assertFalse(importance.empty)
        self.assertEqual(set(importance["Feature"]), {"Age", "Gender"})
        self.assertEqual(
            importance["Method"].iloc[0],
            "permutation_accuracy_original_feature",
        )

    def test_blacklist_contains_all_known_task3_leakage(self):
        for column in [
            "Person_ID",
            "Wellness_Category",
            "Health_Score",
            "Fitness_Level",
        ]:
            self.assertIn(column, EXCLUDED_FEATURES)

    def test_weight_ablation_is_a_controlled_pair(self):
        common = {
            "CV_Accuracy_Mean": 0.80,
            "CV_Macro_F1_Mean": 0.60,
            "CV_Balanced_Accuracy_Mean": 0.62,
            "CV_Recall_Poor_Mean": 0.10,
            "Val_Accuracy": 0.81,
            "Val_Macro_F1": 0.61,
            "Val_Balanced_Accuracy": 0.63,
            "Val_Recall_Poor": 0.12,
        }
        comparison = pd.DataFrame(
            [
                {"Model": "LightGBM (Unweighted)", **common},
                {
                    "Model": "LightGBM (Balanced)",
                    **{
                        **common,
                        "CV_Macro_F1_Mean": 0.68,
                        "CV_Recall_Poor_Mean": 0.35,
                        "Val_Macro_F1": 0.67,
                        "Val_Recall_Poor": 0.30,
                    },
                },
            ]
        )

        ablation = build_weight_ablation(comparison)

        self.assertEqual(ablation["Weighting"].tolist(), ["None", "balanced"])
        self.assertEqual(
            ablation["class_weight"].tolist(),
            ["None", "balanced"],
        )
        for parameter in [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "num_leaves",
            "random_state",
        ]:
            self.assertEqual(ablation[parameter].nunique(), 1)
        self.assertAlmostEqual(
            ablation.loc[
                1,
                "Delta_CV_Recall_Poor_Mean_vs_Unweighted",
            ],
            0.25,
        )
        self.assertAlmostEqual(
            ablation.loc[
                1,
                "Delta_Val_Macro_F1_vs_Unweighted",
            ],
            0.06,
        )

    def test_required_ablation_models_cannot_be_silently_skipped(self):
        with self.assertRaises(ImportError):
            validate_required_models({})

    def test_real_lightgbm_pair_only_changes_class_weight(self):
        if lgb is None:
            self.skipTest("lightgbm is not installed in this environment")
        models = build_model_pipelines(["Age"], ["Gender"])
        unweighted = models["LightGBM (Unweighted)"].named_steps[
            "classifier"
        ].get_params()
        balanced = models["LightGBM (Balanced)"].named_steps[
            "classifier"
        ].get_params()
        self.assertIsNone(unweighted["class_weight"])
        self.assertEqual(balanced["class_weight"], "balanced")
        for parameter in sorted(set(unweighted).union(balanced)):
            if parameter != "class_weight":
                self.assertEqual(
                    unweighted.get(parameter),
                    balanced.get(parameter),
                    msg=f"Unexpected LightGBM parameter drift: {parameter}",
                )


if __name__ == "__main__":
    unittest.main()
