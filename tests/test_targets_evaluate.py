"""统一标签与评价接口测试（D27-06）。"""
import math
import unittest

import pandas as pd

from src.evaluate import (
    METRIC_FIELDS,
    evaluate_classification,
    make_metric_record,
    validate_no_leakage,
)
from src.targets import (
    decode_labels,
    encode_labels,
    generate_target,
    get_class_order,
)


class TestTargets(unittest.TestCase):
    def test_task1_target(self):
        df = pd.DataFrame({"Early_Waker": ["No", "Yes"]})
        self.assertEqual(generate_target(df, "task1").tolist(), ["No", "Yes"])

    def test_task2_boundaries(self):
        df = pd.DataFrame(
            {"Health_Score": [0, 60, 60.1, 70, 70.1, 85, 85.1, 100]}
        )
        self.assertEqual(
            generate_target(df, "task2").tolist(),
            [
                "Poor",
                "Poor",
                "Average",
                "Average",
                "Good",
                "Good",
                "Excellent",
                "Excellent",
            ],
        )

    def test_task2_out_of_range_raises(self):
        df = pd.DataFrame({"Health_Score": [-1, 50]})
        with self.assertRaises(ValueError):
            generate_target(df, "task2")

    def test_task3_class_order(self):
        self.assertEqual(
            get_class_order("task3"),
            ["Poor", "Average", "Good", "Excellent"],
        )

    def test_encode_decode_round_trip(self):
        labels = ["Poor", "Average", "Good", "Excellent"]
        encoded = encode_labels(labels, "task2")
        self.assertEqual(encoded.tolist(), [0, 1, 2, 3])
        self.assertEqual(decode_labels(encoded, "task2").tolist(), labels)


class TestEvaluate(unittest.TestCase):
    def test_perfect_task1_metrics(self):
        metrics = evaluate_classification(
            ["No", "Yes", "No", "Yes"],
            ["No", "Yes", "No", "Yes"],
            "task1",
            y_score=[0.1, 0.9, 0.2, 0.8],
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_task2_ordinal_metrics(self):
        metrics = evaluate_classification(
            ["Poor", "Average", "Good", "Excellent"],
            ["Average", "Average", "Excellent", "Good"],
            "task2",
        )
        self.assertAlmostEqual(metrics["mae"], 0.75)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(0.75))

    def test_leakage_detection(self):
        with self.assertRaises(ValueError):
            validate_no_leakage(["Age", "Health_Score"], "task2")

    def test_clean_feature_list(self):
        validate_no_leakage(["Age", "BMI", "Mood_Score"], "task2")

    def test_metric_record_schema(self):
        metrics = evaluate_classification(
            ["Poor", "Average", "Good", "Excellent"],
            ["Poor", "Average", "Good", "Excellent"],
            "task2",
        )
        record = make_metric_record(
            "task2",
            "Logistic Regression",
            "val",
            20260726,
            4,
            metrics,
            True,
        )
        self.assertEqual(list(record), METRIC_FIELDS)
        self.assertEqual(record["class_order"], "Poor|Average|Good|Excellent")


if __name__ == "__main__":
    unittest.main()
