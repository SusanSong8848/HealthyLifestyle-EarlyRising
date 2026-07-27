"""
数据切分测试 (D27-07)
验证 split_manifest.csv 满足规范。
"""
import os, sys, unittest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from config import SPLITS_DIR

MANIFEST_PATH = os.path.join(SPLITS_DIR, "split_manifest.csv")


class TestSplit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(MANIFEST_PATH)

    def test_row_count(self):
        self.assertEqual(len(self.df), 10000)

    def test_columns(self):
        expected = ["Person_ID", "split", "task1_label", "task2_label", "task3_label"]
        for col in expected:
            self.assertIn(col, self.df.columns)

    def test_person_id_unique(self):
        self.assertEqual(self.df["Person_ID"].nunique(), 10000)
        self.assertEqual(self.df["Person_ID"].duplicated().sum(), 0)

    def test_split_values(self):
        self.assertEqual(set(self.df["split"].unique()), {"train", "val"})

    def test_split_counts(self):
        self.assertEqual(len(self.df[self.df["split"] == "train"]), 8000)
        self.assertEqual(len(self.df[self.df["split"] == "val"]), 2000)

    def test_no_overlap(self):
        train_ids = set(self.df[self.df["split"] == "train"]["Person_ID"])
        val_ids = set(self.df[self.df["split"] == "val"]["Person_ID"])
        self.assertEqual(len(train_ids & val_ids), 0,
                         "Train and val splits should have no overlap!")

    def test_task1_labels_valid(self):
        self.assertTrue(self.df["task1_label"].isin(["Yes", "No"]).all())

    def test_task2_labels_valid(self):
        self.assertTrue(self.df["task2_label"].isin(
            ["Poor", "Average", "Good", "Excellent"]).all())

    def test_task3_labels_valid(self):
        self.assertTrue(self.df["task3_label"].isin(
            ["Average", "Excellent", "Good", "Poor"]).all())


if __name__ == "__main__":
    unittest.main()