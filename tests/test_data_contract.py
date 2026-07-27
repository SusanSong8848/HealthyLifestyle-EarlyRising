"""
数据契约测试 (D27-03)
验证 base_semantic_clean.csv 满足公共清洗规范的全部检查项。
"""
import os, sys, unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from config import ROOT_DIR, ID_COLUMN

CLEAN_PATH = os.path.join(ROOT_DIR, "data", "processed", "base_semantic_clean.csv")
RAW_PATH = os.path.join(ROOT_DIR, "data", "raw", "A题数据集.csv")


class TestDataContract(unittest.TestCase):
    """公共清洗数据契约测试"""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(CLEAN_PATH)
        cls.raw = pd.read_csv(RAW_PATH)

    def test_row_count(self):
        """行数应为 10000"""
        self.assertEqual(len(self.df), 10000)

    def test_column_count(self):
        """列数必须为 66（64 个原始字段 + 2 个分钟衍生字段）"""
        self.assertEqual(len(self.df.columns), 66)

    def test_only_expected_columns_added(self):
        """原始 64 列全部保留，且只新增两个分钟衍生字段"""
        original_columns = list(self.raw.columns)
        expected_added = {"Wake_Up_Time_Minutes", "Sleep_Time_Minutes"}
        added_columns = set(self.df.columns) - set(original_columns)
        removed_columns = set(original_columns) - set(self.df.columns)
        self.assertEqual(removed_columns, set())
        self.assertEqual(added_columns, expected_added)
        self.assertEqual(list(self.df.columns[:len(original_columns)]), original_columns)

    def test_no_missing_cells(self):
        """不应有任何缺失单元格"""
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_no_duplicate_rows(self):
        """不应有完全重复行"""
        self.assertEqual(self.df.duplicated().sum(), 0)

    def test_person_id_unique(self):
        """Person_ID 应无重复无缺失"""
        self.assertIn(ID_COLUMN, self.df.columns)
        self.assertEqual(self.df[ID_COLUMN].isna().sum(), 0)
        self.assertEqual(self.df[ID_COLUMN].duplicated().sum(), 0)

    def test_alcohol_unknown_count(self):
        """Alcohol_Consumption 的 Unknown 应为 3014"""
        if "Alcohol_Consumption" in self.df.columns:
            self.assertEqual((self.df["Alcohol_Consumption"] == "Unknown").sum(), 3014)

    def test_no_exercise_count(self):
        """Exercise_Type 的 No Exercise 应为 824"""
        if "Exercise_Type" in self.df.columns:
            self.assertEqual((self.df["Exercise_Type"] == "No Exercise").sum(), 824)

    def test_no_workout_count(self):
        """Workout_Intensity 的 No Workout 应为 824"""
        if "Workout_Intensity" in self.df.columns:
            self.assertEqual((self.df["Workout_Intensity"] == "No Workout").sum(), 824)

    def test_numeric_values_unchanged(self):
        """所有原始数值应逐值保持不变"""
        num_raw = self.raw.select_dtypes(include=[np.number])
        num_clean = self.df[num_raw.columns]
        self.assertTrue(num_raw.equals(num_clean),
                        "Numeric values were modified during cleaning!")

    def test_column_order_preserved(self):
        """原始64列字段顺序应不变"""
        original_order = list(self.raw.columns)
        clean_order = list(self.df.columns[:len(original_order)])
        self.assertEqual(clean_order, original_order)

    def test_time_format(self):
        """Wake_Up_Time 和 Sleep_Time 应为 HH:MM 格式"""
        for col in ["Wake_Up_Time", "Sleep_Time"]:
            if col in self.df.columns:
                self.assertTrue(
                    self.df[col].str.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d").all(),
                    f"{col} has invalid time format"
                )

    def test_time_minutes_columns_exist(self):
        """应有 _Minutes 衍生列"""
        self.assertIn("Wake_Up_Time_Minutes", self.df.columns)
        self.assertIn("Sleep_Time_Minutes", self.df.columns)

    def test_time_minutes_values(self):
        """分钟衍生列必须与规范化后的 HH:MM 文本逐行一致"""
        for col in ["Wake_Up_Time", "Sleep_Time"]:
            parts = self.df[col].str.split(":", expand=True).astype(int)
            expected = parts[0] * 60 + parts[1]
            actual = self.df[col + "_Minutes"].astype(int)
            self.assertTrue(
                expected.equals(actual),
                f"{col}_Minutes does not match {col}"
            )


if __name__ == "__main__":
    unittest.main()
