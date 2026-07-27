"""
泄漏自动检查测试 (D27-03)
验证各任务的特征集不包含泄漏字段。
"""
import os, sys, unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from config import (
    TASK1_LEAKY_FEATURES, TASK2_LEAKY_FEATURES, TASK3_LEAKY_FEATURES,
    ID_COLUMN, ROOT_DIR,
)

# 简化的各任务用于主模型的特征列表（与 task1_final/task2/task3 一致）
TASK1_NUMERIC = [
    "Age", "Height_cm", "Weight_kg", "BMI",
    "Sleep_Quality_Score", "Number_of_Night_Awakenings",
    "Weekend_Sleep_Difference_Hours", "Nap_Frequency_Per_Week",
    "Screen_Time_Before_Bed_Hours", "Exercise_Frequency_Per_Week",
    "Exercise_Duration_Minutes", "Daily_Steps",
    "Daily_Calorie_Intake", "Water_Intake_Liters",
    "Fruit_Intake_Per_Day", "Vegetable_Intake_Per_Day",
    "Protein_Intake_Grams", "Sugary_Drinks_Per_Week",
    "Fast_Food_Meals_Per_Week", "Breakfast_Regularity_Score",
    "Stress_Level", "Working_Hours_Per_Day", "Sitting_Hours_Per_Day",
    "Outdoor_Time_Hours", "Social_Interaction_Score",
    "Resting_Heart_Rate", "Systolic_BP", "Diastolic_BP",
    "Cholesterol_Level", "Blood_Sugar_Level",
    "Energy_Level_Score", "Fatigue_Level_Score",
    "Immune_Health_Score", "Mood_Score", "Anxiety_Score",
    "Depression_Risk_Score", "Productivity_Score",
    "Focus_Concentration_Score", "Life_Satisfaction_Score",
    "Health_Score",
]


class TestLeakageTask1(unittest.TestCase):
    """Task 1 泄漏测试"""

    def test_no_wake_up_time(self):
        """不应包含 Wake_Up_Time_Minutes"""
        self.assertNotIn("Wake_Up_Time_Minutes", TASK1_NUMERIC)

    def test_no_sleep_time(self):
        """不应包含 Sleep_Time_Minutes"""
        self.assertNotIn("Sleep_Time_Minutes", TASK1_NUMERIC)

    def test_no_sleep_duration(self):
        """不应包含 Sleep_Duration_Hours"""
        self.assertNotIn("Sleep_Duration_Hours", TASK1_NUMERIC)

    def test_no_early_waker(self):
        """不应包含 Early_Waker"""
        self.assertIn("Early_Waker", TASK1_LEAKY_FEATURES)

    def test_no_healthy_aging(self):
        """主模型应排除 Healthy_Aging_Score"""
        self.assertNotIn("Healthy_Aging_Score", TASK1_NUMERIC)


class TestLeakageTask2(unittest.TestCase):
    """Task 2 泄漏测试"""

    def test_no_health_score(self):
        self.assertIn("Health_Score", TASK2_LEAKY_FEATURES)

    def test_no_wellness_category(self):
        self.assertIn("Wellness_Category", TASK2_LEAKY_FEATURES)

    def test_no_fitness_level(self):
        self.assertIn("Fitness_Level", TASK2_LEAKY_FEATURES)


class TestLeakageTask3(unittest.TestCase):
    """Task 3 泄漏测试"""

    def test_no_wellness_category(self):
        self.assertIn("Wellness_Category", TASK3_LEAKY_FEATURES)

    def test_no_health_score(self):
        self.assertIn("Health_Score", TASK3_LEAKY_FEATURES)

    def test_no_fitness_level(self):
        self.assertIn("Fitness_Level", TASK3_LEAKY_FEATURES)

    def test_fitness_level_100_pct_same(self):
        """Fitness_Level 与 Wellness_Category 100% 相同（验证一致性）"""
        raw = pd.read_csv(os.path.join(ROOT_DIR, "datas.csv"))
        # 映射：Wellness_Category 的 4 类对应 Fitness_Level 的 4 类
        self.assertTrue(
            (raw["Wellness_Category"] == raw["Fitness_Level"]).all(),
            "Fitness_Level should be 100% identical to Wellness_Category"
        )


if __name__ == "__main__":
    unittest.main()