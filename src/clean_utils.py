"""
Shared data cleaning utilities (v3)
统一数据清洗 + 特征工程工具函数，消除各脚本中的代码重复。
"""
import pandas as pd
import numpy as np


def normalize_time_column(series: pd.Series) -> pd.Series:
    """统一时间格式为 HH:MM（零填充）"""
    def _norm(val):
        parts = str(val).strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        return f"{h:02d}:{m:02d}"
    return series.apply(_norm)


def time_to_minutes(series: pd.Series) -> pd.Series:
    """HH:MM 格式 → 午夜起分钟数"""
    parts = series.str.split(":", expand=True).astype(float)
    return parts[0] * 60 + parts[1]


def load_and_clean_data(data_path: str) -> pd.DataFrame:
    """
    加载原始数据并执行统一清洗（所有任务共享）。
    规则来自 公共数据要求.txt：
    - 时间统一为 HH:MM，新增 _Minutes 列
    - Exercise_Frequency_Per_Week=0 → Exercise_Type="No Exercise", Workout_Intensity="No Workout"
    - Alcohol_Consumption 缺失 → "Unknown"
    - 不删行、不删列、不编码、不标准化
    """
    df = pd.read_csv(data_path)

    # 时间规范化
    for col in ["Wake_Up_Time", "Sleep_Time"]:
        df[col] = normalize_time_column(df[col])
        df[col + "_Minutes"] = time_to_minutes(df[col])

    # 结构性缺失填充
    mask_no_ex = df["Exercise_Frequency_Per_Week"] == 0
    df.loc[mask_no_ex & df["Exercise_Type"].isna(), "Exercise_Type"] = "No Exercise"
    df.loc[mask_no_ex & df["Workout_Intensity"].isna(), "Workout_Intensity"] = "No Workout"
    df["Alcohol_Consumption"] = df["Alcohol_Consumption"].fillna("Unknown")

    assert df.isnull().sum().sum() == 0, "Missing values remain after cleaning!"
    return df


def get_available_features(df: pd.DataFrame, numeric_cols: list,
                           cat_cols: list, excluded: list) -> tuple:
    """过滤出可用特征列"""
    avail_num = [c for c in numeric_cols if c in df.columns and c not in excluded]
    avail_cat = [c for c in cat_cols if c in df.columns and c not in excluded]
    return avail_num, avail_cat


# ============ 各任务的数值特征列表 ============
# （排除泄漏字段后各任务可用，与 config.py 保持一致）
TASK_NUMERIC_COLS = [
    "Age", "Height_cm", "Weight_kg", "BMI",
    "Sleep_Duration_Hours", "Sleep_Quality_Score",
    "Number_of_Night_Awakenings", "Weekend_Sleep_Difference_Hours",
    "Nap_Frequency_Per_Week", "Screen_Time_Before_Bed_Hours",
    "Exercise_Frequency_Per_Week", "Exercise_Duration_Minutes",
    "Daily_Steps", "Daily_Calorie_Intake", "Water_Intake_Liters",
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

TASK_CATEGORICAL_COLS = [
    "Gender", "Country", "Occupation", "Marital_Status",
    "Exercise_Type", "Morning_Workout", "Workout_Intensity",
    "Gym_Member", "Smoking_Status", "Alcohol_Consumption",
    "Meditation_Practice", "Obesity_Risk", "Hypertension_Risk",
    "Diabetes_Risk", "Cardiovascular_Risk", "Sleep_Disorder_Risk",
    "Fitness_Level",
]