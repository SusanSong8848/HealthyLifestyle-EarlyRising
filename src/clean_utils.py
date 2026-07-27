"""
Shared data cleaning utilities (v3)
统一数据清洗 + 特征工程工具函数，消除各脚本中的代码重复。
"""
import os

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
    df = pd.read_csv(data_path, dtype={"Person_ID": "string"})

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


def load_manifest_split_indices(
    df: pd.DataFrame,
    manifest_path: str,
    id_column: str = "Person_ID",
    label_checks: dict[str, pd.Series] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align the shared split manifest to ``df`` by Person_ID.

    Returns row-position arrays for the ``train`` and ``val`` groups. The
    strict checks stop a task instead of silently creating its own split when
    the manifest is missing or belongs to another data version.
    """
    if id_column not in df.columns:
        raise KeyError(f"Data is missing required ID column: {id_column}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Shared split manifest not found: {manifest_path}. "
            "Run src/preprocess.py (or src/split.py) first."
        )

    manifest = pd.read_csv(manifest_path, dtype={id_column: str})
    required = {id_column, "split"}
    missing_columns = required.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Split manifest is missing columns: {sorted(missing_columns)}"
        )

    data_ids = df[id_column].astype(str).str.strip()
    manifest_ids = manifest[id_column].astype(str).str.strip()

    if data_ids.duplicated().any():
        examples = data_ids[data_ids.duplicated()].head(5).tolist()
        raise ValueError(f"Duplicate {id_column} values in data: {examples}")
    if manifest_ids.duplicated().any():
        examples = manifest_ids[manifest_ids.duplicated()].head(5).tolist()
        raise ValueError(
            f"Duplicate {id_column} values in split manifest: {examples}"
        )

    data_id_set = set(data_ids)
    manifest_id_set = set(manifest_ids)
    if data_id_set != manifest_id_set:
        missing_from_manifest = sorted(data_id_set - manifest_id_set)[:5]
        extra_in_manifest = sorted(manifest_id_set - data_id_set)[:5]
        raise ValueError(
            "Data and split manifest contain different Person_ID sets. "
            f"Missing from manifest: {missing_from_manifest}; "
            f"extra in manifest: {extra_in_manifest}"
        )

    split_map = pd.Series(
        manifest["split"].astype(str).str.strip().values,
        index=manifest_ids,
    )
    aligned_split = data_ids.map(split_map)
    allowed_splits = {"train", "val"}
    actual_splits = set(aligned_split.dropna().unique())
    if actual_splits != allowed_splits:
        raise ValueError(
            "Split manifest must contain exactly 'train' and 'val'; "
            f"found {sorted(actual_splits)}"
        )

    for manifest_column, expected_values in (label_checks or {}).items():
        if manifest_column not in manifest.columns:
            raise ValueError(
                f"Split manifest is missing label column: {manifest_column}"
            )
        expected = pd.Series(expected_values).reset_index(drop=True)
        if len(expected) != len(df):
            raise ValueError(
                f"Label check {manifest_column} has {len(expected)} rows; "
                f"expected {len(df)}"
            )
        expected = expected.astype(str).str.strip()
        manifest_label_map = pd.Series(
            manifest[manifest_column].astype(str).str.strip().values,
            index=manifest_ids,
        )
        observed = data_ids.map(manifest_label_map).reset_index(drop=True)
        mismatch = expected.ne(observed)
        if mismatch.any():
            example_positions = np.flatnonzero(mismatch.to_numpy())[:5].tolist()
            raise ValueError(
                f"Split manifest label {manifest_column} does not match "
                f"the current data/target definition at row positions "
                f"{example_positions}"
            )

    train_idx = np.flatnonzero(aligned_split.eq("train").to_numpy())
    val_idx = np.flatnonzero(aligned_split.eq("val").to_numpy())
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Shared split contains an empty train or val group")

    return train_idx, val_idx


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
