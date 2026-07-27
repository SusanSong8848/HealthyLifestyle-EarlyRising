"""
数据审计脚本（D27-02 / D27-05）

输出：
- docs/data_inventory.csv：原始、历史和当前清洗数据资产清单
- results/data_audit/cleaning_validation.csv：公共清洗数据契约逐项验证表
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PATH, ROOT_DIR


EXPECTED_ROWS = 10000
EXPECTED_RAW_COLUMNS = 64
EXPECTED_CLEAN_COLUMNS = 66
EXPECTED_ADDED_COLUMNS = {
    "Wake_Up_Time_Minutes",
    "Sleep_Time_Minutes",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_file(filepath: str, label: str) -> dict:
    df = pd.read_csv(filepath)
    return {
        "label": label,
        "path": os.path.relpath(filepath, ROOT_DIR),
        "sha256": sha256(filepath),
        "rows": len(df),
        "cols": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "person_id_missing": (
            int(df["Person_ID"].isna().sum()) if "Person_ID" in df.columns else -1
        ),
        "person_id_duplicates": (
            int(df["Person_ID"].duplicated().sum())
            if "Person_ID" in df.columns
            else -1
        ),
        "columns": "|".join(df.columns),
    }


def _validation_row(
    check_id: str,
    check_name: str,
    expected,
    actual,
    passed: bool,
    details: str = "",
) -> dict:
    return {
        "check_id": check_id,
        "check_name": check_name,
        "expected": expected,
        "actual": actual,
        "status": "PASS" if passed else "FAIL",
        "details": details,
    }


def validate_cleaning(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check_id, check_name, expected, actual, passed, details=""):
        rows.append(
            _validation_row(
                check_id, check_name, expected, actual, bool(passed), details
            )
        )

    raw_columns = list(raw.columns)
    clean_columns = list(clean.columns)
    added_columns = set(clean_columns) - set(raw_columns)
    removed_columns = set(raw_columns) - set(clean_columns)

    add("C01", "原始数据行数", EXPECTED_ROWS, len(raw), len(raw) == EXPECTED_ROWS)
    add(
        "C02",
        "原始数据列数",
        EXPECTED_RAW_COLUMNS,
        len(raw_columns),
        len(raw_columns) == EXPECTED_RAW_COLUMNS,
    )
    add("C03", "公共清洗数据行数", EXPECTED_ROWS, len(clean), len(clean) == EXPECTED_ROWS)
    add(
        "C04",
        "公共清洗数据列数",
        EXPECTED_CLEAN_COLUMNS,
        len(clean_columns),
        len(clean_columns) == EXPECTED_CLEAN_COLUMNS,
        "64 个原始字段 + 2 个分钟衍生字段",
    )
    add(
        "C05",
        "原始字段无删除",
        "无",
        "|".join(sorted(removed_columns)) or "无",
        not removed_columns,
    )
    add(
        "C06",
        "新增字段集合",
        "|".join(sorted(EXPECTED_ADDED_COLUMNS)),
        "|".join(sorted(added_columns)),
        added_columns == EXPECTED_ADDED_COLUMNS,
    )
    add(
        "C07",
        "原始字段顺序",
        "与原始数据一致",
        "一致" if clean_columns[: len(raw_columns)] == raw_columns else "不一致",
        clean_columns[: len(raw_columns)] == raw_columns,
    )
    add(
        "C08",
        "缺失单元格",
        0,
        int(clean.isna().sum().sum()),
        clean.isna().sum().sum() == 0,
    )
    add(
        "C09",
        "完全重复行",
        0,
        int(clean.duplicated().sum()),
        clean.duplicated().sum() == 0,
    )
    add(
        "C10",
        "Person_ID 缺失",
        0,
        int(clean["Person_ID"].isna().sum()),
        clean["Person_ID"].isna().sum() == 0,
    )
    add(
        "C11",
        "Person_ID 重复",
        0,
        int(clean["Person_ID"].duplicated().sum()),
        clean["Person_ID"].duplicated().sum() == 0,
    )
    add(
        "C12",
        "Alcohol_Consumption=Unknown",
        3014,
        int((clean["Alcohol_Consumption"] == "Unknown").sum()),
        (clean["Alcohol_Consumption"] == "Unknown").sum() == 3014,
    )
    add(
        "C13",
        "Exercise_Type=No Exercise",
        824,
        int((clean["Exercise_Type"] == "No Exercise").sum()),
        (clean["Exercise_Type"] == "No Exercise").sum() == 824,
    )
    add(
        "C14",
        "Workout_Intensity=No Workout",
        824,
        int((clean["Workout_Intensity"] == "No Workout").sum()),
        (clean["Workout_Intensity"] == "No Workout").sum() == 824,
    )

    no_exercise_valid = (
        clean.loc[
            clean["Exercise_Type"] == "No Exercise",
            "Exercise_Frequency_Per_Week",
        ]
        == 0
    ).all()
    add(
        "C15",
        "No Exercise 结构语义",
        "全部对应 Exercise_Frequency_Per_Week=0",
        "符合" if no_exercise_valid else "不符合",
        no_exercise_valid,
    )

    no_workout_valid = (
        clean.loc[
            clean["Workout_Intensity"] == "No Workout",
            "Exercise_Frequency_Per_Week",
        ]
        == 0
    ).all()
    add(
        "C16",
        "No Workout 结构语义",
        "全部对应 Exercise_Frequency_Per_Week=0",
        "符合" if no_workout_valid else "不符合",
        no_workout_valid,
    )

    numeric_unchanged = raw.select_dtypes(include=[np.number]).equals(
        clean[raw.select_dtypes(include=[np.number]).columns]
    )
    add(
        "C17",
        "原始数值逐值保持",
        "完全一致",
        "一致" if numeric_unchanged else "不一致",
        numeric_unchanged,
    )

    exact_columns = [
        c
        for c in raw_columns
        if c
        not in {
            "Wake_Up_Time",
            "Sleep_Time",
            "Exercise_Type",
            "Workout_Intensity",
            "Alcohol_Consumption",
        }
    ]
    non_transformed_unchanged = all(raw[c].equals(clean[c]) for c in exact_columns)
    add(
        "C18",
        "非转换原始字段逐值保持",
        "完全一致",
        "一致" if non_transformed_unchanged else "不一致",
        non_transformed_unchanged,
    )

    time_format_valid = all(
        clean[col].str.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d").all()
        for col in ["Wake_Up_Time", "Sleep_Time"]
    )
    add(
        "C19",
        "时间格式",
        "HH:MM",
        "全部有效" if time_format_valid else "存在无效值",
        time_format_valid,
    )

    minutes_valid = True
    for col in ["Wake_Up_Time", "Sleep_Time"]:
        parts = clean[col].str.split(":", expand=True).astype(int)
        expected_minutes = parts[0] * 60 + parts[1]
        actual_minutes = clean[col + "_Minutes"].astype(int)
        minutes_valid = minutes_valid and expected_minutes.equals(actual_minutes)
    add(
        "C20",
        "分钟衍生字段",
        "与 HH:MM 逐行一致",
        "一致" if minutes_valid else "不一致",
        minutes_valid,
    )

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 60)
    print("Data Audit Report")
    print("=" * 60)

    inventory = []
    raw = pd.read_csv(DATA_PATH)
    inventory.append(audit_file(DATA_PATH, "官方原始数据"))

    legacy_dir = os.path.join(ROOT_DIR, "data", "legacy")
    if os.path.exists(legacy_dir):
        for filename in sorted(os.listdir(legacy_dir)):
            if filename.lower().endswith(".csv"):
                filepath = os.path.join(legacy_dir, filename)
                inventory.append(audit_file(filepath, f"历史清洗: {filename}"))

    clean_path = os.path.join(
        ROOT_DIR, "data", "processed", "base_semantic_clean.csv"
    )
    if not os.path.exists(clean_path):
        raise FileNotFoundError(
            "缺少 data/processed/base_semantic_clean.csv，请先运行 src/preprocess.py"
        )

    clean = pd.read_csv(clean_path)
    inventory.append(audit_file(clean_path, "当前公共清洗数据"))

    docs_dir = os.path.join(ROOT_DIR, "docs")
    results_dir = os.path.join(ROOT_DIR, "results", "data_audit")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    inventory_path = os.path.join(docs_dir, "data_inventory.csv")
    pd.DataFrame(inventory).to_csv(inventory_path, index=False, encoding="utf-8-sig")

    validation = validate_cleaning(raw, clean)
    validation_path = os.path.join(results_dir, "cleaning_validation.csv")
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")

    failed = validation[validation["status"] != "PASS"]
    print(f"  Raw:   {raw.shape[0]} x {raw.shape[1]}")
    print(f"  Clean: {clean.shape[0]} x {clean.shape[1]}")
    print(f"  Validation: {len(validation) - len(failed)}/{len(validation)} PASS")
    print(f"  Saved: {os.path.relpath(inventory_path, ROOT_DIR)}")
    print(f"  Saved: {os.path.relpath(validation_path, ROOT_DIR)}")

    if not failed.empty:
        raise AssertionError(
            "Cleaning validation failed: "
            + ", ".join(failed["check_id"].astype(str))
        )


if __name__ == "__main__":
    main()
