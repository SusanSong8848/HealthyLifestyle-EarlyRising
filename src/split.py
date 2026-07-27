"""
统一数据切分脚本 (D27-07)
- seed=20260726
- 三任务共用同一份 split
- 输出 split_manifest.csv
- Person_ID 唯一连接
"""
import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_PATH, SPLITS_DIR, RANDOM_STATE, TEST_SIZE,
    HEALTH_SCORE_BINS, HEALTH_SCORE_LABELS, ID_COLUMN,
)
from clean_utils import load_and_clean_data

print("=" * 50)
print("Split script — unified train/val split")
print(f"  seed={RANDOM_STATE}, test_size={TEST_SIZE}")
print("=" * 50)

# Load and clean
df = load_and_clean_data(DATA_PATH)

# Build stratification key (task1 + task2 + task3 labels combined)
df["__hs_level__"] = pd.cut(df["Health_Score"], bins=HEALTH_SCORE_BINS,
                            labels=HEALTH_SCORE_LABELS, include_lowest=True)
stratify_key = (
    df["Early_Waker"].astype(str) + "_" +
    df["__hs_level__"].astype(str) + "_" +
    df["Wellness_Category"].astype(str)
)
print(f"  Unique stratification groups: {stratify_key.nunique()}")

# Split
idx = np.arange(len(df))
train_idx, val_idx = train_test_split(
    idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_key
)

# Build manifest
manifest = []
for i, split_name in [(train_idx, "train"), (val_idx, "val")]:
    for j in i:
        manifest.append({
            "Person_ID": df.iloc[j][ID_COLUMN],
            "split": split_name,
            "task1_label": df.iloc[j]["Early_Waker"],
            "task2_label": df.iloc[j]["__hs_level__"],
            "task3_label": df.iloc[j]["Wellness_Category"],
        })

manifest_df = pd.DataFrame(manifest)
manifest_path = os.path.join(SPLITS_DIR, "split_manifest.csv")
manifest_df.to_csv(manifest_path, index=False)

print(f"  train: {len(train_idx)}, val: {len(val_idx)}")
print(f"  Saved: {manifest_path}")

# Validation check
assert len(manifest_df) == 10000
assert manifest_df["Person_ID"].nunique() == 10000
assert manifest_df["Person_ID"].duplicated().sum() == 0
assert set(manifest_df["split"].unique()) == {"train", "val"}
print("  All validation checks PASSED")