"""
Step 3: Data Preprocessing (统一重构版 v3)
基于 公共数据要求.txt 的公共清洗规范：
- Exercise_Frequency_Per_Week=0 → Exercise_Type="No Exercise", Workout_Intensity="No Workout"
- Alcohol_Consumption 缺失 → "Unknown"（不能填 No Alcohol）
- 时间统一为 HH:MM（保留原始列 + 新增 _Minutes 列）
- 保留全部 10,000 行、64 列和 Person_ID
- 不编码、不标准化、不截断异常值、不删行
- 输出 base_semantic_clean.csv + split_manifest.csv
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_PATH, OUTPUT_DIR, PROCESSED_DIR, SPLITS_DIR,
    RANDOM_STATE, TEST_SIZE, N_FOLDS,
    TARGET_TASK1, TARGET_TASK2, TARGET_TASK3,
    NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, TIME_COLUMNS, ID_COLUMN,
    HEALTH_SCORE_BINS, HEALTH_SCORE_LABELS,
    STRUCTURAL_FILL, ALCOHOL_UNKNOWN, ALL_FEATURES,
)

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

print("=" * 65)
print("Step 3: Data Preprocessing (Unified Clean Rules v3)")
print("=" * 65)

# ============ 1. Load raw data ============
print("\n[1/8] Loading raw data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# ============ 2. Time normalization ============
print("\n[2/8] Normalizing time format to HH:MM...")
for col in TIME_COLUMNS:
    if col in df.columns:
        # Normalize to zero-padded HH:MM
        def _norm_time(val):
            parts = str(val).strip().split(":")
            h, m = int(parts[0]), int(parts[1])
            return f"{h:02d}:{m:02d}"
        df[col] = df[col].apply(_norm_time)
        # Create minutes from midnight column
        parts = df[col].str.split(":", expand=True).astype(float)
        df[col + "_Minutes"] = parts[0] * 60 + parts[1]
        print(f"  {col}: normalized, {col}_Minutes created (range: {df[col+'_Minutes'].min():.0f}-{df[col+'_Minutes'].max():.0f})")

# ============ 3. Missing value imputation (unified rules) ============
print("\n[3/8] Handling missing values (unified rules)...")

# 3a. Structural fills: Exercise_Type, Workout_Intensity
mask_no_exercise = df["Exercise_Frequency_Per_Week"] == 0
ex_type_missing = df["Exercise_Type"].isna()
ex_intensity_missing = df["Workout_Intensity"].isna()

# Safety check: all missing exercise fields correspond to zero frequency
assert (ex_type_missing & ~mask_no_exercise).sum() == 0, \
    "Exercise_Type has NaN but Exercise_Frequency_Per_Week != 0!"
assert (ex_intensity_missing & ~mask_no_exercise).sum() == 0, \
    "Workout_Intensity has NaN but Exercise_Frequency_Per_Week != 0!"

df.loc[mask_no_exercise & ex_type_missing, "Exercise_Type"] = STRUCTURAL_FILL["Exercise_Type"]
df.loc[mask_no_exercise & ex_intensity_missing, "Workout_Intensity"] = STRUCTURAL_FILL["Workout_Intensity"]
print(f"  Exercise_Type: {ex_type_missing.sum()} NaN → '{STRUCTURAL_FILL['Exercise_Type']}'")
print(f"  Workout_Intensity: {ex_intensity_missing.sum()} NaN → '{STRUCTURAL_FILL['Workout_Intensity']}'")

# 3b. Alcohol_Consumption → "Unknown" (NOT "No Alcohol")
alcohol_missing = df["Alcohol_Consumption"].isna()
df.loc[alcohol_missing, "Alcohol_Consumption"] = ALCOHOL_UNKNOWN
print(f"  Alcohol_Consumption: {alcohol_missing.sum()} NaN → '{ALCOHOL_UNKNOWN}'")

# Verify no remaining missing values
remaining_na = df.isnull().sum().sum()
print(f"  Remaining NaN cells: {remaining_na}")
assert remaining_na == 0, f"Still have {remaining_na} missing values!"

# ============ 4. Export base_semantic_clean.csv ============
print("\n[4/8] Exporting base_semantic_clean.csv (human-readable, BEFORE encoding)...")
clean_path = os.path.join(PROCESSED_DIR, "base_semantic_clean.csv")
df.to_csv(clean_path, index=False, encoding="utf-8-sig")
print(f"  Saved: {clean_path} ({len(df)} rows × {len(df.columns)} cols)")

# ============ 5. Encode categorical features ============
print("\n[5/8] Encoding categorical features...")

# Build categorical columns to encode (exclude targets and ID)
cat_cols_to_encode = [c for c in CATEGORICAL_COLUMNS
                      if c in df.columns
                      and c not in [TARGET_TASK1, TARGET_TASK2, TARGET_TASK3, ID_COLUMN]]

encoders = {}
df_encoded = df.copy()

for col in cat_cols_to_encode:
    le = LabelEncoder()
    df_encoded[col + "_Encoded"] = le.fit_transform(df_encoded[col].astype(str))
    encoders[col] = le
    print(f"  {col}: {len(le.classes_)} categories → 0..{len(le.classes_)-1}")

# ============ 6. Encode target variables ============
print("\n[6/8] Encoding target variables...")

# Task 1: Early_Waker → binary
df_encoded["Early_Waker_Encoded"] = df_encoded[TARGET_TASK1].map({"Yes": 1, "No": 0})
print(f"  Task1 Early_Waker: Yes={sum(df_encoded['Early_Waker_Encoded']==1)}, No={sum(df_encoded['Early_Waker_Encoded']==0)}")

# Task 2: Health_Score → 4-class bins
df_encoded["Health_Score_Category"] = pd.cut(
    df_encoded[TARGET_TASK2],
    bins=HEALTH_SCORE_BINS,
    labels=HEALTH_SCORE_LABELS,
    include_lowest=True
)
he_le = LabelEncoder()
df_encoded["Health_Score_Category_Encoded"] = he_le.fit_transform(df_encoded["Health_Score_Category"])
encoders["Health_Score_LabelEncoder"] = he_le
print(f"  Task2 Health_Score bins: {HEALTH_SCORE_BINS}")
print(f"  Category distribution:")
print(df_encoded["Health_Score_Category"].value_counts().sort_index().to_string())

# Task 3: Wellness_Category → 4-class (data has Poor, Average, Good, Excellent)
wc_le = LabelEncoder()
df_encoded["Wellness_Category_Encoded"] = wc_le.fit_transform(df_encoded[TARGET_TASK3])
encoders["Wellness_Category_LabelEncoder"] = wc_le
print(f"  Task3 Wellness_Category mapping: {dict(zip(wc_le.classes_, range(len(wc_le.classes_))))}")
print(f"  Category distribution:")
print(df_encoded[TARGET_TASK3].value_counts().to_string())

# Save encoders
with open(os.path.join(PROCESSED_DIR, "encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)
print(f"  Encoders saved to {PROCESSED_DIR}/encoders.pkl")

# ============ 7. Build feature matrix & split ============
print("\n[7/8] Building feature matrix & joint-stratified split...")

# Build numeric feature list
num_cols = [c for c in NUMERIC_COLUMNS if c in df_encoded.columns
            and c not in [TARGET_TASK1, TARGET_TASK2, TARGET_TASK3, ID_COLUMN]]
time_min_cols = [c + "_Minutes" for c in TIME_COLUMNS if c + "_Minutes" in df_encoded.columns]
num_cols = [c for c in num_cols if c not in TIME_COLUMNS] + time_min_cols

# Build encoded categorical feature list
enc_cat_cols = [c + "_Encoded" for c in cat_cols_to_encode if c + "_Encoded" in df_encoded.columns]

# Combine all feature columns
feat_cols = list(dict.fromkeys(num_cols + enc_cat_cols))  # dedup while preserving order
feat_cols = [c for c in feat_cols if c in df_encoded.columns]

X = df_encoded[feat_cols].copy()
y1 = df_encoded["Early_Waker_Encoded"]
y2 = df_encoded["Health_Score_Category_Encoded"]
y3 = df_encoded["Wellness_Category_Encoded"]
person_ids = df_encoded[ID_COLUMN]

print(f"  Total feature columns: {len(feat_cols)}")

# Joint stratification key: combine task1 + task2 + task3 labels
stratify_key = (
    df_encoded[TARGET_TASK1].astype(str) + "_" +
    df_encoded["Health_Score_Category"].astype(str) + "_" +
    df_encoded[TARGET_TASK3].astype(str)
)
print(f"  Unique stratification groups: {stratify_key.nunique()}")

X_train, X_val, y1_train, y1_val, y2_train, y2_val, y3_train, y3_val, ids_train, ids_val = train_test_split(
    X, y1, y2, y3, person_ids,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=stratify_key
)

print(f"  Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
print(f"  y1_train: No={sum(y1_train==0)}, Yes={sum(y1_train==1)}")
print(f"  y1_val:   No={sum(y1_val==0)}, Yes={sum(y1_val==1)}")

# ============ 8. Scale numeric features & save ============
print("\n[8/8] Scaling numeric features & saving...")

actual_num_cols = [c for c in num_cols if c in feat_cols]
scaler = StandardScaler()
scaler.fit(X_train[actual_num_cols])

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_train_scaled[actual_num_cols] = scaler.transform(X_train[actual_num_cols])
X_val_scaled[actual_num_cols] = scaler.transform(X_val[actual_num_cols])

with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Save processed data
processed = {
    "X_train": X_train_scaled,
    "X_val": X_val_scaled,
    "y1_train": y1_train,
    "y1_val": y1_val,
    "y2_train": y2_train,
    "y2_val": y2_val,
    "y3_train": y3_train,
    "y3_val": y3_val,
    "ids_train": ids_train,
    "ids_val": ids_val,
    "feat_cols": feat_cols,
    "actual_num_cols": actual_num_cols,
    "enc_cat_cols": enc_cat_cols,
}
with open(os.path.join(PROCESSED_DIR, "processed_data.pkl"), "wb") as f:
    pickle.dump(processed, f)

# ============ Save split_manifest.csv ============
print("\n  Generating split_manifest.csv...")

manifest_rows = []

# Train split entries
for i in range(len(ids_train)):
    pid = ids_train.iloc[i] if hasattr(ids_train, 'iloc') else ids_train[i]
    y2_label = he_le.inverse_transform([int(y2_train.iloc[i] if hasattr(y2_train, 'iloc') else y2_train[i])])[0]
    y3_label = wc_le.inverse_transform([int(y3_train.iloc[i] if hasattr(y3_train, 'iloc') else y3_train[i])])[0]
    manifest_rows.append({
        "Person_ID": pid,
        "split": "train",
        "task1_label": "Yes" if (y1_train.iloc[i] if hasattr(y1_train, 'iloc') else y1_train[i]) == 1 else "No",
        "task2_label": y2_label,
        "task3_label": y3_label,
    })

# Val split entries
for i in range(len(ids_val)):
    pid = ids_val.iloc[i] if hasattr(ids_val, 'iloc') else ids_val[i]
    y2_label = he_le.inverse_transform([int(y2_val.iloc[i] if hasattr(y2_val, 'iloc') else y2_val[i])])[0]
    y3_label = wc_le.inverse_transform([int(y3_val.iloc[i] if hasattr(y3_val, 'iloc') else y3_val[i])])[0]
    manifest_rows.append({
        "Person_ID": pid,
        "split": "val",
        "task1_label": "Yes" if (y1_val.iloc[i] if hasattr(y1_val, 'iloc') else y1_val[i]) == 1 else "No",
        "task2_label": y2_label,
        "task3_label": y3_label,
    })

manifest_df = pd.DataFrame(manifest_rows)
manifest_path = os.path.join(SPLITS_DIR, "split_manifest.csv")
manifest_df.to_csv(manifest_path, index=False)
print(f"  split_manifest.csv saved: {manifest_path}")
print(f"    train: {len(manifest_df[manifest_df['split']=='train'])}")
print(f"    val:   {len(manifest_df[manifest_df['split']=='val'])}")

# ============ Summary ============
print(f"\n{'=' * 65}")
print(f"Preprocessing completed!")
print(f"  base_semantic_clean.csv: data/processed/base_semantic_clean.csv")
print(f"  split_manifest.csv:      data/splits/split_manifest.csv")
print(f"  processed_data.pkl:       data/processed/processed_data.pkl")
print(f"  Features: {len(feat_cols)} (numeric: {len(actual_num_cols)}, categorical: {len(enc_cat_cols)})")
print(f"{'=' * 65}")