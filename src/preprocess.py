"""
Step 3: Data Preprocessing
- Missing value imputation
- Time feature conversion (Wake_Up_Time, Sleep_Time -> minutes)
- Categorical encoding
- Numeric scaling
- Health_Score discretization for Task 2
- Train/validation split
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_PATH, OUTPUT_DIR, RANDOM_STATE, TEST_SIZE,
    TARGET_TASK1, TARGET_TASK2, TARGET_TASK3,
    NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, TIME_COLUMNS,
    HEALTH_SCORE_BINS, HEALTH_SCORE_LABELS, ID_COLUMN,
    ALL_FEATURES
)

OUTPUT_PREP = os.path.join(OUTPUT_DIR, "preprocess")
os.makedirs(OUTPUT_PREP, exist_ok=True)

print("=" * 60)
print("Step 3: Data Preprocessing")
print("=" * 60)

# 1. Load raw data
print("\n[1/8] Loading raw data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# 2. Time feature conversion: "HH:MM" -> total minutes since midnight
print("\n[2/8] Converting time features (Wake_Up_Time, Sleep_Time) to minutes...")
for col in TIME_COLUMNS:
    if col in df.columns:
        # Split "H:MM" or "HH:MM"
        parts = df[col].str.split(":", expand=True).astype(float)
        df[col + "_Minutes"] = parts[0] * 60 + parts[1]
        print(f"  {col}: converted to {col}_Minutes (range: {df[col+'_Minutes'].min():.0f} - {df[col+'_Minutes'].max():.0f})")

# 3. Handle missing values
print("\n[3/8] Handling missing values...")

# 3a. Check Exercise_Type and Workout_Intensity: when Exercise_Frequency_Per_Week == 0,
#     these should logically be "None"
mask_no_exercise = (df["Exercise_Frequency_Per_Week"] == 0) | (df["Exercise_Type"].isna())
df.loc[mask_no_exercise, "Exercise_Type"] = df.loc[mask_no_exercise, "Exercise_Type"].fillna("None")
df.loc[mask_no_exercise, "Workout_Intensity"] = df.loc[mask_no_exercise, "Workout_Intensity"].fillna("None")

# Fill remaining NAs in Exercise_Type and Workout_Intensity with mode
for col in ["Exercise_Type", "Workout_Intensity"]:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f"  {col}: remaining NA filled with mode='{mode_val}'")

# 3b. Alcohol_Consumption: 30% missing. Create "Unknown" category to preserve information
df["Alcohol_Consumption"] = df["Alcohol_Consumption"].fillna("Unknown")
print(f"  Alcohol_Consumption: filled 30.14% NA with 'Unknown'")

# Verify no missing values remain
remaining_na = df.isnull().sum().sum()
print(f"  Remaining missing values after imputation: {remaining_na}")

# 4. Encode categorical features
print("\n[4/8] Encoding categorical features...")

# Identify actual categorical columns in df (some may have been converted)
cat_cols_in_df = [c for c in CATEGORICAL_COLUMNS if c in df.columns
                  and c not in [TARGET_TASK1, TARGET_TASK2, TARGET_TASK3, "Wellness_Category"]]
# Add Alcohol_Consumption and Exercise_Type if not already in list
for c in ["Alcohol_Consumption", "Exercise_Type", "Workout_Intensity"]:
    if c not in cat_cols_in_df and c in df.columns:
        cat_cols_in_df.append(c)

encoders = {}
df_encoded = df.copy()

for col in cat_cols_in_df:
    le = LabelEncoder()
    df_encoded[col + "_Encoded"] = le.fit_transform(df_encoded[col].astype(str))
    encoders[col] = le
    n_unique = len(le.classes_)
    print(f"  {col}: {n_unique} categories -> encoded as 0..{n_unique-1}")

print(f"  Total categorical features encoded: {len(cat_cols_in_df)}")

# 5. Encode target variables
print("\n[5/8] Encoding target variables...")
# Task 1: Early_Waker (Yes/No -> 1/0)
df_encoded["Early_Waker_Encoded"] = df_encoded[TARGET_TASK1].map({"Yes": 1, "No": 0})

# Task 2: Health_Score -> binned categories
df_encoded["Health_Score_Category"] = pd.cut(
    df_encoded[TARGET_TASK2],
    bins=HEALTH_SCORE_BINS,
    labels=HEALTH_SCORE_LABELS,
    include_lowest=True
)
he_le = LabelEncoder()
df_encoded["Health_Score_Category_Encoded"] = he_le.fit_transform(df_encoded["Health_Score_Category"])
print(f"  Health_Score bins: {HEALTH_SCORE_BINS}")
print(f"  Health_Score labels: {HEALTH_SCORE_LABELS}")
print(f"  Health_Score category distribution:")
print(df_encoded["Health_Score_Category"].value_counts().to_string())

# Task 3: Wellness_Category
wc_le = LabelEncoder()
df_encoded["Wellness_Category_Encoded"] = wc_le.fit_transform(df_encoded[TARGET_TASK3])
print(f"\n  Wellness_Category mapping: {dict(zip(wc_le.classes_, wc_le.transform(wc_le.classes_)))}")

# Save encoders
encoders["Health_Score_LabelEncoder"] = he_le
encoders["Wellness_Category_LabelEncoder"] = wc_le
with open(os.path.join(OUTPUT_PREP, "encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)

# 6. Build feature matrix
print("\n[6/8] Building feature matrix...")

# Numeric columns (excluding targets and ID)
num_cols = [c for c in NUMERIC_COLUMNS if c in df_encoded.columns
            and c not in [TARGET_TASK1, TARGET_TASK2, TARGET_TASK3,
                          "Wellness_Category", ID_COLUMN]]
# Add time minutes columns
time_min_cols = [c + "_Minutes" for c in TIME_COLUMNS if c + "_Minutes" in df_encoded.columns]
num_cols = [c for c in num_cols if c not in TIME_COLUMNS] + time_min_cols

# Encoded categorical columns
enc_cat_cols = [c + "_Encoded" for c in cat_cols_in_df if c + "_Encoded" in df_encoded.columns]

# Exclude highly leaky features: Healthy_Aging_Score (0.94 corr with Health_Score!)
# and Fitness_Level (encoded separately, correlated with Health_Score)
# We'll keep Healthy_Aging_Score for Task 1 but exclude for Task 2/3
leaky_features_task2 = ["Fitness_Level_Encoded"]
# Healthy_Aging_Score is NOT in numeric (it's in FITNESS_FEATURES via config)

feat_cols = num_cols + enc_cat_cols
feat_cols = [c for c in feat_cols if c in df_encoded.columns]
feat_cols = list(dict.fromkeys(feat_cols))  # deduplicate while preserving order

print(f"  Total feature columns: {len(feat_cols)}")

X = df_encoded[feat_cols].copy()
y1 = df_encoded["Early_Waker_Encoded"]
y2 = df_encoded["Health_Score_Category_Encoded"]
y3 = df_encoded["Wellness_Category_Encoded"]
person_ids = df_encoded[ID_COLUMN]

# 7. Train/validation split
print("\n[7/8] Splitting data (80% train, 20% validation)...")
# Use stratified split for all three targets
X_train, X_val, y1_train, y1_val, y2_train, y2_val, y3_train, y3_val, ids_train, ids_val = train_test_split(
    X, y1, y2, y3, person_ids,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y3  # stratify by Wellness_Category (most balanced)
)

# Split also by Task1 for evaluation
_, _, y1_train_strat, _ = train_test_split(
    X, y1, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y1
)

print(f"  Train set: {X_train.shape[0]} samples")
print(f"  Validation set: {X_val.shape[0]} samples")
print(f"  y1 (Early_Waker) train: {y1_train.value_counts().to_dict()}")
print(f"  y1 (Early_Waker) val:   {y1_val.value_counts().to_dict()}")
print(f"  y2 (Health_Score) train: {y2_train.value_counts().to_dict()}")
print(f"  y2 (Health_Score) val:   {y2_val.value_counts().to_dict()}")
print(f"  y3 (Wellness) train: {y3_train.value_counts().to_dict()}")
print(f"  y3 (Wellness) val:   {y3_val.value_counts().to_dict()}")

# 8. Scale numeric features
print("\n[8/8] Scaling numeric features...")
scaler = StandardScaler()

# Identify which feature columns are numeric (not encoded categoricals)
actual_num_cols = [c for c in num_cols if c in feat_cols]
X_train_num = X_train[actual_num_cols]
X_val_num = X_val[actual_num_cols]

# Fit on train, transform both
scaler.fit(X_train_num)

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_train_scaled[actual_num_cols] = scaler.transform(X_train_num)
X_val_scaled[actual_num_cols] = scaler.transform(X_val_num)

# Save scaler
with open(os.path.join(OUTPUT_PREP, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Save processed data
print("\n  Saving processed datasets...")
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

# Save as pickle for fast reload
with open(os.path.join(OUTPUT_PREP, "processed_data.pkl"), "wb") as f:
    pickle.dump(processed, f)

# Also save as CSV for inspection
df_train = pd.DataFrame(X_train_scaled, columns=feat_cols)
df_train[ID_COLUMN] = ids_train.values
df_train["Early_Waker"] = y1_train.values
df_train["Health_Score_Category"] = y2_train.values
df_train["Wellness_Category"] = y3_train.values
df_train.to_csv(os.path.join(OUTPUT_PREP, "train_processed.csv"), index=False)

df_val = pd.DataFrame(X_val_scaled, columns=feat_cols)
df_val[ID_COLUMN] = ids_val.values
df_val["Early_Waker"] = y1_val.values
df_val["Health_Score_Category"] = y2_val.values
df_val["Wellness_Category"] = y3_val.values
df_val.to_csv(os.path.join(OUTPUT_PREP, "val_processed.csv"), index=False)

print(f"\n  Feature count: {len(feat_cols)}")
print(f"  Numeric features (scaled): {len(actual_num_cols)}")
print(f"  Encoded categorical features: {len(enc_cat_cols)}")
print(f"\n  All outputs saved to: {OUTPUT_PREP}")

print("\n" + "=" * 60)
print("Step 3 completed! Preprocessed data ready for modeling.")
print("=" * 60)