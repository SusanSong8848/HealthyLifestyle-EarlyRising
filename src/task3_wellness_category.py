"""
Task 3 - Wellness Category Prediction (统一重构版 v3 FIXED)
严格遵循 公共数据要求.txt：
- 目标：Wellness_Category 四分类（Excellent / Good / Average / Poor）
  - 题面写三分类，但实际数据有 114 条 Poor，按实际数据做四分类
- 移除泄漏特征：Wellness_Category（目标）、Health_Score（按45/65/80分界可100%还原）、Fitness_Level（与目标100%相同）
- 可疑特征：Healthy_Aging_Score 主模型排除
- random_state=20260726
- 评估指标：Accuracy (ACC3)
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_PATH, TASK3_DIR,
    RANDOM_STATE, N_FOLDS, TASK3_LEAKY_FEATURES, ID_COLUMN,
)

print("=" * 65)
print("Task 3: Wellness Category Prediction (v3 FIXED)")
print(f"Target: Wellness_Category (4-class: Excellent/Good/Average/Poor)")
print(f"random_state={RANDOM_STATE}")
print(f"Leaky excluded: {TASK3_LEAKY_FEATURES}")
print("Also excluded: Healthy_Aging_Score (suspicious composite)")
print("NOTE: Data contains 114 'Poor' samples — model is 4-class, not 3-class")
print("=" * 65)

# ==================== 1. Load & clean data ====================
print("\n[1/6] Loading raw data & cleaning...")
raw = pd.read_csv(DATA_PATH)
print(f"  Shape: {raw.shape}")

# Time normalization + minutes
for col in ["Wake_Up_Time", "Sleep_Time"]:
    def _norm_time(val):
        parts = str(val).strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        return f"{h:02d}:{m:02d}"
    raw[col] = raw[col].apply(_norm_time)
    parts = raw[col].str.split(":", expand=True).astype(float)
    raw[col + "_Minutes"] = parts[0] * 60 + parts[1]

# Structural fills
mask_no_ex = raw["Exercise_Frequency_Per_Week"] == 0
raw.loc[mask_no_ex & raw["Exercise_Type"].isna(), "Exercise_Type"] = "No Exercise"
raw.loc[mask_no_ex & raw["Workout_Intensity"].isna(), "Workout_Intensity"] = "No Workout"
raw["Alcohol_Consumption"] = raw["Alcohol_Consumption"].fillna("Unknown")
assert raw.isnull().sum().sum() == 0, "Missing values remain!"

# ==================== 2. Target variable ====================
print("\n[2/6] Target variable: Wellness_Category...")

# Verify: data has 4 classes, not just 3
wc_dist = raw["Wellness_Category"].value_counts()
print(f"  Wellness_Category distribution:")
print(wc_dist.to_string())

# Encode target
wc_le = LabelEncoder()
y_all = wc_le.fit_transform(raw["Wellness_Category"])
print(f"  Encoding: {dict(zip(wc_le.classes_, range(len(wc_le.classes_))))}")

# Defined excluded features
EXCLUDED_COLUMNS = list(TASK3_LEAKY_FEATURES) + [
    "Healthy_Aging_Score",
    "Early_Waker",
    "Wake_Up_Time", "Sleep_Time",
    "Wake_Up_Time_Minutes", "Sleep_Time_Minutes",
]

# ==================== 3. Feature engineering ====================
print("\n[3/6] Feature engineering...")

numeric_cols = [
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
]

cat_cols = [
    "Gender", "Country", "Occupation", "Marital_Status",
    "Exercise_Type", "Morning_Workout", "Workout_Intensity",
    "Gym_Member", "Smoking_Status", "Alcohol_Consumption",
    "Meditation_Practice", "Obesity_Risk", "Hypertension_Risk",
    "Diabetes_Risk", "Cardiovascular_Risk", "Sleep_Disorder_Risk",
]

available_num = [c for c in numeric_cols if c in raw.columns and c not in EXCLUDED_COLUMNS]
available_cat = [c for c in cat_cols if c in raw.columns and c not in EXCLUDED_COLUMNS]

print(f"  Numeric features: {len(available_num)}")
print(f"  Categorical features: {len(available_cat)}")
print(f"  Total features: {len(available_num) + len(available_cat)}")

# ==================== 4. Split ====================
print("\n[4/6] Stratified split (80/20, stratify=y)...")

idx = np.arange(len(raw))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
)

# Encode categorical features
X_all = raw.copy()
for col in available_cat:
    le = LabelEncoder()
    X_all[col + "_Encoded"] = le.fit_transform(X_all[col].astype(str))

# Build feature matrices
feat_cols = available_num + [c + "_Encoded" for c in available_cat]

X_train_raw = X_all.iloc[train_idx][feat_cols].values.astype(float)
X_test_raw = X_all.iloc[test_idx][feat_cols].values.astype(float)
y_train = y_all[train_idx]
y_test = y_all[test_idx]
person_ids_test = raw.iloc[test_idx][ID_COLUMN].values

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
for i, label in enumerate(wc_le.classes_):
    print(f"    {label}: train={sum(y_train==i)}, test={sum(y_test==i)}")

# ==================== 5. Cross-validation ====================
print("\n[5/6] 5-Fold CV...")

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5, random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                                      random_state=RANDOM_STATE, eval_metric="mlogloss", verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, num_leaves=63,
                                        random_state=RANDOM_STATE, verbose=-1, force_row_wise=True),
    }

cv_baseline = {}
for name, model in make_models().items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_baseline[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
    print(f"  {name:20s}: CV={scores.mean():.4f}±{scores.std():.4f}")

# ==================== 6. Final models + Ensemble ====================
print("\n[6/6] Final model training & evaluation...")

final_models = {}
for name, model in make_models().items():
    model.fit(X_train, y_train)
    final_models[name] = model

ensemble = VotingClassifier(
    estimators=[
        ("lr", final_models["Logistic Regression"]),
        ("rf", final_models["Random Forest"]),
        ("xgb", final_models["XGBoost"]),
        ("lgb", final_models["LightGBM"]),
    ],
    voting="soft"
)
ensemble.fit(X_train, y_train)

all_test_results = {}
for name, model in {**final_models, "Voting Ensemble": ensemble}.items():
    y_pred = model.predict(X_test)
    all_test_results[name] = {
        "ACC3": accuracy_score(y_test, y_pred),
    }

print(f"\n  {'Model':<25s} {'ACC3':>8s}")
print(f"  {'-'*25} {'-'*8}")
for name, r in all_test_results.items():
    marker = " ★" if name == "Voting Ensemble" else ""
    print(f"  {name:<25s} {r['ACC3']:8.4f}{marker}")

best_model = ensemble
y_pred_final = best_model.predict(X_test)
final_acc = all_test_results["Voting Ensemble"]["ACC3"]

# ==================== Save outputs ====================
print("\n" + "=" * 65)
print("Saving outputs...")

# Predictions CSV
label_map = {i: lbl for i, lbl in enumerate(wc_le.classes_)}
pd.DataFrame({
    "Person_ID": person_ids_test,
    "True_Label": pd.Series(y_test).map(label_map).values,
    "Predicted_Label": pd.Series(y_pred_final).map(label_map).values,
}).to_csv(os.path.join(TASK3_DIR, "predictions.csv"), index=False)

# Feature importance
imp_models = ["Random Forest", "XGBoost", "LightGBM"]
all_imps = []
for name in imp_models:
    imps = final_models[name].feature_importances_
    all_imps.append(imps / imps.sum())
avg_imps = np.mean(all_imps, axis=0)
feat_imp = pd.DataFrame({
    "Feature": feat_cols,
    "Importance": avg_imps / avg_imps.sum()
}).sort_values("Importance", ascending=False)
feat_imp.to_csv(os.path.join(TASK3_DIR, "feature_importance.csv"), index=False)

print("\n  Top 15 Features:")
for i in range(min(15, len(feat_imp))):
    row = feat_imp.iloc[i]
    print(f"    {i+1:2d}. {row['Feature']:<35s} {row['Importance']:.6f}")

cm = confusion_matrix(y_test, y_pred_final)
print(f"\n  Confusion Matrix (rows=true, cols=pred):")
lbls = list(label_map.values())
print(f"         {'':>8s} " + " ".join(f"{l:>8s}" for l in lbls))
for i, lbl in enumerate(lbls):
    print(f"  {lbl:8s} " + " ".join(f"{cm[i,j]:8d}" for j in range(len(lbls))))

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=lbls))

pickle.dump(best_model, open(os.path.join(TASK3_DIR, "best_model.pkl"), "wb"))

metrics = {
    "ACC3": float(final_acc),
    "random_state": RANDOM_STATE,
    "n_classes": len(wc_le.classes_),
    "classes": list(wc_le.classes_),
    "note": "4-class classification (includes 114 Poor samples despite problem saying 3-class)",
    "best_model": "Voting Ensemble (LR+RF+XGB+LGB)",
    "features_used": len(feat_cols),
    "excluded": EXCLUDED_COLUMNS,
    "cv_results": cv_baseline,
    "test_results": {name: {k: float(v) for k, v in r.items()} for name, r in all_test_results.items()},
}
pickle.dump(metrics, open(os.path.join(TASK3_DIR, "metrics.pkl"), "wb"))

with open(os.path.join(TASK3_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"ACC3 = {final_acc:.6f}\n")
    f.write(f"Random_State = {RANDOM_STATE}\n")
    f.write(f"Classes = {list(wc_le.classes_)} (4-class, includes Poor)\n")
    f.write(f"Best Model = Voting Ensemble (LR+RF+XGB+LGB)\n")
    f.write(f"Features = {len(feat_cols)}\n")
    f.write(f"Excluded = {EXCLUDED_COLUMNS}\n\n")
    f.write("--- 5-Fold CV Results ---\n")
    for name, r in cv_baseline.items():
        f.write(f"  {name}: CV={r['mean']:.4f}±{r['std']:.4f}\n")
    f.write("\n--- Test Set Results ---\n")
    for name, r in all_test_results.items():
        f.write(f"  {name}: ACC={r['ACC3']:.4f}\n")
    f.write("\n--- Top 15 Features ---\n")
    for i in range(min(15, len(feat_imp))):
        row = feat_imp.iloc[i]
        f.write(f"  {i+1:2d}. {row['Feature']:<35s} {row['Importance']:.6f}\n")
    f.write("\n--- Note ---\n")
    f.write("Data contains 114 'Poor' Wellness_Category samples. Despite the problem\n")
    f.write("description mentioning only Excellent/Good/Average, this model treats it as\n")
    f.write("a 4-class problem as recommended in 公共数据要求.txt.\n")

print(f"\n{'=' * 65}")
print(f"TASK 3 FINAL RESULT (v3 FIXED)")
print(f"  ACC3              = {final_acc:.6f}")
print(f"  Classes           = {list(wc_le.classes_)} (4-class)")
print(f"  Best Model        = Voting Ensemble (LR+RF+XGB+LGB)")
print(f"  Score (40%)       = {final_acc * 100 * 0.40:.2f} / 40.00")
print(f"{'=' * 65}")