"""
Task 1 - Early Waker Classification (统一重构版 v3 FIXED)
严格遵循 公共数据要求.txt 方案：
1. 排除 Wake_Up_Time, Sleep_Time, Sleep_Duration_Hours（三位一体全移除）
2. 排除 Healthy_Aging_Score（可疑综合评分，主模型排除）
3. 5-Fold CV 管道内防泄露
4. 分层抽样 stratify=y
5. random_state=20260726（团队统一随机种子）
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    recall_score, confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_PATH, OUTPUT_DIR, TASK1_DIR,
    RANDOM_STATE, TASK1_LEAKY_FEATURES,
)

print("=" * 65)
print("Task 1 — Early Waker Classification (v3 FIXED)")
print(f"random_state={RANDOM_STATE}")
print("Excluded leaky: Wake_Up_Time, Sleep_Time, Sleep_Duration_Hours")
print("Also excluded: Healthy_Aging_Score (suspicious composite)")
print("=" * 65)

# ==================== 1. Load raw data ====================
print("\n[1/6] Loading raw data...")
raw = pd.read_csv(DATA_PATH)
print(f"  Shape: {raw.shape}")

# ==================== 2. Data cleaning ====================
print("\n[2/6] Data cleaning (unified rules)...")

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
print("  Missing values: 0 [OK]")

# ==================== 3. Feature engineering ====================
print("\n[3/6] Feature engineering...")

y = raw["Early_Waker"].map({"Yes": 1, "No": 0})
person_ids = raw["Person_ID"]

# ---- Defined exclusion list from public requirements + Healthy_Aging_Score ----
EXCLUDED_COLUMNS = list(TASK1_LEAKY_FEATURES) + ["Healthy_Aging_Score"]

# Numeric features (available, non-excluded)
numeric_cols = [
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
    # NOTE: Healthy_Aging_Score intentionally excluded
]

cat_cols = [
    "Gender", "Country", "Occupation", "Marital_Status",
    "Exercise_Type", "Morning_Workout", "Workout_Intensity",
    "Gym_Member", "Smoking_Status", "Alcohol_Consumption",
    "Meditation_Practice", "Obesity_Risk", "Hypertension_Risk",
    "Diabetes_Risk", "Cardiovascular_Risk", "Sleep_Disorder_Risk",
    "Fitness_Level"
]

available_num = [c for c in numeric_cols if c in raw.columns and c not in EXCLUDED_COLUMNS]
available_cat = [c for c in cat_cols if c in raw.columns and c not in EXCLUDED_COLUMNS]

print(f"  Numeric features: {len(available_num)}")
print(f"  Categorical features: {len(available_cat)}")
print(f"  Total features: {len(available_num) + len(available_cat)}")
print(f"  Excluded: {EXCLUDED_COLUMNS}")

# ==================== 4. Stratified split ====================
print("\n[4/6] Stratified split (80/20, stratify=y)...")

idx = np.arange(len(raw))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
print(f"  y_train: No={sum(y.iloc[train_idx]==0)}, Yes={sum(y.iloc[train_idx]==1)}")
print(f"  y_test:  No={sum(y.iloc[test_idx]==0)}, Yes={sum(y.iloc[test_idx]==1)}")

# ==================== 5. 5-Fold CV with in-fold encoding ====================
print("\n[5/6] 5-Fold CV with in-fold encoding (leak-free pipeline)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
X_all = raw.copy()
y_all = y.values

def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, C=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5, random_state=RANDOM_STATE
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05, num_leaves=63,
            random_state=RANDOM_STATE, verbose=-1, force_row_wise=True
        ),
    }

cv_results = {name: {"acc": [], "bal_acc": [], "f1_yes": [], "recall_yes": []} for name in make_models()}
all_oof_preds = {name: np.zeros(len(train_idx)) for name in make_models()}

for fold, (fold_train_idx, fold_val_idx) in enumerate(cv.split(train_idx, y.iloc[train_idx])):
    print(f"\n  Fold {fold+1}/5...")

    ft = train_idx[fold_train_idx]
    fv = train_idx[fold_val_idx]

    X_ft = X_all.iloc[ft].copy()
    X_fv = X_all.iloc[fv].copy()
    y_ft = y_all[ft]
    y_fv = y_all[fv]

    # In-fold encoding
    X_ft_enc = pd.DataFrame(index=X_ft.index)
    X_fv_enc = pd.DataFrame(index=X_fv.index)

    # Numeric scaling
    scaler = StandardScaler()
    X_ft_num = scaler.fit_transform(X_ft[available_num].values)
    X_fv_num = scaler.transform(X_fv[available_num].values)
    for j, col in enumerate(available_num):
        X_ft_enc[col] = X_ft_num[:, j]
        X_fv_enc[col] = X_fv_num[:, j]

    # Categorical encoding
    for col in available_cat:
        le = LabelEncoder()
        X_ft_enc[col] = le.fit_transform(X_ft[col].astype(str))
        X_fv_enc[col] = X_fv[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    for name, model in make_models().items():
        model.fit(X_ft_enc, y_ft)
        y_pred = model.predict(X_fv_enc)

        cv_results[name]["acc"].append(accuracy_score(y_fv, y_pred))
        cv_results[name]["bal_acc"].append(balanced_accuracy_score(y_fv, y_pred))
        cv_results[name]["f1_yes"].append(f1_score(y_fv, y_pred, pos_label=1))
        cv_results[name]["recall_yes"].append(recall_score(y_fv, y_pred, pos_label=1))
        all_oof_preds[name][fold_val_idx] = y_pred

print(f"\n  {'Model':<25s} {'ACC1':>8s} {'BalAcc':>8s} {'F1(Yes)':>8s} {'Recall(Yes)':>8s}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
best_model_name = None
best_oof_acc = 0
for name in cv_results:
    acc_mean = np.mean(cv_results[name]["acc"])
    bal_mean = np.mean(cv_results[name]["bal_acc"])
    f1_mean = np.mean(cv_results[name]["f1_yes"])
    rec_mean = np.mean(cv_results[name]["recall_yes"])
    print(f"  {name:<25s} {acc_mean:8.4f} {bal_mean:8.4f} {f1_mean:8.4f} {rec_mean:8.4f}")
    if acc_mean > best_oof_acc:
        best_oof_acc = acc_mean
        best_model_name = name

print(f"\n  Best OOF Model: {best_model_name} (OOF ACC1={best_oof_acc:.4f})")

# ==================== 6. Final test evaluation + Ensemble ====================
print("\n[6/6] Final test set evaluation + Voting Ensemble...")

X_train_full = X_all.iloc[train_idx].copy()
X_test_full = X_all.iloc[test_idx].copy()
y_train_full = y_all[train_idx]
y_test_full = y_all[test_idx]

# Full encoder
X_train_enc = pd.DataFrame(index=X_train_full.index)
X_test_enc = pd.DataFrame(index=X_test_full.index)

scaler_full = StandardScaler()
X_train_num = scaler_full.fit_transform(X_train_full[available_num].values)
X_test_num = scaler_full.transform(X_test_full[available_num].values)
for j, col in enumerate(available_num):
    X_train_enc[col] = X_train_num[:, j]
    X_test_enc[col] = X_test_num[:, j]

for col in available_cat:
    le = LabelEncoder()
    X_train_enc[col] = le.fit_transform(X_train_full[col].astype(str))
    X_test_enc[col] = X_test_full[col].astype(str).map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )

final_models = {}
for name, model in make_models().items():
    model.fit(X_train_enc, y_train_full)
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
ensemble.fit(X_train_enc, y_train_full)

all_test_results = {}
for name, model in {**final_models, "Voting Ensemble": ensemble}.items():
    y_pred = model.predict(X_test_enc)
    all_test_results[name] = {
        "ACC1": accuracy_score(y_test_full, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_test_full, y_pred),
        "F1_Yes": f1_score(y_test_full, y_pred, pos_label=1),
        "Recall_Yes": recall_score(y_test_full, y_pred, pos_label=1),
    }

print(f"\n  {'Model':<25s} {'ACC1':>8s} {'BalAcc':>8s} {'F1(Yes)':>8s} {'Recall(Yes)':>8s}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for name, r in all_test_results.items():
    marker = " ★" if name == "Voting Ensemble" else ""
    print(f"  {name:<25s} {r['ACC1']:8.4f} {r['Balanced_Accuracy']:8.4f} {r['F1_Yes']:8.4f} {r['Recall_Yes']:8.4f}{marker}")

# ==================== Save outputs ====================
print("\n" + "=" * 65)
print("Saving outputs...")

best_model = ensemble
y_pred_final = best_model.predict(X_test_enc)
final_acc = all_test_results["Voting Ensemble"]["ACC1"]
final_bal = all_test_results["Voting Ensemble"]["Balanced_Accuracy"]
final_f1 = all_test_results["Voting Ensemble"]["F1_Yes"]
final_recall = all_test_results["Voting Ensemble"]["Recall_Yes"]

# Predictions CSV
label_map = {0: "No", 1: "Yes"}
pd.DataFrame({
    "Person_ID": person_ids.iloc[test_idx].values,
    "True_Label": pd.Series(y_test_full).map(label_map).values,
    "Predicted_Label": pd.Series(y_pred_final).map(label_map).values,
}).to_csv(os.path.join(TASK1_DIR, "predictions.csv"), index=False)

# Feature importance
imp_models = ["Random Forest", "XGBoost", "LightGBM"]
all_imps = []
for name in imp_models:
    imps = final_models[name].feature_importances_
    all_imps.append(imps / imps.sum())
avg_imps = np.mean(all_imps, axis=0)
feat_names = list(available_num) + list(available_cat)
feat_imp = pd.DataFrame({
    "Feature": feat_names,
    "Importance": avg_imps / avg_imps.sum()
}).sort_values("Importance", ascending=False)
feat_imp.to_csv(os.path.join(TASK1_DIR, "feature_importance.csv"), index=False)

print("\n  Top 15 Features:")
for i in range(min(15, len(feat_imp))):
    row = feat_imp.iloc[i]
    print(f"    {i+1:2d}. {row['Feature']:<35s} {row['Importance']:.6f}")

cm = confusion_matrix(y_test_full, y_pred_final)
print(f"\n  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
print(f"\n  Classification Report:")
print(classification_report(y_test_full, y_pred_final, target_names=["No", "Yes"]))

pickle.dump(best_model, open(os.path.join(TASK1_DIR, "best_model.pkl"), "wb"))

metrics = {
    "ACC1": float(final_acc),
    "Balanced_Accuracy": float(final_bal),
    "F1_Yes": float(final_f1),
    "Recall_Yes": float(final_recall),
    "random_state": RANDOM_STATE,
    "best_model": "Voting Ensemble (LR+RF+XGB+LGB)",
    "features_used": len(feat_names),
    "excluded": EXCLUDED_COLUMNS,
    "cv_results": {name: {k: float(np.mean(v)) for k, v in res.items()} for name, res in cv_results.items()},
    "test_results": {name: {k: float(v) for k, v in r.items()} for name, r in all_test_results.items()},
}
pickle.dump(metrics, open(os.path.join(TASK1_DIR, "metrics.pkl"), "wb"))

with open(os.path.join(TASK1_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"ACC1 = {final_acc:.6f}\n")
    f.write(f"Balanced_Accuracy = {final_bal:.6f}\n")
    f.write(f"F1_Yes = {final_f1:.6f}\n")
    f.write(f"Recall_Yes = {final_recall:.6f}\n")
    f.write(f"Random_State = {RANDOM_STATE}\n")
    f.write(f"Best Model = Voting Ensemble (LR+RF+XGB+LGB)\n")
    f.write(f"Features = {len(feat_names)} (excluded: Wake_Up_Time, Sleep_Time, Sleep_Duration_Hours, Healthy_Aging_Score)\n\n")
    f.write("--- 5-Fold CV Results (OOF) ---\n")
    for name, res in cv_results.items():
        f.write(f"  {name}: ACC={np.mean(res['acc']):.4f}, BalAcc={np.mean(res['bal_acc']):.4f}, F1={np.mean(res['f1_yes']):.4f}, Recall={np.mean(res['recall_yes']):.4f}\n")
    f.write("\n--- Test Set Results ---\n")
    for name, r in all_test_results.items():
        f.write(f"  {name}: ACC={r['ACC1']:.4f}, BalAcc={r['Balanced_Accuracy']:.4f}, F1={r['F1_Yes']:.4f}, Recall={r['Recall_Yes']:.4f}\n")
    f.write("\n--- Top 15 Features ---\n")
    for i in range(min(15, len(feat_imp))):
        row = feat_imp.iloc[i]
        f.write(f"  {i+1:2d}. {row['Feature']:<35s} {row['Importance']:.6f}\n")

print(f"\n{'=' * 65}")
print(f"TASK 1 FINAL RESULT (v3 FIXED)")
print(f"  ACC1              = {final_acc:.6f}")
print(f"  Balanced Accuracy = {final_bal:.6f}")
print(f"  F1 (Yes)          = {final_f1:.6f}")
print(f"  Recall (Yes)      = {final_recall:.6f}")
print(f"  Best Model        = Voting Ensemble (LR+RF+XGB+LGB)")
print(f"  Score (20%)       = {final_acc * 100 * 0.20:.2f} / 20.00")
print(f"{'=' * 65}")