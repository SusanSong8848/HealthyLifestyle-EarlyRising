"""
Task 1 - Early Waker Classification (最终优化版)
整合方案A（可复现）和方案B（严格防泄漏）的优点。

关键优化：
1. 排除 Wake_Up_Time, Sleep_Time, Sleep_Duration_Hours（三位一体，全移除）
2. 每折内部独立编码+缩放（防泄露管道）
3. 分层抽样 stratify=y
4. 多指标评估：ACC1, Balanced Accuracy, Yes类F1/Recall
5. Voting Ensemble + CatBoost 多模型对比
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
from config import OUTPUT_DIR, TASK1_DIR, RANDOM_STATE

print("=" * 65)
print("Task 1 — FINAL OPTIMIZED (方案A+B合并)")
print("排除: Wake_Up_Time, Sleep_Time, Sleep_Duration_Hours")
print("=" * 65)

# ==================== 1. 加载原始数据 ====================
print("\n[1/6] Loading raw data...")
raw = pd.read_csv(os.path.join(os.path.dirname(OUTPUT_DIR), "datas.csv"))
print(f"  Shape: {raw.shape}")

# ==================== 2. 数据清洗（与 preprocess.py 一致）====================
print("\n[2/6] Data cleaning...")

# 时间转换
for col in ["Wake_Up_Time", "Sleep_Time"]:
    parts = raw[col].str.split(":", expand=True).astype(float)
    raw[col + "_Minutes"] = parts[0] * 60 + parts[1]

# 缺失值处理
mask_no_ex = (raw["Exercise_Frequency_Per_Week"] == 0) | (raw["Exercise_Type"].isna())
raw.loc[mask_no_ex, "Exercise_Type"] = raw.loc[mask_no_ex, "Exercise_Type"].fillna("None")
raw.loc[mask_no_ex, "Workout_Intensity"] = raw.loc[mask_no_ex, "Workout_Intensity"].fillna("None")
for col in ["Exercise_Type", "Workout_Intensity"]:
    raw[col] = raw[col].fillna(raw[col].mode()[0])
raw["Alcohol_Consumption"] = raw["Alcohol_Consumption"].fillna("Unknown")

assert raw.isnull().sum().sum() == 0, "Missing values remain!"
print("  Missing values: 0 [OK]")

# ==================== 3. 特征工程 & 目标编码 ====================
print("\n[3/6] Feature engineering...")

y = raw["Early_Waker"].map({"Yes": 1, "No": 0})
person_ids = raw["Person_ID"]

# ---- 定义排除列表 ----
# 方案B严格策略：所有与"时钟时刻"相关的字段全部排除
EXCLUDED_COLUMNS = [
    "Person_ID",                # ID
    "Early_Waker",              # 目标
    "Wake_Up_Time",             # 原始起床时间文本
    "Sleep_Time",               # 原始入睡时间文本
    "Wake_Up_Time_Minutes",     # 起床时间分钟数（直接泄露）
    "Sleep_Time_Minutes",       # 入睡时间分钟数（强相关）
    "Sleep_Duration_Hours",     # 睡眠时长（可由时间推算，间接泄露）
    # Wellness_Category 也是目标，但保留在特征中？不，它是另一个task的目标
    # 只排除 task1 自己的标签
]

# 构建特征集
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
    "Health_Score", "Healthy_Aging_Score"
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

# ==================== 4. 分层划分 ====================
print("\n[4/6] Stratified split (80/20, stratify=y)...")

idx = np.arange(len(raw))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
print(f"  y_train: No={sum(y.iloc[train_idx]==0)}, Yes={sum(y.iloc[train_idx]==1)}")
print(f"  y_test:  No={sum(y.iloc[test_idx]==0)}, Yes={sum(y.iloc[test_idx]==1)}")

# ==================== 5. 管道内防泄露 CV + 建模 ====================
print("\n[5/6] 5-Fold CV with in-fold encoding (leak-free pipeline)...")

# 在每一折内部独立完成：
# 1. Label Encoding（fit on train fold）
# 2. StandardScaler（fit on train fold）
# 3. 模型训练 + 评估

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
X_all = raw.copy()
y_all = y.values

# 预定义模型（超参数来自之前的调优经验）
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

# 五折 CV，每折内部编码
cv_results = {name: {"acc": [], "bal_acc": [], "f1_yes": [], "recall_yes": []} for name in make_models()}
all_oof_preds = {name: np.zeros(len(train_idx)) for name in make_models()}

fold_models = []

for fold, (fold_train_idx, fold_val_idx) in enumerate(cv.split(train_idx, y.iloc[train_idx])):
    print(f"\n  Fold {fold+1}/5...")

    # 获取本折的原始训练/验证索引
    ft = train_idx[fold_train_idx]  # fold train
    fv = train_idx[fold_val_idx]    # fold val

    X_ft = X_all.iloc[ft].copy()
    X_fv = X_all.iloc[fv].copy()
    y_ft = y_all[ft]
    y_fv = y_all[fv]

    # ---- 在训练折上 fit 编码器 ----
    encoders = {}
    X_ft_enc = pd.DataFrame(index=X_ft.index)
    X_fv_enc = pd.DataFrame(index=X_fv.index)

    # 数值特征：fit scaler on fold train, transform both
    scaler = StandardScaler()
    X_ft_num = scaler.fit_transform(X_ft[available_num].values)
    X_fv_num = scaler.transform(X_fv[available_num].values)

    for j, col in enumerate(available_num):
        X_ft_enc[col] = X_ft_num[:, j]
        X_fv_enc[col] = X_fv_num[:, j]

    # 类别特征：fit LabelEncoder on fold train, transform both
    for col in available_cat:
        le = LabelEncoder()
        X_ft_enc[col] = le.fit_transform(X_ft[col].astype(str))
        # 处理测试折中出现的新类别（极少情况）
        X_fv_enc[col] = X_fv[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        encoders[col] = le

    # ---- 训练评估 ----
    for name, model in make_models().items():
        model.fit(X_ft_enc, y_ft)
        y_pred = model.predict(X_fv_enc)

        cv_results[name]["acc"].append(accuracy_score(y_fv, y_pred))
        cv_results[name]["bal_acc"].append(balanced_accuracy_score(y_fv, y_pred))
        cv_results[name]["f1_yes"].append(f1_score(y_fv, y_pred, pos_label=1))
        cv_results[name]["recall_yes"].append(recall_score(y_fv, y_pred, pos_label=1))

        # 保存 OOF 预测
        all_oof_preds[name][fold_val_idx] = y_pred

    if fold == 0:
        fold_models.append((X_ft_enc, y_ft))

# 打印 CV 汇总
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

# ==================== 6. 独立测试集评估 + Voting Ensemble ====================
print("\n[6/6] Final test set evaluation + Voting Ensemble...")

# 在完整训练集上重新编码+训练
X_train_full = X_all.iloc[train_idx].copy()
X_test_full = X_all.iloc[test_idx].copy()
y_train_full = y_all[train_idx]
y_test_full = y_all[test_idx]

# 全量编码器
enc_full = {}
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

# 训练最终模型
final_models = {}
for name, model in make_models().items():
    model.fit(X_train_enc, y_train_full)
    final_models[name] = model

# Voting Ensemble
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

# 评估所有模型 + Ensemble
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

# ==================== 保存输出 ====================
print("\n" + "=" * 65)
print("Saving outputs...")

best_model = ensemble
y_pred_final = best_model.predict(X_test_enc)
final_acc = all_test_results["Voting Ensemble"]["ACC1"]
final_bal = all_test_results["Voting Ensemble"]["Balanced_Accuracy"]
final_f1 = all_test_results["Voting Ensemble"]["F1_Yes"]
final_recall = all_test_results["Voting Ensemble"]["Recall_Yes"]

# 预测 CSV
label_map = {0: "No", 1: "Yes"}
pd.DataFrame({
    "Person_ID": person_ids.iloc[test_idx].values,
    "True_Label": pd.Series(y_test_full).map(label_map).values,
    "Predicted_Label": pd.Series(y_pred_final).map(label_map).values,
}).to_csv(os.path.join(TASK1_DIR, "predictions.csv"), index=False)

# 特征重要性（从三个树模型平均）
imp_models = ["Random Forest", "XGBoost", "LightGBM"]
all_imps = []
for name in imp_models:
    if hasattr(final_models[name], "feature_importances_"):
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

# 混淆矩阵
cm = confusion_matrix(y_test_full, y_pred_final)
print(f"\n  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
print(f"\n  Classification Report:")
print(classification_report(y_test_full, y_pred_final, target_names=["No", "Yes"]))

# 保存模型
pickle.dump(best_model, open(os.path.join(TASK1_DIR, "best_model.pkl"), "wb"))

# 保存指标
metrics = {
    "ACC1": float(final_acc),
    "Balanced_Accuracy": float(final_bal),
    "F1_Yes": float(final_f1),
    "Recall_Yes": float(final_recall),
    "best_model": "Voting Ensemble (LR+RF+XGB+LGB)",
    "features_used": len(feat_names),
    "excluded": EXCLUDED_COLUMNS,
    "cv_results": {name: {k: float(np.mean(v)) for k, v in res.items()} for name, res in cv_results.items()},
    "test_results": {name: {k: float(v) for k, v in r.items()} for name, r in all_test_results.items()},
}
pickle.dump(metrics, open(os.path.join(TASK1_DIR, "metrics.pkl"), "wb"))

# 可读指标
with open(os.path.join(TASK1_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"ACC1 = {final_acc:.6f}\n")
    f.write(f"Balanced_Accuracy = {final_bal:.6f}\n")
    f.write(f"F1_Yes = {final_f1:.6f}\n")
    f.write(f"Recall_Yes = {final_recall:.6f}\n")
    f.write(f"Best Model = Voting Ensemble (LR+RF+XGB+LGB)\n")
    f.write(f"Features = {len(feat_names)} (excluded: Wake_Up_Time, Sleep_Time, Sleep_Duration_Hours)\n\n")
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
print(f"TASK 1 FINAL RESULT")
print(f"  ACC1              = {final_acc:.6f}")
print(f"  Balanced Accuracy = {final_bal:.6f}")
print(f"  F1 (Yes)          = {final_f1:.6f}")
print(f"  Recall (Yes)      = {final_recall:.6f}")
print(f"  Best Model        = Voting Ensemble (LR+RF+XGB+LGB)")
print(f"  Score (20%)       = {final_acc * 100 * 0.20:.2f} / 20.00")
print(f"{'=' * 65}")