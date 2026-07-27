"""
Task 2 - Health Score Classification (v3.1 最终优化版)
- 使用 clean_utils 共享清洗函数
- 自动选择最优单模型（不再盲目录入 Ensemble）
- 移除所有泄漏特征
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
    DATA_PATH, TASK2_DIR,
    RANDOM_STATE, N_FOLDS, TASK2_LEAKY_FEATURES,
    HEALTH_SCORE_BINS, HEALTH_SCORE_LABELS, ID_COLUMN,
)
from clean_utils import load_and_clean_data, TASK_NUMERIC_COLS, TASK_CATEGORICAL_COLS

print("=" * 65)
print("Task 2: Health Score Classification (v3.1)")
print(f"  Bins: {HEALTH_SCORE_BINS} → {HEALTH_SCORE_LABELS}")
print("=" * 65)

# ==================== 1. Load & clean ====================
print("\n[1/6] Loading & cleaning data...")
raw = load_and_clean_data(DATA_PATH)
print(f"  Shape: {raw.shape}, Missing: {raw.isnull().sum().sum()}")

# ==================== 2. Target ====================
print("\n[2/6] Creating target...")
raw["Health_Score_Level"] = pd.cut(raw["Health_Score"], bins=HEALTH_SCORE_BINS,
                                   labels=HEALTH_SCORE_LABELS, include_lowest=True)
print(f"  Distribution:\n{raw['Health_Score_Level'].value_counts().sort_index().to_string()}")

y_le = LabelEncoder()
y_all = y_le.fit_transform(raw["Health_Score_Level"])

EXCLUDED = list(TASK2_LEAKY_FEATURES) + [
    "Healthy_Aging_Score", "Early_Waker", "Health_Score_Level",
    "Wake_Up_Time", "Sleep_Time", "Wake_Up_Time_Minutes", "Sleep_Time_Minutes",
]

# ==================== 3. Features ====================
print("\n[3/6] Feature engineering...")
avail_num, avail_cat = [], []
for c in TASK_NUMERIC_COLS:
    if c in raw.columns and c not in EXCLUDED:
        avail_num.append(c)
for c in TASK_CATEGORICAL_COLS:
    if c in raw.columns and c not in EXCLUDED:
        avail_cat.append(c)

# Encode categoricals
for col in avail_cat:
    le = LabelEncoder()
    raw[col + "_Encoded"] = le.fit_transform(raw[col].astype(str))

feat_cols = avail_num + [c + "_Encoded" for c in avail_cat]
print(f"  Features: {len(avail_num)} num + {len(avail_cat)} cat → {len(feat_cols)} total")

# ==================== 4. Split ====================
print("\n[4/6] Stratified split...")
idx = np.arange(len(raw))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all)

X_train_raw = raw.iloc[train_idx][feat_cols].values.astype(float)
X_test_raw = raw.iloc[test_idx][feat_cols].values.astype(float)
y_train = y_all[train_idx]
y_test = y_all[test_idx]
ids_test = raw.iloc[test_idx][ID_COLUMN].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ==================== 5. CV + Model Selection ====================
print("\n[5/6] 5-Fold CV...")
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5, random_state=RANDOM_STATE),
    "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                                  random_state=RANDOM_STATE, eval_metric="mlogloss", verbosity=0),
    "LightGBM": lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, num_leaves=63,
                                    random_state=RANDOM_STATE, verbose=-1, force_row_wise=True),
}

cv_results = {}
best_name, best_cv = None, 0
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
    print(f"  {name:20s}: CV={scores.mean():.4f}±{scores.std():.4f}")
    if scores.mean() > best_cv:
        best_cv = scores.mean()
        best_name = name

# ==================== 6. Final ====================
print(f"\n[6/6] Best model: {best_name} (CV={best_cv:.4f}). Training...")

best_model = models[best_name]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
final_acc = accuracy_score(y_test, y_pred)

label_map = {i: lbl for i, lbl in enumerate(y_le.classes_)}
print(f"\n  ACC2 = {final_acc:.4f}")
print(classification_report(y_test, y_pred, target_names=list(label_map.values())))

# ==================== Save ====================
pd.DataFrame({
    "Person_ID": ids_test,
    "True_Label": pd.Series(y_test).map(label_map).values,
    "Predicted_Label": pd.Series(y_pred).map(label_map).values,
}).to_csv(os.path.join(TASK2_DIR, "predictions.csv"), index=False)

imp_df = pd.DataFrame({
    "Feature": feat_cols, "Importance": best_model.feature_importances_ if hasattr(best_model, "feature_importances_") else np.ones(len(feat_cols))
}).sort_values("Importance", ascending=False)
imp_df.to_csv(os.path.join(TASK2_DIR, "feature_importance.csv"), index=False)

pickle.dump(best_model, open(os.path.join(TASK2_DIR, "best_model.pkl"), "wb"))

metrics = {"ACC2": float(final_acc), "best_model": best_name, "features_used": len(feat_cols),
           "excluded": EXCLUDED, "cv_results": cv_results}
pickle.dump(metrics, open(os.path.join(TASK2_DIR, "metrics.pkl"), "wb"))

with open(os.path.join(TASK2_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"ACC2 = {final_acc:.6f}\nBest Model = {best_name}\nFeatures = {len(feat_cols)}\n")
    for name, r in cv_results.items():
        f.write(f"  {name}: CV={r['mean']:.4f}±{r['std']:.4f}\n")

score = final_acc * 100 * 0.40
print(f"\n{'=' * 65}")
print(f"TASK 2: ACC2={final_acc:.4f}, Model={best_name}, Score={score:.2f}/40.00")
print(f"{'=' * 65}")