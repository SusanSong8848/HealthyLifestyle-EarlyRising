"""
Step 5: Task 2 - Health Score Classification (Multi-class: Poor/Average/Good/Excellent)
Models: XGBoost, LightGBM, Random Forest
Evaluation: Accuracy (ACC2)
Note: Excludes leaky features (Healthy_Aging_Score, Fitness_Level)
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, TASK2_DIR, RANDOM_STATE, N_FOLDS

OUTPUT_PREP = os.path.join(OUTPUT_DIR, "preprocess")

print("=" * 60)
print("Task 2: Health Score Classification (4-class)")
print("=" * 60)

# 1. Load data
print("\n[1/5] Loading preprocessed data...")
with open(os.path.join(OUTPUT_PREP, "processed_data.pkl"), "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"].copy()
X_val   = data["X_val"].copy()
y_train = data["y2_train"]
y_val   = data["y2_val"]
ids_train = data["ids_train"]
ids_val   = data["ids_val"]
feat_cols_all = data["feat_cols"]

# 2. Remove leaky features for Task 2
print("\n[2/5] Removing leaky features (correlation > 0.9 with target)...")
leaky_features = [
    "Healthy_Aging_Score",       # 0.94 corr with Health_Score
    "Fitness_Level_Encoded",     # derived from Health_Score
]
feat_cols = [c for c in feat_cols_all if c not in leaky_features]
X_train = X_train[feat_cols]
X_val   = X_val[feat_cols]
print(f"  Removed: {leaky_features}")
print(f"  Features: {len(feat_cols_all)} -> {len(feat_cols)}")
print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

print(f"\n  y_train distribution:")
for i, label in enumerate(["Poor(0)", "Average(1)", "Good(2)", "Excellent(3)"]):
    cnt = (y_train == i).sum()
    print(f"    {label}: {cnt} ({cnt/len(y_train)*100:.1f}%)")

# 3. Train baseline models with class_weight
print("\n[3/5] Training baseline models (class_weight='balanced')...")
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric="mlogloss", verbosity=0,
        # scale_pos_weight doesn't work for multi-class; handled by sample weights
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, class_weight="balanced", verbose=-1
    ),
}

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    results[name] = {"model": model, "cv_mean": cv_scores.mean(),
                     "cv_std": cv_scores.std(), "val_acc": acc, "y_pred": y_pred}
    print(f"  {name:15s}: CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}, Val_Acc={acc:.4f}")

# 4. Hyperparameter tuning for best model
print("\n[4/5] Hyperparameter tuning...")
best_name = max(results, key=lambda k: results[k]["val_acc"])
print(f"  Best baseline: {best_name} (Val_Acc={results[best_name]['val_acc']:.4f})")

if "XGBoost" in best_name:
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    base = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss", verbosity=0)
elif "LightGBM" in best_name:
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
    }
    base = lgb.LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1)
else:
    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [15, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    base = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)

grid = GridSearchCV(base, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)
print(f"  Best params: {grid.best_params_}")
print(f"  Best CV score: {grid.best_score_:.4f}")

best_model = grid.best_estimator_
y_pred_tuned = best_model.predict(X_val)
final_acc = accuracy_score(y_val, y_pred_tuned)
print(f"  Final Val_Acc: {final_acc:.4f}")

# 5. Feature importance & save
print("\n[5/5] Feature importance & saving...")
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_[0])

feat_imp = pd.DataFrame({"Feature": feat_cols, "Importance": importances}).sort_values("Importance", ascending=False)
print("\n  Top 15 Features:")
print(feat_imp.head(15).to_string(index=False))
feat_imp.to_csv(os.path.join(TASK2_DIR, "feature_importance.csv"), index=False)

# Predictions
label_map = {0: "Poor", 1: "Average", 2: "Good", 3: "Excellent"}
y_val_orig = pd.Series(y_val).map(label_map)
y_pred_orig = pd.Series(y_pred_tuned).map(label_map)

results_df = pd.DataFrame({
    "Person_ID": ids_val.values,
    "True_Label": y_val_orig.values,
    "Predicted_Label": y_pred_orig.values,
})
results_df.to_csv(os.path.join(TASK2_DIR, "predictions.csv"), index=False)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred_tuned)
print(f"\n  Confusion Matrix (rows=true, cols=pred):")
print(f"         Poor  Avg   Good  Excel")
for i, lbl in enumerate(["Poor", "Avg", "Good", "Excel"]):
    print(f"  {lbl:6s} {cm[i,0]:5d} {cm[i,1]:5d} {cm[i,2]:5d} {cm[i,3]:5d}")

print(f"\n  Classification Report:")
print(classification_report(y_val, y_pred_tuned, target_names=["Poor", "Average", "Good", "Excellent"]))

# Save model
with open(os.path.join(TASK2_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

acc2 = final_acc
metrics = {"ACC2": acc2, "best_model_name": best_name,
           "best_params": grid.best_params_, "cv_best_score": grid.best_score_}
with open(os.path.join(TASK2_DIR, "metrics.pkl"), "wb") as f:
    pickle.dump(metrics, f)

print(f"\n{'=' * 60}")
print(f"Task 2 Completed! ACC2 = {acc2:.4f}")
print(f"Score contribution (40%): {acc2 * 100 * 0.40:.2f}")
print(f"{'=' * 60}")