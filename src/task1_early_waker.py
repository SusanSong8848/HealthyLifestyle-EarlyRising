"""
Step 4: Task 1 - Early Waker Classification (Binary)
Models: XGBoost, LightGBM, Random Forest, Logistic Regression
Evaluation: Accuracy (ACC1)

CRITICAL FIX — Data Leakage Removal:
Early_Waker is trivially defined by Wake_Up_Time:
  - 4:00-5:59 AM → 100% Yes
  - 6:00-6:59 AM → Mixed (~48% Yes)
  - 7:00-11:59 AM → 0% Yes

If we include Wake_Up_Time / Sleep_Time as features, models achieve ACC1 ≈ 1.0 by
"cheating" — they simply memorize the time threshold instead of learning real patterns.

FIX: We EXCLUDE Wake_Up_Time_Minutes and Sleep_Time_Minutes from the feature set.
Models must now rely on lifestyle, diet, exercise, mental, and physiological features
to predict whether someone is an early waker — this is the true challenge.

We KEEP Sleep_Duration_Hours and Sleep_Quality_Score as they are derived, indirect
sleep metrics that do NOT directly encode the waking hour.
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, TASK1_DIR, RANDOM_STATE, N_FOLDS

OUTPUT_PREP = os.path.join(OUTPUT_DIR, "preprocess")

print("=" * 70)
print("Task 1: Early Waker Classification (Binary)")
print("!!! NO Wake_Up_Time / Sleep_Time — true pattern learning !!!")
print("=" * 70)

# ===== 1. Load Data & Remove Leaky Features =====
print("\n[1/5] Loading preprocessed data & removing leaky time features...")
with open(os.path.join(OUTPUT_PREP, "processed_data.pkl"), "rb") as f:
    data = pickle.load(f)

X_train_raw = data["X_train"]
X_val_raw = data["X_val"]
y_train = data["y1_train"].astype(int)
y_val = data["y1_val"].astype(int)
ids_val = data["ids_val"]
feat_cols_all = data["feat_cols"]

# ---- REMOVE LEAKY FEATURES ----
# Wake_Up_Time_Minutes: directly encodes the answer (waking hour)
# Sleep_Time_Minutes:   also strongly correlated (early wakers sleep earlier)
LEAKY_FEATURES = ["Wake_Up_Time_Minutes", "Sleep_Time_Minutes"]

# Build clean feature list
clean_feat_cols = [c for c in feat_cols_all if c not in LEAKY_FEATURES]

# Build clean train/val matrices (as DataFrames for column indexing)
X_train_df = pd.DataFrame(X_train_raw, columns=feat_cols_all)[clean_feat_cols]
X_val_df = pd.DataFrame(X_val_raw, columns=feat_cols_all)[clean_feat_cols]
X_train = X_train_df.values
X_val = X_val_df.values

print(f"  Original features: {len(feat_cols_all)}")
print(f"  Removed leaky features: {LEAKY_FEATURES}")
print(f"  Clean features: {len(clean_feat_cols)}")
print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
print(f"  y_train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"  y_val:   {dict(zip(*np.unique(y_val, return_counts=True)))}")

# ===== 2. Baseline Models (5-fold CV) =====
print("\n[2/5] Training baseline models (5-fold CV, no leaky features)...")
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_STATE),
    "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0),
    "LightGBM": lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                    random_state=RANDOM_STATE, verbose=-1, force_row_wise=True),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                     learning_rate=0.1, random_state=RANDOM_STATE),
}

baseline_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    baseline_results[name] = {
        "model": model,
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "val_acc": round(float(acc), 4),
        "val_f1": round(float(f1), 4),
        "y_pred": y_pred,
    }
    print(f"  {name:25s}: CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}, Val_Acc={acc:.4f}, F1={f1:.4f}")

# ===== 3. Manual Hyperparameter Tuning =====
print("\n[3/5] Manual hyperparameter tuning...")
tuned_models = {}

# --- XGBoost ---
print("  Tuning XGBoost...")
xgb_params_list = [
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
    {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.2},
]
best_xgb, best_xgb_score = None, 0
for params in xgb_params_list:
    m = xgb.XGBClassifier(**params, random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0)
    cv_scores = cross_val_score(m, X_train, y_train, cv=3, scoring="accuracy")
    score = cv_scores.mean()
    if score > best_xgb_score:
        best_xgb_score = score
        best_xgb = params.copy()
best_xgb["random_state"] = RANDOM_STATE
best_xgb["eval_metric"] = "logloss"
best_xgb["verbosity"] = 0
final_xgb = xgb.XGBClassifier(**best_xgb)
final_xgb.fit(X_train, y_train)
tuned_models["XGBoost"] = {"model": final_xgb, "best_params": best_xgb, "best_cv": round(float(best_xgb_score), 4)}
print(f"    Best: {best_xgb}, CV={best_xgb_score:.4f}")

# --- LightGBM ---
print("  Tuning LightGBM...")
lgb_params_list = [
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "num_leaves": 31},
    {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "num_leaves": 31},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "num_leaves": 31},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "num_leaves": 63},
    {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1, "num_leaves": 63},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "num_leaves": 31},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "num_leaves": 63},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.2, "num_leaves": 31},
]
best_lgb, best_lgb_score = None, 0
for params in lgb_params_list:
    m = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1, force_row_wise=True)
    cv_scores = cross_val_score(m, X_train, y_train, cv=3, scoring="accuracy")
    score = cv_scores.mean()
    if score > best_lgb_score:
        best_lgb_score = score
        best_lgb = params.copy()
best_lgb["random_state"] = RANDOM_STATE
best_lgb["verbose"] = -1
best_lgb["force_row_wise"] = True
final_lgb = lgb.LGBMClassifier(**best_lgb)
final_lgb.fit(X_train, y_train)
tuned_models["LightGBM"] = {"model": final_lgb, "best_params": best_lgb, "best_cv": round(float(best_lgb_score), 4)}
print(f"    Best: {best_lgb}, CV={best_lgb_score:.4f}")

# --- Random Forest ---
print("  Tuning Random Forest...")
rf_params_list = [
    {"n_estimators": 200, "max_depth": 10, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": 15, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 5},
    {"n_estimators": 300, "max_depth": 10, "min_samples_split": 2},
    {"n_estimators": 300, "max_depth": 15, "min_samples_split": 5},
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 2},
    {"n_estimators": 500, "max_depth": 10, "min_samples_split": 2},
    {"n_estimators": 500, "max_depth": None, "min_samples_split": 5},
]
best_rf, best_rf_score = None, 0
for params in rf_params_list:
    m = RandomForestClassifier(**params, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(m, X_train, y_train, cv=3, scoring="accuracy")
    score = cv_scores.mean()
    if score > best_rf_score:
        best_rf_score = score
        best_rf = params.copy()
best_rf["random_state"] = RANDOM_STATE
final_rf = RandomForestClassifier(**best_rf)
final_rf.fit(X_train, y_train)
tuned_models["Random Forest"] = {"model": final_rf, "best_params": best_rf, "best_cv": round(float(best_rf_score), 4)}
print(f"    Best: {best_rf}, CV={best_rf_score:.4f}")

# ===== 4. Select Best Model & Build Ensemble =====
print("\n[4/5] Evaluating tuned models & building Voting Ensemble...")

best_name, best_acc = None, 0
for name, info in tuned_models.items():
    y_pred = info["model"].predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    info["val_acc"] = round(float(acc), 4)
    info["val_f1"] = round(float(f1), 4)
    print(f"  {name}: Val_Acc={acc:.4f}, F1={f1:.4f}")
    if acc > best_acc:
        best_acc, best_name = acc, name

# Voting Ensemble
print("  Building Voting Ensemble (XGB+LGB+RF)...")
ensemble = VotingClassifier(
    estimators=[
        ("xgb", tuned_models["XGBoost"]["model"]),
        ("lgb", tuned_models["LightGBM"]["model"]),
        ("rf", tuned_models["Random Forest"]["model"]),
    ],
    voting="soft"
)
ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_val)
ens_acc = accuracy_score(y_val, y_pred_ens)
ens_f1 = f1_score(y_val, y_pred_ens)
print(f"  Voting Ensemble: Val_Acc={ens_acc:.4f}, F1={ens_f1:.4f}")

# Pick final best
if ens_acc >= best_acc:
    best_model = ensemble
    y_pred_final = y_pred_ens
    final_acc = ens_acc
    final_f1 = ens_f1
    final_name = "Voting Ensemble (XGB+LGB+RF)"
else:
    best_model = tuned_models[best_name]["model"]
    y_pred_final = best_model.predict(X_val)
    final_acc = best_acc
    final_f1 = f1_score(y_val, y_pred_final)
    final_name = best_name

print(f"\n  >>> FINAL MODEL: {final_name}")
print(f"  >>> ACC1       = {final_acc:.6f}")
print(f"  >>> F1 Score   = {final_f1:.6f}")

# ===== 5. Feature Importance & Save =====
print("\n[5/5] Feature importance analysis and saving outputs...")

if hasattr(best_model, "feature_importances_"):
    raw_importances = best_model.feature_importances_
else:
    all_imps = []
    for _, est in best_model.named_estimators_.items():
        if hasattr(est, "feature_importances_"):
            imps = est.feature_importances_
            all_imps.append(imps / imps.sum())  # normalize each first
    raw_importances = np.mean(all_imps, axis=0) if all_imps else np.ones(len(clean_feat_cols))

# Build importance DataFrame (normalized to sum=1)
feat_imp = pd.DataFrame({
    "Feature": clean_feat_cols,
    "Importance": raw_importances / raw_importances.sum()
})
feat_imp = feat_imp.sort_values("Importance", ascending=False).reset_index(drop=True)

print("\n  Top 20 Features (REAL patterns, no time leak):")
for i in range(min(20, len(feat_imp))):
    row = feat_imp.iloc[i]
    print(f"    {i+1:2d}. {row['Feature']:<35s} {row['Importance']:.6f}")

feat_imp.to_csv(os.path.join(TASK1_DIR, "feature_importance.csv"), index=False)

# Also save per-model importance for detailed analysis
per_model_imp = {"Feature": clean_feat_cols}
for name, info in tuned_models.items():
    m = info["model"]
    if hasattr(m, "feature_importances_"):
        imps = m.feature_importances_
        per_model_imp[name] = imps / imps.sum()
pd.DataFrame(per_model_imp).to_csv(os.path.join(TASK1_DIR, "feature_importance_per_model.csv"), index=False)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred_final)
print(f"\n  Confusion Matrix:")
print(f"    TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

print(f"\n  Classification Report:")
print(classification_report(y_val, y_pred_final, target_names=["No", "Yes"]))

# Save predictions
label_map = {0: "No", 1: "Yes"}
predictions_df = pd.DataFrame({
    "Person_ID": ids_val.values,
    "True_Label": pd.Series(y_val).map(label_map).values,
    "Predicted_Label": pd.Series(y_pred_final).map(label_map).values,
})
predictions_df.to_csv(os.path.join(TASK1_DIR, "predictions.csv"), index=False)

# Save model
with open(os.path.join(TASK1_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

# Save metrics
metrics = {
    "ACC1": float(final_acc),
    "F1_Score": float(final_f1),
    "best_model_name": final_name,
    "n_features": len(clean_feat_cols),
    "removed_leaky_features": LEAKY_FEATURES,
    "baseline_results": {
        name: {"val_acc": r["val_acc"], "cv_mean": r["cv_mean"]}
        for name, r in baseline_results.items()
    },
    "tuned_results": {
        name: {
            "val_acc": info.get("val_acc", 0),
            "best_cv": info["best_cv"],
            "best_params": {k: v for k, v in info["best_params"].items()
                            if k not in ["random_state", "eval_metric", "verbosity", "verbose", "force_row_wise"]}
        }
        for name, info in tuned_models.items()
    },
}
with open(os.path.join(TASK1_DIR, "metrics.pkl"), "wb") as f:
    pickle.dump(metrics, f)

# Save human-readable metrics
with open(os.path.join(TASK1_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"ACC1 = {final_acc:.6f}\n")
    f.write(f"F1   = {final_f1:.6f}\n")
    f.write(f"Best Model = {final_name}\n")
    f.write(f"Features used = {len(clean_feat_cols)} (excluded Wake_Up_Time, Sleep_Time)\n\n")
    f.write("--- Baseline Results ---\n")
    for name, r in baseline_results.items():
        f.write(f"  {name}: CV={r['cv_mean']:.4f}±{r['cv_std']:.4f}, Val_Acc={r['val_acc']:.4f}\n")
    f.write("\n--- Tuned Results ---\n")
    for name, r in tuned_models.items():
        params_clean = {k: v for k, v in r["best_params"].items()
                        if k not in ["random_state", "eval_metric", "verbosity", "verbose", "force_row_wise"]}
        f.write(f"  {name}: Best_3CV={r['best_cv']:.4f}, Val_Acc={r.get('val_acc', 0):.4f}, Params={params_clean}\n")
    f.write("\n--- Top 15 Features ---\n")
    for i in range(min(15, len(feat_imp))):
        row = feat_imp.iloc[i]
        f.write(f"  {i+1:2d}. {row['Feature']:<35s} {row['Importance']:.6f}\n")

# Print final summary
print(f"\n{'=' * 70}")
print(f"Task 1 COMPLETED (NO time leakage)!")
print(f"  ACC1 = {final_acc:.6f}")
print(f"  F1   = {final_f1:.6f}")
print(f"  Best Model: {final_name}")
print(f"  Features used: {len(clean_feat_cols)} (excluded Wake_Up_Time, Sleep_Time)")
print(f"  Score contribution (20%): {final_acc * 100 * 0.20:.2f} / 20.00")
print(f"{'=' * 70}")