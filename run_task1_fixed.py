"""Task 1 runner — NO Wake_Up_Time / Sleep_Time leakage"""
import pickle, numpy as np, pandas as pd, os, sys, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb, lightgbm as lgb

ROOT = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.join(ROOT, "outputs", "task1")
PREP_DIR = os.path.join(ROOT, "outputs", "preprocess")

print("=" * 60)
print("Task 1 — NO Wake_Up_Time / Sleep_Time (leak-free)")
print("=" * 60)

# Load
with open(os.path.join(PREP_DIR, "processed_data.pkl"), "rb") as f:
    data = pickle.load(f)

feat_all = data["feat_cols"]
LEAKY = ["Wake_Up_Time_Minutes", "Sleep_Time_Minutes"]
feat_clean = [c for c in feat_all if c not in LEAKY]

X_tr = pd.DataFrame(data["X_train"], columns=feat_all)[feat_clean].values
X_va = pd.DataFrame(data["X_val"], columns=feat_all)[feat_clean].values
y_tr = data["y1_train"].astype(int).values
y_va = data["y1_val"].astype(int).values
ids  = data["ids_val"]

print(f"Train: {X_tr.shape}, Val: {X_va.shape}, Features: {len(feat_clean)}")
print(f"y_train: No={sum(y_tr==0)}, Yes={sum(y_tr==1)}")
print(f"y_val:   No={sum(y_va==0)}, Yes={sum(y_va==1)}")
print()

# Baseline
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {}

for name, m in [
    ("Logistic Regression", LogisticRegression(max_iter=3000, random_state=42)),
    ("Random Forest",       RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)),
    ("XGBoost",             xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss", verbosity=0)),
    ("LightGBM",            lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)),
]:
    cv_s = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy")
    m.fit(X_tr, y_tr)
    yp = m.predict(X_va)
    acc = accuracy_score(y_va, yp)
    f1  = f1_score(y_va, yp)
    models[name] = {"model": m, "cv_mean": float(cv_s.mean()), "cv_std": float(cv_s.std()), "val_acc": float(acc), "val_f1": float(f1)}
    print(f"  {name:25s}  CV={cv_s.mean():.4f}  Val_Acc={acc:.4f}  F1={f1:.4f}")

# Tune (fast: 2 combos each)
print()
print("Tuning (fast 2-param search)...")

# XGB
bx, bp = 0, None
for p in [{"n_estimators":200,"max_depth":6,"learning_rate":0.1},{"n_estimators":300,"max_depth":8,"learning_rate":0.05}]:
    s = cross_val_score(xgb.XGBClassifier(**p, random_state=42, eval_metric="logloss", verbosity=0), X_tr, y_tr, cv=3, scoring="accuracy").mean()
    if s > bx: bx, bp = s, p
mx = xgb.XGBClassifier(**bp, random_state=42, eval_metric="logloss", verbosity=0).fit(X_tr, y_tr)

# LGB
bl, lp = 0, None
for p in [{"n_estimators":200,"max_depth":6,"learning_rate":0.1,"num_leaves":31},{"n_estimators":300,"max_depth":8,"learning_rate":0.05,"num_leaves":63}]:
    s = cross_val_score(lgb.LGBMClassifier(**p, random_state=42, verbose=-1), X_tr, y_tr, cv=3, scoring="accuracy").mean()
    if s > bl: bl, lp = s, p
ml = lgb.LGBMClassifier(**lp, random_state=42, verbose=-1).fit(X_tr, y_tr)

# RF
br, rp = 0, None
for p in [{"n_estimators":200,"max_depth":10},{"n_estimators":500,"max_depth":None}]:
    s = cross_val_score(RandomForestClassifier(**p, random_state=42), X_tr, y_tr, cv=3, scoring="accuracy").mean()
    if s > br: br, rp = s, p
mr = RandomForestClassifier(**rp, random_state=42).fit(X_tr, y_tr)

print(f"  XGBoost:     {bp}  CV={bx:.4f}")
print(f"  LightGBM:    {lp}  CV={bl:.4f}")
print(f"  RandomForest:{rp}  CV={br:.4f}")

# Ensemble
ens = VotingClassifier([("xgb", mx), ("lgb", ml), ("rf", mr)], voting="soft").fit(X_tr, y_tr)
yp_ens = ens.predict(X_va)
ens_acc = accuracy_score(y_va, yp_ens)
ens_f1  = f1_score(y_va, yp_ens)
print(f"\n  Voting Ensemble: Val_Acc={ens_acc:.4f}, F1={ens_f1:.4f}")

# Pick final
best_model = ens
y_pred_final = yp_ens
final_acc = ens_acc
final_f1  = ens_f1
final_name = "Voting Ensemble (XGB+LGB+RF)"

print(f"\n  >>> ACC1 = {final_acc:.6f}")
print(f"  >>> F1   = {final_f1:.6f}")

# Feature importance (avg across 3 models, normalized)
imps = np.mean([mx.feature_importances_, ml.feature_importances_, mr.feature_importances_], axis=0)
imps = imps / imps.sum()
feat_imp = pd.DataFrame({"Feature": feat_clean, "Importance": imps}).sort_values("Importance", ascending=False)

print("\n  Top 15 Features (REAL patterns):")
for i in range(min(15, len(feat_imp))):
    row = feat_imp.iloc[i]
    print(f"    {i+1:2d}. {row['Feature']:<35s} {row['Importance']:.6f}")

# Confusion matrix
cm = confusion_matrix(y_va, y_pred_final)
print(f"\n  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
print(f"\n  Classification Report:")
print(classification_report(y_va, y_pred_final, target_names=["No", "Yes"]))

# ==== SAVE ====
label_map = {0: "No", 1: "Yes"}
pd.DataFrame({
    "Person_ID": ids.values,
    "True_Label": pd.Series(y_va).map(label_map).values,
    "Predicted_Label": pd.Series(y_pred_final).map(label_map).values,
}).to_csv(os.path.join(TASK1_DIR, "predictions.csv"), index=False)

feat_imp.to_csv(os.path.join(TASK1_DIR, "feature_importance.csv"), index=False)
pickle.dump(best_model, open(os.path.join(TASK1_DIR, "best_model.pkl"), "wb"))
pickle.dump({"ACC1": float(final_acc), "F1_Score": float(final_f1), "best_model_name": final_name,
             "n_features": len(feat_clean), "removed_leaky_features": LEAKY},
            open(os.path.join(TASK1_DIR, "metrics.pkl"), "wb"))

with open(os.path.join(TASK1_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"ACC1 = {final_acc:.6f}\n")
    f.write(f"F1   = {final_f1:.6f}\n")
    f.write(f"Best Model = {final_name}\n")
    f.write(f"Features  = {len(feat_clean)} (excluded: {', '.join(LEAKY)})\n\n")
    f.write("--- Baseline Results ---\n")
    for name, r in models.items():
        f.write(f"  {name:25s} CV={r['cv_mean']:.4f} +/- {r['cv_std']:.4f}, Val_Acc={r['val_acc']:.4f}\n")
    f.write("\n--- Top 15 Features ---\n")
    for i in range(min(15, len(feat_imp))):
        row = feat_imp.iloc[i]
        f.write(f"  {i+1:2d}. {row['Feature']:<35s} {row['Importance']:.6f}\n")

print(f"\n{'=' * 60}")
print(f"Done! ACC1 = {final_acc:.6f}, Files saved to: {TASK1_DIR}")
print(f"{'=' * 60}")