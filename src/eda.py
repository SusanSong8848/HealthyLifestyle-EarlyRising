"""
Step 2: Exploratory Data Analysis (EDA)
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")   # 设置后端为Agg，这样可以在没有图形界面的服务器上画图并保存（Agg 后端让图片直接保存成 PNG，而不需要弹出窗口。）
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PATH, OUTPUT_DIR

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_EDA = os.path.join(OUTPUT_DIR, "eda")
os.makedirs(OUTPUT_EDA, exist_ok=True)

print("=" * 60)
print("Step 2: Exploratory Data Analysis (EDA)")
print("=" * 60)

# 1. Load data
print("\n[1/7] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")   #Return a tuple：(row and column counts)
print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# 2. Basic info
print("\n[2/7] Data structure overview...")
print(f"\n  Dtype distribution:")
print(df.dtypes.value_counts())

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["string", "object"]).columns.tolist()
print(f"\n  Numeric features: {len(num_cols)}")
print(f"  Categorical features: {len(cat_cols)}")

desc = df.describe().T
desc_path = os.path.join(OUTPUT_EDA, "numeric_describe.csv")
desc.to_csv(desc_path, encoding="utf-8-sig")
print(f"\n  Numeric describe saved to: {desc_path}")

# 3. Missing values
print("\n[3/7] Missing values check...")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({"Missing": missing, "Pct(%)": missing_pct.round(2)})
missing_df = missing_df[missing_df["Missing"] > 0].sort_values("Missing", ascending=False)

if len(missing_df) == 0:
    print("  [OK] No missing values found!")
else:
    print(f"  [WARNING] Features with missing values ({len(missing_df)}):")
    print(missing_df.to_string())
    missing_path = os.path.join(OUTPUT_EDA, "missing_values.csv")
    missing_df.to_csv(missing_path, encoding="utf-8-sig")

# 4. Target variable distributions
print("\n[4/7] Target variable distribution analysis...")
print("  [Task1] Early_Waker distribution:")
ew_counts = df["Early_Waker"].value_counts()
print(f"    Yes: {ew_counts.get('Yes', 0)} ({ew_counts.get('Yes',0)/len(df)*100:.1f}%)")
print(f"    No:  {ew_counts.get('No', 0)} ({ew_counts.get('No',0)/len(df)*100:.1f}%)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

df["Early_Waker"].value_counts().plot.pie(
    autopct="%1.1f%%", ax=axes[0], colors=["#66b3ff", "#ff9999"],
    labels=["No", "Yes"], title="Early_Waker Distribution"
)

print("\n  [Task2] Health_Score distribution:")
print(df["Health_Score"].describe())
axes[1].hist(df["Health_Score"], bins=40, edgecolor="black", alpha=0.7)
axes[1].axvline(df["Health_Score"].median(), color="red", linestyle="--",
                label=f'Median={df["Health_Score"].median():.1f}')
axes[1].set_title("Health_Score Distribution")
axes[1].set_xlabel("Health_Score")
axes[1].legend()

print("\n  [Task3] Wellness_Category distribution:")
wc_counts = df["Wellness_Category"].value_counts()
print(wc_counts.to_string())
axes[2].bar(wc_counts.index, wc_counts.values, color=["#2ecc71", "#f1c40f", "#e74c3c"])
axes[2].set_title("Wellness_Category Distribution")
axes[2].set_ylabel("Count")

plt.tight_layout()
fig_path = os.path.join(OUTPUT_EDA, "target_distribution.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Distribution chart saved: {fig_path}")

# 5. Categorical feature distributions
print("\n[5/7] Categorical feature distributions...")
key_cats = ["Gender", "Country", "Occupation", "Marital_Status",
            "Smoking_Status", "Alcohol_Consumption", "Exercise_Type",
            "Meditation_Practice", "Morning_Workout", "Workout_Intensity",
            "Gym_Member"]
for col in key_cats:
    if col in df.columns:
        vals = df[col].value_counts()
        print(f"  {col}: {len(vals)} categories, top: {vals.index[0]} ({vals.iloc[0]})")

fig2, axes2 = plt.subplots(3, 3, figsize=(20, 16))
axes2 = axes2.ravel()
plot_cats = ["Gender", "Country", "Smoking_Status", "Exercise_Type",
             "Morning_Workout", "Workout_Intensity", "Meditation_Practice",
             "Alcohol_Consumption", "Gym_Member"]
for i, col in enumerate(plot_cats):
    if col in df.columns:
        ct = pd.crosstab(df[col], df["Early_Waker"])
        ct.plot(kind="bar", stacked=True, ax=axes2[i],
                color=["#66b3ff", "#ff9999"])
        axes2[i].set_title(f"{col} vs Early_Waker")
        axes2[i].tick_params(axis="x", rotation=45)
plt.tight_layout()
cat_fig_path = os.path.join(OUTPUT_EDA, "categorical_vs_target.png")
plt.savefig(cat_fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Categorical chart saved: {cat_fig_path}")

# 6. Numeric feature distributions (KDE by target)
print("\n[6/7] Numeric feature KDE distributions...")
key_num_cols = [
    "Age", "BMI", "Sleep_Duration_Hours", "Sleep_Quality_Score",
    "Daily_Steps", "Daily_Calorie_Intake", "Water_Intake_Liters",
    "Stress_Level", "Resting_Heart_Rate", "Health_Score",
    "Energy_Level_Score", "Mood_Score", "Productivity_Score",
    "Healthy_Aging_Score"
]
fig3, axes3 = plt.subplots(4, 4, figsize=(22, 18))
axes3 = axes3.ravel()
for i, col in enumerate(key_num_cols):
    if col in df.columns:
        for ew in ["Yes", "No"]:
            subset = df[df["Early_Waker"] == ew][col]
            subset.plot.kde(ax=axes3[i], label=ew, alpha=0.6)
        axes3[i].set_title(col)
        axes3[i].legend()
for j in range(len(key_num_cols), len(axes3)):
    axes3[j].set_visible(False)
plt.tight_layout()
num_fig_path = os.path.join(OUTPUT_EDA, "numeric_kde_by_target.png")
plt.savefig(num_fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  KDE chart saved: {num_fig_path}")

# 7. Correlation heatmap
print("\n[7/7] Correlation heatmap...")
corr_cols = ["Age", "BMI", "Sleep_Duration_Hours", "Sleep_Quality_Score",
             "Number_of_Night_Awakenings", "Daily_Steps",
             "Exercise_Frequency_Per_Week", "Exercise_Duration_Minutes",
             "Daily_Calorie_Intake", "Water_Intake_Liters",
             "Protein_Intake_Grams", "Stress_Level",
             "Working_Hours_Per_Day", "Sitting_Hours_Per_Day",
             "Outdoor_Time_Hours", "Resting_Heart_Rate",
             "Systolic_BP", "Diastolic_BP", "Cholesterol_Level",
             "Blood_Sugar_Level", "Energy_Level_Score",
             "Fatigue_Level_Score", "Mood_Score", "Anxiety_Score",
             "Depression_Risk_Score", "Productivity_Score",
             "Health_Score", "Healthy_Aging_Score"]
corr_cols = [c for c in corr_cols if c in df.columns]

corr = df[corr_cols].corr()
plt.figure(figsize=(20, 16))
sns.heatmap(corr, annot=False, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_EDA, "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Heatmap saved: {heatmap_path}")

# Health_Score correlation top 10
print("\n  Health_Score (Task2 raw) vs numeric features - Top 10:")
health_corr = df[corr_cols].corr()["Health_Score"].drop("Health_Score").sort_values(ascending=False)
print(health_corr.head(10).to_string())
print("\n  Bottom 5:")
print(health_corr.tail(5).to_string())

# Early_Waker correlation with numeric features
print("\n  Early_Waker (Task1) vs numeric features - correlation (coded Yes=1, No=0):")
df_temp = df.copy()
df_temp["Early_Waker_num"] = df_temp["Early_Waker"].map({"Yes": 1, "No": 0})
ew_corr_cols = [c for c in corr_cols if c != "Health_Score"]
ew_corr = df_temp[ew_corr_cols + ["Early_Waker_num"]].corr()["Early_Waker_num"].drop("Early_Waker_num").sort_values(ascending=False)
print(ew_corr.head(10).to_string())
print("\n  Bottom 5:")
print(ew_corr.tail(5).to_string())

print("\n" + "=" * 60)
print("EDA completed! Results saved to:", OUTPUT_EDA)
print("=" * 60)