"""
HealthyLifestyle_EarlyRising 项目配置文件（统一重构版 v3）
基于 公共数据要求.txt 统一所有参数的"地基"配置。
"""
import os

# ============ 路径配置 ============
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "datas.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

TASK1_DIR = os.path.join(OUTPUT_DIR, "task1")
TASK2_DIR = os.path.join(OUTPUT_DIR, "task2")
TASK3_DIR = os.path.join(OUTPUT_DIR, "task3")
EDA_DIR = os.path.join(OUTPUT_DIR, "eda")

for d in [TASK1_DIR, TASK2_DIR, TASK3_DIR, EDA_DIR, PROCESSED_DIR, SPLITS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============ 随机种子 & 切分配置 ============
RANDOM_STATE = 20260726  # 团队统一随机种子
TEST_SIZE = 0.2
N_FOLDS = 5

# ============ 目标变量 ============
TARGET_TASK1 = "Early_Waker"
TARGET_TASK2 = "Health_Score"
TARGET_TASK3 = "Wellness_Category"
ID_COLUMN = "Person_ID"

# ============ Task 2: Health_Score 离散化边界 ============
# 团队确认：<60 Poor, 60-70 Average, 70-85 Good, >=85 Excellent
HEALTH_SCORE_BINS = [0, 60, 70, 85, 100]
HEALTH_SCORE_LABELS = ["Poor", "Average", "Good", "Excellent"]

# ============ 时间列 ============
TIME_COLUMNS = ["Wake_Up_Time", "Sleep_Time"]

# ============ 统一清洗规则 ============
# Exercise_Frequency_Per_Week=0 → Exercise_Type="No Exercise", Workout_Intensity="No Workout"
# Alcohol_Consumption 缺失 → "Unknown"
STRUCTURAL_FILL = {
    "Exercise_Type": "No Exercise",
    "Workout_Intensity": "No Workout",
}
ALCOHOL_UNKNOWN = "Unknown"

# ============ 各任务泄漏字段（主模型禁止作为特征）============
# Task 1: 时钟相关字段直接/间接定义标签
TASK1_LEAKY_FEATURES = [
    "Person_ID",
    "Early_Waker",
    "Wake_Up_Time",
    "Sleep_Time",
    "Wake_Up_Time_Minutes",
    "Sleep_Time_Minutes",
    "Sleep_Duration_Hours",
]

# Task 2: Health_Score是标签来源；Fitness_Level/Wellness_Category包含由健康评分形成的等级信息
TASK2_LEAKY_FEATURES = [
    "Person_ID",
    "Health_Score",
    "Wellness_Category",
    "Fitness_Level",
]

# Task 3: Wellness_Category是目标；Fitness_Level与目标100%相同；
#         Health_Score按45/65/80分界可100%还原目标
TASK3_LEAKY_FEATURES = [
    "Person_ID",
    "Wellness_Category",
    "Health_Score",
    "Fitness_Level",
]

# Healthy_Aging_Score 属于可疑综合评分，建议主模型先排除，消融实验时再加入
HEALTHY_AGING_SUSPICIOUS = True  # 设为 False 可保留

# ============ 特征分类 ============
DEMOGRAPHIC_FEATURES = [
    "Age", "Gender", "Height_cm", "Weight_kg", "BMI",
    "Country", "Occupation", "Marital_Status"
]

SLEEP_FEATURES = [
    "Wake_Up_Time", "Sleep_Time", "Sleep_Duration_Hours",
    "Sleep_Quality_Score", "Number_of_Night_Awakenings",
    "Weekend_Sleep_Difference_Hours", "Nap_Frequency_Per_Week",
    "Screen_Time_Before_Bed_Hours"
]

EXERCISE_FEATURES = [
    "Exercise_Frequency_Per_Week", "Exercise_Duration_Minutes",
    "Exercise_Type", "Daily_Steps", "Morning_Workout",
    "Workout_Intensity", "Gym_Member"
]

DIET_FEATURES = [
    "Daily_Calorie_Intake", "Water_Intake_Liters",
    "Fruit_Intake_Per_Day", "Vegetable_Intake_Per_Day",
    "Protein_Intake_Grams", "Sugary_Drinks_Per_Week",
    "Fast_Food_Meals_Per_Week", "Breakfast_Regularity_Score"
]

LIFESTYLE_FEATURES = [
    "Smoking_Status", "Alcohol_Consumption", "Stress_Level",
    "Working_Hours_Per_Day", "Sitting_Hours_Per_Day",
    "Outdoor_Time_Hours", "Social_Interaction_Score",
    "Meditation_Practice"
]

PHYSIOLOGICAL_FEATURES = [
    "Resting_Heart_Rate", "Systolic_BP", "Diastolic_BP",
    "Cholesterol_Level", "Blood_Sugar_Level"
]

MENTAL_FEATURES = [
    "Energy_Level_Score", "Fatigue_Level_Score",
    "Immune_Health_Score", "Mood_Score", "Anxiety_Score",
    "Depression_Risk_Score", "Productivity_Score",
    "Focus_Concentration_Score", "Life_Satisfaction_Score"
]

DISEASE_FEATURES = [
    "Obesity_Risk", "Hypertension_Risk", "Diabetes_Risk",
    "Cardiovascular_Risk", "Sleep_Disorder_Risk"
]

FITNESS_FEATURES = ["Fitness_Level", "Healthy_Aging_Score"]

ALL_FEATURES = (
    DEMOGRAPHIC_FEATURES + SLEEP_FEATURES + EXERCISE_FEATURES +
    DIET_FEATURES + LIFESTYLE_FEATURES + PHYSIOLOGICAL_FEATURES +
    MENTAL_FEATURES + DISEASE_FEATURES + FITNESS_FEATURES
)

# ============ 数值特征列表 ============
NUMERIC_COLUMNS = [
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
    "Health_Score", "Healthy_Aging_Score"
]

# ============ 类别特征列表 ============
CATEGORICAL_COLUMNS = [
    "Gender", "Country", "Occupation", "Marital_Status",
    "Exercise_Type", "Morning_Workout", "Workout_Intensity",
    "Gym_Member", "Smoking_Status", "Alcohol_Consumption",
    "Meditation_Practice", "Obesity_Risk", "Hypertension_Risk",
    "Diabetes_Risk", "Cardiovascular_Risk", "Sleep_Disorder_Risk",
    "Fitness_Level", "Wellness_Category"
]