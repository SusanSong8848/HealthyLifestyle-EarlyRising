"""
HealthyLifestyle_EarlyRising 项目配置文件
"""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # Project root directory name(abspath): eg.D:\ECNU\MathematicalModeling\HealthyLifestyle_EarlyRising 
DATA_PATH = os.path.join(ROOT_DIR, "datas.csv")     # os.path.join : ROOT_DIR + "datas.csv", eg. D:\ECNU\MathematicalModeling\HealthyLifestyle_EarlyRising\src
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

TASK1_DIR = os.path.join(OUTPUT_DIR, "task1")
TASK2_DIR = os.path.join(OUTPUT_DIR, "task2")
TASK3_DIR = os.path.join(OUTPUT_DIR, "task3")

for d in [TASK1_DIR, TASK2_DIR, TASK3_DIR]:
    os.makedirs(d, exist_ok=True)       #"exist_ok=True" indicates that no error will be raised if the file already exists.

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

TARGET_TASK1 = "Early_Waker"
TARGET_TASK2 = "Health_Score"
TARGET_TASK3 = "Wellness_Category"

HEALTH_SCORE_BINS = [0, 50, 65, 80, 100]
HEALTH_SCORE_LABELS = ["Poor", "Average", "Good", "Excellent"]

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

TIME_COLUMNS = ["Wake_Up_Time", "Sleep_Time"]

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

CATEGORICAL_COLUMNS = [
    "Gender", "Country", "Occupation", "Marital_Status",
    "Exercise_Type", "Morning_Workout", "Workout_Intensity",
    "Gym_Member", "Smoking_Status", "Alcohol_Consumption",
    "Meditation_Practice", "Obesity_Risk", "Hypertension_Risk",
    "Diabetes_Risk", "Cardiovascular_Risk", "Sleep_Disorder_Risk",
    "Fitness_Level", "Wellness_Category"
]

ID_COLUMN = "Person_ID"