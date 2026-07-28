# 各脚本输出文件一览（详细版）

---

## 一、`src/eda.py` → `outputs/eda/`

EDA 脚本，用于在建模前了解数据基本情况，可跳过。

### 1-6 输出文件

| 文件 | 说明 |
|------|------|
| `numeric_describe.csv` | 所有数值特征的描述性统计（count/mean/std/min/25%/50%/75%/max） |
| `missing_values.csv` | 缺失值统计（Alcohol_Consumption=3014, Exercise_Type=824, Workout_Intensity=824） |
| `target_distribution.png` | 三类目标变量分布（Early_Waker 41.6%Yes, Health_Score 直方图, Wellness_Category 柱状图含114 Poor） |
| `categorical_vs_target.png` | 9个类别特征与 Early_Waker 堆叠柱状图 |
| `numeric_kde_by_target.png` | 14个核心数值特征的 KDE 曲线（按 Early_Waker 分组） |
| `correlation_heatmap.png` | 28个特征相关性热力图 |

---

## 二、`src/preprocess.py` → `data/processed/` + `data/splits/`

### 1. `base_semantic_clean.csv`
| 项目 | 说明 |
|------|------|
| **内容** | 10,000×66，原始 `A题数据集.csv` (64列) + Wake_Up_Time_Minutes + Sleep_Time_Minutes，0缺失 |
| **规则** | Alcohol→"Unknown", Exercise_Type→"No Exercise", Workout_Intensity→"No Workout", 时间→HH:MM |
| **用途** | 编码前的审计版本，人类可读 |

### 2-4. `encoders.pkl` / `processed_data.pkl` / `scaler.pkl`
预处理后的 pickle 文件，供 `task1_early_waker.py`（备选版）使用。

### 5. `split_manifest.csv`
| 项目 | 说明 |
|------|------|
| **内容** | Person_ID + split + task1_label + task2_label + task3_label |
| **切分** | train=8000, val=2000, seed=20260726, 联合分层 |

---

## 三、`src/task1_final.py` → `outputs/task1/`

### 1. `predictions.csv`
2000行：Person_ID + True_Label + Predicted_Label。**ACC1 = 0.7535**（不含时钟特征的真实预测准确率）。

### 2. `feature_importance.csv`
57个特征（排除 Wake_Up_Time/Sleep_Time/Sleep_Duration_Hours/Healthy_Aging_Score）的重要性排名。Top 3: Productivity_Score(11.5%), Breakfast_Regularity_Score(4.9%), Stress_Level。

### 3-4. `best_model.pkl` / `metrics.pkl` / `metrics.txt`
最优模型：**Logistic Regression**（CV 制导选择，否决 Voting Ensemble）。ACC1=0.7535, BalAcc=0.7514, F1(Yes)=0.6842。

---

## 四、`src/task2_health_score.py` → `outputs/task2/`

### 1. `predictions.csv`
2000行：Person_ID + True_Label + Predicted_Label。**ACC2 = 0.8135**。四分类 Poor/Average/Good/Excellent，分箱边界 [0,60,70,85,100]。

### 2. `feature_importance.csv`
56个特征（排除 Health_Score/Wellness_Category/Fitness_Level/Healthy_Aging_Score）排名。Top 3: BMI, Energy_Level_Score, Mood_Score。

### 3-4. `best_model.pkl` / `metrics.txt`
最优模型：**Logistic Regression**（CV 0.8044±0.0084），CV 自动选最优。**得分：32.54/40.00**。

---

## 五、`src/task3_wellness_category.py` → `outputs/task3/`

### 1. `predictions.csv`
2000行：四分类含 Poor。**ACC3 = 0.8170**（class_weight='balanced' 处理 Poor 类 1.1% 极端不平衡）。

### 2. `feature_importance.csv`
56个特征（排除 Wellness_Category/Health_Score/Fitness_Level/Healthy_Aging_Score）排名。Top 3: Energy_Level_Score, BMI, Mood_Score。

### 3-4. `best_model.pkl` / `metrics.txt`
最优模型：**LightGBM (class_weight='balanced')**（CV 0.8174）。LGB 微优于 LR（0.8171）且 Recall(Poor)>0。**得分：32.68/40.00**。

---

## 六、`src/generate_d28.py` → `results/`（D28 新增）

| 文件 | 说明 |
|------|------|
| `results/metrics/raw/task1_leakage_ablation.csv` | 三场景消融：S1_含泄漏(0.9915)→S2_移除直接泄漏(0.7529)→S3_最终版(0.7531) |
| `results/metrics/raw/task1_model_comparison.csv` | 四模型CV对比：LR(0.7531)>LGB(0.7408)>RF(0.7393)>XGB(0.7357) |
| `results/logs/reproduction_candidates.md` | 复现报告（含SHA256+命令+三任务指标） |

---

## 最终总评分

| 任务 | ACC | 权重 | 得分 | 最优模型 |
|------|:---:|:---:|:---:|------|
| Task 1 | 0.7535 | 20% | 15.07 | Logistic Regression |
| Task 2 | 0.8135 | 40% | 32.54 | Logistic Regression |
| Task 3 | 0.8170 | 40% | 32.68 | LightGBM (balanced) |
| **初赛总分** | | | **80.25 / 100.00** | |