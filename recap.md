# 各脚本输出文件一览（详细版）

---

## 一、`src/eda.py` → `outputs/eda/`

EDA 脚本，用于在建模前了解数据基本情况，可跳过。

### 1-6 输出文件

| 文件 | 说明 |
|------|------|
| **是什么** | 缺失值统计表 |
| **内容** | 每个有缺失值的列名、缺失数量、缺失百分比 |
| **用途** | 确定哪些特征需要填充。本项目原始数据只有 Alcohol_Consumption（3014条）、Exercise_Type（824条）、Workout_Intensity（824条）有缺失 |
| **示例** | `Alcohol_Consumption: 3014 (30.14%)` |

### 3. `target_distribution.png`
| 项目 | 说明 |
|------|------|
| **是什么** | 三类目标变量的可视化分布图 |
| **内容** | 左：Early_Waker 饼图（Yes/No 占比）；中：Health_Score 直方图（连续值分布）；右：Wellness_Category 柱状图（各类计数） |
| **用途** | 了解类别平衡性。Early_Waker 约 41.6% Yes / 58.4% No，基本平衡；Wellness_Category 中 Poor 仅 114 条（1.1%），提示任务3必须比较无权重与 `class_weight='balanced'`，并重点检查 Poor Recall |

### 4. `categorical_vs_target.png`
| 项目 | 说明 |
|------|------|
| **是什么** | 9个类别特征与 Early_Waker 目标的堆叠柱状图 |
| **内容** | 横轴是每个类别特征的取值，纵轴是样本数，颜色区分 Early_Waker=Yes/No 的堆叠比例 |
| **用途** | 观察哪些类别特征与早起行为相关。例如 Gender 中 Yes/No 比例差异较小，而 Smoking_Status 可能呈现一定差异 |

### 5. `numeric_kde_by_target.png`
| 项目 | 说明 |
|------|------|
| **是什么** | 14个核心数值特征的核密度估计（KDE）曲线图 |
| **内容** | 每条曲线是一个特征在 Early_Waker=Yes（蓝色）和 No（橙色）两组的概率密度分布。两曲线重叠越多，说明该特征区分早起/非早起的能力越弱 |
| **用途** | 定性判断哪些数值特征对分类有帮助。如 Productivity_Score 两条曲线明显分离（早起者生产力更高），而 Age 两条曲线几乎重叠（年龄与早起无关） |

### 6. `correlation_heatmap.png`
| 项目 | 说明 |
|------|------|
| **是什么** | 28个特征的相关性热力图 |
| **内容** | 每格颜色表示两个特征之间的皮尔逊相关系数。红色=正相关，蓝色=负相关，白色≈0 |
| **用途** | 发现强相关特征对（如 Height_cm 与 Weight_kg 正相关、Energy_Level_Score 与 Fatigue_Level_Score 负相关）以及各特征与 Health_Score 的关联强度。同时可以看出 Wake_Up_Time_Minutes 与 Sleep_Time_Minutes 高度共线（r ≈ -0.82）——这也是把它们排除的原因 |

---

## 二、`src/preprocess.py` → `data/processed/` + `data/splits/`

预处理脚本，执行统一语义清洗并生成共享切分。**是所有后续任务的前置依赖**。任务脚本（task1_final.py、task2_health_score.py、task3.py）读取同一份 `split_manifest.csv`；其中任务3的编码、缺失填补和标准化在模型 Pipeline 内按训练折拟合。

> `processed_data.pkl`、`encoders.pkl`、`scaler.pkl` 是 Task 1/2 与旧代码的兼容产物；Task 3 不读取这三项。

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

## 五、`src/task3.py` → 统一结果目录

> v3.1 的 `ACC3=0.8160` 和 LightGBM 最优仅为历史记录。Task 3 v3.4 已完成并冻结：最终模型为无权重逻辑回归（`C=1`），五折 CV Accuracy=0.8481，验证集 `ACC3=0.8485`。

### 1. 基线结果

| 文件 | 说明 |
|------|------|
| `results/metrics/raw/task3_baseline.csv` | 两行基线：Dummy Most Frequent 与 Logistic Regression (Unweighted)。包含 CV/验证 Accuracy、Macro-F1、Balanced Accuracy 和四类 Recall |
| `results/figures/baseline/task3_confusion_matrix.png` | 无权重逻辑回归在共享验证集上的四分类混淆矩阵 |
| `results/figures/baseline/task3_confusion_matrix.csv` | 与图片对应的精确计数，便于论文核对 |

### 2. 模型比较与类别权重消融

| 文件 | 说明 |
|------|------|
| `results/metrics/raw/task3_model_comparison.csv` | 全部候选模型的五折 CV 与验证指标；只允许一行 `Selected=True` |
| `results/metrics/raw/task3_tuning.csv` | 逻辑回归 `C∈{0.1,0.3,1,3,10}` 的训练集五折搜索结果；不使用验证集选参 |
| `results/metrics/raw/task3_weight_ablation.csv` | 固定 LightGBM 其他参数，只比较 `class_weight=None` 与 `"balanced"`；保存两组原始指标及相对无权重组的差值 |
| `results/metrics/raw/task3_classification_report.csv` | 选中模型的 Poor/Average/Good/Excellent precision、recall、F1 |
| `results/metrics/raw/task3_metrics.json` | 最终模型名、ACC3、Macro-F1、Balanced Accuracy、四类 Recall、样本数、seed、折数和特征数的机器可读摘要 |
| `results/metrics/raw/task3_run_complete.json` | 仅在全部结果成功保存后生成的完成标记 |

### 3. 图、预测与模型

| 文件 | 说明 |
|------|------|
| `results/figures/candidate/task3_confusion_matrix.png` | CV 选中模型在共享验证集上的混淆矩阵 |
| `results/figures/candidate/task3_confusion_matrix.csv` | 与候选模型混淆矩阵图片对应的精确计数 |
| `results/figures/candidate/task3_feature_importance.png` | 选中模型 Top 20 特征重要性图 |
| `results/metrics/raw/task3_feature_importance.csv` | 56 个原始输入字段的验证集置换重要性；以打乱字段后的 Accuracy 平均下降量衡量，只解释预测关联 |
| `results/metrics/raw/task3_features_used.csv` | 实际进入 Task 3 主模型的原始字段及数值/类别类型 |
| `results/predictions/candidate/task3_predictions.csv` | 本地验证文件：`Person_ID + True_Label + Predicted_Label`，2000 行；包含逐人信息，不上传公开仓库 |
| `models/candidate/task3/task3_best_model.pkl` | 包含填补、One-Hot、标准化和分类器的完整 Pipeline |

### 4. 论文引用规则

1. ACC3、Macro-F1、Balanced Accuracy 和四类 Recall：读取 `task3_model_comparison.csv` 中唯一的 `Selected=True` 行；当前分别为 0.8485、0.7832、0.7652，Poor Recall 为 0.5217。
2. 类别权重结论：读取 `task3_weight_ablation.csv`，不能只看 Accuracy，必须同时比较 Macro-F1 与 Poor Recall。
3. 编码后特征数：读取 `task3_metrics.json` 的 `transformed_features_used`，不写死。
4. 任何运行后手工改动的数字都不能作为正式结果；结果变化时重新运行脚本并同步更新论文。
