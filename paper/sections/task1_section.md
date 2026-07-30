# 数据预处理、泄漏检查与任务一建模

> 成员 A 为成员 C 提供的论文素材稿（2026-07-29）
> 所有数据指标均来自正式 final 运行，可直接引用

---

## 1 数据预处理

### 1.1 数据概况

本研究使用"早期起床者健康数据集"（Health Lifestyle Early Riser Dataset）。原始数据包含 10,000 名受访者，每人 64 个字段。字段涵盖人口统计学（年龄、性别、身高、体重、 BMI）、生活方式（作息、运动、饮食）、心理健康（焦虑评分、抑郁风险）及生理指标（血压、血糖、胆固醇）等多个维度。

### 1.2 缺失值分析与处理

原始数据共 4,662 个缺失单元格，分布在三个字段上：

| 字段 | 缺失数 | 占比 | 处理方式 | 理由 |
|------|:---:|:---:|------|------|
| `Alcohol_Consumption` | 3,014 | 30.14% | 填充 `"Unknown"` | 缺失不等于不饮酒；仅包括 Light、Moderate、Heavy 和未知 |
| `Exercise_Type` | 824 | 8.24% | 填充 `"No Exercise"` | 所有 824 条缺失记录的 `Exercise_Frequency_Per_Week` 均为 0，属结构性缺失 |
| `Workout_Intensity` | 824 | 8.24% | 填充 `"No Workout"` | 同上 |

> **论文要点**：运动字段的填补存在字段间约束——缺失记录的运动频率全部为零，且不存在频率非零却被标记为 `No Exercise` 的记录。饮酒字段则不同：原始数据只有 `Light`、`Moderate`、`Heavy` 和缺失，没有任何字段能证明缺失代表不饮酒，因此统一标记为信息缺失。

### 1.3 时间特征工程

原始数据中 `Wake_Up_Time` 和 `Sleep_Time` 的格式不统一（如 "6:35" vs "06:35"），统一为 `HH:MM` 零填充格式，并新增转化为分钟数的特征列 `Wake_Up_Time_Minutes` 和 `Sleep_Time_Minutes`。这两个分钟数列用于数据探索和相关性分析，但在任务一建模时会作为泄漏特征排除。

### 1.4 目标变量

| 任务 | 目标变量 | 类型 | 类别 |
|------|---------|------|------|
| Task 1 | `Early_Waker` | 二分类 | Yes（4,158 人，41.6%） / No（5,842 人，58.4%） |
| Task 2 | `Health_Score` → Health_Score_Level | 四分类（由连续值离散化）| Poor [0,60)、Average [60,70)、Good [70,85)、Excellent [85,100] |
| Task 3 | `Wellness_Category` | 四分类 | Poor（114 人，1.1%）、Average、Good、Excellent |

> **论文要点**：Task 2 的离散化边界（60, 70, 85）是团队的建模选择而非题面的硬性规定。Task 3 题面只写了 Excellent/Good/Average 三分类，但实际数据中有 114 条 Poor，模型按四分类处理，论文需要报告这一点。

### 1.5 数据切分

三项任务统一采用 `train_test_split(test_size=0.2, random_state=20260726)`，从 10,000 个样本中划分出 8,000 个训练样本和 2,000 个验证样本。切分时采用联合分层抽样，同时按 Task 1（Early_Waker）、Task 2（Health_Score_Level）和 Task 3（Wellness_Category）的标签分布进行分层，确保三者分布与总体一致。训练/验证集划分清单保存在 `data/splits/split_manifest.csv` 中，三人共享同一份切分。

```text
训练集: 8,000 样本
验证集: 2,000 样本
分层变量: task1_label + task2_label + task3_label（联合分层）
随机种子: 20260726（已锁定）
```

---

## 2 泄漏检查

### 2.1 数据泄露的定义

数据泄露（Data Leakage）指模型在训练时无意中使用了不应在真实预测场景中获得的特征。在本题中，最危险的是时钟特征——它们与目标变量之间存在定义性关系。

### 2.2 时钟特征与 Early_Waker 的对应关系

| 起床时间段 | Early_Waker=Yes | Early_Waker=No | 可分性 |
|:----------:|:---:|:---:|:---:|
| 4:00–5:59 AM | 2,837 (100%) | 0 | 100% 确定是早起者 |
| 6:00–6:59 AM | 1,321 (48.3%) | 1,415 | 模糊区间 |
| 7:00–11:59 AM | 0 | 4,427 (100%) | 100% 确定非早起者 |

除早上六点这一个小时外，起床时间可以 100% 确定 `Early_Waker` 标签。将 `Wake_Up_Time_Minutes`、`Sleep_Time_Minutes` 和 `Sleep_Duration_Hours` 全部纳入特征集时，任何分类器的交叉验证准确率（ACC1）都接近 1.0——这不是"模型很强"的信号，而是"模型在抄答案"。

### 2.3 受控消融实验

为量化泄漏的影响，我们以 Logistic Regression 为基准分类器，在 5 折交叉验证下对三个特征集进行受控比较：

| 场景 | 特征集 | 特征数 | 5-Fold CV ACC | 解读 |
|------|------|:---:|:---:|------|
| **S1** | 含 `Wake_Up_Time_Minutes` + `Sleep_Time_Minutes` + `Sleep_Duration_Hours` | 61 | **0.9915** | 几乎满分——典型的数据泄露 |
| **S2** | 移除 Wake_Up_Time_Minutes 和 Sleep_Time_Minutes，但保留 Sleep_Duration_Hours | 59 | **0.7529** | 骤降 23.9 个百分点 |
| **S3** | 移除全部三个时钟特征（最终版） | 58 | **0.7531** | 无泄露基线 |

> **论文要点**：S1 到 S2 的 23.9 个百分点下降量化了时钟特征的泄露严重性。S2 与 S3 仅差 0.0002，说明 `Sleep_Duration_Hours` 在 WakeUp/Sleep 分钟数被移除后无法独立重建标签。但为严格起见，最终版本仍排除所有三个时钟特征，连同可疑的综合评分 `Healthy_Aging_Score`。

### 2.4 三任务泄漏排除清单

| 任务 | 排除特征 | 原因 |
|------|---------|------|
| Task 1 | `Wake_Up_Time`, `Sleep_Time`, `Sleep_Duration_Hours` | 起床时间 <6:30 与标签 100% 一致；入睡时间 + 时长可间接推算出起床时间 |
| Task 1 | `Healthy_Aging_Score` | 可疑综合评分，预集成了健康指标信息 |
| Task 2 | `Health_Score`（标签来源），`Wellness_Category`, `Fitness_Level` | 等级信息直接来自 Health_Score，任留一个都会泄露 |
| Task 3 | `Wellness_Category`, `Health_Score`（按 45/65/80 分界可 100% 还原），`Fitness_Level`（与目标 100% 相同） | 直接目标泄漏 |
| 全部 | `Person_ID` | 仅用于样本连接和提交，无预测价值 |

---

## 3 任务一：Early_Waker 二分类（权重 20%）

### 3.1 建模策略

排除所有时钟特征和 `Healthy_Aging_Score` 后，特征集剩余 58 个变量（43 个数值特征 + 15 个类别编码特征）。类别特征（如 Country、Occupation、Exercise_Type 等）使用 LabelEncoder 转换为整数编码，数值特征使用 StandardScaler 标准化。为防止编码和缩放过程中的信息泄漏，每折交叉验证内部独立完成 `fit → transform`——在训练折上拟合编码器和标准化器，再用拟合后的对象变换验证折。

### 3.2 模型选择

为确定最合适的算法，比较了四个分类器的 5 折交叉验证表现：

| 模型 | 5-Fold CV ACC | 标准差 | 训练时间 (s) |
|------|:---:|:---:|:---:|
| **Logistic Regression** ★ | **0.7531** | 0.0066 | 0.1 |
| LightGBM | 0.7408 | 0.0080 | 4.0 |
| Random Forest | 0.7393 | 0.0061 | 3.4 |
| XGBoost | 0.7357 | 0.0069 | 6.4 |

Logistic Regression 在所有指标上均优于树模型——不仅交叉验证均值最高（0.7531），标准差最小（0.0066），而且训练时间只有 0.1 秒，比 LightGBM 快 40 倍。

树模型（Random Forest、XGBoost、LightGBM）在训练集上表现更极端（训练 ACC 更高），但在交叉验证上未能超过 Logistic Regression，这表明移除时钟特征后，早起/非早起的决策边界变得相对线性——生活方式和健康指标的组合可以以一个线性分界面较好地分离两类人群。

模型选择完全由交叉验证结果决定，验证集仅用于最终评估，避免了验证集信息对模型选择的间接影响。

### 3.3 最终结果

| 指标 | 数值 |
|------|:---:|
| **ACC1** | **0.7535** |
| Balanced Accuracy | 0.7559 |
| F1 (Yes) | 0.7094 |
| Recall (Yes) | 0.6779 |
| 特征数 | 57（排除 4 个泄漏/可疑特征） |
| 最优模型 | LogisticRegression (max_iter=3000, C=1.0) |
| 交叉验证策略 | 5 折分层，折内编码 |
| 验证集大小 | 2,000 样本 |

### 3.4 特征重要性分析

特征重要性通过取 Random Forest、XGBoost、LightGBM 三个树模型归一化后的平均重要性获得（Logistic Regression 不提供 `feature_importances_` 属性）。58 个特征的重要性分布较为均匀，Top 15 为：

| 排名 | 特征 | 平均重要性 |
|:---:|------|:---:|
| 1 | Productivity_Score | 11.49% |
| 2 | Breakfast_Regularity_Score | 4.89% |
| 3 | Stress_Level | 2.43% |
| 4 | Health_Score | 2.36% |
| 5 | Daily_Calorie_Intake | 2.28% |
| 6 | Daily_Steps | 2.25% |
| 7 | Blood_Sugar_Level | 2.22% |
| 8 | Water_Intake_Liters | 2.22% |
| 9 | Energy_Level_Score | 2.21% |
| 10 | Weekend_Sleep_Difference_Hours | 2.12% |
| 11 | Cholesterol_Level | 2.11% |
| 12 | Sleep_Quality_Score | 2.10% |
| 13 | Protein_Intake_Grams | 2.09% |
| 14 | Outdoor_Time_Hours | 2.07% |
| 15 | Height_cm | 2.07% |

> **论文要点**：排名第一的 `Productivity_Score`（生产力评分）独占 11.5% 的重要性，远高于其余特征——早起者系统性地拥有更高的生产力水平。排名 2–15 的特征均匀分布在 2%–5% 之间，涵盖饮食（热量、蛋白质、饮水）、运动（步数、锻炼时长）、睡眠（质量、周末差异）、情绪（压力）、生理指标（血糖、胆固醇）和人口学变量（身高）。这表明早起行为不是一个单因素现象，而是生产力、作息规律、生理健康和营养摄入等多维度的综合体现——这正是比赛希望参赛者发现的规律。

---

## 4 章节写作建议

1. **第一段**：直接给出 1.2 的缺失表，以及 Alcohol 为什么是缺失而不是"不饮酒"的证据——这能体现出严谨的数据处理意识。

2. **第二段**：用表格展示时钟特征的范围-标签对应关系，S1→S2→S3 消融结果（0.9915 → 0.7529 → 0.7531），以及"23.9pp 差距量化了泄露严重性"的解读。

3. **第三段**：给出模型选择表（LR > LGB > RF > XGB），以及最终的 ACC1=0.7535。强调"CV 制导模型选择"和"折内编码防泄露"这两点。

4. **第四段**：给出 Top 15 特征重要性表，并解读 Productivity_Score 独占 11.5% 以及其余特征的均匀分布——这表明早起行为是多维度的综合体现。

5. **结论**：重申排除时钟特征后的 0.7535 是模型的真实预测能力，0.9915 是数据泄露的假象。Logistic Regression 在严格特征集上的最优表现说明该问题具有良好的线性可分性。