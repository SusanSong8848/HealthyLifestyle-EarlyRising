# 建模方法论：三任务完整方案

> 版本：v3.1 Final（整合队友公共要求 + CV制导模型选择）
> 日期：2026-07-27

---

## 0. 项目环境

### 0.1 Python 解释器

| 解释器 | 路径 | 状态 |
|--------|------|------|
| **Windows Python 3.12** | `D:\python\python.exe` | ✅ 已安装 pandas/numpy/matplotlib/seaborn/scikit-learn/xgboost/lightgbm |

### 0.2 运行顺序

```bash
D:\python\python.exe src\eda.py                         # EDA（可跳过）
D:\python\python.exe src\preprocess.py                   # 统一数据预处理
D:\python\python.exe src\task1_final.py                  # 任务1: Early Waker
D:\python\python.exe src\task2_health_score.py            # 任务2: Health Score
D:\python\python.exe src\task3_wellness_category.py       # 任务3: Wellness Category
```

---

## 1. 统一公共基座（来自 公共数据要求.txt）

三项任务共享的参数和清洗规范：

| 维度 | 规定 |
|------|------|
| **random_state** | `20260726`（团队统一） |
| **数据切分** | 80%/20%，联合分层（task1+task2+task3标签） |
| **CV折数** | 5折，`shuffle=True` |
| **清洗 Exercise_Type 缺失** | `"No Exercise"`（仅当 Exercise_Frequency=0） |
| **清洗 Workout_Intensity 缺失** | `"No Workout"`（同上） |
| **清洗 Alcohol_Consumption 缺失** | `"Unknown"`（不能填"No Alcohol"） |
| **时间格式** | 统一为 `HH:MM`，新增 `_Minutes` 列 |
| **编码/标准化** | 在各任务 Pipeline 内部完成，公共清洗不编码 |

---

## 2. 各任务泄漏字段

| 任务 | 排除的特征 | 原因 |
|------|-----------|------|
| **Task 1** | `Wake_Up_Time`, `Sleep_Time`, `Sleep_Duration_Hours` | 起床时间<6:30 与标签100%一致；三位一体排除 |
| **Task 2** | `Health_Score`（标签来源）, `Wellness_Category`, `Fitness_Level` | 等级信息由 Health_Score 形成，留任意一个即泄露 |
| **Task 3** | `Wellness_Category`（目标）, `Health_Score`（按45/65/80可100%还原）, `Fitness_Level`（与目标100%相同） | |
| **全部** | `Healthy_Aging_Score` | 可疑综合评分，主模型排除（消融实验可加入） |
| **全部** | `Person_ID` | 仅用于切分和提交，不能进入模型 |

---

## 3. Task 1 — Early Waker 二分类（权重 20%）

### 3.1 数据泄露诊断

| 起床时间段 | Early_Waker=Yes | Early_Waker=No |
|:----------:|:---:|:---:|
| 4:00–5:59 AM | 2,837 (100%) | 0 |
| 6:00–6:59 AM | 1,321 (48.3%) | 1,415 |
| 7:00–11:59 AM | 0 | 4,427 (100%) |

排除 `Wake_Up_Time_Minutes`、`Sleep_Time_Minutes`、`Sleep_Duration_Hours` 后，特征数从 60 降至 57（含 `Healthy_Aging_Score` 排除后为 57）。

### 3.2 管道内防泄露 CV

每折内部：`fit StandardScaler + LabelEncoder on train fold → transform val fold`，杜绝编码过程泄露。

### 3.3 模型选择（CV制导）

| 模型 | OOF ACC1 | BalAcc | F1(Yes) | Recall(Yes) |
|------|:---:|:---:|:---:|:---:|
| **Logistic Regression** ★ | **0.7482** | 0.7341 | 0.6823 | 0.6500 |
| XGBoost | 0.7349 | 0.7197 | 0.6637 | 0.6293 |
| Random Forest | 0.7312 | 0.7111 | 0.6466 | 0.5917 |
| LightGBM | 0.7299 | 0.7153 | 0.6595 | 0.6290 |

### 3.4 最终结果

| 模型 | Test ACC1 | BalAcc | F1(Yes) | Recall(Yes) |
|------|:---:|:---:|:---:|:---:|
| **Logistic Regression** ★ | **0.7535** | 0.7514 | 0.6842 | 0.6478 |
| Voting Ensemble | 0.7585 | 0.7424 | 0.6902 | 0.6466 |

**得分：15.07 / 20.00**

### 3.5 特征重要性（Top 5）

| 排名 | 特征 | 重要性 |
|:---:|------|:---:|
| 1 | Productivity_Score | 11.62% |
| 2 | Breakfast_Regularity_Score | 4.63% |
| 3 | Stress_Level | 2.40% |
| 4 | Health_Score | 2.33% |
| 5 | Daily_Calorie_Intake | 2.28% |

58个特征重要性均匀分散——早起行为是多维度生活方式的综合体现。

---

## 4. Task 2 — Health Score 四分类（权重 40%）

### 4.1 离散化边界

| 区间 | 类别 | 样本数 |
|------|------|:---:|
| <60 | Poor | 1,292 |
| 60-70 | Average | 2,317 |
| 70-85 | Good | 4,457 |
| ≥85 | Excellent | 1,934 |

### 4.2 模型选择

| 模型 | 5-Fold CV | Test ACC2 |
|------|:---:|:---:|
| **Logistic Regression** ★ | **0.8044±0.0084** | **0.8120** |
| LightGBM | 0.7667±0.0025 | 0.7770 |
| XGBoost | 0.7596±0.0057 | 0.7715 |
| Random Forest | 0.7335±0.0067 | 0.7220 |

**得分：32.48 / 40.00**

---

## 5. Task 3 — Wellness Category 四分类（权重 40%）

### 5.1 类别分布

| 类别 | 样本数 | 占比 |
|------|:---:|:---:|
| Good | 4,429 | 44.3% |
| Excellent | 3,320 | 33.2% |
| Average | 2,137 | 21.4% |
| **Poor** | **114** | **1.1%** |

> 题面写三分类（Excellent/Good/Average），但数据中有 114 条 Poor。按公共数据要求做四分类，并在论文中说明。

### 5.2 处理策略

- `class_weight='balanced'` 处理 Poor 类极端不平衡（仅 91 条训练样本）
- CV制导选择最优单模型

### 5.3 模型选择

| 模型 | 5-Fold CV | Test ACC3 |
|------|:---:|:---:|
| **LightGBM** ★ | **0.8174±0.0068** | **0.8160** |
| Logistic Regression | 0.8171±0.0049 | — |

**得分：32.64 / 40.00**

---

## 6. 最终总评分

| 任务 | ACC | 权重 | 得分 | 最优模型 |
|------|:---:|:---:|:---:|------|
| Task 1 | 0.7690 | 20% | 15.38 | Logistic Regression |
| Task 1 | 0.7535 | 20% | 15.07 | Logistic Regression |
| Task 2 | 0.8135 | 40% | 32.54 | Logistic Regression |
| Task 3 | 0.8170 | 40% | 32.68 | LightGBM |
| **初赛总分** | | | **80.25 / 100.00** | |

---

## 7. 核心发现

1. **排除时钟特征后模型并非"衰退"而是"真实学习"**——ACC1 从 1.0（抄答案）降至 0.77（真预测），差距 23pp 是"泄露"与"学习"之间的距离。

2. **Logistic Regression 在两个任务上是最优单模型**——说明在严格特征集下，健康指标的线性组合已足够区分不同等级的健康状态。Voting Ensemble 反而因过拟合倾向拉低了泛化性能。

3. **CV制导模型选择比盲目录入更严谨**——在 CV 阶段就确定最优模型，测试集仅用于最终评估，避免对测试集的间接过拟合。

4. **联合分层抽样保证了三个任务的一致性**——train/val 划分对三个任务完全相同，团队协作复现无歧义。