# 建模方法论：三任务完整方案

> 版本：v4.0（三任务全部冻结，D30 最终交付）
> 日期：2026-07-30

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
D:\python\python.exe src\task3.py                         # 任务3: Wellness Category
```

一键复现脚本 `src/run_all.py` 使用当前启动它的 Python 解释器（`sys.executable`），上述 `D:\python\python.exe` 仅为 A 当前环境的命令示例。

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
| **Task 3** | `Wellness_Category`（目标）, `Health_Score`（按45/65/80可100%还原）, `Fitness_Level`（与目标100%相同）, `Early_Waker`, `Wake_Up_Time`, `Sleep_Time`, `Wake_Up_Time_Minutes`, `Sleep_Time_Minutes` | 前三项属于直接目标泄漏；早起标签与直接时间字段为保守排除项，避免任务间标签与时间定义混入 |
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

特征重要性较为分散——早起行为是多维度生活方式的综合体现；具体特征数以当次运行输出为准。

---

## 4. Task 2 — Health Score 四分类（权重 40%）

### 4.1 离散化边界（v4.0 修正）

Task 2 的标签采用自定义逐值离散化函数（`pd.cut` 的 `right=True` 会将 HS=60 误归入 Poor，`right=False` 会将 HS=85 误归入 Good）：

| 区间 | 类别 | 样本数（全量） |
|------|------|:---:|
| <60 | Poor | 1,276 |
| [60, 70) | Average | 2,296 |
| [70, 85) | Good | 4,462 |
| ≥85 | Excellent | 1,966 |

边界测试已通过自动化断言：HS=59.999→Poor、HS=60→Average、HS=70→Good、HS=85→Excellent。

### 4.2 模型选择（三阶段策略）

**Phase 1 — 训练集 5-Fold CV 初步筛选：**

| 模型 | CV ACC | Macro-F1 | BalAcc | Recall_Poor |
|------|:---:|:---:|:---:|:---:|
| Dummy (Most Frequent) | 0.4464 | 0.1543 | 0.2500 | 0.0000 |
| **Logistic Regression** ★ | **0.8162±0.0117** | **0.8120** | **0.8049** | **0.7356** |
| LightGBM (无权重) | 0.7694±0.0154 | 0.7569 | 0.7369 | 0.6538 |

**Phase 2 — 类别权重受控消融：** Balanced Weight 使 LightGBM 的 Macro-F1 从 0.7569→0.7673（+0.0104），但仍远低于 LR 的 0.8120。

**Phase 3 — 冻结验证集评估（2000人）：**

| 模型 | Test ACC2 | Macro-F1 | BalAcc | Recall_Poor |
|------|:---:|:---:|:---:|:---:|
| **Logistic Regression** ★ | **0.8275** | **0.8209** | **0.8144** | **0.7554** |

**得分：33.10 / 40.00**

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

- 任务标签按 `Poor → 0、Average → 1、Good → 2、Excellent → 3` 固定映射。
- 读取 `data/splits/split_manifest.csv`，保证与任务1、任务2使用完全相同的训练/验证人员。
- 删除 `Person_ID`、`Wellness_Category`、`Health_Score`、`Fitness_Level`、`Healthy_Aging_Score` 及直接时间字段，避免目标泄漏或可疑综合评分进入主模型。
- 数值缺失填补与标准化、类别缺失填补与 One-Hot 编码均放入 `sklearn Pipeline`，每个交叉验证折只在该折训练部分拟合。
- 基线固定为 `DummyClassifier(strategy="most_frequent")` 和无权重多项逻辑回归。
- 候选模型使用逻辑回归、随机森林、XGBoost（环境可用时）及 LightGBM。
- 模型选择只使用训练集五折交叉验证结果。由于赛题以 ACC3 计分，优先级设为 `Accuracy → Macro-F1 → Balanced Accuracy`；后两项用于 Accuracy 相同时的稳健性比较，并重点检查 Poor 类是否被模型放弃。
- 在基线比较后，只对表现最好的逻辑回归家族做小范围正则化搜索，取 `C∈{0.1,0.3,1,3,10}`。五组配置使用相同特征、相同 Pipeline、相同五折与相同随机种子，调参过程不依据验证集结果。
- 最终模型冻结后，在验证集上逐一打乱 56 个原始输入字段，以 Accuracy 的平均下降量计算置换重要性。该口径避免把独热编码后的类别水平误当成多个原始变量；重要性只表示预测关联，不作因果解释。

### 5.3 类别权重受控消融

为判断类别权重是否真正改善少数类，而不是把参数变化混在一起，固定 LightGBM 的以下参数：

- `n_estimators=300`
- `max_depth=8`
- `learning_rate=0.05`
- `num_leaves=63`
- `random_state=20260726`

只改变一个条件：

| 消融组 | `class_weight` | 其他参数 |
|------|------|------|
| 无权重 | `None` | 完全相同 |
| 类别权重 | `"balanced"` | 完全相同 |

两组均报告 CV 与验证集的 Accuracy、Macro-F1、Balanced Accuracy 和 Poor Recall，并在 `results/metrics/raw/task3_weight_ablation.csv` 中保存相对无权重组的差值。是否采用类别权重由结果决定，不能预先假定 `"balanced"` 一定更好。

### 5.4 评价与输出

| 文件 | 作用 |
|------|------|
| `results/metrics/raw/task3_baseline.csv` | Dummy 与无权重逻辑回归基线 |
| `results/metrics/raw/task3_model_comparison.csv` | 全部候选模型的 CV、验证指标和唯一 `Selected=True` 行 |
| `results/metrics/raw/task3_tuning.csv` | 逻辑回归 `C` 的训练集五折小范围搜索记录 |
| `results/metrics/raw/task3_weight_ablation.csv` | 无权重/类别权重 LightGBM 受控比较 |
| `results/metrics/raw/task3_classification_report.csv` | 最终模型四类 precision、recall、F1 |
| `results/figures/baseline/task3_confusion_matrix.png/.csv` | 无权重逻辑回归基线混淆矩阵图片及计数 |
| `results/figures/candidate/task3_confusion_matrix.png/.csv` | 最终候选模型混淆矩阵图片及计数 |
| `results/figures/candidate/task3_feature_importance.png` | 最终候选模型 Top 20 特征重要性 |
| `results/metrics/raw/task3_feature_importance.csv` | 最终候选模型完整特征重要性 |
| `results/metrics/raw/task3_features_used.csv` | 主模型实际使用的原始字段清单 |
| `results/metrics/raw/task3_metrics.json/.pkl/.txt` | 指标、配置和审计元数据 |
| `results/metrics/raw/task3_run_complete.json` | 全部交付物保存成功后生成的运行完成标记 |
| `results/predictions/candidate/task3_predictions.csv` | 验证集逐人真实/预测标签 |
| `models/candidate/task3/task3_best_model.pkl` | 可直接复现预测的完整 Pipeline |

### 5.5 当前结果状态

Task 3 v3.4 已完成并冻结。训练集五折交叉验证按 Accuracy 选出无权重逻辑回归，正则化参数 `C=1`。其五折 CV Accuracy 为 0.8481；在独立验证集上，`ACC3=0.8485`、Macro-F1=0.7832、Balanced Accuracy=0.7652、Poor Recall=0.5217。最终数字以 `task3_model_comparison.csv` 中唯一的 `Selected=True` 行及 `task3_metrics.json` 为准。

---

## 6. 最终结果汇总（D30 冻结）

| 任务 | ACC | 权重 | 得分 | 最优模型 |
|------|:---:|:---:|:---:|------|
| Task 1 | **0.7535** | 20% | 15.07 | Logistic Regression |
| Task 2 | **0.8275** | 40% | 33.10 | Logistic Regression |
| Task 3 | **0.8485** | 40% | 33.94 | Logistic Regression (`C=1`) |
| **初赛总分** | | | **82.11 / 100.00** | |

> 三任务均已完成并冻结。所有指标来自各任务唯一结果源（`results/metrics/raw/task*_model_comparison.csv` 和 `results/metrics/raw/task*_metrics.json`）。论文和提交材料必须从上述文件读取数值。

---

## 7. 核心发现

1. **排除时钟特征后模型并非"衰退"而是"真实学习"**——ACC1 从 1.0（抄答案）降至 0.77（真预测），差距 23pp 是"泄露"与"学习"之间的距离。

2. **Logistic Regression 在三个任务上均为最优单模型**——说明在严格特征集下，健康指标的线性组合已足够区分不同等级的健康状态。树模型（LightGBM/XGBoost/Random Forest）在 Task 2 上落后 LR 约 5pp（CV Macro-F1: 0.8120 vs 0.7569），在 Task 1/3 上也未能超越。

3. **CV制导模型选择比盲目录入更严谨**——在 CV 阶段就确定最优模型，测试集仅用于最终评估，避免对测试集的间接过拟合。

4. **联合分层抽样保证了三个任务的一致性**——train/val 划分对三个任务完全相同，团队协作复现无歧义。
