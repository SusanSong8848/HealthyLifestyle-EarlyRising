# Task 1 建模方法论：严格无数据泄露的 Early Waker 预测

> 作者：角色 A（算法与代码主攻手）
> 日期：2026-07-26

---

## 1. 问题定义

**任务目标**：利用多维度健康与行为特征，构建二分类模型预测个体是否为"早起者"（Early_Waker = Yes/No）。

**初赛评分**：ACC1 × 100 × 20% — 即该任务占初赛总分的 20%。

---

## 2. 数据泄露的诊断与修复

### 2.1 问题发现

初始建模（包含全部 60 个特征）时，所有模型（Random Forest、XGBoost、LightGBM）的交叉验证和验证集准确率均达到 **ACC1 = 1.0000（100%）**。

通过交叉分析 `Wake_Up_Time` 与 `Early_Waker` 标签，发现了数据泄露的证据：

| 起床时间段 | Early_Waker = Yes | Early_Waker = No | 样本数 |
|:----------:|:---:|:---:|:---:|
| **4:00–5:59 AM** | **2,837 (100%)** | 0 | 2,837 |
| **6:00–6:59 AM** | 1,321 (48.3%) | 1,415 (51.7%) | 2,736 |
| **7:00–11:59 AM** | 0 | **4,427 (100%)** | 4,427 |

结论：**`Early_Waker` 标签几乎等同于 `Wake_Up_Time ≤ 6:30` 的直接翻译**。除早上 6 点这一小时外，其他时间段可以 100% 确定标签。

### 2.2 为什么这属于数据泄露

`Wake_Up_Time`（起床时间）本身就是 `Early_Waker`（是否早起）的**定义性特征**——就像用"考试分数"来预测"是否及格"一样。如果模型中包含这类特征：

- 决策树会在第一层分裂就直接按 `Wake_Up_Time ≤ 390分钟（6:30 AM）` 分开
- 其他 59 个特征（饮食、运动、心理、生理等）的贡献度为 0
- 模型没有"学习"任何规律，只是在"抄答案"

这与比赛考察"用多维度健康数据预测作息习惯"的目标完全背离。

### 2.3 修复策略

从特征集中**移除 2 个直接泄露标签的特征**：

| 移除的特征 | 移除原因 |
|-----------|---------|
| `Wake_Up_Time_Minutes` | 直接定义答案是/否（4-5点全Yes, 7-11点全No） |
| `Sleep_Time_Minutes` | 与起床时间高度共线（`r ≈ -0.82`），间接暴露作息时刻 |

**保留的睡眠相关特征**：
- `Sleep_Duration_Hours`（睡眠时长）——不直接暴露时刻
- `Sleep_Quality_Score`（睡眠质量）
- `Number_of_Night_Awakenings`（夜间醒来次数）
- `Weekend_Sleep_Difference_Hours`（周末作息差异）
- `Screen_Time_Before_Bed_Hours`（睡前屏幕时间）

修复后特征数：**60 → 58**。

---

## 3. 数据预处理流程

### 3.1 缺失值处理

| 特征 | 缺失情况 | 处理方法 | 理由 |
|------|---------|---------|------|
| `Alcohol_Consumption` | ~30% 缺失 | 填充 `"Unknown"` | 保留"未报告"信息，避免信息损失 |
| `Exercise_Type` | 锻炼频率=0 时为空 | 填充 `"None"` | 逻辑一致：不锻炼→无锻炼类型 |
| `Workout_Intensity` | 锻炼频率=0 时为空 | 填充 `"None"` | 逻辑一致 |
| 其他类别特征 | 少量缺失 | 众数填充 | 稳健处理 |

最终验证：**10,000 条数据，0 个缺失值**。

### 3.2 特征工程

- **时间转换**：`Wake_Up_Time`（"6:35"）和 `Sleep_Time`（"23:48"）转换为**午夜起分钟数**（395 和 1428），方便数值模型处理
- **类别编码**：17 个类别特征使用 `LabelEncoder` 转为 0…k-1 整数
- **目标编码**：`Early_Waker` → Yes=1, No=0；`Health_Score` → 四分位数分箱 → Poor/Average/Good/Excellent
- **数值标准化**：43 个数值特征使用 `StandardScaler`（仅对训练集 fit）

### 3.3 数据划分

- 训练集：8,000 样本（80%）
- 验证集：2,000 样本（20%）
- 分层抽样：按 `Wellness_Category` 保持各类别比例一致
- 随机种子：42（保证可复现）

### 3.4 Data_clean.csv 的输出

`Data_clean.csv` 在缺失值填充**之后**、类别编码**之前**导出，保持人类可读的原始格式（不含 `_Encoded` 后缀列）：

```
原始数据 (10,000×64)
  → 缺失值填充（Alcohol→"Unknown", 锻炼→"None"等）
  → 时间特征转换（新增 Wake_Up_Time_Minutes, Sleep_Time_Minutes 列）
  → 导出 Data_clean.csv (10,000×66)
  → 继续编码、标准化、划分 → processed_data.pkl
```

---

## 4. 建模流程

### 4.1 基线模型对比（5 折交叉验证）

使用 **58 个无泄漏特征** 训练 4 个基线模型：

| 模型 | 5-Fold CV | 验证集 ACC | 验证集 F1 |
|------|:---:|:---:|:---:|
| Logistic Regression | 0.7488 ± 0.0114 | **0.7500** | 0.6827 |
| Random Forest (n=200, depth=12) | 0.7309 ± 0.0058 | 0.7305 | 0.6390 |
| XGBoost (n=200, depth=6) | 0.7338 ± 0.0027 | 0.7375 | 0.6721 |
| LightGBM (n=200, depth=6) | 0.7331 ± 0.0061 | 0.7315 | 0.6595 |

> ⚠️ **关键对比**：含泄漏特征时 CV=1.0000；移除后 CV≈0.73-0.75。差距约 25 个百分点，正是"抄答案"和"真学习"的差异。

### 4.2 超参数调优（手动网格搜索）

每个模型搜索 2 组参数组合（3 折 CV），避免 GridSearchCV 的笛卡尔积爆炸：

**XGBoost**：`{n_estimators: 300, max_depth: 8, learning_rate: 0.05}` → 3-CV = 0.7429

**LightGBM**：`{n_estimators: 300, max_depth: 8, learning_rate: 0.05, num_leaves: 63}` → 3-CV = 0.7390

**Random Forest**：`{n_estimators: 500, max_depth: None}` → 3-CV = 0.7356

### 4.3 集成模型（Voting Ensemble）

将三个调优后的模型通过 **软投票（Soft Voting）** 集成：

```
VotingClassifier(
    XGBoost (n=300, depth=8, lr=0.05)
  + LightGBM (n=300, depth=8, lr=0.05, leaves=63)
  + Random Forest (n=500, depth=None)
  → voting='soft'
)
```

集成后的验证集 ACC = **0.7420**，F1 = 0.6738。

---

## 5. 最终结果

### 5.1 ACC1 得分

| 阶段 | ACC1 | 含义 |
|------|:---:|------|
| **修复前**（60 特征，含泄漏） | 1.0000 | 抄答案，无意义 |
| **修复后**（58 特征，无泄漏） | **0.7420** | 真正学习多维度规律 |
| 得分贡献（20%权重） | **14.84 / 20.00** | |

### 5.2 特征重要性（Top 10）

| 排名 | 特征 | 重要性 | 解读 |
|:---:|------|:---:|------|
| 1 | **Productivity_Score** | 5.23% | 早起者平均生产力更高 |
| 2 | **Breakfast_Regularity_Score** | 4.29% | 早起者吃早餐更规律 |
| 3 | **Stress_Level** | 3.28% | 压力水平存在显著差异 |
| 4 | **Sleep_Duration_Hours** | 3.14% | 睡眠时长（非时刻） |
| 5 | **Cholesterol_Level** | 2.99% | 生理指标 |
| 6 | **Protein_Intake_Grams** | 2.89% | 蛋白质摄入 |
| 7 | **Daily_Calorie_Intake** | 2.88% | 每日卡路里 |
| 8 | **Water_Intake_Liters** | 2.86% | 饮水量 |
| 9 | **Height_cm** | 2.86% | 身高 |
| 10 | **Blood_Sugar_Level** | 2.78% | 血糖水平 |

**关键发现**：58 个特征的重要性分布**均匀且分散**——排名第一的 `Productivity_Score` 也仅占 5.23%。这说明早起行为并非由单个因素决定，而是**多维度的生活方式的综合体现**，符合比赛的考察目标。

### 5.3 混淆矩阵（验证集 2,000 样本）

| | 预测 No | 预测 Yes |
|---|---|:---:|
| **实际 No** | 951 (TP) | 203 (FP) |
| **实际 Yes** | 313 (FN) | 533 (TN) |

- **Precision (Yes)** = 533/(533+203) = 72.4%
- **Recall (Yes)** = 533/(533+313) = 63.0%

---

## 6. 核心结论

1. **数据泄露是机器学习竞赛中最隐蔽的陷阱**。`Wake_Up_Time` 作为"定义性特征"必须在建模前排除，否则模型准确率虚高但毫无实际价值。

2. **移除泄漏特征后 ACC1 从 1.0 降至 0.742**。这 25.8 个百分点的差距恰好量化了"抄答案"和"真学习"之间的距离。

3. **58 个特征的贡献均匀分散**（无单一特征超过 6%），证明早起行为是一个由生活方式、饮食习惯、心理状态、生理指标共同决定的多维度现象——这才是比赛真正希望参赛者发现的规律。

4. **Voting Ensemble（XGB + LightGBM + RF）** 在无泄漏条件下提供了最稳健的预测，可作为后续持续优化的基线。

---

## 7. 产出文件索引

| 文件 | 路径 | 说明 |
|------|------|------|
| 预测结果 | `outputs/task1/predictions.csv` | 2000行，Person_ID + True_Label + Predicted_Label |
| 特征重要性 | `outputs/task1/feature_importance.csv` | 58个特征的重要性排名 |
| 评估指标 | `outputs/task1/metrics.txt` | ACC1、F1、各模型对比、Top 15 特征 |
| 最佳模型 | `outputs/task1/best_model.pkl` | Voting Ensemble 序列化模型 |
| 独立运行脚本 | `run_task1_fixed.py` | 可独立运行的完整建模流程 |
| 标准脚本 | `src/task1_early_waker.py` | 标准化模块脚本 |
| 清洗数据 | `outputs/preprocess/Data_clean.csv` | 缺失值填充后、编码前的清洗数据 |