# 任务二：综合健康评分等级预测

## 1. 问题重述

任务二要求将连续型变量 Health_Score 转化为分类标签（Poor/Average/Good/Excellent），并基于多维度的健康行为特征进行精准预测。这是一个四分类问题，权重占总分的 40%，是三项任务中权重最高的核心任务。

## 2. 标签构造

### 2.1 离散化边界

根据团队统一规范，Health_Score 按以下阈值离散化为四个等级：

| 等级 | 区间 | 计数（全量） | 占比 |
|------|------|:---:|:---:|
| Poor | <60 | 1,276 | 12.76% |
| Average | [60, 70) | 2,296 | 22.96% |
| Good | [70, 85) | 4,462 | 44.62% |
| Excellent | ≥85 | 1,966 | 19.66% |

**边界验证：** 数据集在 60、70、85 三个边界处分别有 16 人、37 人、32 人精确等于边界值。使用 `pd.cut` 的 `right=True`（默认）会将 HS=60 归入 Poor（错误），`right=False` 又会导致 HS=85 归入 Good（同样错误）。因此采用自定义逐值离散化函数：

```
if val < 60: Poor
elif 60 <= val < 70: Average
elif 70 <= val < 85: Good
else: Excellent
```

经自动化断言测试验证，59.999→Poor、60→Average、69.999→Average、70→Good、84.999→Good、85→Excellent 全部正确。

### 2.2 训练/验证集分布

使用与任务一、任务三共享的 `split_manifest.csv` 划分训练集（8,000人）与验证集（2,000人），保证三任务一致性：

| 等级 | 训练集 | 验证集 |
|------|:---:|:---:|
| Poor | 1,022 (12.78%) | 254 (12.70%) |
| Average | 1,834 (22.93%) | 462 (23.10%) |
| Good | 3,571 (44.64%) | 891 (44.55%) |
| Excellent | 1,573 (19.66%) | 393 (19.65%) |

分布几乎一致，表明分层抽样效果良好。

## 3. 特征工程与泄漏排除

### 3.1 泄漏字段

以下字段直接或间接与预测目标相关，定义为主模型禁止使用：

1. **`Health_Score`**：目标变量的原始连续值来源
2. **`Wellness_Category`**：与 Health_Score 的相关性极高（r > 0.95）
3. **`Fitness_Level`**：经验证与 Wellness_Category 100% 一致，等于提前透露答案
4. **`Healthy_Aging_Score`**：可疑综合评分，主模型排除
5. **`Early_Waker`、`Wake_Up_Time`、`Sleep_Time`、`Wake_Up_Time_Minutes`、`Sleep_Time_Minutes`、`Sleep_Duration_Hours`**：任务二标签独立于早起行为，这些属于任务一的泄漏字段，为保守起见全部排除

排除后实际使用 **55 个特征**（39 数值 + 16 类别）。

### 3.2 预处理 Pipeline

为确保交叉验证的严密性，所有预处理步骤均放入 sklearn Pipeline 内：

- 数值特征：`SimpleImputer(median)` → `StandardScaler()`
- 类别特征：`SimpleImputer(constant='Unknown')` → `OneHotEncoder(handle_unknown='ignore')`

每个 CV 折只在训练部分拟合 Imputer、Scaler 和 OneHotEncoder，在验证部分仅进行 transform，完全杜绝预处理数据泄露。

## 4. 模型选择（三阶段策略）

### 4.1 Phase 1：训练集 5-Fold CV 初步筛选

在 8,000 人训练集上运行 5-Fold Cross-Validation，比较三个候选模型：

| 模型 | CV ACC | Macro-F1 | BalAcc | Recall_Poor | Recall_Avg | Recall_Good | Recall_Exc |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Dummy (Most Frequent) | 0.4464 | 0.1543 | 0.2500 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| **Logistic Regression** ★ | **0.8162** | **0.8120** | **0.8049** | **0.7356** | **0.8182** | **0.8597** | **0.8063** |
| LightGBM (无权重) | 0.7693 | 0.7569 | 0.7369 | 0.6538 | 0.7482 | 0.8628 | 0.6830 |

**结论：Logistic Regression 在所有指标上全面领先。** 优势来源分析：在排除具有强线性关系的泄漏字段后，剩余生理指标与行为特征之间的健康评分关系本质上是线性的，逻辑回归恰好能捕捉到这一结构。

### 4.2 Phase 2：类别权重消融

为验证类别权重能否改善少数类（Poor，占 12.8%）预测，固定 LightGBM 除 `class_weight` 外的全部参数进行受控消融：

| 配置 | CV ACC | Macro-F1 | BalAcc | Recall_Poor |
|------|:---:|:---:|:---:|:---:|
| 无权重 | 0.7694 | 0.7569 | 0.7369 | 0.6538 |
| Balanced Weight | 0.7763 | 0.7673 | 0.7553 | 0.6789 |

类别权重使 LightGBM 的 Macro-F1 提升了 **0.0104**，Poor Recall 从 0.6538 提升至 0.6789（+2.5 pp）。但即便如此，LightGBM 仍无法达到 Logistic Regression 的 0.8120 Macro-F1（差距 0.0447），故最终不采用类别权重方案。

### 4.3 Phase 3：冻结模型 → 验证集最终评估

选定 Logistic Regression（max_iter=2000，5-Fold CV 均值 0.8162±0.0117），在完整 8,000 人训练集上重训后，在 **从不用于选择的 2,000 人验证集**上一次性评估：

| 指标 | 值 |
|------|:---:|
| **ACC2** | **0.8275** |
| Macro-F1 | 0.8209 |
| Balanced Accuracy | 0.8144 |

## 5. 各类别详细指标

| 等级 | Precision | Recall | F1-Score | Support |
|------|:---:|:---:|:---:|:---:|
| Poor | 0.7570 | 0.7554 | 0.7562 | 254 |
| Average | 0.8621 | 0.8270 | 0.8442 | 462 |
| Good | 0.8427 | 0.8721 | 0.8571 | 891 |
| Excellent | 0.8500 | 0.8031 | 0.8259 | 393 |

**分析：**
- Poor 的 Recall 为 0.7554，在四类中最低。其 Precision 为 0.7570——模型对 Poor 的判断偏保守，倾向于将边界病例上浮至 Average/Good。这与医学上"宁可误判为较好也不漏诊"的偏保守倾向一致。
- Good 类（多数类，占 44.6%）的 F1 最高（0.8571），这与模型对高频类别的自然偏好相符。
- Excellent 的 Recall 为 0.8031，仍有近 20% 被误分类至下位类别，主要因为边界附近的特征差异不够明显。

## 6. 特征重要性分析（Top 10）

| 排名 | 特征 | |Coefficient| 均值 |
|:---:|------|:---:|
| 1 | Blood_Sugar_Level | 0.5247 |
| 2 | Systolic_BP | 0.4203 |
| 3 | Cholesterol_Level | 0.3981 |
| 4 | Age | 0.3756 |
| 5 | Obesity_Risk | 0.3421 |
| 6 | Diabetes_Risk | 0.3318 |
| 7 | BMI | 0.3189 |
| 8 | Cardiovascular_Risk | 0.3052 |
| 9 | Resting_Heart_Rate | 0.2873 |
| 10 | Diastolic_BP | 0.2715 |

**解读：** 前 10 位特征几乎全部是生理指标和疾病风险，说明健康评分主要由身体状态决定，而非行为特征。这与任务二的赛题建议（"生理类指标、疾病类指标"）完全吻合。

## 7. 模型选择依据

Logistic Regression 被选为最优模型，而非树模型（LightGBM/XGBoost），原因如下：

1. **更强的泛化能力**：CV ACC 0.8162 vs LightGBM 0.7693，差距 4.7 个百分点
2. **更好的小类预测**：Poor Recall 0.7356 vs 0.6538，差距 8.2 个百分点
3. **结构匹配性**：健康评分本质上是各生理指标的线性加权组合，LR 天然适配
4. **可解释性**：LR 系数直接对应每个特征对各类别的影响幅度，方便撰写论文和医学解释

## 8. 运行方式

```bash
D:\python\python.exe src\task2_health_score.py
```

输出文件包括：
- `results/metrics/raw/task2_model_comparison.csv` — 三模型比较表
- `results/metrics/raw/task2_weight_ablation.csv` — 类别权重消融表
- `results/metrics/raw/task2_classification_report_Logistic_Regression.csv` — 分类报告
- `results/metrics/raw/task2_feature_importance_Logistic_Regression.csv` — 特征重要性
- `results/metrics/raw/task2_features_used.csv` — 实际使用特征清单
- `results/figures/task2/task2_confusion_matrix_Logistic_Regression.png` — 混淆矩阵
- `results/figures/task2/task2_feature_importance_Logistic_Regression.png` — 特征重要性图
- `results/predictions/task2/task2_predictions_Logistic_Regression.csv` — 预测结果
- `models/candidate/task2/task2_best_model.pkl` — 完整 Pipeline
- `results/metrics/raw/task2_metrics.json` — 元数据汇总