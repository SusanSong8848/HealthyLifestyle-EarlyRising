# HealthyLifestyle_EarlyRising

> 2026年第五届"创新杯"大学生大数据挑战赛 **初赛A题**：健康生活方式与早起习惯数据分析

## 项目简介

本项目基于"早期起床者健康数据集"（10,000样本 × 64个原始字段），通过机器学习方法完成三项分类任务。

### 历史基线结果（v3.1，仅供追溯）

| 任务 | 预测目标 | 类型 | ACC | 最优模型 | 权重 |
|------|---------|------|:---:|------|:---:|
| Task 1 | Early_Waker | 二分类 | 0.7690 | Logistic Regression | 20% |
| Task 2 | Health_Score Level | 四分类 | 0.8120 | Logistic Regression | 40% |
| Task 3 | Wellness_Category | 四分类 | 0.8160 | LightGBM (balanced) | 40% |

**历史基线总分：81.64 / 100.00**

> Task 3 已升级为 v3.3，统一使用 `src/task3.py`、共享切分表和折内预处理，并新增无权重/类别权重消融。上表中的 Task 3 指标必须在 v3.3 完整重跑后替换，不能直接作为最终论文结果。

---

## 目录结构

```
HealthyLifestyle_EarlyRising/
├── data/
│   ├── raw/A题数据集.csv                 # 官方原始数据（只读）
│   ├── legacy/                            # 历史数据版本归档
│   ├── processed/
│   │   ├── base_semantic_clean.csv        # 公共清洗数据（编码前，人类可读）
│   │   ├── processed_data.pkl             # Task 1/2及旧代码兼容产物
│   │   ├── encoders.pkl                   # Task 1/2及旧代码兼容产物
│   │   └── scaler.pkl                     # Task 1/2及旧代码兼容产物
│   └── splits/split_manifest.csv          # 三任务共用切分明细
├── src/
│   ├── config.py                          # 全局配置（路径、seed、泄漏列表、分箱）
│   ├── clean_utils.py                     # 共享清洗函数
│   ├── eda.py                             # 探索性数据分析
│   ├── preprocess.py                      # 数据预处理（清洗+编码+标准化+切分）
│   ├── split.py                           # 独立数据切分脚本
│   ├── data_audit.py                      # 数据审计脚本
│   ├── task1_final.py                     # 任务1：Early Waker 二分类（★推荐）
│   ├── task1_early_waker.py               # 任务1：备选模块化版本
│   ├── task2_health_score.py              # 任务2：Health Score 四分类
│   ├── task3.py                            # 任务3统一入口：Wellness Category 四分类
│   └── run_all.py                         # 一键复现
├── tests/
│   ├── test_data_contract.py              # 数据契约测试（12 tests）
│   ├── test_leakage.py                    # 泄漏检查测试（12 tests）
│   ├── test_split.py                      # 切分验证测试（9 tests）
│   ├── test_shared_split_usage.py         # 三任务共享切分检查
│   └── test_task3_pipeline.py             # 任务3管道、指标和消融检查
├── config/cleaning_spec.yaml              # 公共清洗规范
├── docs/
│   ├── data_inventory.csv                 # 数据资产清单
│   ├── cleaning_comparison.csv            # 三版清洗对比
│   ├── metrics_schema.csv                 # 评价指标口径
│   └── decision_log.md                    # 决策日志
├── results/
│   ├── data_audit/cleaning_validation.csv # 清洗验证结果
│   ├── metrics/raw/                        # 唯一指标源
│   │   ├── task1_baseline.csv
│   │   ├── task3_baseline.csv
│   │   ├── task3_model_comparison.csv
│   │   └── task3_weight_ablation.csv
│   ├── figures/baseline/                  # 基线混淆矩阵
│   ├── figures/candidate/                 # 候选模型混淆矩阵与特征重要性图
│   └── predictions/candidate/             # 候选模型验证集预测
├── models/candidate/task3/                # 任务3冻结候选模型
├── outputs/                               # Task 1/2及EDA历史输出（.gitignore）
├── Plan.md                                # 项目计划
├── method.md                              # 建模方法论
├── recap.md                               # 输出文件详解
├── 要求.md                                # 赛题要求
└── README.md                              # 本文件
```

---

## 环境要求

- **Python**：`D:\python\python.exe`（Python 3.12）
- **依赖**：pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm

```bash
D:\python\python.exe -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
```

---

## 快速开始

### 一键复现

```bash
D:\python\python.exe src\run_all.py
```

`run_all.py` 会自动使用启动它的同一个 Python 解释器，不再写死某位队员电脑上的安装路径。

### 分步执行

```bash
# Step 1: 数据预处理（必须在此之后执行模型）
D:\python\python.exe src\preprocess.py

# Step 2: 任务1 — Early Waker 二分类
D:\python\python.exe src\task1_final.py

# Step 3: 任务2 — Health Score 四分类
D:\python\python.exe src\task2_health_score.py

# Step 4: 任务3 — Wellness Category 四分类
D:\python\python.exe src\task3.py

# Step 5: 探索性数据分析（可选）
D:\python\python.exe src\eda.py
```

### 测试

```bash
D:\python\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

验收标准：命令末尾显示 **OK**，且不得有 failed/error。测试总数随测试文件增加而变化，不再写死。

---

## 数据输入

| 文件 | 路径 | 用途 |
|------|------|------|
| 官方原始数据 | `data/raw/A题数据集.csv`（10,000行×64列） | 所有脚本的输入 |
| 公共清洗数据 | `data/processed/base_semantic_clean.csv` | 审计用（编码前，人类可读） |
| 切分明细 | `data/splits/split_manifest.csv` | Person_ID + split(train/val) |

---

## 核心配置

| 参数 | 值 | 说明 |
|------|:---:|------|
| `RANDOM_STATE` | 20260726 | 团队统一随机种子 |
| `TEST_SIZE` | 0.2 | 80%训练/20%验证 |
| `N_FOLDS` | 5 | K折交叉验证折数 |
| Task2 分箱 | [0,60,70,85,100] | Poor/Average/Good/Excellent |

---

## 各任务泄漏排除

| 任务 | 排除特征 |
|------|---------|
| Task 1 | Wake_Up_Time, Sleep_Time, Sleep_Duration_Hours, Healthy_Aging_Score |
| Task 2 | Health_Score, Wellness_Category, Fitness_Level, Healthy_Aging_Score |
| Task 3 | Person_ID, Wellness_Category, Health_Score, Fitness_Level, Healthy_Aging_Score, Early_Waker, Wake_Up_Time, Sleep_Time, Wake_Up_Time_Minutes, Sleep_Time_Minutes |

---

## 结果输出

| 任务 | 输出文件 | 内容 |
|------|---------|------|
| Task 1 | `outputs/task1/predictions.csv` | 2000行：Person_ID + True_Label + Predicted_Label |
| Task 1 | `outputs/task1/feature_importance.csv` | 57个特征重要性排名 |
| Task 2 | `outputs/task2/predictions.csv` | 2000行：Person_ID + True_Label + Predicted_Label |
| Task 2 | `outputs/task2/feature_importance.csv` | 56个特征重要性排名 |
| Task 3 | `results/metrics/raw/task3_baseline.csv` | Dummy 与无权重逻辑回归基线的 CV/验证指标 |
| Task 3 | `results/metrics/raw/task3_model_comparison.csv` | 所有候选模型的 Accuracy、Macro-F1、Balanced Accuracy、四类 Recall 与最终选择 |
| Task 3 | `results/metrics/raw/task3_weight_ablation.csv` | 同参数 LightGBM 的无权重/`balanced` 受控消融及指标差值 |
| Task 3 | `results/figures/baseline/task3_confusion_matrix.png/.csv` | 无权重逻辑回归基线混淆矩阵图片及精确计数 |
| Task 3 | `results/figures/candidate/task3_confusion_matrix.png/.csv` | CV 选中模型的验证集混淆矩阵图片及精确计数 |
| Task 3 | `results/figures/candidate/task3_feature_importance.png` | CV 选中模型的 Top 20 特征重要性 |
| Task 3 | `results/metrics/raw/task3_features_used.csv` | 主模型实际使用的原始字段及类型 |
| Task 3 | `results/metrics/raw/task3_metrics.json/.pkl/.txt` | 最终指标、配置和审计元数据的三种格式 |
| Task 3 | `results/predictions/candidate/task3_predictions.csv` | 2000行：Person_ID + True_Label + Predicted_Label |
| Task 3 | `models/candidate/task3/task3_best_model.pkl` | 在完整训练集重拟合的预处理器与分类器 Pipeline |

> 更多输出文件详解见 `recap.md`

Task 3 的编码后特征数由训练数据中的实际类别水平决定，以 `results/metrics/raw/task3_metrics.json` 中的 `transformed_features_used` 为准，不再在文档中写死为 56。

`.gitignore` 只继续屏蔽原始/加工数据；共享切分表、`results/` 下的审计 CSV 和 Task 3 最终模型已明确放行，普通 `git add` 即可同步这些交付物。

---

## 团队分工

| 成员 | 职责 |
|------|------|
| **成员 A**（数据与代码负责人） | 数据清洗、切分、任务1、一键复现、测试 |
| **成员 B**（模型与实验负责人） | 任务2、模型比较、参数搜索、指标复核 |
| **成员 C**（论文与提交负责人） | 任务3、图表、论文统稿、AI使用说明 |

> 详见 `part_1.txt` 和 `docs/decision_log.md`

---

## 历史基线评分公式

```
初赛成绩 = ACC1 × 100 × 20% + ACC2 × 100 × 40% + ACC3 × 100 × 40%
         = 76.90 × 0.2 + 81.20 × 0.4 + 81.60 × 0.4
         = 15.38 + 32.48 + 32.64
         = 81.64
```

> 该公式只记录 v3.1 历史基线。最终论文必须从 `results/metrics/raw/task3_metrics.json` 或 `task3_model_comparison.csv` 读取 v3.3 的 ACC3，再重新计算总分。

---

## 注意事项

1. **所有脚本统一读取** `data/raw/A题数据集.csv`，不要手动替换数据文件。
2. **random_state=20260726 已锁定**，三人不得私自更改。
3. **编码、标准化、缺失填补**在各自任务 Pipeline 内部完成（fit on train fold only）。
4. **禁止在公共清洗阶段**编码、标准化、截断异常值或删除行/列。
5. 提交预测时只包含 `Person_ID + Predicted_Label`，不要带 `True_Label`。

`processed_data.pkl`、`encoders.pkl`、`scaler.pkl` 目前保留给 Task 1/2 和旧代码兼容；Task 3 不读取它们，以 `src/task3.py` 内的折内 Pipeline 为准。
