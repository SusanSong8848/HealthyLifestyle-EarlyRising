# HealthyLifestyle_EarlyRising

> 2026年第五届"创新杯"大学生大数据挑战赛 **初赛A题**：健康生活方式与早起习惯数据分析

## 项目简介

本项目基于"早期起床者健康数据集"（10,000样本 × 64个原始字段），通过机器学习方法完成三项分类任务。

### 最终结果（v4.0, D30 冻结）

| 任务 | 预测目标 | 类型 | ACC | 最优模型 | 权重 | 得分 |
|------|---------|------|:---:|------|:---:|:---:|
| Task 1 | Early_Waker | 二分类 | 0.7535 | Logistic Regression | 20% | 15.07 |
| Task 2 | Health_Score Level | 四分类 | 0.8275 | Logistic Regression | 40% | 33.10 |
| Task 3 | Wellness_Category | 四分类 | 0.8485 | Logistic Regression (`C=1`) | 40% | 33.94 |
| **初赛总分** | | | | | | **82.11 / 100.00** |

> 三任务均已完成并冻结。所有指标来自各任务唯一结果源（`results/metrics/raw/task*_model_comparison.csv` 和 `results/metrics/raw/task*_metrics.json`）。论文和提交材料不得直接写入数值，必须从上述文件读取。

---

## 目录结构

```
HealthyLifestyle_EarlyRising/
├── data/
│   ├── raw/A题数据集.csv                 # 官方原始数据（只读）
│   ├── processed/
│   │   └── base_semantic_clean.csv        # 公共清洗数据（编码前，人类可读）
│   └── splits/split_manifest.csv          # 三任务共用切分明细
├── src/
│   ├── config.py                          # 全局配置（路径、seed、泄漏列表、分箱）
│   ├── clean_utils.py                     # 共享清洗函数
│   ├── eda.py                             # 探索性数据分析
│   ├── preprocess.py                      # 数据预处理
│   ├── split.py                           # 独立数据切分脚本
│   ├── data_audit.py                      # 数据审计脚本
│   ├── task1_final.py                     # 任务1：Early Waker 二分类（推荐入口）
│   ├── task2_health_score.py              # 任务2：Health Score 四分类（v4.0）
│   ├── task3.py                           # 任务3：Wellness Category 四分类
│   └── run_all.py                         # 一键复现
├── tests/
│   ├── test_data_contract.py              # 数据契约测试
│   ├── test_leakage.py                    # 泄漏检查测试
│   ├── test_split.py                      # 切分验证测试
│   ├── test_shared_split_usage.py         # 三任务共享切分检查
│   └── test_task3_pipeline.py             # 任务3管道、指标和消融检查
├── config/cleaning_spec.yaml              # 公共清洗规范
├── docs/
│   ├── data_inventory.csv                 # 数据资产清单
│   ├── cleaning_comparison.csv            # 三版清洗对比
│   ├── metrics_schema.csv                 # 评价指标口径
│   └── decision_log.md                    # 决策日志
├── results/
│   ├── final/                             # 三任务最终交付物汇总
│   │   ├── final_metrics.csv              # 正式指标
│   │   ├── predictions/                   # 三任务预测CSV（各2000行）
│   │   ├── final_manifest.csv             # 文件清单+SHA256
│   │   └── reproduction_report.md         # 复现报告
│   ├── metrics/raw/                        # 各任务唯一指标源
│   │   ├── task1_model_comparison.csv
│   │   ├── task2_model_comparison.csv
│   │   ├── task2_weight_ablation.csv
│   │   ├── task3_model_comparison.csv
│   │   ├── task3_tuning.csv
│   │   └── task3_weight_ablation.csv
│   ├── figures/task2/                     # 任务2混淆矩阵与特征重要性图
│   └── predictions/task2/                 # 任务2预测
├── models/candidate/
│   ├── task1/task1_best_model.pkl
│   ├── task2/task2_best_model.pkl
│   └── task3/task3_best_model.pkl
├── paper/sections/                        # 论文初稿
│   ├── task1_section.md
│   ├── task2_section.md
│   └── task3_section.md
├── Plan.md                                # 项目计划
├── method.md                              # 建模方法论（v4.0）
├── recap.md                               # 输出文件详解
├── 要求.md                                # 赛题要求
├── progress.md                            # 成员B进度日志
├── A_require_B_progress.md                # 成员A进度日志
├── README.md                              # 本文件
└── .gitignore
```

---

## 环境要求

- **Python**：Python 3.12
- **核心依赖**：pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 一键复现

```bash
python src\run_all.py
```

`run_all.py` 会自动使用启动它的同一个 Python 解释器。

### 分步执行

```bash
python src\preprocess.py              # Step 1: 数据预处理
python src\task1_final.py              # Step 2: 任务1
python src\task2_health_score.py       # Step 3: 任务2
python src\task3.py                    # Step 4: 任务3
python src\eda.py                      # Step 5: EDA（可选）
```

### 测试

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 核心配置

| 参数 | 值 | 说明 |
|------|:---:|------|
| `RANDOM_STATE` | 20260726 | 团队统一随机种子 |
| `TEST_SIZE` | 0.2 | 80%训练/20%验证 |
| `N_FOLDS` | 5 | K折交叉验证折数 |
| Task2 分箱 | 自定义离散化（<60, [60,70), [70,85), ≥85） | Poor/Average/Good/Excellent |
| Task3 选模 | CV Accuracy 优先 | 对齐官方 ACC3；Macro-F1、Balanced Accuracy 为并列时的次级指标 |

---

## 各任务泄漏排除

| 任务 | 排除特征 |
|------|---------|
| Task 1 | Wake_Up_Time, Sleep_Time, Sleep_Duration_Hours, Healthy_Aging_Score |
| Task 2 | Person_ID, Health_Score, Wellness_Category, Fitness_Level, Healthy_Aging_Score, Early_Waker, Wake_Up_Time, Sleep_Time, Wake_Up_Time_Minutes, Sleep_Time_Minutes, Sleep_Duration_Hours |
| Task 3 | Person_ID, Wellness_Category, Health_Score, Fitness_Level, Healthy_Aging_Score, Early_Waker, Wake_Up_Time, Sleep_Time, Wake_Up_Time_Minutes, Sleep_Time_Minutes |

---

## D30 最终评分

```
初赛成绩 = ACC1 × 100 × 20% + ACC2 × 100 × 40% + ACC3 × 100 × 40%
         = 75.35 × 0.2 + 82.75 × 0.4 + 84.85 × 0.4
         = 15.07 + 33.10 + 33.94
         = 82.11 / 100.00
```

| 任务 | ACC | 得分 | 模型 |
|------|:---:|:---:|------|
| Task 1 | 0.7535 | 15.07 / 20.00 | Logistic Regression |
| Task 2 | 0.8275 | 33.10 / 40.00 | Logistic Regression |
| Task 3 | 0.8485 | 33.94 / 40.00 | Logistic Regression (`C=1`) |
| **总分** | | **82.11 / 100.00** | |

---

## 团队分工

| 成员 | 职责 |
|------|------|
| **成员 A**（数据与代码负责人） | 数据清洗、切分、任务1、一键复现、测试 |
| **成员 B**（模型与实验负责人） | 任务2、模型比较、参数搜索、指标复核 |
| **成员 C**（论文与提交负责人） | 任务3、图表、论文统稿、AI使用说明 |

> 详见 `part_1.txt`

---

## 注意事项

1. **所有脚本统一读取** `data/raw/A题数据集.csv`，不要手动替换数据文件。
2. **random_state=20260726 已锁定**，三人不得私自更改。
3. **编码、标准化、缺失填补**在各自任务 Pipeline 内部完成（fit on train fold only）。
4. **禁止在公共清洗阶段**编码、标准化、截断异常值或删除行/列。
5. `.gitignore` 已屏蔽 *.csv 和 *.pkl，仅放行汇总指标、混淆矩阵计数及冻结模型。