# 项目文件结构（v3.3 — Task 3 统一版）

```
HealthyLifestyle_EarlyRising/
├── data/
│   ├── raw/A题数据集.csv               # 官方原始数据副本 (D27-02)
│   ├── legacy/                        # 历史清洗版本 (D27-02)
│   ├── processed/
│   │   ├── base_semantic_clean.csv    # 统一清洗数据，编码前 (D27-05)
│   │   ├── processed_data.pkl         # Task 1/2及旧代码兼容产物
│   │   ├── encoders.pkl               # Task 1/2及旧代码兼容产物
│   │   └── scaler.pkl                 # Task 1/2及旧代码兼容产物
│   └── splits/
│       └── split_manifest.csv         # 三任务共用切分明细 (D27-07)
├── outputs/                          # Task 1/2 与 EDA 历史输出（.gitignore 中）
│   ├── eda/                          # EDA 图表
│   ├── task1/                        # 任务1输出
│   └── task2/                        # 任务2输出
├── results/
│   ├── data_audit/cleaning_validation.csv  # 清洗验证结果 (D27-05)
│   ├── metrics/raw/
│   │   ├── task1_baseline.csv              # 任务1基线指标 (D27-08)
│   │   ├── task3_baseline.csv              # 任务3基线指标 (D27-10)
│   │   ├── task3_model_comparison.csv      # 任务3候选模型比较 (D28-03)
│   │   ├── task3_weight_ablation.csv       # 无权重/类别权重消融 (D28-03)
│   │   ├── task3_classification_report.csv # 任务3四分类报告
│   │   ├── task3_feature_importance.csv    # 任务3特征重要性
│   │   ├── task3_features_used.csv         # 任务3实际字段清单
│   │   └── task3_metrics.json/.pkl/.txt    # 任务3指标与审计摘要
│   ├── figures/
│   │   ├── baseline/task3_confusion_matrix.png/.csv
│   │   └── candidate/
│   │       ├── task3_confusion_matrix.png/.csv
│   │       └── task3_feature_importance.png
│   ├── predictions/candidate/task3_predictions.csv
│   ├── issues/                              # 问题记录
│   └── logs/                                # 复现日志
├── models/candidate/task3/
│   └── task3_best_model.pkl                 # 任务3完整 Pipeline
├── paper/figures/
│   └── captions_draft.md                    # D28-06：图注人工审核稿
├── config/
│   └── cleaning_spec.yaml             # 公共清洗规范 (D27-03)
├── docs/
│   ├── cleaning_comparison.csv        # 三版清洗对比 (D27-03)
│   ├── data_inventory.csv             # 数据资产清单 (D27-02)
│   ├── decision_log.md                # 决策日志
│   └── metrics_schema.csv             # 评价指标口径 (D27-06)
├── tests/
│   ├── test_data_contract.py          # 数据契约测试 (D27-05)
│   ├── test_leakage.py                # 泄漏检查测试 (D27-03)
│   ├── test_split.py                  # 切分验证测试 (D27-07)
│   ├── test_shared_split_usage.py     # 三任务共用切分检查
│   └── test_task3_pipeline.py         # 任务3管道、指标与消融检查
├── src/
│   ├── __init__.py
│   ├── config.py                      # 全局配置
│   ├── clean_utils.py                 # 共享清洗函数
│   ├── eda.py                         # EDA
│   ├── preprocess.py                  # 公共清洗+预处理 (D27-05)
│   ├── split.py                       # 数据切分 (D27-07)
│   ├── data_audit.py                  # 数据审计
│   ├── task1.py                       # 任务1（主脚本）(D27-08)
│   ├── task1_final.py                 # 任务1（详细版）
│   ├── task1_early_waker.py           # 任务1（备选模块化版）
│   ├── task2_health_score.py          # 任务2
│   ├── task3.py                       # 任务3统一入口
│   └── run_all.py                     # 一键复现
├── .gitignore
├── Plan.md                           # 本文件
├── method.md                         # 建模方法论
├── recap.md                          # 输出文件说明
├── 要求.md                           # 赛题要求
└── 2026年第五届"创新杯"大学生大数据挑战赛A题.pdf
```

## 环境要求

- **Python 解释器**：`D:\python\python.exe`（Python 3.12）
- **依赖**：pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm

## 运行方式

```bash
D:\python\python.exe src\eda.py                         # EDA（可跳过）
D:\python\python.exe src\preprocess.py                   # 统一预处理
D:\python\python.exe src\task1.py                        # 任务1 (D27-08)
D:\python\python.exe src\task2_health_score.py            # 任务2
D:\python\python.exe src\task3.py                         # 任务3
D:\python\python.exe src\run_all.py                       # 一键复现
```

`src/run_all.py` 自动使用启动它的当前 Python 解释器，不写死电脑路径。`src/task1.py` 是 D27-08 基线交付入口；`src/task1_final.py` 是当前一键复现使用的详细版。二者暂不删除，最终冻结模型时再由三人确定唯一入口。Task 3 不读取 `processed_data.pkl`、`encoders.pkl` 或 `scaler.pkl`。

## D27 验收交付件汇总

| 任务ID | 交付文件 | 状态 |
|--------|---------|:---:|
| D27-02 | `data/raw/A题数据集.csv`, `docs/data_inventory.csv` | ✅ |
| D27-03 | `docs/cleaning_comparison.csv`, `config/cleaning_spec.yaml`, `tests/test_leakage.py` | ✅ |
| D27-05 | `src/preprocess.py`, `data/processed/base_semantic_clean.csv`, `results/data_audit/cleaning_validation.csv`, `tests/test_data_contract.py` | ✅ |
| D27-07 | `src/split.py`, `data/splits/split_manifest.csv`, `tests/test_split.py` | ✅ |
| D27-08 | `src/task1.py`, `results/metrics/raw/task1_baseline.csv`, `results/figures/baseline/task1_confusion_matrix.png` | ✅ |
| D27-10 | `src/task3.py`, `results/metrics/raw/task3_baseline.csv`, `results/figures/baseline/task3_confusion_matrix.png` | 代码就绪，待完整运行验收 |
| D28-03 | `results/metrics/raw/task3_model_comparison.csv`, `results/metrics/raw/task3_weight_ablation.csv` | 代码就绪，待完整运行验收 |
| — | `src/run_all.py`, `docs/decision_log.md`, `docs/metrics_schema.csv` | ✅ |

## 单元测试

```bash
D:\python\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

验收标准为命令末尾显示 `OK`，且没有 failed/error；测试总数不再写死。

## v3.3 核心改进

| 改进 | 说明 |
|------|------|
| **CV制导模型选择** | 不再盲目录入 Voting Ensemble，CV 阶段选最优单模型 |
| **Task 3 单一入口** | 统一为 `src/task3.py`，`src/run_all.py` 只调用该文件 |
| **Task 3 单一结果源** | 指标进入 `results/metrics/raw/`，图进入 `results/figures/`，预测进入 `results/predictions/`，模型进入 `models/` |
| **折内预处理** | 缺失填补、One-Hot 和标准化均在 Pipeline 内按 CV 折拟合，避免预处理泄漏 |
| **不平衡指标完整** | 同时报告 Accuracy、Macro-F1、Balanced Accuracy 和四类 Recall，重点复核 Poor Recall |
| **受控类别权重消融** | 固定 LightGBM 其余参数，仅比较 `class_weight=None` 与 `"balanced"` |
| **共享清洗函数** | `src/clean_utils.py` 消除三处重复 |
| **自动化测试扩充** | 新增共享切分与任务3管道/消融测试 |

## Task 3 验收顺序

1. 先运行全部单元测试。
2. 再运行 `D:\python\python.exe src\task3.py`，中途不得手工改 CSV。
3. 检查 `task3_baseline.csv` 恰有 Dummy 和无权重逻辑回归两行。
4. 检查 `task3_weight_ablation.csv` 恰有无权重和 `balanced` LightGBM 两行，且两行除 `class_weight` 外参数一致。
5. 检查 `task3_model_comparison.csv` 恰有一个 `Selected=True`。
6. 检查两个混淆矩阵、特征重要性图、预测表和模型文件均已生成。
7. 最终论文中的 Task 3 数值只能从上述结果文件读取，旧的 `ACC3=0.8160` 不得直接沿用。
