# 项目文件结构（v3.2 Final — D28 交付版）

```
HealthyLifestyle_EarlyRising/
├── data/raw/A题数据集.csv             # 原始数据（本地，只读）
├── data/
│   ├── raw/A题数据集.csv               # 官方原始数据副本 (D27-02)
│   ├── legacy/                        # 历史清洗版本 (D27-02)
│   ├── processed/
│   │   ├── base_semantic_clean.csv    # 统一清洗数据，编码前 (D27-05)
│   │   ├── processed_data.pkl         # 预处理训练/验证数据
│   │   ├── encoders.pkl               # 编码器
│   │   └── scaler.pkl                 # 标准化器
│   └── splits/
│       └── split_manifest.csv         # 三任务共用切分明细 (D27-07)
├── outputs/                          # 输出（.gitignore 中）
│   ├── eda/                          # EDA 图表
│   ├── task1/                        # 任务1输出
│   ├── task2/                        # 任务2输出
│   └── task3/                        # 任务3输出
├── results/
│   ├── data_audit/cleaning_validation.csv  # 清洗验证结果 (D27-05)
│   ├── metrics/raw/task1_baseline.csv      # 任务1基线指标 (D27-08)
│   ├── figures/baseline/task1_confusion_matrix.png  # 任务1混淆矩阵 (D27-08)
│   ├── issues/                              # 问题记录
│   └── logs/                                # 复现日志
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
│   └── test_split.py                  # 切分验证测试 (D27-07)
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
│   ├── task3_wellness_category.py     # 任务3
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
D:\python\python.exe src\task3_wellness_category.py       # 任务3
D:\python\python.exe src\run_all.py                       # 一键复现
```

## D27 验收交付件汇总

| 任务ID | 交付文件 | 状态 |
|--------|---------|:---:|
| D27-02 | `data/raw/A题数据集.csv`, `docs/data_inventory.csv` | ✅ |
| D27-03 | `docs/cleaning_comparison.csv`, `config/cleaning_spec.yaml`, `tests/test_leakage.py` | ✅ |
| D27-05 | `src/preprocess.py`, `data/processed/base_semantic_clean.csv`, `results/data_audit/cleaning_validation.csv`, `tests/test_data_contract.py` | ✅ |
| D27-07 | `src/split.py`, `data/splits/split_manifest.csv`, `tests/test_split.py` | ✅ |
| D27-08 | `src/task1.py`, `results/metrics/raw/task1_baseline.csv`, `results/figures/baseline/task1_confusion_matrix.png` | ✅ |
| — | `src/run_all.py`, `docs/decision_log.md`, `docs/metrics_schema.csv` | ✅ |

## 单元测试

```
Ran 33 tests in 0.185s — OK
  test_data_contract.py — 12 tests PASSED
  test_leakage.py       — 12 tests PASSED
  test_split.py          —  9 tests PASSED
```

## v3.1 核心改进

| 改进 | 说明 |
|------|------|
| **CV制导模型选择** | 不再盲目录入 Voting Ensemble，CV 阶段选最优单模型 |
| **Task1/2 用 LR，Task3 用 LGB+balanced** | 最优模型为数据驱动选择，非强行集成 |
| **共享清洗函数** | `src/clean_utils.py` 消除三处重复 |
| **33项自动化测试** | 数据契约、泄漏检查、切分验证 |