# 成员 A — D27 交付状态

> 日期：2026-07-27
> 职责：数据、公共代码与复现负责人

---

## D27 交付件逐项验收

### D27-02：登记官方原始数据、三份历史清洗结果及相应代码 ✅

| 输出 | 路径 | 状态 |
|------|------|:---:|
| 官方原始数据 | `data/raw/A题数据集.csv` | ✅ |
| 数据资产清单 | `docs/data_inventory.csv`（含 v1/v2/v3 三版 sha256、行列数与差异摘要） | ✅ |
| 数据审计脚本 | `src/data_audit.py` | ✅ |

### D27-03：三版清洗对比 + 锁定公共清洗规则 ✅

| 输出 | 路径 | 状态 |
|------|------|:---:|
| 三版清洗对比表 | `docs/cleaning_comparison.csv`（v1 原始 → v2 A旧方案 → v3 统一） | ✅ |
| 公共清洗规范 | `config/cleaning_spec.yaml`（缺失、语义填充、禁止项、验证清单） | ✅ |
| 泄漏自动检查 | `tests/test_leakage.py`（12 tests） | ✅ |

**已锁定规则**：Alcohol 缺失 → `Unknown`；Exercise_Type/Workout_Intensity 缺失 → `No Exercise`/`No Workout`（仅当频率=0）；不编码、不标准化、不截断、不删行。

### D27-05：用脚本生成唯一公共清洗数据和验证表 ✅

| 输出 | 路径 | 状态 |
|------|------|:---:|
| 预处理脚本 | `src/preprocess.py`（含联合分层 + split_manifest 输出） | ✅ |
| 公共清洗数据 | `data/processed/base_semantic_clean.csv`（10,000×66，0 缺失） | ✅ |
| 清洗验证表 | `results/data_audit/cleaning_validation.csv`（17 项验证全 PASS） | ✅ |
| 数据契约测试 | `tests/test_data_contract.py`（12 tests） | ✅ |

### D27-07：生成三任务共用数据切分清单 ✅

| 输出 | 路径 | 状态 |
|------|------|:---:|
| 切分脚本 | `src/split.py` | ✅ |
| 切分明细 | `data/splits/split_manifest.csv`（train=8000, val=2000） | ✅ |
| 切分测试 | `tests/test_split.py`（9 tests） | ✅ |

**参数**：`seed=20260726`，联合分层（task1_label + task2_label + task3_label），Person_ID 唯一连接。严禁私自重切。

### D27-08：任务一基线 ✅

| 输出 | 路径 | 状态 |
|------|------|:---:|
| 任务脚本 | `src/task1.py`（= task1_final.py 副本） | ✅ |
| 基线指标 | `results/metrics/raw/task1_baseline.csv` | ✅ |
| 混淆矩阵 | `results/figures/baseline/task1_confusion_matrix.png` | ✅ |

| 指标 | 数值 |
|------|:---:|
| **ACC1** | **0.7690** |
| Balanced Accuracy | 0.7559 |
| Macro F1 | 0.7617 |
| Recall(Yes) | 0.6779 |
| 最优模型 | Logistic Regression（CV 制导，否决 Voting Ensemble） |
| 特征数 | 57（排除 Wake_Up_Time/Sleep_Time/Sleep_Duration/Healthy_Aging_Score） |

### 辅助交付

| 输出 | 路径 | 状态 |
|------|------|:---:|
| 一键复现 | `src/run_all.py` | ✅ |
| 共享清洗函数 | `src/clean_utils.py` | ✅ |
| 评价指标口径 | `docs/metrics_schema.csv` | ✅ |
| 决策日志 | `docs/decision_log.md`（14 条决策） | ✅ |
| 全局配置 | `src/config.py`（泄漏列表、分箱、路径） | ✅ |

---

## 测试汇总

```
Ran 33 tests in 0.169s — OK
  tests/test_data_contract.py — 12 tests PASSED
  tests/test_leakage.py       — 12 tests PASSED
  tests/test_split.py          —  9 tests PASSED
```

---

## 模型指标总览（供 B 核验）

| 任务 | ACC | 最优模型 |
|------|:---:|------|
| Task 1 | 0.7690 | Logistic Regression |
| Task 2 | 0.8120 | Logistic Regression |
| Task 3 | 0.8160 | LightGBM (class_weight=balanced) |
| **初赛总分** | **81.64** | |