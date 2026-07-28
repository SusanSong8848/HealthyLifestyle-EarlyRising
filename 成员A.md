# 成员 A — D27 + D28 交付状态

> 日期：2026-07-28
> 职责：数据、公共代码与复现负责人

---

## D28 交付件

### D28-01：任务一泄漏消融 + 模型比较 ✅

| 输出 | 路径 | 状态 |
|------|------|:---:|
| 消融CSV | `results/metrics/raw/task1_leakage_ablation.csv` | ✅ |
| 模型比较CSV | `results/metrics/raw/task1_model_comparison.csv` | ✅ |
| 生成脚本 | `src/d28_ablation.py`（完全独立，仅依赖 raw 数据） | ✅ |

**泄漏消融（5-Fold CV, LR）**

| 场景 | N | CV_ACC | Verdict |
|------|:---:|:---:|------|
| S1 含直接泄漏 (WakeUp+Sleep分钟) | 61 | **0.9915** | ❌ 几乎满分 → 严重数据泄露 |
| S2 移除直接泄漏 (保留Duration) | 59 | **0.7529** | ⚠️ 骤降 23.9pp |
| S3 移除所有时钟特征 (最终版) | 58 | **0.7531** | ✅ 无泄露基线 |

> S2≈S3 (仅差 0.0002)：`Sleep_Duration_Hours` 在 WakeUp/Sleep 分钟数被移除后无法独立重构标签。但为严格性最终版仍排除三者。

**模型比较（S3最终特征集, 5-Fold CV)**

| 模型 | CV_ACC | Std | Time(s) |
|------|:---:|:---:|:---:|
| **Logistic Regression** ★ | **0.7531** | 0.0066 | 0.1 |
| LightGBM | 0.7408 | 0.0080 | 4.0 |
| Random Forest | 0.7393 | 0.0061 | 3.4 |
| XGBoost | 0.7357 | 0.0069 | 6.4 |

> LR 在 ACC、稳定性和速度上全面最优，选为最终模型。

### D28-05：复现三任务候选 ✅

| 输出 | 路径 | 状态 |
|------|------|:---:|
| 复现报告 | `results/logs/reproduction_candidates.md`（含 SHA256） | ✅ |

| 任务 | ACC | 权重 | 得分 | 最优模型 |
|------|:---:|:---:|:---:|------|
| Task1 (LR) | 0.7535 | 20% | 15.07 | Logistic Regression |
| Task2 (LR) | 0.8135 | 40% | 32.54 | Logistic Regression |
| Task3 (LGB+balanced) | 0.8170 | 40% | 32.68 | LightGBM (balanced) |
| **初赛总分** | | | **80.29 / 100.00** | |

### Git 清理 ✅

- `git filter-repo --invert-paths` 从全部 30 个提交历史中彻底移除 `*.csv`、`*.pkl`、`*.xlsx`
- `.gitignore` 中已移除所有例外放行规则（`!data/splits/...`、`!results/...` 等）
- Force push 已完成，远程仓库历史已重写

---

## D27 交付件（已完成）

| 任务ID | 交付文件 | 状态 |
|--------|---------|:---:|
| D27-02 | `data/raw/A题数据集.csv`, `docs/data_inventory.csv`, `src/data_audit.py` | ✅ |
| D27-03 | `docs/cleaning_comparison.csv`, `config/cleaning_spec.yaml`, `tests/test_leakage.py` (12 tests) | ✅ |
| D27-05 | `src/preprocess.py`, `base_semantic_clean.csv`, `cleaning_validation.csv`, `tests/test_data_contract.py` (12 tests) | ✅ |
| D27-07 | `src/split.py`, `split_manifest.csv`, `tests/test_split.py` (9 tests) | ✅ |
| D27-08 | `src/task1.py`, `task1_baseline.csv`, `task1_confusion_matrix.png` | ✅ |
| 辅助 | `src/clean_utils.py`, `docs/metrics_schema.csv`, `docs/decision_log.md`, `src/config.py`, `src/run_all.py` | ✅ |

---

## 测试汇总

```
Ran 33 tests in 0.201s — OK
  tests/test_data_contract.py — 12 tests PASSED
  tests/test_leakage.py       — 12 tests PASSED
  tests/test_split.py          —  9 tests PASSED
```

---

## 交付件清单（供 B / C 核验）

| D27-ID | 文件 | D28-ID | 文件 |
|--------|------|--------|------|
| D27-02 | `data/raw/A题数据集.csv`, `docs/data_inventory.csv` | D28-01 | `task1_leakage_ablation.csv`, `task1_model_comparison.csv` |
| D27-03 | `cleaning_comparison.csv`, `cleaning_spec.yaml`, `test_leakage.py` | D28-05 | `reproduction_candidates.md` |
| D27-05 | `preprocess.py`, `base_semantic_clean.csv`, `cleaning_validation.csv`, `test_data_contract.py` | — | `src/d28_ablation.py` (生成脚本) |
| D27-07 | `split.py`, `split_manifest.csv`, `test_split.py` | — | — |
| D27-08 | `task1.py`, `task1_baseline.csv`, `confusion_matrix.png` | — | — |