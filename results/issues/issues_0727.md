# 7 月 27 日基线问题登记

更新时间：2026-07-27 22:46（Asia/Shanghai）

## 当前结论

- D27-04：B 已完成标签、指标和泄漏配置，等待三人复核。
- D27-06：B 已完成统一标签与评价接口，45 项测试全部通过，等待 C 复核。
- D27-09：B 已完成任务二正式基线，等待 C 复核。
- D27-11：阻塞。任务一和任务三的正式基线产物尚未提供，不能生成完整的三任务 `baseline_raw.csv`。

## 已完成的任务二基线

| 模型 | Accuracy | Macro-F1 | Balanced Accuracy | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Dummy Most Frequent | 0.4455 | 0.1541 | 0.2500 | 0.6835 | 0.9703 |
| Logistic Regression | 0.8265 | 0.8202 | 0.8123 | 0.1735 | 0.4165 |

公共切分：train=8,000、val=2,000，seed=20260726。验证集 Person_ID 与 `split_manifest.csv` 完全一致，训练集交集为 0。

## 阻塞项

| 问题ID | 依赖任务 | 缺失产物 | 负责人 | 最晚修复时间 | 状态 |
|---|---|---|---|---|---|
| I-0727-01 | D27-08 | `results/metrics/raw/task1_baseline.csv`、`results/figures/baseline/task1_confusion_matrix.png` | A | 2026-07-27 22:30 | 未收到 |
| I-0727-02 | D27-10 | `src/task3.py`、`results/metrics/raw/task3_baseline.csv`、`results/figures/baseline/task3_confusion_matrix.png` | C | 2026-07-27 22:30 | 未收到 |
| I-0727-03 | D27-11 | 三任务字段、类别顺序、样本量和 seed 的统一汇总 | B | 依赖 I-0727-01/02 | 阻塞 |

## 解除阻塞条件

1. A、C 分别提供符合 `docs/metrics_schema.csv` 的正式基线 CSV。
2. 两项任务均使用公共 `split_manifest.csv`，验证集样本量为 2,000。
3. 类别顺序、seed 和指标能够通过统一接口复核。
4. 条件满足后，由 B 生成 `results/metrics/baseline_raw.csv` 并重新验收 D27-11。

在依赖满足前，不创建只有任务二数据的伪三任务汇总文件。
