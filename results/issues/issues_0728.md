# 7 月 28 日角色 B 进展与阻塞

更新时间：2026-07-28 17:30（Asia/Shanghai）

## 已完成

- D27 角色 B 交付已创建独立 PR #2：
  `https://github.com/SusanSong8848/HealthyLifestyle-EarlyRising/pull/2`
- D28-02 任务二模型比较与方法消融已完成，等待 C 复核。
- 六个模型族、16 组确定性配置均使用公共训练集内 5 折分层 CV。
- 直接分类优胜者：Logistic Regression。
- 回归后分档优胜者：Ridge Regression。

## D28-02 验证集结果

| 路线 | 模型 | Accuracy | Macro-F1 | Balanced Accuracy | MAE | RMSE |
|---|---|---:|---:|---:|---:|---:|
| 直接分类 | Logistic Regression | 0.8265 | 0.8202 | 0.8123 | 0.1735 | 0.4165 |
| 回归后分档 | Ridge Regression | 0.8340 | 0.8244 | 0.8145 | 0.1660 | 0.4074 |

公共切分：train=8,000、val=2,000、seed=20260726。两条路线的验证集
Person_ID 均与 `split_manifest.csv` 完全一致，指标已由测试独立重算。

## 当前阻塞

| 问题 ID | 影响任务 | 缺失或异常 | 负责人 | 解除条件 | 状态 |
|---|---|---|---|---|---|
| I-0728-00 | 分支同步 | 首次连接曾出现 504、DNS 线程和 SSH 拒绝；17:30 已成功 fetch，确认 origin/main=e761cfe | B | 将 origin/main 依次并入 D27、D28 并推送 | 已解除 |
| I-0728-01 | D28-01 / D28-04 | `task1_model_comparison.csv`、`task1_leakage_ablation.csv` 未收到 | A | A 按统一 schema 提交正式文件 | 未收到 |
| I-0728-02 | D28-03 / D28-04 | `task3_model_comparison.csv`、`task3_weight_ablation.csv` 未收到 | C | C 按统一 schema 提交正式文件 | 未收到 |
| I-0728-03 | D28-05 | 三任务入围候选复现报告未收到 | A | 复现差异不超过 1e-4 且结论一致 | 未收到 |
| I-0728-04 | D28-06 / D28-09 | 候选图、图注及最终图表清单未收到 | C | 图表输入、脚本、模型和类别顺序可追溯 | 未收到 |
| I-0728-05 | D28-07 / D28-08 | 三任务聚合、复现和三人确认尚未完成 | B / 三人 | D28-04、D28-05 PASS 并完成三人签字 | 阻塞 |

## 聚合门禁

`src/aggregate_model_results.py` 会在三个任务的比较、调参和消融产物全部
存在且数据版本、split、seed、样本量和字段一致时，才生成：

- `results/metrics/model_comparison.csv`
- `results/metrics/tuning_log.csv`
- `results/metrics/ablation.csv`

当前门禁按预期拒绝生成上述全任务文件；不会把任务二单独结果伪装成三任务汇总。
