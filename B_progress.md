# 项目进度日志 — 成员B (模型实验与任务二)

> 最后更新: 2026-07-30 11:45 UTC+8  
> 维护人: 成员B (via Cline AI)

---

## 2026-07-30 上午 — Task 2 v4.0 重构与完整交付

### 所做工作

1. **全面审查项目现状**
   - 检查了 `task2_model.txt`（队友提出的改进方案）
   - 确认现有两版 `task2_health_score.py` 的差异：21KB版（Pipeline+CV）和6KB版（引用config+clean_utils）
   - 确认 `config.py` 的标签定义和泄漏字段需对齐队友要求

2. **数据探查**
   - 边界值验证：HS=60 有16人，HS=70 有37人，HS=85 有32人
   - 确认 `pd.cut(right=True)` 和 `right=False` 各有问题，改用自定义离散化函数
   - Fitness_Level 与 Wellness_Category 完全一致（都=Good 4429/Excellent 3320/Average 2137/Poor 114），确认为泄漏
   - 数据管线正常：10000条 → 8000训练 / 2000验证

3. **重构 `src/task2_health_score.py` (v4.0)**
   - 严格按 `task2_model.txt` 的7条规则重构
   - 三阶段策略：Phase 1 (CV筛选) → Phase 2 (权重消融) → Phase 3 (验证集最终评估)
   - 自定义离散化函数，自动化边界测试断言
   - 预处理 Pipeline 折内拟合，杜绝泄露
   - 泄漏字段：Person_ID, Health_Score, Wellness_Category, Fitness_Level, Healthy_Aging_Score, Early_Waker, Wake_Up_Time, Sleep_Time, Wake_Up_Time_Minutes, Sleep_Time_Minutes, Sleep_Duration_Hours

4. **Phase 1 结果**

| 模型 | CV ACC | Macro-F1 | BalAcc | Recall_Poor | Recall_Avg | Recall_Good | Recall_Exc |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Dummy | 0.4464 | 0.1543 | 0.2500 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| **LR** ★ | **0.8162** | **0.8120** | **0.8049** | **0.7356** | **0.8182** | **0.8597** | **0.8063** |
| LGBM | 0.7694 | 0.7569 | 0.7369 | 0.6538 | 0.7482 | 0.8628 | 0.6830 |

**LR 全面最优**，Macro-F1 领先 LGBM 约 5.5pp。

5. **Phase 2 消融**

| 配置 | CV ACC | Macro-F1 | Recall_Poor |
|------|:---:|:---:|:---:|
| LGBM 无权重 | 0.7694 | 0.7569 | 0.6538 |
| LGBM Balanced | 0.7763 | 0.7673 | 0.6789 |
| **LR** | **0.8162** | **0.8120** | **0.7356** |

类别权重有助但不足以超越 LR。

6. **Phase 3 最终验证 (2000人)**

| 指标 | 值 |
|------|:---:|
| **ACC2** | **0.8275** |
| Macro-F1 | 0.8209 |
| Balanced Accuracy | 0.8144 |
| **得分** | **33.10 / 40.00** |

7. **产出交付物（全部就绪）**
   - ✅ `results/metrics/raw/task2_model_comparison.csv`
   - ✅ `results/metrics/raw/task2_weight_ablation.csv`
   - ✅ `results/metrics/raw/task2_classification_report_Logistic_Regression.csv`
   - ✅ `results/metrics/raw/task2_feature_importance_Logistic_Regression.csv`
   - ✅ `results/metrics/raw/task2_features_used.csv`
   - ✅ `results/metrics/raw/task2_metrics.json`
   - ✅ `results/figures/task2/task2_confusion_matrix_Logistic_Regression.png`
   - ✅ `results/figures/task2/task2_feature_importance_Logistic_Regression.png`
   - ✅ `results/predictions/task2/task2_predictions_Logistic_Regression.csv`
   - ✅ `models/candidate/task2/task2_best_model.pkl`
   - ✅ `paper/sections/task2_section.md` (论文初稿)

### 遇到的问题及解决

| 问题 | 解决方案 |
|------|---------|
| Windows GBK 编码错误（✓符号） | 替换为纯 ASCII "OK" |
| `pd.cut` 边界错误（60/70/85） | 改用自定义逐值离散化 |
| Phase 3 超时中断 | 单独写 `_complete_task2.py` 补全缺失产物 |
| Person_ID 已被泄漏字段排除 | drop(..., errors='ignore') |

### 下一步建议

1. **队员确认 Phase 1 结果**（如需要不同的模型选择方向）
2. **Task 2 的 C 值微调**：当前使用默认 C=1.0，CV 标准差 0.0117 表明模型较稳定，不需要大范围调参
3. **Task 2 论文补充**：添加"回归后分级 vs 直接分类"的对比实验（`task2_model.txt` 第7条提到但未强制要求，为加分项）
4. **Task 1/3 复核**（见下）

---

## 待做：Task 1 复核

- Task 1 当前 ACC1=0.7535 (LR), 得分 15.07/20.00
- 状态：✅ 可交付，产出物见 `results/metrics/raw/task1_model_comparison.csv`

## 待做：Task 3 复核

- Task 3 历史 ACC3=0.8160 (旧)，需完整重跑确认
- 产出物见 `results/metrics/raw/task3_*`

## 三任务总分估计（暂定）

| 任务 | ACC | 权重 | 得分 | 模型 |
|------|:---:|:---:|:---:|------|
| Task 1 | 0.7535 | 20% | 15.07 | Logistic Regression |
| Task 2 | 0.8275 | 40% | 33.10 | Logistic Regression |
| Task 3 | ~0.8160 | 40% | ~32.64 | 待确认 |
| **初赛总分** | | | **~80.81** | |

> Task 3 数值为历史值，待最终重跑后更新。

---

## 运行方式

```bash
# Task 2 一键运行
D:\python\python.exe src\task2_health_score.py
```

详细步骤请参考 `Plan.md`。