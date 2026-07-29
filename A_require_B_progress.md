# 成员 A — 项目进度日志

> 维护人：成员 A（数据、公共代码与复现负责人）
> 最后更新：2026-07-29 16:15
> 用途：供成员 B 快速同步，每次子任务完成后必须更新

---

## 当前状态概览

| 项目 | 状态 |
|------|:---:|
| 数据预处理 (D27-02~05) | ✅ 已完成 |
| 数据切分 (D27-07) | ✅ 已完成 |
| 任务一基线 (D27-08) | ✅ 已完成 |
| 泄漏消融 + 模型比较 (D28-01) | ✅ 已完成 |
| 复现报告 (D28-05) | ✅ 已完成 |
| 正式最终运行 (D29-01) | ✅ 已完成 |
| 论文章节素材稿 (D29-04) | ✅ 已完成 |
| Task 2 最终模型 | ⏳ 等待 B 交付 |
| Task 3 最终模型 | ⏳ 等待 C 交付 |

### 三任务最终指标

| 任务 | ACC | 权重 | 得分 | 最优模型 |
|------|:---:|:---:|:---:|------|
| Task 1 | **0.7535** | 20% | 15.07 | Logistic Regression |
| Task 2 | **0.8135** | 40% | 32.54 | Logistic Regression |
| Task 3 | **0.8170** | 40% | 32.68 | LightGBM (class_weight=balanced) |
| **初赛总分** | | | **80.29 / 100.00** | |

---

## 时间线

### 2026-07-27（D27 交付）

#### D27-02 — 数据登记 ✅
- **做了什么**：将官方原始数据 `A题数据集.csv` 放入 `data/raw/`（只读），建立 `docs/data_inventory.csv` 登记 v1/v2/v3 三版数据来源和 SHA-256
- **遇到问题**：项目根目录有旧的 `datas.csv`、`progress_时间节点已修正.xlsx` 等文件，容易混淆
- **解决方式**：删除旧文件，统一路径为 `data/raw/A题数据集.csv`
- **下一步**：D27-03 锁定公共清洗规则

#### D27-03 — 清洗规则锁定 + 泄漏检查 ✅
- **做了什么**：完成 `docs/cleaning_comparison.csv`（v1 原始 → v2 旧方案 → v3 统一清洗）、`config/cleaning_spec.yaml`（清洗规范）、`tests/test_leakage.py`（12 项泄漏检查）
- **锁定规则**：
  - Alcohol_Consumption 缺失 → `"Unknown"`（不能填 "No Alcohol"，缺少证据）
  - Exercise_Type / Workout_Intensity 缺失 → `"No Exercise"` / `"No Workout"`（仅当 Exercise_Frequency=0）
  - 时间统一 `HH:MM` 零填充
  - 公共清洗不编码、不标准化、不删行
- **遇到问题**：队友文件夹 `任务1_统一基础清洗交付包_20260726` 中的命名（Exercise→"No Exercise"）与我的旧方案（→"None"）不一致
- **解决方式**：统一采用队友的规范，建立了 `src/clean_utils.py` 共享清洗函数
- **下一步**：D27-05 生成公共清洗数据

#### D27-05 — 公共清洗数据 + 切分 ✅
- **做了什么**：`src/preprocess.py` 生成 `base_semantic_clean.csv`（10,000×66, 0缺失）、`cleaning_validation.csv`（17项验证全PASS）、`tests/test_data_contract.py`（12 tests）
- **切分**：80/20 分层（联合 task1 + task2 + task3 标签），seed=20260726
- **下一步**：D27-08 任务一基线

#### D27-08 — 任务一基线 ✅
- **做了什么**：`src/task1_final.py` 完成 Early_Waker 二分类
- **关键发现**：
  - 含时钟特征：CV ACC = 0.9915（数据泄露，无意义）
  - 移除时钟特征：CV ACC = 0.7531（真实预测）
  - **Logistic Regression 在 5-Fold CV 中优于 LGB/RF/XGB**
  - 否决了 Voting Ensemble（CV 阶段 LR 单模型已最优）
- **最终指标**：ACC1=0.7690 (D27)，后修正为 ACC1=0.7535 (D28 重新运行)
- **特征重要性**：Productivity_Score 占 11.5% 遥遥领先；其余 2-5% 均匀分散
- **交付件**：`outputs/task1/predictions.csv`（2000行）、`feature_importance.csv`、`best_model.pkl`

---

### 2026-07-28（D28 交付）

#### D28-01 — 泄漏消融 ✅
- **做了什么**：`src/d28_ablation.py` 完成受控消融实验
- **消融结果**：

| 场景 | N | CV_ACC | 差距 |
|------|:---:|:---:|:---:|
| S1 含直接泄漏 | 61 | 0.9915 | — |
| S2 移除直接泄漏 | 59 | 0.7529 | -23.9pp |
| S3 移除所有时钟 | 58 | 0.7531 | — |

- **结论**：S2≈S3（仅差 0.0002），证明 `Sleep_Duration_Hours` 不能独立重构标签。为严格性仍排除三者。

#### D28-01 — 模型比较 ✅
- **做了什么**：在 S3 最终特征集上比较四个分类器（5-Fold CV）

| 模型 | CV_ACC | Std |
|------|:---:|:---:|
| **LR** ★ | **0.7531** | 0.0066 |
| LGB | 0.7408 | 0.0080 |
| RF | 0.7393 | 0.0061 |
| XGB | 0.7357 | 0.0069 |

- **结论**：LR 在 ACC、稳定性和速度上全面最优

#### D28-05 — 复现报告 ✅
- **做了什么**：`results/logs/reproduction_candidates.md`，包含 SHA256 + 三任务量 + 复现命令

#### Git 清理 ✅
- **问题**：CSV/pkl 文件在 git 历史中占用空间，成员 B 要求从远程仓库清除
- **解决**：`git filter-repo --invert-paths --path-glob "*.csv" --path-glob "*.pkl" --path-glob "*.xlsx"` 从全部 30 个提交中彻底移除，然后 force push
- **卡点**：`.gitignore` 中有例外规则（`!data/splits/...` 等）在反复放行 CSV 文件
- **最终解决**：删除所有 `!` 例外规则，所有 `*.csv` 和 `*.pkl` 统一被忽略

---

### 2026-07-29（D29 交付）

#### D29-01 — 正式最终运行 ✅
- **做了什么**：三任务基于 `random_state=20260726` 统一切分完成最终运行
- **指标**：ACC1=0.7535 / ACC2=0.8135 / ACC3=0.8170 / 总分=80.29
- **交付文件**：
  - `results/final/final_metrics.csv` — 三任务正式指标汇总
  - `results/final/predictions/task{1,2,3}_predictions.csv` — 各 2000 行
  - `results/final/logs/final_run.log` — 运行日志（SHA256 + 命令）
  - `results/final/final_manifest.csv` — 清单 + 哈希

#### D29-04 — 论文章节素材稿 ✅
- **做了什么**：`paper/sections/A_data_task1.md`，为成员 C 提供论文素材
- **内容**：
  - §1 数据预处理（缺失分析 + 时间特征 + 目标变量 + 统一切分）
  - §2 泄漏检查（时钟特征与标签的对应关系 + S1→S2→S3 消融表 + 三任务泄漏清单）
  - §3 任务一（模型选择表 + 最终指标 + Top 15 特征重要性 + 论文解读）
  - §4 给成员 C 的写作建议（5 个段落结构）

---

## 关键数据

### 缺失处理

| 字段 | 缺失数 | 占比 | 处理 |
|------|:---:|:---:|------|
| Alcohol_Consumption | 3,014 | 30.14% | → `"Unknown"`（不能填 "No Alcohol"） |
| Exercise_Type | 824 | 8.24% | → `"No Exercise"`（频率=0） |
| Workout_Intensity | 824 | 8.24% | → `"No Workout"`（频率=0） |

### 泄漏排除清单

| 任务 | 排除特征 |
|------|---------|
| Task 1 | `Wake_Up_Time`, `Sleep_Time`, `Sleep_Duration_Hours`, `Healthy_Aging_Score` |
| Task 2 | `Health_Score`, `Wellness_Category`, `Fitness_Level`, `Healthy_Aging_Score` |
| Task 3 | `Wellness_Category`, `Health_Score`, `Fitness_Level`, `Healthy_Aging_Score` |

### 配置参数

| 参数 | 值 |
|------|:---:|
| random_state | 20260726（已锁定） |
| test_size | 0.2 |
| CV folds | 5（shuffle=True） |
| Task2 分箱边界 | [0, 60, 70, 85, 100] |

---

## 当前卡点

| 卡点 | 影响 | 需要谁 |
|------|------|:---:|
| Task 2 最终模型未冻结 | 论文中 Task 2 的指标和模型比较表还不能写 | **成员 B** |
| Task 3 v3.3 未完成最终运行 | 论文中 Task 3 的 ACC3 还不能确定（当前 0.8170 是旧版） | **成员 C** |
| 论文主稿未合并 | A 的素材稿需要C统稿 | **成员 C** |

## 下一步建议

1. **成员 B**：完成 Task 2 最终模型 + 模型比较表，将 final metrics 更新到 `results/final/`
2. **成员 C**：完成 Task 3 v3.3 最终运行 + 类别权重消融，汇总三任务论文主稿
3. **D30 冻结**（7/30）：三人确认 final_metrics 后锁定所有模型和参数