# 项目文件结构（v3.1 Final）

```
HealthyLifestyle_EarlyRising/
├── datas.csv                        # 原始数据（本地，不在 git 中）
├── data/
│   ├── processed/
│   │   ├── base_semantic_clean.csv   # 基础清洗数据（编码前人类可读）
│   │   ├── processed_data.pkl        # 预处理训练/验证数据
│   │   ├── encoders.pkl              # 编码器
│   │   └── scaler.pkl                # 标准化器
│   └── splits/
│       └── split_manifest.csv        # 统一数据切分明细
├── outputs/                         # 输出（本地，.gitignore 中）
│   ├── eda/                         # EDA 图表
│   ├── task1/                       # 任务1: predictions.csv, metrics, model
│   ├── task2/                       # 任务2: predictions.csv, metrics, model
│   └── task3/                       # 任务3: predictions.csv, metrics, model
├── src/
│   ├── __init__.py
│   ├── config.py                    # 全局配置（统一 random_state、泄漏列表、分箱）
│   ├── clean_utils.py               # 共享清洗函数（消除代码重复）
│   ├── eda.py                       # EDA 可视化
│   ├── preprocess.py                # 数据预处理（联合分层 + split_manifest）
│   ├── task1_final.py               # 任务1（CV制导最优模型：Logistic Regression）
│   ├── task1_early_waker.py         # 任务1（备选模块化版本）
│   ├── task2_health_score.py        # 任务2（CV制导最优模型：Logistic Regression）
│   └── task3_wellness_category.py   # 任务3（CV制导最优模型：LightGBM + balanced）
├── .gitignore
├── Plan.md                          # 本文件
├── method.md                        # 建模方法论（v3.1）
├── part.txt                         # 角色分工
├── 要求.md                          # 赛题
└── 2026年第五届"创新杯"大学生大数据挑战赛A题.pdf
```

## 环境要求

- **Python 解释器**：`D:\python\python.exe`（Python 3.12）
- **依赖**：pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm

## 运行方式

```bash
D:\python\python.exe src\eda.py                         # EDA（可跳过）
D:\python\python.exe src\preprocess.py                   # 统一预处理
D:\python\python.exe src\task1_final.py                  # 任务1 ★ 推荐独立版
D:\python\python.exe src\task2_health_score.py            # 任务2
D:\python\python.exe src\task3_wellness_category.py       # 任务3
```

## 实现步骤

### Step 1: 环境搭建 ✅

### Step 2: EDA ✅ — `src/eda.py`

### Step 3: 数据预处理 ✅ — `src/preprocess.py`

统一清洗规范（基于 `公共数据要求.txt`）：
- `Exercise_Frequency_Per_Week=0` → `"No Exercise"`, `"No Workout"`
- `Alcohol_Consumption` 缺失 → `"Unknown"`
- 时间统一 `HH:MM`，保留原始列并新增 `_Minutes`
- 联合分层抽样（task1+task2+task3 标签）
- 输出 `split_manifest.csv`

### Step 4: 任务1 ✅ — `src/task1_final.py`

| 指标 | 数值 |
|------|:---:|
| **ACC1** | 0.7690 |
| 最优模型 | Logistic Regression |
| 得分 | 15.38 / 20.00 |

### Step 5: 任务2 ✅ — `src/task2_health_score.py`

| 指标 | 数值 |
|------|:---:|
| **ACC2** | 0.8120 |
| 最优模型 | Logistic Regression |
| 分箱 | [0,60,70,85,100] |
| 得分 | 32.48 / 40.00 |

### Step 6: 任务3 ✅ — `src/task3_wellness_category.py`

| 指标 | 数值 |
|------|:---:|
| **ACC3** | 0.8160 |
| 最优模型 | LightGBM (class_weight=balanced) |
| 类别 | 4类（含 Poor 114条） |
| 得分 | 32.64 / 40.00 |

## 最终评分

| 任务 | ACC | 权重 | 得分 |
|------|:---:|:---:|:---:|
| Task 1 | 0.7690 | 20% | 15.38 |
| Task 2 | 0.8120 | 40% | 32.48 |
| Task 3 | 0.8160 | 40% | 32.64 |
| **总分** | | | **81.64 / 100.00** |

## v3.1 相比 v3 的关键改进

| 改进 | 说明 |
|------|------|
| **CV制导模型选择** | 不再盲目录入 Voting Ensemble，CV 阶段选最优单模型（Task1/2 选 LR，Task3 选 LGB） |
| **共享清洗函数** | `src/clean_utils.py` 消除三处重复代码 |
| **Task3 class_weight=balanced** | 处理 Poor 类 114 条的极端不平衡 |
| **队友文件夹清理** | `任务1_统一基础清洗交付包_20260726/` 已移除，其规范已整合到 config.py |
| **Task1 得分提升** | 0.7585 → 0.7690（+1.05pp，因选用 LR 而非 Ensemble） |
| **Task2 得分提升** | 0.7940 → 0.8120（+1.80pp，因选用 LR 而非 Ensemble） |