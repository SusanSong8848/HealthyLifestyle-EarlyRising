# 项目文件结构（统一重构版 v3）

```
HealthyLifestyle_EarlyRising/
├── datas.csv                        # 原始数据（本地，不在 git 中）
├── data/                            # 统一数据目录
│   ├── processed/
│   │   ├── base_semantic_clean.csv   # 基础清洗数据（编码前）
│   │   ├── processed_data.pkl        # 预处理后的训练/验证数据
│   │   ├── encoders.pkl              # 编码器
│   │   └── scaler.pkl                # 标准化器
│   └── splits/
│       └── split_manifest.csv        # 数据切分明细
├── outputs/                         # 输出文件（本地，不在 git 中）
│   ├── eda/                         # EDA 图表
│   ├── task1/                       # 任务1输出
│   ├── task2/                       # 任务2输出
│   └── task3/                       # 任务3输出
├── src/
│   ├── __init__.py
│   ├── config.py                    # 全局配置（v3 统一版）
│   ├── eda.py                       # Step 2: EDA 可视化
│   ├── preprocess.py                # Step 3: 数据预处理（v3 统一清洗规范）
│   ├── task1_final.py               # Step 4: 任务1（推荐使用）
│   ├── task1_early_waker.py         # Step 4: 任务1（备选，依赖 preprocess.py）
│   ├── task2_health_score.py        # Step 5: 任务2（v3 修复泄漏+更新分箱）
│   └── task3_wellness_category.py   # Step 6: 任务3（新增，v3 四分类）
├── .gitignore
├── Plan.md                          # 本文件
├── method.md                        # 建模方法论详细文档
├── part.txt                         # 角色分工说明
├── 要求.md                          # 赛题要求
└── 2026年第五届"创新杯"大学生大数据挑战赛A题.pdf
```

## 环境要求

- **Python 解释器**：`D:\python\python.exe`（Python 3.12）
- **已安装依赖**：pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm

### 运行方式

```bash
D:\python\python.exe src\eda.py
D:\python\python.exe src\preprocess.py
D:\python\python.exe src\task1_final.py              # 推荐：任务1 独立版
D:\python\python.exe src\task2_health_score.py       # 任务2
D:\python\python.exe src\task3_wellness_category.py  # 任务3
```

## 实现步骤

### Step 1: 环境搭建 ✅ 已完成

依赖库已安装：pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm

### Step 2: EDA与数据探索

**脚本**：`src/eda.py`

- 数据基本统计量与结构概览
- 缺失值检查
- 目标变量（Early_Waker, Health_Score, Wellness_Category）分布可视化
- 类别特征与目标变量关系分析
- 数值特征 KDE 分布（按 Early_Waker 分组）
- 相关性热力图

### Step 3: 数据预处理（v3 统一清洗规范）

**脚本**：`src/preprocess.py`

基于队友的 `公共数据要求.txt`，所有任务统一的"地基"：

- **统一清洗规则**：
  - `Exercise_Frequency_Per_Week=0` → `Exercise_Type="No Exercise"`, `Workout_Intensity="No Workout"`
  - `Alcohol_Consumption` 缺失 → `"Unknown"`（不能填"No Alcohol"）
  - 时间统一为 `HH:MM`（新增 `_Minutes` 列保留原始列）
  - 保留全部 10,000 行、64 列和 Person_ID
  - 不编码、不标准化、不截断异常值、不删行
- **统一随机种子**：`random_state=20260726`（团队统一）
- **联合分层抽样**：根据 Task1标签 + Task2等级 + Task3类别 联合分层
- 输出 `base_semantic_clean.csv`（编码前人类可读版本）
- 输出 `split_manifest.csv`（Person_ID, split, task1_label, task2_label, task3_label）
- 编码和标准化放入各任务 Pipeline 中完成

### Step 4: 任务1 - Early Waker 二分类（权重 20%）

**推荐脚本**：`src/task1_final.py`

**关键设计**：
- 排除泄漏特征：`Wake_Up_Time`, `Sleep_Time`, `Sleep_Duration_Hours`（三位一体全移除）
- 排除可疑综合评分：`Healthy_Aging_Score`（主模型排除，消融实验可加入）
- 5-Fold CV 管道内每折独立编码（fit on train fold, transform val fold）
- 分层抽样 stratify=y
- Voting Ensemble（LR + RF + XGB + LGB）
- 评估指标：ACC1, Balanced Accuracy, F1 (Yes), Recall (Yes)

### Step 5: 任务2 - Health Score 等级预测（权重 40%）

**脚本**：`src/task2_health_score.py`

- Health_Score 离散化边界（团队确认）：
  - `<60`: Poor
  - `60-70`: Average
  - `70-85`: Good
  - `>=85`: Excellent
- 移除泄漏特征：`Health_Score`, `Wellness_Category`, `Fitness_Level`
- 移除可疑特征：`Healthy_Aging_Score`
- Voting Ensemble（LR + RF + XGB + LGB）
- 生成预测结果 CSV

### Step 6: 任务3 - Wellness Category 预测（权重 40%）

**脚本**：`src/task3_wellness_category.py`（新增）

- 目标：四分类（Excellent / Good / Average / Poor）
  - 题面写三分类，但实际数据包含 114 条 Poor，按实际数据做四分类
- 移除泄漏特征：`Wellness_Category`, `Health_Score`, `Fitness_Level`
- 移除可疑特征：`Healthy_Aging_Score`
- Voting Ensemble（LR + RF + XGB + LGB）
- 生成预测结果 CSV

### Step 7: 结果汇总与提交文件

- 特征重要性分析
- 模型选择依据文档（method.md）
- 调参过程记录
- 各任务预测 CSV
- 最终评分预估

## v3 与之前版本的关键差异

| 项目 | v2（旧方案） | v3（统一重构版） |
|------|-------------|-----------------|
| random_state | 42 | **20260726**（团队统一） |
| 清洗 Alcohol 缺失 | "Unknown" ✅ | "Unknown" ✅ |
| 清洗 Exercise_Type 缺失 | "None" | **"No Exercise"**（团队统一） |
| 清洗 Workout_Intensity 缺失 | "None" | **"No Workout"**（团队统一） |
| Health_Score 分箱 | [0,50,65,80,100] | **[0,60,70,85,100]**（团队确认） |
| Task2 泄漏排除 | Healthy_Aging_Score, Fitness_Level | **Health_Score, Wellness_Category, Fitness_Level** |
| Task3 | 未实现 | **已实现（四分类）** |
| 数据切分 | 单层 stratify=y | **联合分层**（task1+task2+task3） |
| split_manifest.csv | 无 | **已生成** |
| targets.yaml | 无 | **待三人确认** |