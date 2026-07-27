## 项目文件结构

```
HealthyLifestyle_EarlyRising/
├── datas.csv                        # 原始数据（本地，不在 git 中）
├── outputs/                         # 输出文件（本地，不在 git 中）
│   ├── eda/                         # EDA 图表
│   ├── preprocess/                  # 预处理数据
│   ├── task1/                       # 任务1输出
│   ├── task2/                       # 任务2输出
│   └── task3/                       # 任务3输出
├── src/
│   ├── __init__.py
│   ├── config.py                    # 全局配置（路径、特征列表、超参数）
│   ├── eda.py                       # Step 2: EDA 可视化
│   ├── preprocess.py                # Step 3: 数据预处理
│   ├── task1_early_waker.py         # Step 4: 任务1（模块化版本，依赖 preprocess.py）
│   ├── task1_final.py              # Step 4: 任务1（最终独立版，整合方案A+B，推荐使用）
│   └── task2_health_score.py       # Step 5: 任务2（依赖 preprocess.py）
├── .gitignore
├── Plan.md                          # 本文件
├── method.md                        # 任务1 建模方法论详细文档
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
D:\python\python.exe src\task1_final.py          # 推荐：最终独立版
D:\python\python.exe src\task1_early_waker.py    # 备选：模块化版本
D:\python\python.exe src\task2_health_score.py
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

### Step 3: 数据预处理

**脚本**：`src/preprocess.py`

- 时间特征转换（Wake_Up_Time, Sleep_Time → 午夜起分钟数）
- 缺失值处理（Alcohol_Consumption → "Unknown"，Exercise_Type/Workout_Intensity → "None"）
- 类别特征 Label Encoding
- 目标变量编码（Early_Waker → 0/1，Health_Score → 四分类区间，Wellness_Category → 编码）
- 导出 Data_clean.csv（编码前的人类可读版本）
- 数值特征 StandardScaler 标准化
- 80/20 分层划分（stratify=Wellness_Category）
- 输出 processed_data.pkl, encoders.pkl, scaler.pkl

### Step 4: 任务1 - Early Waker 二分类（权重 20%）

**推荐脚本**：`src/task1_final.py`（整合方案 A + 方案 B）

**关键设计**：
- 排除 Wake_Up_Time, Sleep_Time, Sleep_Duration_Hours（三位一体，防止数据泄露）
- 5-Fold CV 管道内每折独立编码（fit on train fold, transform val fold）
- 分层抽样 stratify=y
- Voting Ensemble（LR + RF + XGB + LGB）
- 评估指标：ACC1, Balanced Accuracy, F1 (Yes), Recall (Yes)

**更多细节**见 `method.md`

### Step 5: 任务2 - Health Score 等级预测（权重 40%）

**脚本**：`src/task2_health_score.py`

- Health_Score 离散化为 Poor/Average/Good/Excellent 四分类
- 移除泄露特征（Healthy_Aging_Score, Fitness_Level_Encoded）
- 使用 class_weight='balanced' 处理类别不平衡
- XGBoost / LightGBM / Random Forest + GridSearchCV 调参
- 生成预测结果 CSV

### Step 6: 任务3 - Wellness Category 预测（权重 40%）

**状态**：待实现 ⏳

- 目标：三分类（Excellent/Good/Average）
- 建议特征：生活方式+心理情绪+行为特征
- 需创建 `src/task3_wellness_category.py`

### Step 7: 结果汇总与提交文件

- 特征重要性分析
- 模型选择依据文档（method.md）
- 调参过程记录
- 各任务预测 CSV
- 最终评分预估