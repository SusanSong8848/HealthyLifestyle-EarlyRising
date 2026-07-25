## 📋 完整项目实施计划

### 项目文件结构

```javascript
HealthyLifestyle_EarlyRising/
├── data/
│   └── datas.csv                    # 原始数据
├── notebooks/
│   └── eda.ipynb                    # EDA (可选)
├── src/
│   ├── __init__.py
│   ├── config.py                    # 配置文件（路径、参数）
│   ├── preprocess.py                # 数据预处理
│   ├── eda.py                       # 可视化EDA脚本
│   ├── task1_early_waker.py         # 任务1：早起行为预测
│   ├── task2_health_score.py        # 任务2：健康评分等级预测
│   ├── task3_wellness_category.py   # 任务3：健康综合类别预测
│   └── utils.py                     # 工具函数
├── outputs/
│   ├── task1/                       # 任务1输出
│   ├── task2/                       # 任务2输出
│   └── task3/                       # 任务3输出
├── main.py                          # 主执行脚本
├── requirements.txt
└── README.md
```

### 实现步骤

#### Step 1: 环境搭建 → 安装依赖库（进行中）

#### Step 2: EDA与数据探索

- 观察基本统计量
- 检查缺失值
- 目标变量分布可视化
- 特征分布与相关性分析

#### Step 3: 数据预处理

- 时间特征转换（Wake_Up_Time, Sleep_Time）
- 类别特征编码 (One-Hot / Label Encoding)
- 数值特征标准化
- 缺失值与异常值处理

#### Step 4: 任务1 - Early Waker 二分类 (权重20%)

- 特征选择
- 训练多个模型（XGBoost, LightGBM, Random Forest）
- 5折交叉验证
- 最优模型选择与调参
- 生成预测结果CSV

#### Step 5: 任务2 - Health Score 等级预测 (权重40%)

- 将Health_Score离散化为 Excellent/Good/Average/Poor（四分类）
- 特征工程（生理特征、疾病风险特征为主）
- 多分类模型训练与调优
- 生成预测结果CSV

#### Step 6: 任务3 - Wellness Category 预测 (权重40%)

- 三分类（Excellent/Good/Average）
- 特征工程（生活方式+心理情绪+行为特征为主）
- 多分类模型训练与调优
- 生成预测结果CSV

#### Step 7: 结果汇总与提交文件

- 特征重要性分析
- 模型选择依据文档
- 调参过程记录
- 各任务预测CSV
- 最终评分预估
