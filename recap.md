# 各脚本输出文件一览（详细版）

---

## 一、`src/eda.py` → `outputs/eda/`

EDA（Exploratory Data Analysis，探索性数据分析）脚本，用于在建模前了解数据的基本面貌。不参与任何后续建模步骤，可跳过。

### 1. `numeric_describe.csv`
| 项目 | 说明 |
|------|------|
| **是什么** | 所有数值特征的描述性统计表 |
| **内容** | 每个数值列的 count（非空样本数）、mean（均值）、std（标准差）、min/max（最小/最大值）、25%/50%/75%分位数 |
| **用途** | 快速了解数据量纲差异。比如 `Daily_Steps` 均值约 8000 而 `BMI` 均值约 25，提示后续建模需要 StandardScaler 标准化 |
| **示例** | 打开 CSV 可以看到 Age 均值约 48 岁，Sleep_Duration_Hours 均值约 6.8 小时 |

### 2. `missing_values.csv`
| 项目 | 说明 |
|------|------|
| **是什么** | 缺失值统计表 |
| **内容** | 每个有缺失值的列名、缺失数量、缺失百分比 |
| **用途** | 确定哪些特征需要填充。本项目原始数据只有 Alcohol_Consumption（3014条）、Exercise_Type（824条）、Workout_Intensity（824条）有缺失 |
| **示例** | `Alcohol_Consumption: 3014 (30.14%)` |

### 3. `target_distribution.png`
| 项目 | 说明 |
|------|------|
| **是什么** | 三类目标变量的可视化分布图 |
| **内容** | 左：Early_Waker 饼图（Yes/No 占比）；中：Health_Score 直方图（连续值分布）；右：Wellness_Category 柱状图（各类计数） |
| **用途** | 了解类别平衡性。Early_Waker 约 41.6% Yes / 58.4% No，基本平衡；Wellness_Category 中 Poor 仅 114 条（1.1%），提示任务3需要 class_weight='balanced' 处理 |

### 4. `categorical_vs_target.png`
| 项目 | 说明 |
|------|------|
| **是什么** | 9个类别特征与 Early_Waker 目标的堆叠柱状图 |
| **内容** | 横轴是每个类别特征的取值，纵轴是样本数，颜色区分 Early_Waker=Yes/No 的堆叠比例 |
| **用途** | 观察哪些类别特征与早起行为相关。例如 Gender 中 Yes/No 比例差异较小，而 Smoking_Status 可能呈现一定差异 |

### 5. `numeric_kde_by_target.png`
| 项目 | 说明 |
|------|------|
| **是什么** | 14个核心数值特征的核密度估计（KDE）曲线图 |
| **内容** | 每条曲线是一个特征在 Early_Waker=Yes（蓝色）和 No（橙色）两组的概率密度分布。两曲线重叠越多，说明该特征区分早起/非早起的能力越弱 |
| **用途** | 定性判断哪些数值特征对分类有帮助。如 Productivity_Score 两条曲线明显分离（早起者生产力更高），而 Age 两条曲线几乎重叠（年龄与早起无关） |

### 6. `correlation_heatmap.png`
| 项目 | 说明 |
|------|------|
| **是什么** | 28个特征的相关性热力图 |
| **内容** | 每格颜色表示两个特征之间的皮尔逊相关系数。红色=正相关，蓝色=负相关，白色≈0 |
| **用途** | 发现强相关特征对（如 Height_cm 与 Weight_kg 正相关、Energy_Level_Score 与 Fatigue_Level_Score 负相关）以及各特征与 Health_Score 的关联强度。同时可以看出 Wake_Up_Time_Minutes 与 Sleep_Time_Minutes 高度共线（r ≈ -0.82）——这也是把它们排除的原因 |

---

## 二、`src/preprocess.py` → `data/processed/` + `data/splits/`

预处理脚本，执行统一清洗+编码+标准化+切分。**是所有后续任务的前置依赖**，三个任务脚本（task1_final.py、task2_health_score.py、task3_wellness_category.py）都是独立版，不依赖此脚本的输出——但 task1_early_waker.py（备选）依赖 `processed_data.pkl`。

### 1. `base_semantic_clean.csv`
| 项目 | 说明 |
|------|------|
| **是什么** | 缺失值填充后、任何编码/标准化**之前**的清洗数据 |
| **内容** | 完整的 10,000 行样本，每行一个人。相比原始 `datas.csv`（64列），多了 `Wake_Up_Time_Minutes` 和 `Sleep_Time_Minutes` 两列（66列）。Alcohol_Consumption 的 3014 个缺失值已替换为 "Unknown"，Exercise_Type/Workout_Intensity 的 824 个缺失值已替换为 "No Exercise"/"No Workout" |
| **为什么叫"semantic clean"** | 此时数据仍是**人类可读的原始文本**（如 Gender="Male"、Country="Italy"），没有被转成 0/1/2 的整数编码。方便队员审计清洗是否正确 |
| **用途** | 团队内部检查清洗质量；也可以作为各任务的原始输入（但需各自完成编码） |

### 2. `encoders.pkl`
| 项目 | 说明 |
|------|------|
| **是什么** | Python pickle 文件，包含所有 LabelEncoder 对象 |
| **内容** | 字典 `{特征名: LabelEncoder对象}`，共 17 个类别特征的编码器 + Health_Score_LabelEncoder + Wellness_Category_LabelEncoder |
| **用途** | 将编码后的整数 0/1/2 还原为原始文本标签。例如 `encoders['Gender'].inverse_transform([0,1])` → `['Female','Male']` |
| **加载方式** | `pickle.load(open("encoders.pkl","rb"))` |

### 3. `processed_data.pkl`
| 项目 | 说明 |
|------|------|
| **是什么** | 完整的预处理后数据打包文件 |
| **内容** | 字典，包含：<br>• `X_train` / `X_val`：标准化后的特征矩阵（8000×60 / 2000×60）<br>• `y1_train` / `y1_val`：Task1 标签（0/1）<br>• `y2_train` / `y2_val`：Task2 标签（0-3）<br>• `y3_train` / `y3_val`：Task3 标签（0-3）<br>• `ids_train` / `ids_val`：Person_ID<br>• `feat_cols`：60个特征名列表<br>• `actual_num_cols`：43个数值特征名<br>• `enc_cat_cols`：17个编码后类别特征名 |
| **注意事项** | 此时 60 列**已经包含所有可能泄漏的特征**（如 Wake_Up_Time_Minutes、Fitness_Level_Encoded 等）。各任务脚本需各自从中排除泄漏特征。`task1_early_waker.py`（备选版）依赖此文件 |
| **加载方式** | `pickle.load(open("processed_data.pkl","rb"))` |

### 4. `scaler.pkl`
| 项目 | 说明 |
|------|------|
| **是什么** | StandardScaler 对象 |
| **内容** | 在训练集 8000 个样本上拟合好的标准化器（每个数值特征的 mean 和 std） |
| **用途** | 对新数据（如测试集）做标准化时，必须用同样的 scaler。`scaler.transform(X_new)` 将新数据缩放到与训练集相同的分布 |

### 5. `split_manifest.csv`
| 项目 | 说明 |
|------|------|
| **是什么** | 统一数据切分明细表 |
| **内容** | 10,000 行，每行一个 Person_ID，包含列：<br>• `Person_ID`：P00001 ~ P10000<br>• `split`：`train`（8000人）或 `val`（2000人）<br>• `task1_label`：Yes / No<br>• `task2_label`：Poor / Average / Good / Excellent<br>• `task3_label`：Poor / Average / Good / Excellent |
| **用途** | 三人共享同一份切分，确保每个人的模型在**完全相同的数据**上训练和验证。禁止各自重新随机切分 |

---

## 三、`src/task1_final.py` → `outputs/task1/`

### 1. `predictions.csv`
| 项目 | 说明 |
|------|------|
| **是什么** | 任务1的最终预测结果表（比赛提交文件之一） |
| **格式** | 3列 × 2000行：`Person_ID`（个人标识）、`True_Label`（真实标签 Yes/No）、`Predicted_Label`（模型预测标签 Yes/No） |
| **评分方式** | 官方在测试集上计算 `(预测正确数 / 总测试样本数)` 得到 ACC1 |
| **注意事项** | True_Label 是验证集的真实标签（比赛提交时应删除此列，只提交 Person_ID + Predicted_Label） |

### 2. `feature_importance.csv`
| 项目 | 说明 |
|------|------|
| **是什么** | 57个特征的重要性排名表 |
| **计算方式** | 取 Random Forest、XGBoost、LightGBM 三个树模型各自的特征重要性（归一化后），求平均。因为 Logistic Regression 没有 `feature_importances_` 属性 |
| **Top 3 特征** | Productivity_Score（11.62%）、Breakfast_Regularity_Score（4.63%）、Stress_Level（2.40%） |
| **解读** | 排名第一的特征贡献达 11.6%，其余特征均在 5% 以下且分布均匀——说明早起行为不是由某个单一因素决定，而是**多维度生活方式综合体现** |

### 3. `best_model.pkl`
| 项目 | 说明 |
|------|------|
| **是什么** | 训练好的最优模型序列化文件（可以重新加载做预测） |
| **模型类型** | `LogisticRegression(max_iter=3000, C=1.0, random_state=20260726)` |
| **为什么是 Logistic Regression** | 5折CV中 LR 的 OOF ACC1=0.7482 高于其他三个模型；在测试集上 ACC1=0.7690 也最高。树模型（RF/XGB/LGB）有一定过拟合倾向 |
| **加载方式** | `pickle.load(open("best_model.pkl","rb"))`，然后 `model.predict(X_new)` |

### 4. `metrics.pkl` / `metrics.txt`
| 项目 | 说明 |
|------|------|
| **是什么** | 模型评估指标的完整记录（`.`pkl` 给程序读，`.txt` 给人读） |
| `.pkl` 内容 | 字典，包含 ACC1（0.7690）、Balanced_Accuracy（0.7559）、F1_Yes（0.7094）、Recall_Yes（0.6779）、5折CV结果、各模型测试集对比、排除特征列表 |
| `.txt` 内容 | 同上，纯文本格式，方便直接打开查看 |

---

## 四、`src/task2_health_score.py` → `outputs/task2/`

### 1. `predictions.csv`
| 项目 | 说明 |
|------|------|
| **是什么** | 任务2的最终预测结果表 |
| **格式** | `Person_ID` + `True_Label` + `Predicted_Label`（2000行，标签为 Poor/Average/Good/Excellent 四类） |
| **目标来源** | Health_Score 原始连续值按 [0,60,70,85,100] 边界离散化而来 |
| **得分** | ACC2 = 0.8120 |

### 2. `feature_importance.csv`
| 项目 | 说明 |
|------|------|
| **是什么** | 56个特征的重要性排名（排除了 Health_Score、Wellness_Category、Fitness_Level、Healthy_Aging_Score 等泄露/可疑特征） |
| **Top 3 特征** | BMI、Energy_Level_Score、Mood_Score |
| **解读** | BMI 排第一符合医学常识（肥胖直接影响健康评分）；精力水平和情绪评分紧随其后，说明心理健康是综合健康评分的重要组成部分 |

### 3. `best_model.pkl`
| 项目 | 说明 |
|------|------|
| **模型类型** | `LogisticRegression(max_iter=3000, C=1.0, random_state=20260726)` |
| **CV** | 5折 CV 均值 0.8044±0.0084 |

### 4. `metrics.pkl` / `metrics.txt`
| 项目 | 说明 |
|------|------|
| **内容** | ACC2=0.8120、各模型 CV 对比（LR 0.8044 > LGB 0.7667 > XGB 0.7596 > RF 0.7335）、分箱边界、四类 precision/recall/f1 |

---

## 五、`src/task3_wellness_category.py` → `outputs/task3/`

### 1. `predictions.csv`
| 项目 | 说明 |
|------|------|
| **是什么** | 任务3的最终预测结果表 |
| **格式** | `Person_ID` + `True_Label` + `Predicted_Label`（2000行，四类含 Poor） |
| **特殊说明** | 题面写"三分类 Excellent/Good/Average"，但数据中实际有 114 条 Poor。模型按四分类处理，论文中需说明这点 |
| **得分** | ACC3 = 0.8160（使用了 class_weight='balanced' 处理 Poor 类的极端不平衡） |

### 2. `feature_importance.csv`
| 项目 | 说明 |
|------|------|
| **是什么** | 56个特征的重要性排名（排除了 Wellness_Category、Health_Score、Fitness_Level 等泄漏特征） |
| **Top 3 特征** | Energy_Level_Score、BMI、Mood_Score（与 Task 2 类似，因为两个目标高度相关） |

### 3. `best_model.pkl`
| 项目 | 说明 |
|------|------|
| **模型类型** | `LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, class_weight='balanced')` |
| **为什么不是 Logistic Regression** | Task3 中 LightGBM 的 CV（0.8174）微优于 LR（0.8171），且 LR 的 Recall(Poor) 接近 0。LGB + class_weight=balanced 组合在保持整体准确率的同时，对 Poor 类的 Recall 有实质提升 |

### 4. `metrics.pkl` / `metrics.txt`
| 项目 | 说明 |
|------|------|
| **内容** | ACC3=0.8160、四类 precision/recall/f1（Poor 的 Precision=0.80 但 Recall=0.17，因为只有 23 条测试样本）、5折CV结果、类别分布 |