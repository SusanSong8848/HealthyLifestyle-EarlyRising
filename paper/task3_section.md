# 任务三：健康综合类别预测

## 1 问题分析与类别口径

任务三要求根据生活方式、心理情绪、行为习惯及生理指标预测个体的综合健康类别 `Wellness_Category`，属于监督式多分类问题。题面将目标描述为 Excellent、Good、Average 三类，但对原始数据核验后发现实际还包含 114 个 Poor 样本，因此本文以数据中的四类标签 Poor、Average、Good、Excellent 为准，不删除少数类样本，并在结果中单独报告 Poor 类召回率。

四类样本数依次为 114、2 137、4 429 和 3 320，其中 Poor 仅占 1.14%，存在显著类别不平衡。若仅报告总体准确率，模型可能通过忽略 Poor 类获得较高分数。考虑到竞赛最终以 ACC3 计分，本文使用训练集五折交叉验证 Accuracy 作为首要选模指标，同时报告 Macro-F1、Balanced Accuracy 和四类 Recall，以检验模型对少数类的识别能力。

## 2 数据预处理与泄漏控制

公共清洗阶段不删除任何样本，不改变原始数值，不进行编码或标准化。对每周运动频率为 0 的样本，将运动类型和运动强度的结构性缺失分别解释为 `No Exercise` 和 `No Workout`；无法确认含义的饮酒缺失统一标记为 `Unknown`；时间字符串规范为 `HH:MM`。清洗后共有 10 000 行、66 列，其中新增的两个分钟变量仅用于审计和其他任务，任务三主模型不使用直接时间字段。

泄漏审查发现，`Fitness_Level` 与 `Wellness_Category` 的逐行一致率为 100%；同时，依据 `Health_Score` 的区间 `[27.3,44.9]`、`[45.0,64.9]`、`[65.0,79.9]` 和 `[80.0,100.0]` 可完整还原四类标签。因此，将 `Fitness_Level` 或 `Health_Score` 放入特征会使模型间接读取答案，造成虚假的高准确率。主模型删除 `Person_ID`、`Wellness_Category`、`Fitness_Level`、`Health_Score`，并保守排除可疑综合评分 `Healthy_Aging_Score`、跨任务标签 `Early_Waker` 及起床/入睡时间和对应分钟变量。最终使用 40 个数值字段和 16 个类别字段，共 56 个原始输入特征。

数值缺失填补与标准化、类别缺失填补与独热编码均封装在 `Pipeline` 中，并在每个交叉验证训练折内单独拟合，防止验证折信息提前进入预处理参数。类别编码后共形成 118 个模型输入维度。

## 3 数据划分与评价方法

三个任务共同使用按 `Person_ID` 对齐的唯一切分表，随机种子固定为 20260726。训练集包含 8 000 人，验证集包含 2 000 人；任务三各类别分布如下。

| 类别 | 训练集 | 验证集 |
|---|---:|---:|
| Poor | 91 | 23 |
| Average | 1 710 | 427 |
| Good | 3 543 | 886 |
| Excellent | 2 656 | 664 |

模型选择只使用训练集内部的分层五折交叉验证结果，验证集不参与模型或参数选择，仅在方案冻结后进行一次最终评价。主要指标定义如下：

\[
\mathrm{Accuracy}=\frac{\sum_{i=1}^{n}I(y_i=\hat y_i)}{n},
\]

\[
\mathrm{Macro\text{-}F1}=\frac{1}{K}\sum_{k=1}^{K}F1_k,
\qquad
\mathrm{Balanced\ Accuracy}=\frac{1}{K}\sum_{k=1}^{K}\mathrm{Recall}_k.
\]

其中 \(K=4\)。Accuracy 与竞赛 ACC3 完全一致；Macro-F1 和 Balanced Accuracy 对每一类赋予相同权重，可反映类别不平衡条件下的稳健性。

## 4 模型建立

### 4.1 Dummy 基线

Dummy 模型始终预测训练集中数量最多的 Good 类，用于衡量“不学习任何特征”时的最低基准。其五折 Accuracy 为 0.4429，Poor、Average 和 Excellent 的召回率均为 0，说明单纯利用类别比例无法完成任务。

### 4.2 多分类逻辑回归

对第 \(k\) 类，Softmax 逻辑回归给出

\[
P(y=k\mid \mathbf{x})
=
\frac{\exp(\mathbf{w}_k^\mathrm{T}\mathbf{x}+b_k)}
{\sum_{j=1}^{K}\exp(\mathbf{w}_j^\mathrm{T}\mathbf{x}+b_j)}.
\]

模型通过最小化多分类交叉熵与 L2 正则项之和估计参数。逻辑回归可直接处理标准化后的连续变量和独热编码后的类别变量，训练稳定、复现成本低，也便于进行特征解释。本文同时比较无权重版本与 `class_weight="balanced"` 版本，以分析类别权重对 Poor 类识别和总体准确率的影响。

### 4.3 树模型

选取平衡随机森林和 LightGBM 作为非线性对比模型。随机森林通过多棵决策树的集成降低单树方差；LightGBM 通过梯度提升逐步拟合残差，可捕捉变量之间的非线性关系和交互效应。LightGBM 的无权重版与平衡权重版除 `class_weight` 外保持全部参数相同，用于受控消融。

## 5 模型比较与参数选择

各候选模型在训练集五折交叉验证及冻结验证集上的结果如下。

| 模型 | CV Accuracy | CV Macro-F1 | CV Balanced Acc. | CV Poor Recall | Val Accuracy |
|---|---:|---:|---:|---:|---:|
| Dummy Most Frequent | 0.4429 | 0.1535 | 0.2500 | 0.0000 | 0.4430 |
| Logistic Regression（无权重） | **0.8481** | 0.7830 | 0.7676 | 0.5275 | **0.8485** |
| Logistic Regression（平衡） | 0.8355 | 0.7800 | **0.8171** | **0.7351** | 0.8385 |
| Random Forest（平衡） | 0.7750 | 0.6309 | 0.6170 | 0.1088 | 0.7865 |
| LightGBM（无权重） | 0.8084 | 0.6541 | 0.6308 | 0.1094 | 0.8160 |
| LightGBM（平衡） | 0.8140 | 0.7104 | 0.6763 | 0.2632 | 0.8245 |

无权重逻辑回归取得最高的交叉验证 Accuracy。平衡逻辑回归虽然将 Poor 类五折召回率从 0.5275 提高到 0.7351，但 Accuracy 从 0.8481 降至 0.8355，表明类别权重改善少数类识别的同时牺牲了官方评分指标。因此，本文不因少数类指标提高而直接选择平衡模型，而是将其作为公平性与稳健性的补充方案。

对表现最佳的逻辑回归家族进一步在训练集内搜索 \(C\in\{0.1,0.3,1,3,10\}\)。五折 Accuracy 分别为 0.8442、0.8481、0.8481、0.8471 和 0.8458。按 Accuracy 优先、Macro-F1 次之的规则，最终选取 \(C=1\)。小范围搜索未获得超过默认正则化强度的 ACC3 增益，因此不再进行大规模网格搜索。

在 LightGBM 受控消融中，平衡权重相对无权重版本使五折 Accuracy 提高 0.0056、Macro-F1 提高 0.0563、Balanced Accuracy 提高 0.0455、Poor Recall 提高 0.1538；验证集 Accuracy 提高 0.0085、Poor Recall 提高 0.0870。说明类别权重对 LightGBM 有稳定帮助，但两种 LightGBM 的总体表现仍低于逻辑回归。

## 6 最终结果与误差分析

最终模型为 \(C=1\) 的无权重多分类逻辑回归。其验证集结果为：

| 指标 | 数值 |
|---|---:|
| ACC3 / Accuracy | **0.8485** |
| Macro-F1 | 0.7832 |
| Balanced Accuracy | 0.7652 |
| Poor Recall | 0.5217 |
| Average Recall | 0.8080 |
| Good Recall | 0.8578 |
| Excellent Recall | 0.8735 |

混淆矩阵如下：

| 真实类别 \ 预测类别 | Poor | Average | Good | Excellent |
|---|---:|---:|---:|---:|
| Poor | 12 | 11 | 0 | 0 |
| Average | 6 | 345 | 76 | 0 |
| Good | 0 | 54 | 760 | 72 |
| Excellent | 0 | 0 | 84 | 580 |

模型错误全部发生在相邻等级之间，未出现 Poor 被判为 Good/Excellent 或 Excellent 被判为 Average/Poor 的跨两级错误，说明模型学习到了健康等级的有序结构。主要混淆来自 Average 与 Good、Good 与 Excellent 的分界区域。Poor 类仅有 23 个验证样本，其中 12 个被正确识别、11 个被判为 Average；由于一个样本即可使 Poor Recall 变化约 4.35 个百分点，该指标的不确定性明显高于其他类别，解释时不宜过度外推。

## 7 特征重要性与解释

在最终模型冻结后，本文在验证集上逐一打乱原始字段，记录 Accuracy 的平均下降量作为置换重要性。排名前十的字段为：睡眠质量评分（0.1415）、每周运动频率（0.1324）、BMI（0.1215）、压力水平（0.0759）、每周快餐次数（0.0675）、每日步数（0.0634）、肥胖风险（0.0450）、吸烟状态（0.0329）、每日蔬菜摄入量（0.0199）和情绪评分（0.0157）。

结果表明，任务三标签主要由睡眠、运动、体重状态、压力、饮食和吸烟等多维因素共同驱动，与题目所强调的生活方式、心理情绪及行为特征一致。需要强调的是，置换重要性衡量的是字段对当前模型预测的贡献，不代表改变该字段一定会造成健康类别变化，也不能作因果解释。

## 8 模型评价与局限

本方案的优点是：严格排除可直接还原目标的字段；统一使用可追溯的 Person_ID 切分；所有编码、填补和标准化均在交叉验证折内完成；模型、调参、消融、预测、图表和指标文件可由同一脚本复现。最终逻辑回归结构简单、训练快速，且在本数据上优于更复杂的树模型，说明复杂模型并不必然带来更高得分。

局限主要包括：Poor 类样本极少，少数类指标方差较大；验证结论来自同一数据集的随机划分，尚未经过外部数据检验；部分综合指标被排除后，模型侧重于可解释的原始行为与生理字段，可能牺牲一定上限；置换重要性存在相关特征相互替代的影响。后续若有更多时间，可在不接触验证集的前提下尝试有序分类损失或概率校准，但不建议在当前截止时间前加入深度学习、Stacking 或大规模参数搜索。

## 9 本节数据来源

- 模型比较：`results/metrics/raw/task3_model_comparison.csv`
- 参数搜索：`results/metrics/raw/task3_tuning.csv`
- 权重消融：`results/metrics/raw/task3_weight_ablation.csv`
- 分类报告：`results/metrics/raw/task3_classification_report.csv`
- 混淆矩阵：`results/figures/candidate/task3_confusion_matrix.csv`
- 特征重要性：`results/metrics/raw/task3_feature_importance.csv`
- 最终摘要：`results/metrics/raw/task3_metrics.json`
- 预测结果：`results/predictions/candidate/task3_predictions.csv`
