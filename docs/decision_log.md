# 决策日志 (Decision Log)

> 项目: 2026年第五届"创新杯"大学生大数据挑战赛 A 题
> 创建: 2026-07-27 15:00

| 时间 | 决策ID | 决策内容 | 决策人 | 复核人 | 状态 |
|------|--------|---------|--------|--------|------|
| 07-27 15:00 | D-001 | 确认 A/B/C 三人分工 (part_1.txt) | A(队长) | 三人 | 已确认 |
| 07-27 15:30 | D-002 | 统一 random_state=20260726 | A | B | 已锁定 |
| 07-27 15:30 | D-003 | 三份历史清洗对比: v1(原始)→v2(A旧方案"None")→v3(统一"No Exercise"/"No Workout") | A | B | 已锁定 |
| 07-27 15:30 | D-004 | Alcohol_Consumption 缺失 → "Unknown"（禁止填"No Alcohol"） | A | 三人 | 已锁定 |
| 07-27 15:30 | D-005 | 公共清洗数据不编码、不标准化、不截断、不删行 | A | 三人 | 已锁定 |
| 07-27 15:30 | D-006 | Task2 Health_Score 分箱边界: [0,60,70,85,100] | B(提案) | 三人 | 已确认 |
| 07-27 15:30 | D-007 | Task3 按四分类处理（含 Poor 114条），论文中说明冲突 | C(提案) | 三人 | 已确认 |
| 07-27 15:30 | D-008 | 三任务泄漏黑名单: T1-Wake_Up_Time族, T2-Health_Score/Fitness_Level/Wellness_Category, T3-Wellness_Category/Health_Score/Fitness_Level | A(提案) | B | 已锁定 |
| 07-27 15:30 | D-009 | Healthy_Aging_Score 作为可疑综合评分，主模型排除，消融实验可加入 | A(提案) | B | 已锁定 |
| 07-27 16:00 | D-010 | 采用联合分层 (task1+task2+task3标签) 而非单层 stratify | A(提案) | B | 已锁定 |
| 07-27 17:00 | D-011 | Task1 最优模型: LogisticRegression (CV ACC1=0.7482)，否决 Voting Ensemble | A | B | 已确认 |
| 07-27 17:00 | D-012 | Task2 最优模型: LogisticRegression (CV ACC2=0.8044)，否决 Voting Ensemble | A | B | 已确认 |
| 07-27 17:00 | D-013 | Task3 最优模型: LightGBM + class_weight=balanced (CV ACC3=0.8174) | A | B | 已确认 |
| 07-27 17:00 | D-014 | 创建 src/clean_utils.py 消除三处重复清洗代码 | A | B | 已确认 |
| 07-27 17:30 | D-015 | 统一 D27-05 数据契约：官方原始数据为 10,000×64；公共清洗表保留全部原始字段并新增两个分钟衍生字段，固定为 10,000×66 | B（数据契约复核） | 待团队同步 | 已执行 |
| 07-27 22:40 | D-016 | 锁定 Task2 当前可复现分档口径：[0,60] Poor、(60,70] Average、(70,85] Good、(85,100] Excellent | B | 三人 | 待复核 |
| 07-27 22:46 | D-017 | D27-11 暂不生成不完整的三任务汇总；等待 A/C 正式基线后再合并 | B | C | 已登记阻塞 |
