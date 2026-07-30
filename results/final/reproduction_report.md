# 复现报告

> 生成时间: 2026-07-30T06:46:35.068598+00:00
> random_state: 20260726

## 运行环境

- Python: 3.12 (D:\python\python.exe)
- 依赖: pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn
- 数据: data/raw/A题数据集.csv (10,000×66)
- 切分: 8,000训练 / 2,000验证 (split_manifest.csv, 联合分层)

## 一键复现

```bash
D:\python\python.exe src\run_all.py
```

或分步运行：

```bash
D:\python\python.exe src\preprocess.py          # 预处理
D:\python\python.exe src\split.py                # 切分
D:\python\python.exe src\task1_final.py           # Task 1
D:\python\python.exe src\task2_health_score.py    # Task 2
D:\python\python.exe src\task3.py                 # Task 3
```

## 三任务最终指标

| 任务 | ACC | 权重 | 得分 | 最优模型 |
|------|:---:|:---:|:---:|------|
| Task 1 | 0.7535 | 20% | 15.07 | Logistic Regression |
| Task 2 | 0.8275 | 40% | 33.10 | Logistic Regression |
| Task 3 | 0.8485 | 40% | 33.94 | Logistic Regression (C=1) |
| **初赛总分** | | | **82.11** | |

## 数据完整性

- 输入: `data/processed/base_semantic_clean.csv` (10,000×66, 0缺失)
- 切分: `data/splits/split_manifest.csv` (8,000 train / 2,000 val)
- 单元测试: `python -m unittest discover -s tests -p "test_*.py" -v` → OK

## 模型文件

- Task 1: models/candidate/task1/task1_best_model.pkl
- Task 2: models/candidate/task2/task2_best_model.pkl
- Task 3: models/candidate/task3/task3_best_model.pkl

## 预测文件

- results/final/predictions/task1_predictions.csv (2,000行)
- results/final/predictions/task2_predictions.csv (2,000行)
- results/final/predictions/task3_predictions.csv (2,000行)
