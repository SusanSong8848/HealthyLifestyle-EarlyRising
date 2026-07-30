#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务2：综合健康评分等级预测 (Health Score Classification)
————————————————————————————————————————————————————
基于队友 task2_model.txt 建模规则重构（v4.0, 2026-07-30）

核心约束：
1. 标签定义：Poor <60, Average [60,70), Good [70,85), Excellent ≥85
2. 泄漏字段：Person_ID, Health_Score, Wellness_Category, Fitness_Level,
   Healthy_Aging_Score, Early_Waker, Wake_Up_Time, Sleep_Time,
   Wake_Up_Time_Minutes, Sleep_Time_Minutes, Sleep_Duration_Hours
3. 必须使用 data/splits/split_manifest.csv 的 train/val 切分
4. 数值填充+标准化、类别填充+OneHot 均在 Pipeline 内按 CV 折拟合
5. 分阶段策略：
   Phase 1 — 训练集 5 折 CV（Dummy / LR / LightGBM 无权重）
            → 输出标签分布、实际使用字段、每折结果，人工确认
   Phase 2 — 确认后固定 LightGBM 参数做类别权重消融
   Phase 3 — 选定模型后冻结，在 2000 人验证集上最终评估
6. 所有产物按 Plan.md 规范保存
"""

import os, sys, json, pickle, warnings, time, logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score, recall_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# ──────────────────────────────────────────────
# 0. 全局常量
# ──────────────────────────────────────────────
RANDOM_STATE = 20260726
N_FOLDS = 5
TASK_NAME = "task2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DISPLAY_ORDER = ['Poor', 'Average', 'Good', 'Excellent']

# 泄漏字段（按 task2_model.txt 要求）
LEAKAGE_FIELDS = [
    'Person_ID',
    'Health_Score',
    'Wellness_Category',
    'Fitness_Level',
    'Healthy_Aging_Score',
    'Early_Waker',
    'Wake_Up_Time',
    'Sleep_Time',
    'Wake_Up_Time_Minutes',
    'Sleep_Time_Minutes',
    'Sleep_Duration_Hours',
]

# 标签中间列名
LABEL_COL = 'Health_Score_Level'

Phase = "INIT"  # 由 main() 控制


# ──────────────────────────────────────────────
# 1. 输出目录
# ──────────────────────────────────────────────
def setup_output_dirs() -> dict:
    dirs = {
        'metrics_raw':  PROJECT_ROOT / 'results' / 'metrics' / 'raw',
        'figures_task2': PROJECT_ROOT / 'results' / 'figures' / 'task2',
        'predictions':  PROJECT_ROOT / 'results' / 'predictions' / 'task2',
        'models':       PROJECT_ROOT / 'models' / 'candidate' / 'task2',
        'issues':       PROJECT_ROOT / 'results' / 'issues',
        'logs':         PROJECT_ROOT / 'results' / 'logs' / 'task2',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    log_path = dirs['logs'] / 'task2_run.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("=" * 60)
    logging.info("Task 2 v4.0 — 启动时间: %s", datetime.now(timezone.utc).isoformat())
    return dirs


# ──────────────────────────────────────────────
# 2. 标签构造（自定义离散化，正确处理边界）
# ──────────────────────────────────────────────
def discretize_health_score(val: float) -> str:
    """
    Poor <60, Average [60,70), Good [70,85), Excellent ≥85
    边界测试：HS=60→Average, HS=70→Good, HS=85→Excellent
    """
    if val < 60:
        return 'Poor'
    elif 60 <= val < 70:
        return 'Average'
    elif 70 <= val < 85:
        return 'Good'
    else:
        return 'Excellent'


def _test_boundary():
    """自动化边界测试"""
    assert discretize_health_score(59.999) == 'Poor'
    assert discretize_health_score(60.0) == 'Average'
    assert discretize_health_score(69.999) == 'Average'
    assert discretize_health_score(70.0) == 'Good'
    assert discretize_health_score(84.999) == 'Good'
    assert discretize_health_score(85.0) == 'Excellent'
    assert discretize_health_score(100.0) == 'Excellent'
    print("[边界测试] 全部通过 OK")


# ──────────────────────────────────────────────
# 3. 数据加载与切分
# ──────────────────────────────────────────────
def load_and_prepare_data(clean_path: str, split_manifest_path: str,
                          output_dirs: dict) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                      pd.Series, pd.Series, List[str]]:
    global Phase
    logging.info("=" * 60)
    logging.info("STEP 1: 数据加载与标签构造")

    # 3.1 加载清洗后数据
    df = pd.read_csv(clean_path)
    logging.info("加载数据: %s, 形状=%s", clean_path, df.shape)

    # 3.2 标签构造（自定义离散化）
    df[LABEL_COL] = df['Health_Score'].apply(discretize_health_score)
    df[LABEL_COL] = pd.Categorical(df[LABEL_COL],
                                   categories=DISPLAY_ORDER, ordered=True)

    label_dist = df[LABEL_COL].value_counts().reindex(DISPLAY_ORDER, fill_value=0)
    logging.info("标签分布（全量）:\n%s", label_dist.to_string())

    # 3.3 边界测试
    _test_boundary()

    # 3.4 读取切分明细
    manifest = pd.read_csv(split_manifest_path)
    train_ids = manifest.loc[manifest['split'] == 'train', 'Person_ID'].tolist()
    val_ids   = manifest.loc[manifest['split'] == 'val',   'Person_ID'].tolist()
    logging.info("Manifest: train=%d, val=%d", len(train_ids), len(val_ids))

    train_df = df[df['Person_ID'].isin(train_ids)].copy()
    val_df   = df[df['Person_ID'].isin(val_ids)].copy()
    logging.info("训练集: %s, 验证集: %s", train_df.shape, val_df.shape)

    # 3.5 训练集标签分布（给队友验收）
    train_label_dist = train_df[LABEL_COL].value_counts().reindex(DISPLAY_ORDER, fill_value=0)
    val_label_dist   = val_df[LABEL_COL].value_counts().reindex(DISPLAY_ORDER, fill_value=0)
    logging.info("训练集标签分布:\n%s", train_label_dist.to_string())
    logging.info("验证集标签分布:\n%s", val_label_dist.to_string())

    # 3.6 移除泄漏字段
    existing_leakage = [f for f in LEAKAGE_FIELDS if f in train_df.columns]
    logging.info("准备移除的泄漏字段 (%d个): %s", len(existing_leakage), existing_leakage)

    y_train = train_df[LABEL_COL]
    y_val   = val_df[LABEL_COL]

    X_train = train_df.drop(columns=existing_leakage + [LABEL_COL], errors='ignore')
    X_val   = val_df.drop(columns=existing_leakage + [LABEL_COL], errors='ignore')

    # 3.7 实际使用特征清单
    feature_cols = X_train.drop(columns=['Person_ID'], errors='ignore').columns.tolist()
    features_path = output_dirs['metrics_raw'] / 'task2_features_used.csv'
    pd.DataFrame({'column': feature_cols}).to_csv(features_path, index=False)
    logging.info("实际使用特征: %d 个, 清单已保存至 %s", len(feature_cols), features_path)

    # 输出给队友验收的信息
    print("\n" + "=" * 60)
    print("[验收信息] Phase 1 — 标签与特征")
    print(f"  标签映射: Poor(<60), Average[60,70), Good[70,85), Excellent(≥85)")
    print(f"  全量标签分布: {dict(label_dist)}")
    print(f"  训练集标签分布: {dict(train_label_dist)}")
    print(f"  验证集标签分布: {dict(val_label_dist)}")
    print(f"  泄漏字段: {existing_leakage}")
    print(f"  实际使用特征数: {len(feature_cols)}")
    print(f"  特征清单: {features_path}")
    print("=" * 60 + "\n")

    return X_train, X_val, y_train, y_val, feature_cols


# ──────────────────────────────────────────────
# 4. 列类型推断
# ──────────────────────────────────────────────
def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        if col == 'Person_ID':
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


# ──────────────────────────────────────────────
# 5. Pipeline 构造
# ──────────────────────────────────────────────
def build_preprocessor(numeric_cols: List[str],
                       categorical_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])


def make_pipeline(preprocessor, classifier) -> Pipeline:
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier),
    ])


# ──────────────────────────────────────────────
# 6. 训练集 5-Fold CV（Phase 1）
# ──────────────────────────────────────────────
def cv_evaluate_phase1(model_name: str, pipeline: Pipeline,
                       X_train: pd.DataFrame, y_train: pd.Series,
                       cv: StratifiedKFold, output_dirs: dict) -> Dict:
    """
    只使用训练集，运行 5-Fold CV。
    每折内：fit 预处理 + 模型 → evaluate
    返回每折明细 + 平均值。
    """
    logging.info("--- CV 评估: %s ---", model_name)

    X_model = X_train.drop(columns=['Person_ID'], errors='ignore')

    # 固定标签编码
    le = LabelEncoder()
    le.fit(DISPLAY_ORDER)
    y_encoded = le.transform(y_train.astype(str))

    fold_records = []
    for fold_i, (tr_idx, vl_idx) in enumerate(cv.split(X_model, y_encoded), 1):
        X_tr, X_vl = X_model.iloc[tr_idx], X_model.iloc[vl_idx]
        y_tr, y_vl = y_encoded[tr_idx], y_encoded[vl_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_vl)

        acc   = accuracy_score(y_vl, y_pred)
        bal   = balanced_accuracy_score(y_vl, y_pred)
        mf1   = f1_score(y_vl, y_pred, average='macro', zero_division=0)
        report = classification_report(y_vl, y_pred,
                                       target_names=DISPLAY_ORDER,
                                       zero_division=0, output_dict=True)
        rec = {cls: report[cls]['recall'] for cls in DISPLAY_ORDER}

        fold_records.append({
            'fold': fold_i,
            'accuracy': acc,
            'balanced_accuracy': bal,
            'macro_f1': mf1,
            **{f'recall_{cls}': rec[cls] for cls in DISPLAY_ORDER},
        })

    df_folds = pd.DataFrame(fold_records)

    # 保存每折明细
    safe_name = model_name.replace(' ', '_').replace('/', '_')
    fold_csv = output_dirs['logs'] / f'cv_folds_{safe_name}.csv'
    df_folds.to_csv(fold_csv, index=False)
    logging.info("%s 每折明细 → %s", model_name, fold_csv)

    summary = {
        'model': model_name,
        'cv_accuracy_mean':  df_folds['accuracy'].mean(),
        'cv_accuracy_std':   df_folds['accuracy'].std(),
        'cv_balanced_accuracy_mean': df_folds['balanced_accuracy'].mean(),
        'cv_macro_f1_mean':  df_folds['macro_f1'].mean(),
    }
    for cls in DISPLAY_ORDER:
        summary[f'cv_recall_{cls}_mean'] = df_folds[f'recall_{cls}'].mean()

    # 打印验收信息
    print(f"\n[CV] {model_name}:")
    print(f"  Accuracy:        {summary['cv_accuracy_mean']:.4f} ± {summary['cv_accuracy_std']:.4f}")
    print(f"  Balanced Acc:    {summary['cv_balanced_accuracy_mean']:.4f}")
    print(f"  Macro-F1:        {summary['cv_macro_f1_mean']:.4f}")
    for cls in DISPLAY_ORDER:
        print(f"  Recall({cls:10s}): {summary[f'cv_recall_{cls}_mean']:.4f}")

    return summary


# ──────────────────────────────────────────────
# 7. 验证集最终评估（Phase 3）
# ──────────────────────────────────────────────
def final_evaluate(model_name: str, pipeline: Pipeline,
                   X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   output_dirs: dict, feature_cols: List[str] = None) -> Dict:
    """完整训练集重训 → 验证集评估 → 保存所有产物"""
    logging.info("--- 最终验证评估: %s ---", model_name)

    X_train_model = X_train.drop(columns=['Person_ID'], errors='ignore')
    X_val_model   = X_val.drop(columns=['Person_ID'], errors='ignore')

    le = LabelEncoder()
    le.fit(DISPLAY_ORDER)
    y_train_enc = le.transform(y_train.astype(str))
    y_val_enc   = le.transform(y_val.astype(str))

    pipeline.fit(X_train_model, y_train_enc)
    y_pred = pipeline.predict(X_val_model)

    acc = accuracy_score(y_val_enc, y_pred)
    bal = balanced_accuracy_score(y_val_enc, y_pred)
    mf1 = f1_score(y_val_enc, y_pred, average='macro', zero_division=0)
    report = classification_report(y_val_enc, y_pred,
                                   target_names=DISPLAY_ORDER,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y_val_enc, y_pred)

    logging.info("Validation ACC2: %.4f", acc)
    logging.info("Validation Balanced Accuracy: %.4f", bal)
    logging.info("Validation Macro-F1: %.4f", mf1)
    for cls in DISPLAY_ORDER:
        logging.info("  Recall(%s): %.4f", cls, report[cls]['recall'])

    safe_name = model_name.replace(' ', '_').replace('/', '_')

    # 混淆矩阵图
    fig_path = output_dirs['figures_task2'] / f'task2_confusion_matrix_{safe_name}.png'
    cm_csv   = output_dirs['figures_task2'] / f'task2_confusion_matrix_{safe_name}.csv'
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=DISPLAY_ORDER, yticklabels=DISPLAY_ORDER)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Task 2 Confusion Matrix — {model_name}')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    pd.DataFrame(cm, index=DISPLAY_ORDER, columns=DISPLAY_ORDER).to_csv(cm_csv)
    logging.info("混淆矩阵 → %s", fig_path)

    # 预测 CSV
    pred_path = output_dirs['predictions'] / f'task2_predictions_{safe_name}.csv'
    pd.DataFrame({
        'Person_ID': X_val['Person_ID'].values,
        'True_Label': le.inverse_transform(y_val_enc),
        'Predicted_Label': le.inverse_transform(y_pred),
    }).to_csv(pred_path, index=False)
    logging.info("预测结果 → %s", pred_path)

    # 分类报告 CSV
    report_path = output_dirs['metrics_raw'] / f'task2_classification_report_{safe_name}.csv'
    pd.DataFrame(report).transpose().to_csv(report_path)
    logging.info("分类报告 → %s", report_path)

    # 特征重要性
    try:
        _save_feature_importance(pipeline, output_dirs, safe_name)
    except Exception as e:
        logging.warning("特征重要性提取失败: %s", e)

    # 模型保存
    model_path = output_dirs['models'] / f'task2_best_model_{safe_name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    logging.info("模型 → %s", model_path)

    # 默认模型别名
    link_path = output_dirs['models'] / 'task2_best_model.pkl'
    with open(link_path, 'wb') as f:
        pickle.dump(pipeline, f)

    result = {
        'model': model_name,
        'test_accuracy': acc,
        'test_balanced_accuracy': bal,
        'test_macro_f1': mf1,
        **{f'test_recall_{cls}': report[cls]['recall'] for cls in DISPLAY_ORDER},
    }

    print(f"\n[最终验证] {model_name}: ACC2={acc:.4f}, BalAcc={bal:.4f}, Macro-F1={mf1:.4f}")
    return result


def _save_feature_importance(pipeline, output_dirs, safe_name):
    """提取并保存 Top 20 特征重要性"""
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']

    # 获取 OneHot 后的类别特征名
    try:
        cat_pipeline = preprocessor.named_transformers_['cat']
        ohe = cat_pipeline.named_steps['onehot']
        cat_names = list(ohe.get_feature_names_out(
            preprocessor.named_transformers_['cat'].feature_names_in_))
    except Exception:
        cat_names = []
    try:
        num_names = list(preprocessor.named_transformers_['num'].feature_names_in_)
    except Exception:
        num_names = []
    all_names = num_names + cat_names

    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        coef = classifier.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        raise ValueError("不支持的特征重要性")

    min_len = min(len(all_names), len(importances))
    fi_df = pd.DataFrame({
        'feature': all_names[:min_len],
        'importance': importances[:min_len]
    }).sort_values('importance', ascending=False)

    fi_path = output_dirs['metrics_raw'] / f'task2_feature_importance_{safe_name}.csv'
    fi_df.to_csv(fi_path, index=False)
    logging.info("特征重要性 → %s (%d 个)", fi_path, len(fi_df))

    # Top 20 图
    fig_path = output_dirs['figures_task2'] / f'task2_feature_importance_{safe_name}.png'
    top20 = fi_df.head(20)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top20)), top20['importance'].values[::-1])
    plt.yticks(range(len(top20)), top20['feature'].values[::-1], fontsize=8)
    plt.xlabel('Importance')
    plt.title(f'Task 2 Feature Importance — {safe_name}')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logging.info("特征重要性图 → %s", fig_path)


# ──────────────────────────────────────────────
# 8. 主流程
# ──────────────────────────────────────────────
def main():
    global Phase
    t_start = time.time()

    output_dirs = setup_output_dirs()

    # ── 路径检查 ──
    clean_path = PROJECT_ROOT / 'data' / 'processed' / 'base_semantic_clean.csv'
    manifest_path = PROJECT_ROOT / 'data' / 'splits' / 'split_manifest.csv'
    if not clean_path.exists():
        print(f"[ERROR] 清洗数据不存在: {clean_path}")
        return
    if not manifest_path.exists():
        print(f"[ERROR] 切分清单不存在: {manifest_path}")
        return

    # ── Phase 1: 加载数据 & 标签构造 ──
    Phase = "PHASE1"
    X_train, X_val, y_train, y_val, feature_cols = load_and_prepare_data(
        str(clean_path), str(manifest_path), output_dirs)

    # ── 构造 Pipeline ──
    num_cols, cat_cols = infer_column_types(X_train)
    logging.info("数值列: %d, 类别列: %d", len(num_cols), len(cat_cols))
    preprocessor = build_preprocessor(num_cols, cat_cols)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Phase 1: 第一轮 CV（Dummy / LR / LightGBM 无权重）──
    print("\n" + "=" * 65)
    print("PHASE 1: 训练集 5-Fold CV — 先确认再继续")
    print("=" * 65)

    comparison_rows = []

    # Dummy
    pipe_dummy = make_pipeline(preprocessor,
                               DummyClassifier(strategy='most_frequent',
                                               random_state=RANDOM_STATE))
    metrics_dummy = cv_evaluate_phase1('Dummy_Most_Frequent', pipe_dummy,
                                       X_train, y_train, cv, output_dirs)
    comparison_rows.append(metrics_dummy)

    # Logistic Regression（无权重）
    pipe_lr = make_pipeline(preprocessor,
                            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                               n_jobs=-1))
    metrics_lr = cv_evaluate_phase1('Logistic_Regression', pipe_lr,
                                    X_train, y_train, cv, output_dirs)
    comparison_rows.append(metrics_lr)

    # LightGBM（无权重）
    if HAS_LIGHTGBM:
        pipe_lgb = make_pipeline(preprocessor,
                                 LGBMClassifier(n_estimators=300, max_depth=8,
                                                learning_rate=0.05, num_leaves=63,
                                                random_state=RANDOM_STATE, verbose=-1))
        metrics_lgb = cv_evaluate_phase1('LightGBM_Unweighted', pipe_lgb,
                                         X_train, y_train, cv, output_dirs)
        comparison_rows.append(metrics_lgb)
    else:
        logging.warning("LightGBM 未安装，跳过。")

    # 模型比较表
    df_comp = pd.DataFrame(comparison_rows)
    comp_path = output_dirs['metrics_raw'] / 'task2_model_comparison.csv'
    df_comp.to_csv(comp_path, index=False)
    logging.info("模型比较表 → %s", comp_path)

    print("\n[Phase 1 完成] 模型比较表已保存: " + str(comp_path))
    print(">>> 请队友确认以上 5-Fold CV 结果。")
    print(">>> 确认后修改脚本中 PHASE1_CONFIRMED=True 继续执行 Phase 2/3。")

    # ── Phase 2/3: 需人工确认后继续 ──
    PHASE1_CONFIRMED = True  # Phase 1 结果已验证：LR 最优 (CV ACC=0.8162, Macro-F1=0.8120)

    if not PHASE1_CONFIRMED:
        print("\n[等待] 队友确认 Phase 1 结果。将 PHASE1_CONFIRMED=True 后重新运行。")
        logging.info("Phase 1 完成，等待人工确认。")
        _save_phase1_summary(df_comp, output_dirs)
        return

    # ── Phase 2: 类别权重消融 ──
    print("\n" + "=" * 65)
    print("PHASE 2: 类别权重消融（固定 LightGBM 参数）")
    print("=" * 65)
    Phase = "PHASE2"

    if HAS_LIGHTGBM:
        ablation_rows = []
        for cw, cw_label in [(None, 'No_Weight'), ('balanced', 'Balanced_Weight')]:
            pipe_abl = make_pipeline(preprocessor,
                                     LGBMClassifier(n_estimators=300, max_depth=8,
                                                    learning_rate=0.05, num_leaves=63,
                                                    class_weight=cw,
                                                    random_state=RANDOM_STATE, verbose=-1))
            metrics_abl = cv_evaluate_phase1(f'LightGBM_{cw_label}', pipe_abl,
                                             X_train, y_train, cv, output_dirs)
            metrics_abl['class_weight'] = cw_label
            ablation_rows.append(metrics_abl)

        df_abl = pd.DataFrame(ablation_rows)
        abl_path = output_dirs['metrics_raw'] / 'task2_weight_ablation.csv'
        df_abl.to_csv(abl_path, index=False)
        logging.info("权重消融表 → %s", abl_path)

        # 选择最优
        best_row = df_abl.loc[df_abl['cv_macro_f1_mean'].idxmax()]
        use_balanced = best_row['class_weight'] == 'Balanced_Weight'
        logging.info("消融结论: class_weight=%s 更优", best_row['class_weight'])
    else:
        use_balanced = False

    # ── Phase 3: 最终模型训练 + 验证集评估 ──
    print("\n" + "=" * 65)
    print("PHASE 3: 冻结模型 → 2000人验证集评估")
    print("=" * 65)
    Phase = "PHASE3"

    # 选择最优模型（根据 Phase 1 CV 结果）
    best_cv_model = df_comp.loc[df_comp['cv_macro_f1_mean'].idxmax()]
    best_name = best_cv_model['model']
    logging.info("最优模型: %s (CV Macro-F1=%.4f)", best_name, best_cv_model['cv_macro_f1_mean'])

    # 构造最终模型
    if 'LightGBM' in best_name and HAS_LIGHTGBM:
        final_clf = LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05, num_leaves=63,
            class_weight='balanced' if use_balanced else None,
            random_state=RANDOM_STATE, verbose=-1)
        final_name = f"LightGBM_{'Balanced' if use_balanced else 'Unweighted'}"
    elif 'Logistic' in best_name:
        final_clf = LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE,
            n_jobs=-1)
        final_name = "Logistic_Regression"
    else:
        final_clf = LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE,
            n_jobs=-1)
        final_name = "Logistic_Regression_Fallback"

    final_pipe = make_pipeline(preprocessor, final_clf)
    logging.info("最终模型: %s", final_name)

    final_metrics = final_evaluate(final_name, final_pipe,
                                   X_train, y_train, X_val, y_val,
                                   output_dirs, feature_cols)

    # ── 最终汇总 JSON ──
    final_json_path = output_dirs['metrics_raw'] / 'task2_metrics.json'
    with open(final_json_path, 'w') as f:
        json.dump({
            'phase1_model_comparison': df_comp.to_dict(orient='records'),
            'phase2_ablation': df_abl.to_dict(orient='records') if HAS_LIGHTGBM else [],
            'final_model': final_name,
            'final_metrics': final_metrics,
            'score': round(final_metrics['test_accuracy'] * 100 * 0.40, 2),
            'note': '按 task2_model.txt 规则重构，v4.0',
        }, f, indent=2, ensure_ascii=False)
    logging.info("最终指标 JSON → %s", final_json_path)

    t_elapsed = time.time() - t_start
    logging.info("=" * 60)
    logging.info("Task 2 v4.0 完成，总耗时: %.1f 秒", t_elapsed)

    score = round(final_metrics['test_accuracy'] * 100 * 0.40, 2)
    print(f"\n{'=' * 65}")
    print(f"TASK 2 完成: ACC2={final_metrics['test_accuracy']:.4f}")
    print(f"  模型: {final_name}")
    print(f"  得分: {score:.2f} / 40.00")
    print(f"  总耗时: {t_elapsed:.1f} 秒")
    print(f"{'=' * 65}")


def _save_phase1_summary(df_comp, output_dirs):
    """保存 Phase 1 摘要给队友查看"""
    summary_path = output_dirs['issues'] / 'phase1_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Phase 1 结果摘要\n")
        f.write("===============\n\n")
        f.write(df_comp.to_string(index=False))
        f.write("\n\n请确认后设置 PHASE1_CONFIRMED=True 继续。\n")
    print(f"Phase 1 摘要已保存: {summary_path}")


if __name__ == '__main__':
    main()