"""
任务二正式基线（D27-09）。

- 使用公共清洗表和公共 split_manifest.csv
- 至少比较 Dummy 与 Logistic Regression
- 统一输出 Accuracy、Macro-F1、Balanced Accuracy、MAE、RMSE
- 严格执行任务二泄漏黑名单
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import RANDOM_STATE
from evaluate import (
    METRIC_FIELDS,
    evaluate_classification,
    get_forbidden_features,
    make_metric_record,
    plot_confusion_matrix,
    validate_no_leakage,
)
from targets import generate_target, get_class_order


ROOT_DIR = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT_DIR / "data" / "processed" / "base_semantic_clean.csv"
SPLIT_PATH = ROOT_DIR / "data" / "splits" / "split_manifest.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "task2"
METRICS_PATH = ROOT_DIR / "results" / "metrics" / "raw" / "task2_baseline.csv"
FIGURE_PATH = (
    ROOT_DIR
    / "results"
    / "figures"
    / "baseline"
    / "task2_confusion_matrix.png"
)


def build_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    """创建仅在训练集拟合的预处理器。"""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def load_public_split() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """读取公共清洗表，并按 Person_ID 对齐公共切分。"""
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing {CLEAN_PATH}; run src/preprocess.py first")
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Missing {SPLIT_PATH}; run src/split.py first")

    clean = pd.read_csv(CLEAN_PATH)
    manifest = pd.read_csv(SPLIT_PATH)
    required_manifest = {"Person_ID", "split", "task2_label"}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ValueError(f"Split manifest missing columns: {sorted(missing_manifest)}")

    merged = clean.merge(
        manifest[["Person_ID", "split", "task2_label"]],
        on="Person_ID",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(clean):
        raise ValueError("Public split does not cover every Person_ID")
    if set(merged["split"]) != {"train", "val"}:
        raise ValueError("Public split must contain train and val")

    generated_target = generate_target(merged, "task2")
    manifest_target = merged["task2_label"].astype("string")
    mismatch_count = int((generated_target != manifest_target).sum())
    if mismatch_count:
        raise ValueError(
            f"Task2 target disagrees with split manifest for {mismatch_count} rows"
        )

    forbidden = get_forbidden_features("task2")
    feature_columns = [col for col in clean.columns if col not in forbidden]
    validate_no_leakage(feature_columns, "task2")
    if not feature_columns:
        raise ValueError("Task2 feature list is empty")

    train_mask = merged["split"] == "train"
    val_mask = merged["split"] == "val"
    if int(train_mask.sum()) != 8000 or int(val_mask.sum()) != 2000:
        raise ValueError(
            f"Unexpected split sizes: train={train_mask.sum()}, val={val_mask.sum()}"
        )

    X = merged[feature_columns].copy()
    return X, generated_target, merged["split"], merged


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    X, y, split, merged = load_public_split()
    train_mask = split == "train"
    val_mask = split == "val"
    X_train = X.loc[train_mask]
    X_val = X.loc[val_mask]
    y_train = y.loc[train_mask]
    y_val = y.loc[val_mask]

    numeric_columns = list(X.select_dtypes(include=[np.number]).columns)
    categorical_columns = [
        col for col in X.columns if col not in set(numeric_columns)
    ]

    models = {
        "Dummy Most Frequent": DummyClassifier(
            strategy="most_frequent",
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
    }

    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, pd.Series] = {}
    metric_rows = []

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            [
                (
                    "preprocess",
                    build_preprocessor(numeric_columns, categorical_columns),
                ),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pd.Series(
            pipeline.predict(X_val),
            index=y_val.index,
            dtype="string",
        )
        metrics = evaluate_classification(y_val, y_pred, "task2")
        is_primary = model_name == "Logistic Regression"
        metric_rows.append(
            make_metric_record(
                task_id="task2",
                model=model_name,
                split="val",
                seed=RANDOM_STATE,
                n_samples=len(y_val),
                metrics=metrics,
                is_primary=is_primary,
            )
        )
        fitted_models[model_name] = pipeline
        predictions[model_name] = y_pred

    metrics_df = pd.DataFrame(metric_rows, columns=METRIC_FIELDS)
    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    primary_name = "Logistic Regression"
    primary_model = fitted_models[primary_name]
    primary_pred = predictions[primary_name]
    primary_row = metrics_df.loc[metrics_df["is_primary"]].iloc[0]

    prediction_df = pd.DataFrame(
        {
            "Person_ID": merged.loc[val_mask, "Person_ID"].values,
            "True_Label": y_val.values,
            "Predicted_Label": primary_pred.values,
        }
    )
    prediction_df.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_confusion_matrix(
        y_val,
        primary_pred,
        "task2",
        FIGURE_PATH,
        "Task 2 Baseline - Logistic Regression",
    )

    preprocessor = primary_model.named_steps["preprocess"]
    logistic_model = primary_model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    mean_abs_coef = np.abs(logistic_model.coef_).mean(axis=0)
    importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": mean_abs_coef}
    ).sort_values("Importance", ascending=False)
    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (OUTPUT_DIR / "best_model.pkl").open("wb") as f:
        pickle.dump(primary_model, f)

    run_summary = {
        "task_id": "task2",
        "seed": RANDOM_STATE,
        "split_manifest": str(SPLIT_PATH.relative_to(ROOT_DIR)),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "features": list(X.columns),
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "class_order": get_class_order("task2"),
        "primary_model": primary_name,
        "primary_metrics": {
            key: (
                None
                if pd.isna(primary_row[key])
                else float(primary_row[key])
            )
            for key in [
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "mae",
                "rmse",
            ]
        },
    }
    with (OUTPUT_DIR / "baseline_run.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("Task 2 formal baseline completed")
    print(f"  Split: train={train_mask.sum()}, val={val_mask.sum()}")
    print(
        f"  Features: {len(X.columns)} "
        f"(numeric={len(numeric_columns)}, categorical={len(categorical_columns)})"
    )
    for _, row in metrics_df.iterrows():
        print(
            f"  {row['model']}: "
            f"ACC={row['accuracy']:.4f}, "
            f"Macro-F1={row['macro_f1']:.4f}, "
            f"BalAcc={row['balanced_accuracy']:.4f}, "
            f"MAE={row['mae']:.4f}, "
            f"RMSE={row['rmse']:.4f}"
        )
    print(f"  Saved: {METRICS_PATH.relative_to(ROOT_DIR)}")
    print(f"  Saved: {FIGURE_PATH.relative_to(ROOT_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
