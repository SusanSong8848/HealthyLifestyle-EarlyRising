"""统一评价、泄漏检查和混淆矩阵接口（D27-06）。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    roc_auc_score,
)

try:
    from .targets import encode_labels, get_class_order, validate_labels
except ImportError:
    from targets import encode_labels, get_class_order, validate_labels


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LEAKAGE_PATH = ROOT_DIR / "config" / "leakage_blacklist.yaml"

METRIC_FIELDS = [
    "task_id",
    "model",
    "split",
    "seed",
    "n_samples",
    "class_order",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "roc_auc",
    "mae",
    "rmse",
    "recall_poor",
    "recall_average",
    "recall_good",
    "recall_excellent",
    "is_primary",
]


def load_leakage_config(path: str | Path = DEFAULT_LEAKAGE_PATH) -> dict:
    """加载泄漏黑名单配置。"""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "tasks" not in config:
        raise ValueError(f"Invalid leakage config: {config_path}")
    return config


def get_forbidden_features(
    task_id: str, path: str | Path = DEFAULT_LEAKAGE_PATH
) -> dict[str, str]:
    """取得任务禁止使用的字段及原因。"""
    tasks = load_leakage_config(path)["tasks"]
    if task_id not in tasks:
        raise KeyError(f"Unknown task_id: {task_id}")
    return dict(tasks[task_id]["forbidden"])


def validate_no_leakage(
    feature_columns: Iterable[str],
    task_id: str,
    path: str | Path = DEFAULT_LEAKAGE_PATH,
) -> None:
    """如特征包含黑名单字段则抛出异常。"""
    forbidden = get_forbidden_features(task_id, path)
    leaked = sorted(set(feature_columns) & set(forbidden))
    if leaked:
        reasons = "; ".join(f"{name}: {forbidden[name]}" for name in leaked)
        raise ValueError(f"{task_id} leakage fields detected: {reasons}")


def evaluate_classification(
    y_true: Iterable,
    y_pred: Iterable,
    task_id: str,
    y_score: Iterable[float] | None = None,
) -> dict[str, float | None]:
    """按统一口径计算任务指标。"""
    true_labels = validate_labels(y_true, task_id)
    pred_labels = validate_labels(y_pred, task_id)
    if len(true_labels) != len(pred_labels):
        raise ValueError("y_true and y_pred must have the same length")

    class_order = get_class_order(task_id)
    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "macro_f1": float(
            f1_score(
                true_labels,
                pred_labels,
                labels=class_order,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, pred_labels)
        ),
        "roc_auc": None,
        "mae": None,
        "rmse": None,
        "recall_poor": None,
        "recall_average": None,
        "recall_good": None,
        "recall_excellent": None,
    }

    if task_id == "task1" and y_score is not None:
        scores = np.asarray(y_score)
        if scores.ndim == 2:
            scores = scores[:, 1]
        true_encoded = encode_labels(true_labels, task_id)
        result["roc_auc"] = float(roc_auc_score(true_encoded, scores))

    if task_id == "task2":
        true_encoded = encode_labels(true_labels, task_id)
        pred_encoded = encode_labels(pred_labels, task_id)
        result["mae"] = float(mean_absolute_error(true_encoded, pred_encoded))
        result["rmse"] = float(
            np.sqrt(mean_squared_error(true_encoded, pred_encoded))
        )

    if task_id == "task3":
        recalls = recall_score(
            true_labels,
            pred_labels,
            labels=class_order,
            average=None,
            zero_division=0,
        )
        recall_by_class = dict(zip(class_order, recalls))
        result["recall_poor"] = float(recall_by_class["Poor"])
        result["recall_average"] = float(recall_by_class["Average"])
        result["recall_good"] = float(recall_by_class["Good"])
        result["recall_excellent"] = float(recall_by_class["Excellent"])

    return result


def make_metric_record(
    task_id: str,
    model: str,
    split: str,
    seed: int,
    n_samples: int,
    metrics: dict[str, float | None],
    is_primary: bool,
) -> dict:
    """将指标包装为 metrics_schema.csv 规定的单行记录。"""
    record = {
        "task_id": task_id,
        "model": model,
        "split": split,
        "seed": int(seed),
        "n_samples": int(n_samples),
        "class_order": "|".join(get_class_order(task_id)),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "roc_auc": metrics.get("roc_auc"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "recall_poor": metrics.get("recall_poor"),
        "recall_average": metrics.get("recall_average"),
        "recall_good": metrics.get("recall_good"),
        "recall_excellent": metrics.get("recall_excellent"),
        "is_primary": bool(is_primary),
    }
    if list(record) != METRIC_FIELDS:
        raise AssertionError("Metric record field order does not match schema")
    return record


def plot_confusion_matrix(
    y_true: Iterable,
    y_pred: Iterable,
    task_id: str,
    output_path: str | Path,
    title: str,
) -> None:
    """按固定类别顺序保存混淆矩阵图片。"""
    true_labels = validate_labels(y_true, task_id)
    pred_labels = validate_labels(y_pred, task_id)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ConfusionMatrixDisplay.from_predictions(
        true_labels,
        pred_labels,
        labels=get_class_order(task_id),
        cmap="Blues",
        colorbar=False,
        values_format="d",
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
