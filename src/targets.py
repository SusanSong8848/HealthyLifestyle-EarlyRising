"""统一目标标签生成接口（D27-06）。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS_PATH = ROOT_DIR / "config" / "targets.yaml"


@lru_cache(maxsize=None)
def load_targets_config(path: str | Path = DEFAULT_TARGETS_PATH) -> dict:
    """加载并返回目标配置。"""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "tasks" not in config:
        raise ValueError(f"Invalid targets config: {config_path}")
    return config


def get_task_spec(task_id: str, path: str | Path = DEFAULT_TARGETS_PATH) -> dict:
    """取得单个任务的配置。"""
    tasks = load_targets_config(path)["tasks"]
    if task_id not in tasks:
        raise KeyError(f"Unknown task_id: {task_id}")
    return tasks[task_id]


def get_class_order(
    task_id: str, path: str | Path = DEFAULT_TARGETS_PATH
) -> list[str]:
    """取得任务固定类别顺序。"""
    return list(get_task_spec(task_id, path)["class_order"])


def validate_labels(
    labels: Iterable,
    task_id: str,
    path: str | Path = DEFAULT_TARGETS_PATH,
) -> pd.Series:
    """验证标签不缺失且全部属于配置中的类别。"""
    series = pd.Series(labels, dtype="string")
    if series.isna().any():
        raise ValueError(f"{task_id} labels contain missing values")
    allowed = set(get_class_order(task_id, path))
    unknown = set(series.unique()) - allowed
    if unknown:
        raise ValueError(f"{task_id} labels contain unknown classes: {sorted(unknown)}")
    return series


def generate_target(
    df: pd.DataFrame,
    task_id: str,
    path: str | Path = DEFAULT_TARGETS_PATH,
) -> pd.Series:
    """按统一配置从公共清洗数据生成任务标签。"""
    spec = get_task_spec(task_id, path)
    source_column = spec["source_column"]
    target_name = spec["target_name"]
    if source_column not in df.columns:
        raise KeyError(f"Missing target source column: {source_column}")

    if task_id == "task2":
        bins = spec["bins"]
        target = pd.cut(
            df[source_column],
            bins=bins["edges"],
            labels=spec["class_order"],
            right=bool(bins["right_closed"]),
            include_lowest=bool(bins["include_lowest"]),
        ).astype("string")
    else:
        target = df[source_column].astype("string")

    target = validate_labels(target, task_id, path)
    target.name = target_name
    target.index = df.index
    return target


def encode_labels(
    labels: Iterable,
    task_id: str,
    path: str | Path = DEFAULT_TARGETS_PATH,
) -> pd.Series:
    """按固定类别顺序将文本标签编码为整数。"""
    series = validate_labels(labels, task_id, path)
    mapping = {
        label: index for index, label in enumerate(get_class_order(task_id, path))
    }
    return series.map(mapping).astype(int)


def decode_labels(
    encoded: Iterable[int],
    task_id: str,
    path: str | Path = DEFAULT_TARGETS_PATH,
) -> pd.Series:
    """将整数标签还原为固定类别文本。"""
    class_order = get_class_order(task_id, path)
    values = pd.Series(encoded)
    invalid = sorted(set(values.dropna().astype(int)) - set(range(len(class_order))))
    if invalid:
        raise ValueError(f"{task_id} encoded labels out of range: {invalid}")
    return values.astype(int).map(dict(enumerate(class_order))).astype("string")
