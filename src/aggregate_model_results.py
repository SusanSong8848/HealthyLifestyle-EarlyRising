"""
Strict three-task aggregation gate for D28-04 and freeze readiness checks.

Global metric files are written only when all task-level dependencies exist and
pass the shared data/split/seed/schema checks.  Missing task artifacts never
produce a partial file that could be mistaken for a complete three-task result.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .config import RANDOM_STATE
except ImportError:
    from config import RANDOM_STATE


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "results" / "metrics" / "raw"
METRICS_DIR = ROOT_DIR / "results" / "metrics"

TASK_IDS = ("task1", "task2", "task3")

COMPARISON_SOURCES = {
    task_id: RAW_DIR / f"{task_id}_model_comparison.csv"
    for task_id in TASK_IDS
}
ABLATION_SOURCES = {
    "task1": RAW_DIR / "task1_leakage_ablation.csv",
    "task2": RAW_DIR / "task2_method_ablation.csv",
    "task3": RAW_DIR / "task3_weight_ablation.csv",
}
TUNING_SOURCES = {
    task_id: RAW_DIR / f"{task_id}_tuning_log.csv"
    for task_id in TASK_IDS
}

GLOBAL_OUTPUTS = {
    "comparison": METRICS_DIR / "model_comparison.csv",
    "tuning": METRICS_DIR / "tuning_log.csv",
    "ablation": METRICS_DIR / "ablation.csv",
}

COMMON_REQUIRED_FIELDS = {
    "task_id",
    "data_version",
    "split_version",
    "approach",
    "model_family",
    "model",
    "params_json",
    "split",
    "seed",
    "n_samples",
    "cv_folds",
    "class_order",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "runtime_sec",
    "output_path",
    "selected_within_family",
    "selected_within_approach",
    "is_primary",
}


class MissingDependencyError(RuntimeError):
    """Raised when a complete three-task aggregation is not possible."""

    def __init__(self, missing: Iterable[Path]):
        self.missing = tuple(Path(path) for path in missing)
        relative = []
        for path in self.missing:
            try:
                relative.append(str(path.relative_to(ROOT_DIR)))
            except ValueError:
                relative.append(str(path))
        super().__init__("Missing aggregation dependencies: " + ", ".join(relative))


def _resolve_tuning_sources() -> dict[str, Path]:
    """Prefer explicit tuning logs, otherwise accept comparison logs."""
    resolved = {}
    for task_id in TASK_IDS:
        explicit = TUNING_SOURCES[task_id]
        fallback = COMPARISON_SOURCES[task_id]
        resolved[task_id] = explicit if explicit.exists() else fallback
    return resolved


def dependency_report() -> dict[str, list[str]]:
    groups = {
        "comparison": COMPARISON_SOURCES,
        "ablation": ABLATION_SOURCES,
        "tuning": _resolve_tuning_sources(),
    }
    report = {}
    for group_name, sources in groups.items():
        report[group_name] = [
            str(path.relative_to(ROOT_DIR))
            for path in sources.values()
            if not path.exists()
        ]
    return report


def _load_task_frames(
    sources: dict[str, Path],
    *,
    required_fields: set[str] = COMMON_REQUIRED_FIELDS,
) -> list[pd.DataFrame]:
    missing = [path for path in sources.values() if not path.exists()]
    if missing:
        raise MissingDependencyError(missing)

    frames = []
    for expected_task, path in sources.items():
        frame = pd.read_csv(path)
        missing_fields = required_fields - set(frame.columns)
        if missing_fields:
            raise ValueError(
                f"{path.relative_to(ROOT_DIR)} missing required fields: "
                f"{sorted(missing_fields)}"
            )
        actual_tasks = set(frame["task_id"].astype(str))
        if actual_tasks != {expected_task}:
            raise ValueError(
                f"{path.relative_to(ROOT_DIR)} contains task IDs "
                f"{sorted(actual_tasks)}, expected only {expected_task}"
            )
        frames.append(frame)
    return frames


def validate_shared_contract(frame: pd.DataFrame) -> None:
    if set(frame["task_id"].astype(str)) != set(TASK_IDS):
        raise ValueError("Aggregate must contain task1, task2 and task3")

    seeds = set(pd.to_numeric(frame["seed"], errors="raise").astype(int))
    if seeds != {int(RANDOM_STATE)}:
        raise ValueError(f"Seed mismatch across tasks: {sorted(seeds)}")

    data_versions = set(frame["data_version"].dropna().astype(str))
    if len(data_versions) != 1:
        raise ValueError(
            f"Data version mismatch across tasks: {sorted(data_versions)}"
        )

    split_versions = set(frame["split_version"].dropna().astype(str))
    if len(split_versions) != 1:
        raise ValueError(
            f"Split version mismatch across tasks: {sorted(split_versions)}"
        )

    invalid_samples = frame.loc[
        frame["split"].eq("val") & frame["n_samples"].ne(2000)
    ]
    if not invalid_samples.empty:
        raise ValueError("Every validation row must contain 2,000 samples")

    duplicate_fields = [
        "task_id",
        "approach",
        "model",
        "params_json",
        "split",
    ]
    if frame.duplicated(duplicate_fields).any():
        raise ValueError(
            f"Duplicate aggregate records by {duplicate_fields}"
        )


def _atomic_to_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix=output_path.stem + "_",
        dir=output_path.parent,
        delete=False,
        newline="",
        encoding="utf-8-sig",
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            frame.to_csv(handle, index=False)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def build_aggregate(
    sources: dict[str, Path],
    output_path: Path,
) -> pd.DataFrame:
    frames = _load_task_frames(sources)
    aggregate = pd.concat(frames, ignore_index=True, sort=False)
    validate_shared_contract(aggregate)
    _atomic_to_csv(aggregate, output_path)
    return aggregate


def aggregate_all() -> dict[str, pd.DataFrame]:
    report = dependency_report()
    missing = []
    seen_missing = set()
    for values in report.values():
        for relative in values:
            path = ROOT_DIR / relative
            if path not in seen_missing:
                missing.append(path)
                seen_missing.add(path)
    if missing:
        raise MissingDependencyError(missing)

    source_groups = {
        "comparison": COMPARISON_SOURCES,
        "tuning": _resolve_tuning_sources(),
        "ablation": ABLATION_SOURCES,
    }
    # Validate all groups before writing any global output.
    validated = {}
    for group_name, sources in source_groups.items():
        frames = _load_task_frames(sources)
        aggregate = pd.concat(frames, ignore_index=True, sort=False)
        validate_shared_contract(aggregate)
        validated[group_name] = aggregate

    for group_name, aggregate in validated.items():
        _atomic_to_csv(aggregate, GLOBAL_OUTPUTS[group_name])
    return validated


def freeze_readiness() -> dict[str, object]:
    required = [
        *GLOBAL_OUTPUTS.values(),
        ROOT_DIR / "results" / "logs" / "reproduction_candidates.md",
        ROOT_DIR / "docs" / "final_model_decision.md",
        ROOT_DIR / "config" / "final" / "task1.yaml",
        ROOT_DIR / "config" / "final" / "task2.yaml",
        ROOT_DIR / "config" / "final" / "task3.yaml",
    ]
    missing = [
        str(path.relative_to(ROOT_DIR))
        for path in required
        if not path.exists()
    ]
    return {
        "ready": not missing,
        "missing": missing,
        "requires_three_person_signoff": True,
    }


def main() -> None:
    print("=" * 72)
    print("D28-04 strict three-task aggregation gate")
    try:
        aggregates = aggregate_all()
    except MissingDependencyError as error:
        print("  BLOCKED")
        print(f"  {error}")
        print("  No global metric files were created or overwritten.")
        print("  Freeze readiness:")
        print(json.dumps(freeze_readiness(), ensure_ascii=False, indent=2))
        raise SystemExit(2)

    for group_name, frame in aggregates.items():
        print(
            f"  {group_name}: {len(frame)} rows -> "
            f"{GLOBAL_OUTPUTS[group_name].relative_to(ROOT_DIR)}"
        )
    print("  Freeze readiness:")
    print(json.dumps(freeze_readiness(), ensure_ascii=False, indent=2))
    print("=" * 72)


if __name__ == "__main__":
    main()
