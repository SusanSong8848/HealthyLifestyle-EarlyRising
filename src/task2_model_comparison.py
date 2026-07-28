"""
Task 2 model comparison and method ablation (D28-02).

The script compares two routes on the shared public split:

1. Direct ordinal classification.
2. Continuous Health_Score regression followed by the locked score bins.

Hyper-parameters are selected by five-fold CV inside the public training split.
Only the winning model from each route is evaluated on the public validation
split.  The D27 baseline script is intentionally left unchanged.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import pickle
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .config import N_FOLDS, RANDOM_STATE
    from .evaluate import (
        evaluate_classification,
        get_forbidden_features,
        validate_no_leakage,
    )
    from .targets import generate_target, get_class_order, get_task_spec
except ImportError:
    from config import N_FOLDS, RANDOM_STATE
    from evaluate import (
        evaluate_classification,
        get_forbidden_features,
        validate_no_leakage,
    )
    from targets import generate_target, get_class_order, get_task_spec


ROOT_DIR = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT_DIR / "data" / "processed" / "base_semantic_clean.csv"
SPLIT_PATH = ROOT_DIR / "data" / "splits" / "split_manifest.csv"
RAW_METRICS_DIR = ROOT_DIR / "results" / "metrics" / "raw"
MODEL_COMPARISON_PATH = RAW_METRICS_DIR / "task2_model_comparison.csv"
METHOD_ABLATION_PATH = RAW_METRICS_DIR / "task2_method_ablation.csv"
TUNING_LOG_PATH = RAW_METRICS_DIR / "task2_tuning_log.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "task2" / "comparison"

APPROACH_DIRECT = "direct_classification"
APPROACH_REGRESSION = "regression_then_binning"

COMPARISON_FIELDS = [
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
    "roc_auc",
    "mae",
    "rmse",
    "recall_poor",
    "recall_average",
    "recall_good",
    "recall_excellent",
    "runtime_sec",
    "output_path",
    "selected_within_family",
    "selected_within_approach",
    "is_primary",
]

ABLATION_DELTA_FIELDS = [
    "accuracy_delta_vs_direct",
    "macro_f1_delta_vs_direct",
    "balanced_accuracy_delta_vs_direct",
    "mae_delta_vs_direct",
    "rmse_delta_vs_direct",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
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


def score_predictions_to_labels(scores: Any) -> pd.Series:
    """Clip continuous predictions and apply the locked D-016 bins."""
    values = pd.Series(np.asarray(scores, dtype=float))
    if values.isna().any():
        raise ValueError("Regression predictions contain missing values")
    clipped = values.clip(lower=0.0, upper=100.0)
    spec = get_task_spec("task2")
    bins = spec["bins"]
    labels = pd.cut(
        clipped,
        bins=bins["edges"],
        labels=spec["class_order"],
        right=bool(bins["right_closed"]),
        include_lowest=bool(bins["include_lowest"]),
    ).astype("string")
    if labels.isna().any():
        raise ValueError("Regression predictions could not be binned")
    return labels


def load_comparison_data() -> dict[str, Any]:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing {CLEAN_PATH}; run src/preprocess.py first")
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Missing {SPLIT_PATH}; run src/split.py first")

    clean = pd.read_csv(CLEAN_PATH, dtype={"Person_ID": "string"})
    manifest = pd.read_csv(SPLIT_PATH, dtype={"Person_ID": "string"})
    required_manifest = {"Person_ID", "split", "task2_label"}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest missing columns: {sorted(missing)}")

    merged = clean.merge(
        manifest[["Person_ID", "split", "task2_label"]],
        on="Person_ID",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(clean):
        raise ValueError("Public split does not cover every Person_ID")
    if merged["Person_ID"].duplicated().any():
        raise ValueError("Person_ID is not unique after split alignment")

    generated_target = generate_target(merged, "task2")
    manifest_target = merged["task2_label"].astype("string")
    if not generated_target.equals(manifest_target):
        mismatch = int(generated_target.ne(manifest_target).sum())
        raise ValueError(
            f"Task2 target disagrees with split manifest for {mismatch} rows"
        )

    train_mask = merged["split"].eq("train")
    val_mask = merged["split"].eq("val")
    if int(train_mask.sum()) != 8000 or int(val_mask.sum()) != 2000:
        raise ValueError(
            f"Unexpected split sizes: train={train_mask.sum()}, "
            f"val={val_mask.sum()}"
        )

    forbidden = get_forbidden_features("task2")
    feature_columns = [column for column in clean.columns if column not in forbidden]
    validate_no_leakage(feature_columns, "task2")
    if not feature_columns:
        raise ValueError("Task2 feature list is empty")

    X = merged[feature_columns].copy()
    numeric_columns = list(X.select_dtypes(include=[np.number]).columns)
    categorical_columns = [
        column for column in X.columns if column not in set(numeric_columns)
    ]
    return {
        "X": X,
        "labels": generated_target,
        "scores": merged["Health_Score"].astype(float),
        "split": merged["split"].astype("string"),
        "merged": merged,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_columns": feature_columns,
        "data_version": sha256_file(CLEAN_PATH),
        "split_version": sha256_file(SPLIT_PATH),
    }


def _grid(**parameters: list[Any]) -> list[dict[str, Any]]:
    keys = list(parameters)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(parameters[key] for key in keys))
    ]


def model_specs() -> list[dict[str, Any]]:
    """Return the six locked model families and their compact parameter grids."""
    return [
        {
            "approach": APPROACH_DIRECT,
            "model_family": "Dummy Classifier",
            "model": "Dummy Most Frequent",
            "params": [{}],
        },
        {
            "approach": APPROACH_DIRECT,
            "model_family": "Logistic Regression",
            "model": "Logistic Regression",
            "params": _grid(C=[0.1, 1.0, 10.0]),
        },
        {
            "approach": APPROACH_DIRECT,
            "model_family": "LightGBM Classifier",
            "model": "LightGBM Classifier",
            "params": _grid(
                n_estimators=[200, 400],
                num_leaves=[15, 31],
                learning_rate=[0.05],
            ),
        },
        {
            "approach": APPROACH_REGRESSION,
            "model_family": "Dummy Regressor",
            "model": "Dummy Mean Regressor",
            "params": [{}],
        },
        {
            "approach": APPROACH_REGRESSION,
            "model_family": "Ridge",
            "model": "Ridge Regression",
            "params": _grid(alpha=[0.1, 1.0, 10.0]),
        },
        {
            "approach": APPROACH_REGRESSION,
            "model_family": "LightGBM Regressor",
            "model": "LightGBM Regressor",
            "params": _grid(
                n_estimators=[200, 400],
                num_leaves=[15, 31],
                learning_rate=[0.05],
            ),
        },
    ]


def build_estimator(
    approach: str,
    model_family: str,
    params: dict[str, Any],
):
    if approach == APPROACH_DIRECT:
        if model_family == "Dummy Classifier":
            return DummyClassifier(
                strategy="most_frequent",
                random_state=RANDOM_STATE,
            )
        if model_family == "Logistic Regression":
            return LogisticRegression(
                C=float(params["C"]),
                max_iter=3000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )
        if model_family == "LightGBM Classifier":
            return lgb.LGBMClassifier(
                objective="multiclass",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
                **params,
            )
    elif approach == APPROACH_REGRESSION:
        if model_family == "Dummy Regressor":
            return DummyRegressor(strategy="mean")
        if model_family == "Ridge":
            return Ridge(alpha=float(params["alpha"]))
        if model_family == "LightGBM Regressor":
            return lgb.LGBMRegressor(
                objective="regression",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
                **params,
            )
    raise KeyError(f"Unknown model family: {approach}/{model_family}")


def prepare_cv_folds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    score_train: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> list[dict[str, Any]]:
    """Fit preprocessing separately inside every training fold."""
    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    folds = []
    for fit_positions, hold_positions in splitter.split(X_train, y_train):
        X_fit = X_train.iloc[fit_positions]
        X_hold = X_train.iloc[hold_positions]
        preprocessor = build_preprocessor(numeric_columns, categorical_columns)
        folds.append(
            {
                "X_fit": preprocessor.fit_transform(X_fit),
                "X_hold": preprocessor.transform(X_hold),
                "y_fit": y_train.iloc[fit_positions].reset_index(drop=True),
                "y_hold": y_train.iloc[hold_positions].reset_index(drop=True),
                "score_fit": score_train.iloc[fit_positions].reset_index(drop=True),
                "score_hold": score_train.iloc[hold_positions].reset_index(drop=True),
            }
        )
    return folds


def _mean_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, float | None]:
    fields = [
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
    ]
    result: dict[str, float | None] = {}
    for field in fields:
        values = [row[field] for row in fold_metrics if row[field] is not None]
        result[field] = float(np.mean(values)) if values else None
    return result


def select_best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No records supplied for model selection")

    def key(record: dict[str, Any]) -> tuple:
        mae = float(record["mae"]) if record["mae"] is not None else float("inf")
        return (
            -float(record["accuracy"]),
            -float(record["macro_f1"]),
            mae,
            str(record["model"]),
            str(record["params_json"]),
        )

    return min(records, key=key)


def make_comparison_record(
    *,
    context: dict[str, Any],
    approach: str,
    model_family: str,
    model: str,
    params: dict[str, Any],
    split: str,
    n_samples: int,
    metrics: dict[str, Any],
    runtime_sec: float,
    output_path: str = "",
) -> dict[str, Any]:
    record = {
        "task_id": "task2",
        "data_version": context["data_version"],
        "split_version": context["split_version"],
        "approach": approach,
        "model_family": model_family,
        "model": model,
        "params_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
        "split": split,
        "seed": int(RANDOM_STATE),
        "n_samples": int(n_samples),
        "cv_folds": int(N_FOLDS) if split == "cv_train_mean" else 0,
        "class_order": "|".join(get_class_order("task2")),
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
        "runtime_sec": float(runtime_sec),
        "output_path": output_path,
        "selected_within_family": False,
        "selected_within_approach": False,
        "is_primary": False,
    }
    if list(record) != COMPARISON_FIELDS:
        raise AssertionError("Task2 comparison record does not match schema")
    return record


def evaluate_cv_configuration(
    *,
    spec: dict[str, Any],
    params: dict[str, Any],
    folds: list[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    fold_metrics = []
    for fold in folds:
        estimator = build_estimator(
            spec["approach"],
            spec["model_family"],
            params,
        )
        if spec["approach"] == APPROACH_DIRECT:
            estimator.fit(fold["X_fit"], fold["y_fit"])
            predictions = pd.Series(
                estimator.predict(fold["X_hold"]),
                dtype="string",
            )
        else:
            estimator.fit(fold["X_fit"], fold["score_fit"])
            predictions = score_predictions_to_labels(
                estimator.predict(fold["X_hold"])
            )
        fold_metrics.append(
            evaluate_classification(fold["y_hold"], predictions, "task2")
        )

    return make_comparison_record(
        context=context,
        approach=spec["approach"],
        model_family=spec["model_family"],
        model=spec["model"],
        params=params,
        split="cv_train_mean",
        n_samples=8000,
        metrics=_mean_metrics(fold_metrics),
        runtime_sec=time.perf_counter() - started,
    )


def fit_route_winner(
    *,
    winner: dict[str, Any],
    context: dict[str, Any],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    score_train: pd.Series,
    val_person_ids: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    X_fit = preprocessor.fit_transform(X_train)
    X_hold = preprocessor.transform(X_val)
    params = json.loads(winner["params_json"])
    estimator = build_estimator(
        winner["approach"],
        winner["model_family"],
        params,
    )

    if winner["approach"] == APPROACH_DIRECT:
        estimator.fit(X_fit, y_train)
        predictions = pd.Series(estimator.predict(X_hold), dtype="string")
        predicted_scores = np.full(len(predictions), np.nan)
    else:
        estimator.fit(X_fit, score_train)
        predicted_scores = np.clip(
            np.asarray(estimator.predict(X_hold), dtype=float),
            0.0,
            100.0,
        )
        predictions = score_predictions_to_labels(predicted_scores)

    metrics = evaluate_classification(y_val.reset_index(drop=True), predictions, "task2")
    approach_slug = (
        "direct"
        if winner["approach"] == APPROACH_DIRECT
        else "regression_binning"
    )
    prediction_path = OUTPUT_DIR / f"{approach_slug}_predictions.csv"
    model_path = OUTPUT_DIR / f"{approach_slug}_best_model.pkl"

    prediction_df = pd.DataFrame(
        {
            "Person_ID": val_person_ids.astype("string").reset_index(drop=True),
            "True_Label": y_val.astype("string").reset_index(drop=True),
            "Predicted_Label": predictions.reset_index(drop=True),
            "Predicted_Score": predicted_scores,
        }
    )
    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "preprocessor": preprocessor,
                "model": estimator,
                "approach": winner["approach"],
                "model_family": winner["model_family"],
                "params": params,
            },
            handle,
        )

    record = make_comparison_record(
        context=context,
        approach=winner["approach"],
        model_family=winner["model_family"],
        model=winner["model"],
        params=params,
        split="val",
        n_samples=len(y_val),
        metrics=metrics,
        runtime_sec=time.perf_counter() - started,
        output_path=str(prediction_path.relative_to(ROOT_DIR)),
    )
    record["selected_within_family"] = True
    record["selected_within_approach"] = True
    return record, prediction_df, {
        "prediction_path": str(prediction_path.relative_to(ROOT_DIR)),
        "model_path": str(model_path.relative_to(ROOT_DIR)),
    }


def _add_ablation_deltas(records: list[dict[str, Any]]) -> pd.DataFrame:
    direct = next(
        record for record in records if record["approach"] == APPROACH_DIRECT
    )
    rows = []
    for record in records:
        row = dict(record)
        for metric in [
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "mae",
            "rmse",
        ]:
            row[f"{metric}_delta_vs_direct"] = (
                float(record[metric]) - float(direct[metric])
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=COMPARISON_FIELDS + ABLATION_DELTA_FIELDS)


def run_comparison() -> dict[str, Any]:
    RAW_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context = load_comparison_data()

    train_mask = context["split"].eq("train")
    val_mask = context["split"].eq("val")
    X_train = context["X"].loc[train_mask].reset_index(drop=True)
    X_val = context["X"].loc[val_mask].reset_index(drop=True)
    y_train = context["labels"].loc[train_mask].reset_index(drop=True)
    y_val = context["labels"].loc[val_mask].reset_index(drop=True)
    score_train = context["scores"].loc[train_mask].reset_index(drop=True)
    val_person_ids = context["merged"].loc[val_mask, "Person_ID"].reset_index(drop=True)

    folds = prepare_cv_folds(
        X_train,
        y_train,
        score_train,
        context["numeric_columns"],
        context["categorical_columns"],
    )

    tuning_records = []
    for spec in model_specs():
        for params in spec["params"]:
            record = evaluate_cv_configuration(
                spec=spec,
                params=params,
                folds=folds,
                context=context,
            )
            tuning_records.append(record)
            print(
                f"  CV {record['approach']} / {record['model_family']} "
                f"{record['params_json']}: ACC={record['accuracy']:.4f}, "
                f"Macro-F1={record['macro_f1']:.4f}, MAE={record['mae']:.4f}"
            )

    family_winners = []
    for spec in model_specs():
        candidates = [
            row
            for row in tuning_records
            if row["approach"] == spec["approach"]
            and row["model_family"] == spec["model_family"]
        ]
        winner = select_best_record(candidates)
        winner["selected_within_family"] = True
        family_winners.append(winner)

    route_winners = []
    for approach in [APPROACH_DIRECT, APPROACH_REGRESSION]:
        winner = select_best_record(
            [row for row in family_winners if row["approach"] == approach]
        )
        winner["selected_within_approach"] = True
        route_winners.append(winner)

    validation_records = []
    artifacts = {}
    prediction_frames = {}
    for route_winner in route_winners:
        record, predictions, route_artifacts = fit_route_winner(
            winner=route_winner,
            context=context,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            score_train=score_train,
            val_person_ids=val_person_ids,
            numeric_columns=context["numeric_columns"],
            categorical_columns=context["categorical_columns"],
        )
        validation_records.append(record)
        artifacts[record["approach"]] = route_artifacts
        prediction_frames[record["approach"]] = predictions

    primary = select_best_record(validation_records)
    primary["is_primary"] = True

    tuning_df = pd.DataFrame(tuning_records, columns=COMPARISON_FIELDS)
    comparison_df = pd.DataFrame(
        family_winners + validation_records,
        columns=COMPARISON_FIELDS,
    )
    ablation_df = _add_ablation_deltas(validation_records)

    tuning_df.to_csv(TUNING_LOG_PATH, index=False, encoding="utf-8-sig")
    comparison_df.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    ablation_df.to_csv(
        METHOD_ABLATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "task_id": "task2",
        "seed": RANDOM_STATE,
        "data_version": context["data_version"],
        "split_version": context["split_version"],
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "features": context["feature_columns"],
        "class_order": get_class_order("task2"),
        "route_winners": validation_records,
        "primary_approach": primary["approach"],
        "primary_model": primary["model"],
        "artifacts": artifacts,
    }
    with (OUTPUT_DIR / "comparison_run.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return {
        "context": context,
        "tuning": tuning_df,
        "comparison": comparison_df,
        "ablation": ablation_df,
        "validation_records": validation_records,
        "prediction_frames": prediction_frames,
        "primary": primary,
    }


def main() -> None:
    print("=" * 78)
    print("Task 2 model comparison: direct classification vs regression + binning")
    result = run_comparison()
    print("-" * 78)
    for record in result["validation_records"]:
        print(
            f"  {record['approach']} / {record['model']}: "
            f"ACC={record['accuracy']:.4f}, "
            f"Macro-F1={record['macro_f1']:.4f}, "
            f"BalAcc={record['balanced_accuracy']:.4f}, "
            f"MAE={record['mae']:.4f}, RMSE={record['rmse']:.4f}"
        )
    print(
        f"  Primary: {result['primary']['approach']} / "
        f"{result['primary']['model']}"
    )
    print(f"  Saved: {MODEL_COMPARISON_PATH.relative_to(ROOT_DIR)}")
    print(f"  Saved: {METHOD_ABLATION_PATH.relative_to(ROOT_DIR)}")
    print(f"  Saved: {TUNING_LOG_PATH.relative_to(ROOT_DIR)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
