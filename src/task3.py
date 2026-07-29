"""
Task 3 - Wellness Category Prediction (v3.4)

Guarantees:
- Uses the shared Person_ID split manifest.
- Excludes target and known leakage fields.
- Fits encoding/scaling inside every cross-validation fold.
- Selects the final model by CV Accuracy, matching the official ACC3 score.
- Uses Macro-F1 and Balanced Accuracy as tie-breakers and diagnostics.
- Uses the shared validation split only for final reporting.
- Writes the exact D27/D28 filenames defined in progress.xlsx.
- Compares unweighted and class-balanced LightGBM with all else fixed.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import warnings
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    DATA_PATH,
    FIGURES_BASELINE_DIR,
    FIGURES_CANDIDATE_DIR,
    ID_COLUMN,
    METRICS_RAW_DIR,
    N_FOLDS,
    PREDICTIONS_CANDIDATE_DIR,
    RANDOM_STATE,
    SPLITS_DIR,
    TASK3_CANDIDATE_MODEL_DIR,
    TASK3_LEAKY_FEATURES,
)
from clean_utils import (  # noqa: E402
    TASK_CATEGORICAL_COLS,
    TASK_NUMERIC_COLS,
    load_and_clean_data,
    load_manifest_split_indices,
)

CLASS_ORDER = ["Poor", "Average", "Good", "Excellent"]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_ORDER)}
INDEX_TO_CLASS = {index: label for label, index in CLASS_TO_INDEX.items()}
BASELINE_MODEL_NAMES = [
    "Dummy Most Frequent",
    "Logistic Regression (Unweighted)",
]
WEIGHT_ABLATION_MODEL_NAMES = [
    "LightGBM (Unweighted)",
    "LightGBM (Balanced)",
]
LOGISTIC_C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
TUNED_LOGISTIC_PREFIX = "Logistic Regression (Tuned C="
LIGHTGBM_COMMON_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "force_row_wise": True,
    "n_jobs": -1,
}
EXCLUDED_FEATURES = list(
    dict.fromkeys(
        list(TASK3_LEAKY_FEATURES)
        + [
            "Healthy_Aging_Score",
            "Early_Waker",
            "Wake_Up_Time",
            "Sleep_Time",
            "Wake_Up_Time_Minutes",
            "Sleep_Time_Minutes",
        ]
    )
)


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense encoder compatible with old and new sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Build preprocessing that can be cloned and fitted inside each CV fold."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def select_feature_columns(
    frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    numeric_columns = [
        c
        for c in TASK_NUMERIC_COLS
        if c in frame.columns and c not in EXCLUDED_FEATURES
    ]
    categorical_columns = [
        c
        for c in TASK_CATEGORICAL_COLS
        if c in frame.columns and c not in EXCLUDED_FEATURES
    ]
    feature_columns = numeric_columns + categorical_columns
    duplicates = pd.Index(feature_columns)[pd.Index(feature_columns).duplicated()]
    if len(duplicates):
        raise ValueError(f"Duplicate Task 3 features: {duplicates.tolist()}")
    leaked = sorted(set(feature_columns).intersection(EXCLUDED_FEATURES))
    if leaked:
        raise ValueError(f"Task 3 leakage features remain in X: {leaked}")
    if not feature_columns:
        raise ValueError("Task 3 has no usable feature columns")
    return numeric_columns, categorical_columns


def build_model_pipelines(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> OrderedDict[str, Pipeline]:
    """Create a separate end-to-end preprocessing Pipeline per candidate."""
    estimators: OrderedDict[str, object] = OrderedDict(
        [
            ("Dummy Most Frequent", DummyClassifier(strategy="most_frequent")),
            (
                "Logistic Regression (Unweighted)",
                LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                    C=1.0,
                    class_weight=None,
                ),
            ),
            (
                "Logistic Regression (Balanced)",
                LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                    C=1.0,
                    class_weight="balanced",
                ),
            ),
            (
                "Random Forest (Balanced)",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_split=5,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    if xgb is not None:
        estimators["XGBoost (Unweighted)"] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=len(CLASS_ORDER),
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            tree_method="hist",
            verbosity=0,
            n_jobs=-1,
        )
    else:
        warnings.warn("xgboost is unavailable; XGBoost will be skipped.")
    if lgb is not None:
        estimators["LightGBM (Unweighted)"] = lgb.LGBMClassifier(
            **LIGHTGBM_COMMON_PARAMS,
            class_weight=None,
        )
        estimators["LightGBM (Balanced)"] = lgb.LGBMClassifier(
            **LIGHTGBM_COMMON_PARAMS,
            class_weight="balanced",
        )
    else:
        warnings.warn("lightgbm is unavailable; LightGBM will be skipped.")

    pipelines: OrderedDict[str, Pipeline] = OrderedDict()
    for name, estimator in estimators.items():
        pipelines[name] = Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(numeric_columns, categorical_columns),
                ),
                ("classifier", estimator),
            ]
        )
    return pipelines


def build_logistic_tuning_pipelines(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> OrderedDict[str, Pipeline]:
    """Create a small, training-CV-only regularization search."""
    pipelines: OrderedDict[str, Pipeline] = OrderedDict()
    for c_value in LOGISTIC_C_GRID:
        model_name = f"{TUNED_LOGISTIC_PREFIX}{c_value:g})"
        pipelines[model_name] = Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(numeric_columns, categorical_columns),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                        C=c_value,
                        class_weight=None,
                    ),
                ),
            ]
        )
    return pipelines


def validate_required_models(models: OrderedDict[str, Pipeline]) -> None:
    """Fail before training rather than silently omit the required ablation."""
    missing = [
        name for name in WEIGHT_ABLATION_MODEL_NAMES if name not in models
    ]
    if missing:
        raise ImportError(
            "Task 3 requires lightgbm for the controlled class-weight "
            f"ablation; missing models: {missing}. Install lightgbm before "
            "running src/task3.py."
        )


def _single_class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_index: int,
) -> float:
    return float(
        recall_score(
            y_true,
            y_pred,
            labels=[class_index],
            average="macro",
            zero_division=0,
        )
    )


def build_scoring() -> dict[str, object]:
    scoring: dict[str, object] = {
        "accuracy": make_scorer(accuracy_score),
        "macro_f1": make_scorer(
            f1_score,
            average="macro",
            labels=list(range(len(CLASS_ORDER))),
            zero_division=0,
        ),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }
    for class_index, class_name in enumerate(CLASS_ORDER):
        scoring[f"recall_{class_name.lower()}"] = make_scorer(
            _single_class_recall,
            class_index=class_index,
        )
    return scoring


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return Accuracy, Macro-F1, Balanced Accuracy and all class recalls."""
    labels = list(range(len(CLASS_ORDER)))
    recalls = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro_F1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    for class_name, value in zip(CLASS_ORDER, recalls):
        metrics[f"Recall_{class_name}"] = float(value)
    return metrics


def _cv_summary_row(
    model_name: str,
    cv_output: dict[str, np.ndarray],
) -> dict[str, object]:
    metric_columns = OrderedDict(
        [
            ("accuracy", "CV_Accuracy"),
            ("macro_f1", "CV_Macro_F1"),
            ("balanced_accuracy", "CV_Balanced_Accuracy"),
            ("recall_poor", "CV_Recall_Poor"),
            ("recall_average", "CV_Recall_Average"),
            ("recall_good", "CV_Recall_Good"),
            ("recall_excellent", "CV_Recall_Excellent"),
        ]
    )
    row: dict[str, object] = {"Model": model_name}
    for scorer_name, prefix in metric_columns.items():
        values = np.asarray(cv_output[f"test_{scorer_name}"], dtype=float)
        row[f"{prefix}_Mean"] = float(values.mean())
        row[f"{prefix}_Std"] = float(values.std())
    row["CV_Fit_Time_Seconds"] = float(
        np.asarray(cv_output["fit_time"], dtype=float).sum()
    )
    return row


def run_cross_validation(
    models: OrderedDict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> pd.DataFrame:
    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    rows = []
    for model_name, model in models.items():
        cv_output = cross_validate(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring=build_scoring(),
            return_train_score=False,
            error_score="raise",
            n_jobs=1,
        )
        row = _cv_summary_row(model_name, cv_output)
        rows.append(row)
        print(
            f"  {model_name:22s}: "
            f"Macro-F1={row['CV_Macro_F1_Mean']:.4f}"
            f"±{row['CV_Macro_F1_Std']:.4f}, "
            f"Balanced Acc={row['CV_Balanced_Accuracy_Mean']:.4f}, "
            f"Accuracy={row['CV_Accuracy_Mean']:.4f}, "
            f"Poor Recall={row['CV_Recall_Poor_Mean']:.4f}"
        )
    return pd.DataFrame(rows)


def select_best_model(comparison: pd.DataFrame) -> str:
    """Select by training CV only, led by the official Accuracy metric."""
    required = {
        "Model",
        "CV_Macro_F1_Mean",
        "CV_Balanced_Accuracy_Mean",
        "CV_Accuracy_Mean",
    }
    missing = required.difference(comparison.columns)
    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}")
    ranked = comparison.sort_values(
        [
            "CV_Accuracy_Mean",
            "CV_Macro_F1_Mean",
            "CV_Balanced_Accuracy_Mean",
            "Model",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return str(ranked.iloc[0]["Model"])


def fit_and_evaluate_candidates(
    models: OrderedDict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[dict[str, Pipeline], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    validation_metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        y_pred = np.asarray(fitted.predict(x_val), dtype=int)
        fitted_models[name] = fitted
        predictions[name] = y_pred
        validation_metrics[name] = evaluate_predictions(y_val, y_pred)
    return fitted_models, predictions, validation_metrics


def add_validation_metrics(
    comparison: pd.DataFrame,
    validation_metrics: dict[str, dict[str, float]],
    selected_model: str,
) -> pd.DataFrame:
    result = comparison.copy()
    metric_names = [
        "Accuracy",
        "Macro_F1",
        "Balanced_Accuracy",
        "Recall_Poor",
        "Recall_Average",
        "Recall_Good",
        "Recall_Excellent",
    ]
    for metric_name in metric_names:
        result[f"Val_{metric_name}"] = result["Model"].map(
            lambda name: validation_metrics[str(name)][metric_name]
        )
    result["Selected"] = result["Model"].eq(selected_model)
    result = result.sort_values(
        [
            "CV_Accuracy_Mean",
            "CV_Macro_F1_Mean",
            "CV_Balanced_Accuracy_Mean",
            "Model",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result.insert(1, "CV_Rank", np.arange(1, len(result) + 1))
    return result


def extract_feature_importance(
    fitted_pipeline: Pipeline,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> pd.DataFrame:
    """Measure paper-facing importance at the original-feature level."""
    result = permutation_importance(
        fitted_pipeline,
        x_val,
        y_val,
        scoring="accuracy",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    signed_importance = np.asarray(
        result.importances_mean,
        dtype=float,
    ).reshape(-1)
    raw_importance = np.maximum(signed_importance, 0.0)
    feature_names = np.asarray(x_val.columns, dtype=str)
    method = "permutation_accuracy_original_feature"
    if len(feature_names) != len(raw_importance):
        raise ValueError(
            "Feature importance/name length mismatch: "
            f"{len(raw_importance)} vs {len(feature_names)}"
        )
    total = float(raw_importance.sum())
    normalized = raw_importance / total if total > 0 else raw_importance
    return (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": normalized,
                "Raw_Importance": raw_importance,
                "Signed_Mean_Accuracy_Decrease": signed_importance,
                "Std_Accuracy_Decrease": np.asarray(
                    result.importances_std,
                    dtype=float,
                ),
                "Method": method,
            }
        )
        .sort_values(["Importance", "Feature"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_weight_ablation(comparison: pd.DataFrame) -> pd.DataFrame:
    """Controlled LightGBM ablation: only class_weight may differ."""
    indexed = comparison.set_index("Model", drop=False)
    missing = [
        name for name in WEIGHT_ABLATION_MODEL_NAMES if name not in indexed.index
    ]
    if missing:
        raise ValueError(
            "Weight ablation cannot be produced; missing models: "
            f"{missing}"
        )
    ablation = indexed.loc[WEIGHT_ABLATION_MODEL_NAMES].copy().reset_index(
        drop=True
    )
    ablation.insert(1, "Weighting", ["None", "balanced"])
    ablation.insert(
        2,
        "Controlled_Change",
        "LightGBM parameters fixed; only class_weight changes",
    )
    ablation.insert(3, "class_weight", ["None", "balanced"])
    parameter_position = 4
    for parameter in [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "num_leaves",
        "random_state",
    ]:
        ablation.insert(
            parameter_position,
            parameter,
            LIGHTGBM_COMMON_PARAMS[parameter],
        )
        parameter_position += 1
    reference = ablation.iloc[0]
    for metric in [
        "CV_Accuracy_Mean",
        "CV_Macro_F1_Mean",
        "CV_Balanced_Accuracy_Mean",
        "CV_Recall_Poor_Mean",
        "Val_Accuracy",
        "Val_Macro_F1",
        "Val_Balanced_Accuracy",
        "Val_Recall_Poor",
    ]:
        ablation[f"Delta_{metric}_vs_Unweighted"] = (
            ablation[metric] - float(reference[metric])
        )
    return ablation


def save_confusion_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    *,
    title: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    labels = list(range(len(CLASS_ORDER)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(
        matrix,
        index=[f"True_{label}" for label in CLASS_ORDER],
        columns=[f"Pred_{label}" for label in CLASS_ORDER],
    ).to_csv(
        os.path.join(output_dir, "task3_confusion_matrix.csv"),
        encoding="utf-8-sig",
    )
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_ORDER,
    ).plot(ax=axis, cmap="Blues", values_format="d", colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(
        os.path.join(output_dir, "task3_confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_feature_importance_plot(
    importance: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save a paper-ready top-20 feature-importance chart."""
    os.makedirs(output_dir, exist_ok=True)
    top = importance.head(20).sort_values("Importance", ascending=True)
    figure, axis = plt.subplots(figsize=(9.2, 7.2))
    axis.barh(top["Feature"], top["Importance"], color="#2F75B5")
    axis.set_xlabel("Normalized importance")
    axis.set_title("Task 3 Candidate Model: Top 20 Feature Importance")
    figure.tight_layout()
    figure.savefig(
        os.path.join(output_dir, "task3_feature_importance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_outputs(
    *,
    ids_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray,
    comparison: pd.DataFrame,
    tuning_comparison: pd.DataFrame,
    weight_ablation: pd.DataFrame,
    best_model_name: str,
    best_pipeline: Pipeline,
    x_val: pd.DataFrame,
    validation_metrics: dict[str, float],
    feature_columns: list[str],
) -> None:
    for directory in [
        METRICS_RAW_DIR,
        FIGURES_BASELINE_DIR,
        FIGURES_CANDIDATE_DIR,
        PREDICTIONS_CANDIDATE_DIR,
        TASK3_CANDIDATE_MODEL_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)

    pd.DataFrame(
        {
            ID_COLUMN: ids_val,
            "True_Label": [INDEX_TO_CLASS[int(value)] for value in y_val],
            "Predicted_Label": [INDEX_TO_CLASS[int(value)] for value in y_pred],
        }
    ).to_csv(
        os.path.join(
            PREDICTIONS_CANDIDATE_DIR,
            "task3_predictions.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    baseline = comparison[
        comparison["Model"].isin(BASELINE_MODEL_NAMES)
    ].copy()
    missing_baselines = sorted(
        set(BASELINE_MODEL_NAMES).difference(baseline["Model"])
    )
    if missing_baselines:
        raise ValueError(f"Missing Task 3 baseline models: {missing_baselines}")
    baseline["Baseline_Reference"] = baseline["Model"].eq(
        "Logistic Regression (Unweighted)"
    )
    baseline.to_csv(
        os.path.join(METRICS_RAW_DIR, "task3_baseline.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        os.path.join(METRICS_RAW_DIR, "task3_model_comparison.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    tuning_comparison.to_csv(
        os.path.join(METRICS_RAW_DIR, "task3_tuning.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    weight_ablation.to_csv(
        os.path.join(METRICS_RAW_DIR, "task3_weight_ablation.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    report = classification_report(
        y_val,
        y_pred,
        labels=list(range(len(CLASS_ORDER))),
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose()
    report_frame.index.name = "Class_or_Aggregate"
    report_frame.to_csv(
        os.path.join(
            METRICS_RAW_DIR,
            "task3_classification_report.csv",
        ),
        encoding="utf-8-sig",
    )
    save_confusion_outputs(
        y_val,
        baseline_pred,
        FIGURES_BASELINE_DIR,
        title=(
            "Task 3 Baseline: Logistic Regression "
            "(Unweighted) Validation Confusion Matrix"
        ),
    )
    save_confusion_outputs(
        y_val,
        y_pred,
        FIGURES_CANDIDATE_DIR,
        title=f"Task 3 Candidate: {best_model_name} Validation Confusion Matrix",
    )

    importance = extract_feature_importance(best_pipeline, x_val, y_val)
    importance.to_csv(
        os.path.join(
            METRICS_RAW_DIR,
            "task3_feature_importance.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )
    save_feature_importance_plot(importance, FIGURES_CANDIDATE_DIR)

    pd.DataFrame(
        {
            "Feature": feature_columns,
            "Input_Type": [
                "numeric" if feature in TASK_NUMERIC_COLS else "categorical"
                for feature in feature_columns
            ],
        }
    ).to_csv(
        os.path.join(METRICS_RAW_DIR, "task3_features_used.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    transformed_count = len(
        best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    metrics = {
        "ACC3": validation_metrics["Accuracy"],
        **validation_metrics,
        "best_model": best_model_name,
        "selection_rule": (
            "Highest 5-fold CV Accuracy, matching official ACC3; tie-break "
            "by CV Macro-F1 and CV Balanced Accuracy. Validation metrics are "
            "not used for selection."
        ),
        "class_order": CLASS_ORDER,
        "random_state": RANDOM_STATE,
        "n_folds": N_FOLDS,
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_val)),
        "train_class_counts": {
            class_name: int(np.sum(y_train == class_index))
            for class_index, class_name in enumerate(CLASS_ORDER)
        },
        "validation_class_counts": {
            class_name: int(np.sum(y_val == class_index))
            for class_index, class_name in enumerate(CLASS_ORDER)
        },
        "input_features_used": len(feature_columns),
        "transformed_features_used": transformed_count,
        "excluded": EXCLUDED_FEATURES,
        "model_comparison": json.loads(comparison.to_json(orient="records")),
        "weight_ablation": json.loads(
            weight_ablation.to_json(orient="records")
        ),
        "note": (
            "Encoding and scaling are inside sklearn Pipelines and fitted "
            "separately in every CV fold."
        ),
    }
    with open(
        os.path.join(
            TASK3_CANDIDATE_MODEL_DIR,
            "task3_best_model.pkl",
        ),
        "wb",
    ) as handle:
        pickle.dump(best_pipeline, handle)
    with open(
        os.path.join(METRICS_RAW_DIR, "task3_metrics.pkl"),
        "wb",
    ) as handle:
        pickle.dump(metrics, handle)
    with open(
        os.path.join(METRICS_RAW_DIR, "task3_metrics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(METRICS_RAW_DIR, "task3_metrics.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(f"ACC3 = {validation_metrics['Accuracy']:.6f}\n")
        handle.write(f"Accuracy = {validation_metrics['Accuracy']:.6f}\n")
        handle.write(f"Macro-F1 = {validation_metrics['Macro_F1']:.6f}\n")
        handle.write(
            f"Balanced Accuracy = "
            f"{validation_metrics['Balanced_Accuracy']:.6f}\n"
        )
        for class_name in CLASS_ORDER:
            handle.write(
                f"Recall {class_name} = "
                f"{validation_metrics[f'Recall_{class_name}']:.6f}\n"
            )
        handle.write(f"Best Model = {best_model_name}\n")
        handle.write(
            "Selection = CV Accuracy > CV Macro-F1 > CV Balanced Accuracy\n"
        )
        handle.write(f"Input Features = {len(feature_columns)}\n")
        handle.write(f"Transformed Features = {transformed_count}\n")
    with open(
        os.path.join(METRICS_RAW_DIR, "task3_run_complete.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "status": "complete",
                "best_model": best_model_name,
                "ACC3": validation_metrics["Accuracy"],
                "prediction_rows": int(len(y_val)),
                "selection_rule": metrics["selection_rule"],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    run_complete_path = os.path.join(
        METRICS_RAW_DIR,
        "task3_run_complete.json",
    )
    if os.path.exists(run_complete_path):
        os.remove(run_complete_path)
    print("=" * 72)
    print("Task 3: Wellness Category Prediction (v3.4)")
    print("Selection: 5-fold CV Accuracy; preprocessing: fold-local Pipeline")
    print("=" * 72)

    print("\n[1/7] Loading and cleaning data...")
    raw = load_and_clean_data(DATA_PATH)
    print(f"  Shape: {raw.shape}, Missing: {raw.isnull().sum().sum()}")

    print("\n[2/7] Validating the four-class target...")
    observed = set(raw["Wellness_Category"].astype(str).unique())
    if observed != set(CLASS_ORDER):
        raise ValueError(
            f"Expected classes {CLASS_ORDER}; found {sorted(observed)}"
        )
    y_all = (
        raw["Wellness_Category"]
        .astype(str)
        .map(CLASS_TO_INDEX)
        .to_numpy(dtype=int)
    )
    print(raw["Wellness_Category"].value_counts().to_string())

    print("\n[3/7] Selecting leakage-safe raw features...")
    numeric_columns, categorical_columns = select_feature_columns(raw)
    feature_columns = numeric_columns + categorical_columns
    x_all = raw[feature_columns].copy()
    print(
        f"  {len(numeric_columns)} numeric + "
        f"{len(categorical_columns)} categorical = "
        f"{len(feature_columns)} raw inputs"
    )
    print(f"  Leakage intersection: {set(feature_columns) & set(EXCLUDED_FEATURES)}")

    print("\n[4/7] Loading shared Person_ID split...")
    manifest_path = os.path.join(SPLITS_DIR, "split_manifest.csv")
    train_idx, val_idx = load_manifest_split_indices(
        raw,
        manifest_path,
        id_column=ID_COLUMN,
        label_checks={"task3_label": raw["Wellness_Category"]},
    )
    x_train = x_all.iloc[train_idx].reset_index(drop=True)
    x_val = x_all.iloc[val_idx].reset_index(drop=True)
    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    ids_val = raw.iloc[val_idx][ID_COLUMN].astype(str).to_numpy()
    print(f"  Train={len(train_idx)}, Validation={len(val_idx)}")
    for class_index, class_name in enumerate(CLASS_ORDER):
        print(
            f"    {class_name:9s}: train={np.sum(y_train == class_index)}, "
            f"val={np.sum(y_val == class_index)}"
        )

    print("\n[5/7] Running leakage-safe 5-fold comparison and tuning...")
    models = build_model_pipelines(numeric_columns, categorical_columns)
    validate_required_models(models)
    base_cv_comparison = run_cross_validation(models, x_train, y_train)
    print("  Logistic Regression C tuning (training CV only):")
    tuning_models = build_logistic_tuning_pipelines(
        numeric_columns,
        categorical_columns,
    )
    tuning_cv_comparison = run_cross_validation(
        tuning_models,
        x_train,
        y_train,
    )
    best_tuning_name = select_best_model(tuning_cv_comparison)
    models.update(tuning_models)
    cv_comparison = pd.concat(
        [base_cv_comparison, tuning_cv_comparison],
        ignore_index=True,
    )
    best_name = select_best_model(cv_comparison)
    print(f"  Best tuning configuration: {best_tuning_name}")
    print(f"  Selected by CV only: {best_name}")

    print("\n[6/7] Fitting all candidates and reporting validation...")
    fitted, predictions, validation = fit_and_evaluate_candidates(
        models,
        x_train,
        y_train,
        x_val,
        y_val,
    )
    comparison = add_validation_metrics(cv_comparison, validation, best_name)
    tuning_comparison = comparison[
        comparison["Model"].isin(tuning_models)
    ].copy()
    tuning_comparison.insert(
        2,
        "C",
        tuning_comparison["Model"].str.extract(
            r"Tuned C=([0-9.]+)",
            expand=False,
        ).astype(float),
    )
    tuning_comparison.insert(
        3,
        "Best_Within_Tuning",
        tuning_comparison["Model"].eq(best_tuning_name),
    )
    tuning_comparison.insert(
        4,
        "Tuning_Rule",
        "5-fold training CV Accuracy; no validation metrics used",
    )
    tuning_comparison = tuning_comparison[
        [
            "Model",
            "CV_Rank",
            "C",
            "Best_Within_Tuning",
            "Tuning_Rule",
        ]
        + [
            column
            for column in tuning_comparison.columns
            if column.startswith("CV_")
            and column != "CV_Rank"
        ]
    ]
    weight_ablation = build_weight_ablation(comparison)
    best_pipeline = fitted[best_name]
    best_prediction = predictions[best_name]
    best_metrics = validation[best_name]
    baseline_prediction = predictions[
        "Logistic Regression (Unweighted)"
    ]

    print("\n[7/7] Saving complete outputs...")
    save_outputs(
        ids_val=ids_val,
        y_train=y_train,
        y_val=y_val,
        y_pred=best_prediction,
        baseline_pred=baseline_prediction,
        comparison=comparison,
        tuning_comparison=tuning_comparison,
        weight_ablation=weight_ablation,
        best_model_name=best_name,
        best_pipeline=best_pipeline,
        x_val=x_val,
        validation_metrics=best_metrics,
        feature_columns=feature_columns,
    )
    for metric_name, value in best_metrics.items():
        print(f"  {metric_name:20s}: {value:.4f}")
    print(f"\nMetrics saved to: {METRICS_RAW_DIR}")
    print(f"Baseline figure: {FIGURES_BASELINE_DIR}")
    print(f"Candidate figures: {FIGURES_CANDIDATE_DIR}")
    print(f"Task 3 score contribution: {best_metrics['Accuracy'] * 40:.2f}/40.00")


if __name__ == "__main__":
    main()
