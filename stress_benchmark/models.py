from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import BenchmarkConfig
from .decision_support import apply_decision_support


warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed`.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")


def _dependency_versions() -> Dict[str, Optional[str]]:
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "imbalanced-learn",
        "xgboost",
        "lightgbm",
        "joblib",
    ]
    versions: Dict[str, Optional[str]] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


EXCLUDE_EXACT = {
    "_target",
    "target3",
    "target_binary",
    "nurse_label",
    "wesad_protocol_label",
    "label_purity",
    "label_overlap_ratio",
    "window_start_ts",
    "window_end_ts",
    "survey_offset_hours",
}

EXCLUDE_SUBSTRINGS = (
    "label",
    "target",
    "subject",
    "session",
    "source",
    "group",
    "utc",
    "recommendation",
    "alert",
    "window",
    "offset",
)


@dataclass
class ModelSpec:
    name: str
    estimator: object
    needs_scaling: bool = False
    use_external_resampling: bool = True


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for benchmarking. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc


def feature_columns(df: pd.DataFrame, include_time_features: bool = False) -> List[str]:
    cols = []
    for col in df.columns:
        lowered = col.lower()
        if col in EXCLUDE_EXACT:
            continue
        if any(part in lowered for part in EXCLUDE_SUBSTRINGS):
            continue
        if not include_time_features and lowered in {"hour", "dayofweek", "month"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def available_model_specs(config: BenchmarkConfig) -> Dict[str, ModelSpec]:
    _require_sklearn()
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier

    specs: Dict[str, ModelSpec] = {
        "rf": ModelSpec(
            "rf",
            RandomForestClassifier(
                n_estimators=500,
                criterion="entropy",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=config.random_state,
            ),
            needs_scaling=False,
        ),
        "extratrees": ModelSpec(
            "extratrees",
            ExtraTreesClassifier(
                n_estimators=600,
                criterion="entropy",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=config.random_state,
            ),
            needs_scaling=False,
        ),
        "knn": ModelSpec("knn", KNeighborsClassifier(n_neighbors=9, weights="distance"), needs_scaling=True),
        "gnb": ModelSpec("gnb", GaussianNB(var_smoothing=1e-8), needs_scaling=False),
        "gb": ModelSpec(
            "gb",
            GradientBoostingClassifier(random_state=config.random_state, learning_rate=0.05, n_estimators=250, max_depth=3),
            needs_scaling=False,
        ),
    }

    try:
        from imblearn.ensemble import BalancedRandomForestClassifier

        specs["brf"] = ModelSpec(
            "brf",
            BalancedRandomForestClassifier(
                n_estimators=600,
                criterion="entropy",
                min_samples_leaf=2,
                sampling_strategy="all",
                replacement=True,
                bootstrap=False,
                n_jobs=-1,
                random_state=config.random_state,
            ),
            needs_scaling=False,
            use_external_resampling=False,
        )
    except Exception:
        pass

    try:
        from xgboost import XGBClassifier

        specs["xgb"] = ModelSpec(
            "xgb",
            XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                tree_method="hist",
                random_state=config.random_state,
            ),
            needs_scaling=False,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        specs["lgbm"] = ModelSpec(
            "lgbm",
            LGBMClassifier(
                n_estimators=600,
                learning_rate=0.035,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight="balanced",
                random_state=config.random_state,
                verbose=-1,
            ),
            needs_scaling=False,
        )
    except Exception:
        pass

    return specs


def _requested_model_specs(config: BenchmarkConfig) -> List[ModelSpec]:
    specs = available_model_specs(config)
    requested = [name.strip() for name in config.models if name.strip()]
    missing = [name for name in requested if name not in specs]
    if missing:
        raise RuntimeError(
            "Requested model(s) are unavailable in this Python environment: "
            f"{', '.join(missing)}. Install the full requirements or remove them "
            "explicitly from --models before running a reduced benchmark."
        )
    return [specs[name] for name in requested]


def _make_splits(df: pd.DataFrame, config: BenchmarkConfig):
    _require_sklearn()
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold

    groups = df["group_id"].astype(str).to_numpy() if "group_id" in df else None
    y = df["_target"].to_numpy()
    n_groups = len(np.unique(groups)) if groups is not None else 0
    if config.protocol == "loso" and groups is not None:
        return list(LeaveOneGroupOut().split(df, y, groups))
    if config.protocol == "groupkfold" and groups is not None and n_groups >= 2:
        n_splits = min(config.n_splits, n_groups)
        try:
            from sklearn.model_selection import StratifiedGroupKFold

            return list(
                StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state).split(df, y, groups)
            )
        except Exception:
            return list(GroupKFold(n_splits=n_splits).split(df, y, groups))
    n_splits = min(config.n_splits, max(2, int(pd.Series(y).value_counts().min())))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state).split(df, y))


def _maybe_resample(X: np.ndarray, y: np.ndarray, use_smote: bool, random_state: int) -> Tuple[np.ndarray, np.ndarray, str]:
    if not use_smote:
        return X, y, "none"
    counts = pd.Series(y).value_counts()
    min_count = int(counts.min()) if not counts.empty else 0
    if min_count < 3:
        return X, y, "skipped_minority_too_small"
    try:
        from imblearn.over_sampling import SMOTE

        k_neighbors = max(1, min(5, min_count - 1))
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res, f"smote_k{k_neighbors}"
    except Exception:
        return X, y, "skipped_imblearn_missing"


def _group_class_cap_indices(
    y: np.ndarray,
    groups: Optional[np.ndarray],
    cap: Optional[int],
    random_state: int,
) -> Tuple[np.ndarray, str]:
    if cap is None or cap <= 0 or groups is None:
        return np.arange(len(y)), "none"
    rng = np.random.default_rng(random_state)
    selected = []
    for group_value in np.unique(groups):
        group_idx = np.flatnonzero(groups == group_value)
        for class_value in np.unique(y[group_idx]):
            idx = group_idx[y[group_idx] == class_value]
            if len(idx) > cap:
                idx = rng.choice(idx, size=cap, replace=False)
            selected.extend(idx.tolist())
    selected = np.asarray(sorted(selected), dtype=int)
    if selected.size == 0:
        return np.arange(len(y)), "skipped_empty"
    return selected, f"cap_{cap}_per_group_class"


def _source_balance_indices(
    y: np.ndarray,
    sources: Optional[np.ndarray],
    mode: str,
    random_state: int,
) -> Tuple[np.ndarray, str]:
    if mode == "none" or sources is None:
        return np.arange(len(y)), "none"
    source_values = np.asarray(sources).astype(str)
    unique_sources = np.unique(source_values)
    if len(unique_sources) < 2:
        return np.arange(len(y)), "none_single_source"

    rng = np.random.default_rng(random_state)
    selected: List[int] = []
    if mode == "source":
        counts = {source: int(np.sum(source_values == source)) for source in unique_sources}
        cap = min(counts.values())
        for source in unique_sources:
            idx = np.flatnonzero(source_values == source)
            if len(idx) > cap:
                idx = rng.choice(idx, size=cap, replace=False)
            selected.extend(idx.tolist())
        selected = np.asarray(sorted(selected), dtype=int)
        return selected, f"source_balanced_total_{cap}"

    if mode == "source_class":
        for class_value in np.unique(y):
            class_idx = np.flatnonzero(y == class_value)
            class_sources = np.unique(source_values[class_idx])
            if len(class_sources) < 2:
                selected.extend(class_idx.tolist())
                continue
            counts = {
                source: int(np.sum((source_values == source) & (y == class_value)))
                for source in class_sources
            }
            cap = min(counts.values())
            for source in class_sources:
                idx = np.flatnonzero((source_values == source) & (y == class_value))
                if len(idx) > cap:
                    idx = rng.choice(idx, size=cap, replace=False)
                selected.extend(idx.tolist())
        selected = np.asarray(sorted(set(selected)), dtype=int)
        if selected.size:
            return selected, "source_balanced_per_class"
    return np.arange(len(y)), "skipped_source_balance"


def _candidate_values(primary: object, grid: Sequence[object]) -> Tuple[object, ...]:
    return tuple(grid) if grid else (primary,)


def _fit_feature_selector(
    X_np: np.ndarray,
    y_encoded: np.ndarray,
    feature_k: Optional[int],
    random_state: int,
) -> Tuple[Optional[np.ndarray], str]:
    n_features = X_np.shape[1]
    if feature_k is None or feature_k <= 0 or feature_k >= n_features:
        return None, f"all_{n_features}"
    try:
        from sklearn.ensemble import ExtraTreesClassifier

        selector = ExtraTreesClassifier(
            n_estimators=300,
            criterion="entropy",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
        selector.fit(X_np, y_encoded)
        importances = np.nan_to_num(selector.feature_importances_, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.any(importances > 0):
            return None, f"skipped_zero_importance_all_{n_features}"
        selected = np.argsort(importances)[::-1][: min(feature_k, n_features)]
        selected = np.asarray(sorted(selected.tolist()), dtype=int)
        return selected, f"extratrees_top_{len(selected)}"
    except Exception as exc:
        return None, f"skipped_feature_selection_{type(exc).__name__}"


def _apply_feature_selector(X_np: np.ndarray, selected_indices: Optional[np.ndarray]) -> np.ndarray:
    if selected_indices is None:
        return X_np
    return X_np[:, selected_indices]


def _score_predictions(y_true: np.ndarray, pred: np.ndarray, metric: str) -> float:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    if metric == "macro_f1":
        return float(f1_score(y_true, pred, average="macro", zero_division=0))
    return float(balanced_accuracy_score(y_true, pred))


def _set_model_objective(model: object, n_classes: int) -> None:
    if model.__class__.__name__ == "XGBClassifier":
        if n_classes <= 2:
            model.set_params(objective="binary:logistic", eval_metric="logloss")
        else:
            model.set_params(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")


def _calibration_split(
    y_encoded: np.ndarray,
    groups: Optional[np.ndarray],
    random_state: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    _require_sklearn()
    classes = set(np.unique(y_encoded).tolist())
    if len(classes) < 2 or len(y_encoded) < 20:
        return None

    if groups is not None and len(np.unique(groups)) >= 4:
        from sklearn.model_selection import StratifiedGroupKFold

        for n_splits in [5, 4, 3, 2]:
            if len(np.unique(groups)) < n_splits:
                continue
            try:
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                for model_idx, cal_idx in splitter.split(np.zeros_like(y_encoded), y_encoded, groups):
                    if set(np.unique(y_encoded[cal_idx]).tolist()) == classes and set(np.unique(y_encoded[model_idx]).tolist()) == classes:
                        return model_idx, cal_idx
            except Exception:
                continue

    from sklearn.model_selection import StratifiedShuffleSplit

    for test_size in [0.2, 0.25, 0.33]:
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            model_idx, cal_idx = next(splitter.split(np.zeros_like(y_encoded), y_encoded))
            if set(np.unique(y_encoded[cal_idx]).tolist()) == classes and set(np.unique(y_encoded[model_idx]).tolist()) == classes:
                return model_idx, cal_idx
        except Exception:
            continue
    return None


def _fit_base_model(
    spec: ModelSpec,
    X_np: np.ndarray,
    y_encoded: np.ndarray,
    n_classes: int,
    config: BenchmarkConfig,
) -> Tuple[object, str]:
    from sklearn.base import clone

    if spec.use_external_resampling:
        X_train_np, y_train_resampled, resample_note = _maybe_resample(
            X_np, y_encoded, config.use_smote, config.random_state
        )
    else:
        X_train_np, y_train_resampled, resample_note = X_np, y_encoded, "model_internal_balancing"
    model = clone(spec.estimator)
    _set_model_objective(model, n_classes)
    model.fit(X_train_np, y_train_resampled)
    return model, resample_note


def _tune_binary_threshold(
    y_true_encoded: np.ndarray,
    proba: Optional[np.ndarray],
    classes: np.ndarray,
    metric: str,
    positive_recall_floor: float = 0.70,
) -> Tuple[Optional[float], str]:
    if proba is None or proba.shape[1] != 2 or len(np.unique(y_true_encoded)) < 2:
        return None, "skipped"
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

    positive_encoded = 1 if 1 in classes else int(classes[-1])
    pos_idx = int(np.flatnonzero(classes == positive_encoded)[0])
    best_threshold = 0.5
    best_score = -np.inf
    best_precision = 0.0
    best_recall = 0.0
    best_balanced_accuracy = 0.0
    positive_proba = proba[:, pos_idx]
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = np.where(positive_proba >= threshold, positive_encoded, int([cls for cls in classes.tolist() if cls != positive_encoded][0]))
        recall = recall_score(y_true_encoded, pred, pos_label=positive_encoded, zero_division=0)
        precision = precision_score(y_true_encoded, pred, pos_label=positive_encoded, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true_encoded, pred)
        if metric == "macro_f1":
            score = f1_score(y_true_encoded, pred, average="macro", zero_division=0)
        elif metric == "balanced_accuracy_recall_floor":
            score = balanced_acc if recall >= positive_recall_floor else -np.inf
        else:
            score = balanced_acc
        if not np.isfinite(score):
            continue
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_precision = float(precision)
            best_recall = float(recall)
            best_balanced_accuracy = float(balanced_acc)
    if best_score == -np.inf:
        for threshold in np.linspace(0.05, 0.95, 91):
            pred = np.where(positive_proba >= threshold, positive_encoded, int([cls for cls in classes.tolist() if cls != positive_encoded][0]))
            recall = recall_score(y_true_encoded, pred, pos_label=positive_encoded, zero_division=0)
            precision = precision_score(y_true_encoded, pred, pos_label=positive_encoded, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_true_encoded, pred)
            score = recall
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_precision = float(precision)
                best_recall = float(recall)
                best_balanced_accuracy = float(balanced_acc)
        return best_threshold, (
            f"{metric}_fallback_recall:{best_score:.6f};"
            f"precision:{best_precision:.6f};recall:{best_recall:.6f};balanced_accuracy:{best_balanced_accuracy:.6f}"
        )
    return best_threshold, (
        f"{metric}:{best_score:.6f};"
        f"precision:{best_precision:.6f};recall:{best_recall:.6f};balanced_accuracy:{best_balanced_accuracy:.6f}"
    )


def _predict_encoded_with_threshold(
    model: object,
    X_np: np.ndarray,
    threshold: Optional[float],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    proba = model.predict_proba(X_np) if hasattr(model, "predict_proba") else None
    if threshold is not None and proba is not None and proba.shape[1] == 2:
        classes = np.asarray(model.classes_, dtype=int)
        positive_encoded = 1 if 1 in classes else int(classes[-1])
        pos_idx = int(np.flatnonzero(classes == positive_encoded)[0])
        neg_encoded = int([cls for cls in classes.tolist() if cls != positive_encoded][0])
        pred_encoded = np.where(proba[:, pos_idx] >= threshold, positive_encoded, neg_encoded)
        return pred_encoded.astype(int), proba
    pred_encoded = model.predict(X_np)
    return pred_encoded.astype(int), proba


def _confidence_for_predictions(pred: np.ndarray, proba: Optional[np.ndarray], classes: np.ndarray) -> np.ndarray:
    if proba is None:
        return np.full(len(pred), np.nan)
    class_to_idx = {int(label): idx for idx, label in enumerate(classes.tolist())}
    confidence = np.full(len(pred), np.nan)
    for row_idx, label in enumerate(pred):
        class_idx = class_to_idx.get(int(label))
        if class_idx is not None and class_idx < proba.shape[1]:
            confidence[row_idx] = proba[row_idx, class_idx]
    return confidence


def _maybe_calibrate_model(model: object, X_cal: np.ndarray, y_cal: np.ndarray, config: BenchmarkConfig) -> Tuple[object, str]:
    if not config.calibrate:
        return model, "none"
    if len(np.unique(y_cal)) < 2:
        return model, "skipped_calibration_missing_classes"
    try:
        from sklearn.calibration import CalibratedClassifierCV
        try:
            from sklearn.frozen import FrozenEstimator
        except Exception:
            from sklearn.calibration import FrozenEstimator

        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method=config.calibration_method,
            cv=None,
        )
        calibrated.fit(X_cal, y_cal)
        return calibrated, f"{config.calibration_method}_holdout"
    except Exception as exc:
        return model, f"skipped_calibration_{type(exc).__name__}"


def _transform_features_for_artifact(artifact: Dict[str, object], X: pd.DataFrame) -> np.ndarray:
    X_np = artifact["imputer"].transform(X)
    selected_indices = artifact.get("selected_feature_indices")
    if selected_indices is not None:
        X_np = _apply_feature_selector(X_np, np.asarray(selected_indices, dtype=int))
    scaler = artifact.get("scaler")
    if scaler is not None:
        X_np = scaler.transform(X_np)
    return X_np


def _fit_candidate_artifact(
    spec: ModelSpec,
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    model_idx: np.ndarray,
    cal_idx: Optional[np.ndarray],
    config: BenchmarkConfig,
    groups: Optional[np.ndarray],
    sources: Optional[np.ndarray],
    n_classes: int,
    class_cap: Optional[int],
    feature_k: Optional[int],
) -> Dict[str, object]:
    _require_sklearn()
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    cap_positions = np.asarray(model_idx)
    if n_classes == 2:
        rel_cap_idx, cap_note = _group_class_cap_indices(
            y_encoded[model_idx],
            groups[model_idx] if groups is not None else None,
            class_cap,
            config.random_state,
        )
        cap_positions = cap_positions[rel_cap_idx]
    else:
        cap_note = "none"

    source_positions = cap_positions
    if n_classes == 2:
        rel_source_idx, source_note = _source_balance_indices(
            y_encoded[source_positions],
            sources[source_positions] if sources is not None else None,
            config.source_balance,
            config.random_state,
        )
        source_positions = source_positions[rel_source_idx]
    else:
        source_note = "none"

    imputer = SimpleImputer(strategy="median")
    X_model_np = imputer.fit_transform(X.iloc[source_positions])
    y_model = y_encoded[source_positions]
    selected_indices, feature_note = _fit_feature_selector(X_model_np, y_model, feature_k, config.random_state)
    X_model_np = _apply_feature_selector(X_model_np, selected_indices)
    scaler = None
    if spec.needs_scaling:
        scaler = StandardScaler()
        X_model_np = scaler.fit_transform(X_model_np)

    model, resample_note = _fit_base_model(spec, X_model_np, y_model, n_classes, config)
    if cal_idx is not None:
        X_cal_np = imputer.transform(X.iloc[cal_idx])
        X_cal_np = _apply_feature_selector(X_cal_np, selected_indices)
        if scaler is not None:
            X_cal_np = scaler.transform(X_cal_np)
        model, calibration_note = _maybe_calibrate_model(model, X_cal_np, y_encoded[cal_idx], config)
        _, cal_proba = _predict_encoded_with_threshold(model, X_cal_np, None)
        threshold_metrics = _candidate_values(config.threshold_metric, config.threshold_metric_grid)
        best_threshold = None
        best_threshold_note = "none"
        best_threshold_score = -np.inf
        if config.tune_threshold and n_classes == 2:
            for threshold_metric in threshold_metrics:
                threshold, threshold_note = _tune_binary_threshold(
                    y_encoded[cal_idx],
                    cal_proba,
                    np.asarray(model.classes_, dtype=int),
                    str(threshold_metric),
                    config.positive_recall_floor,
                )
                cal_pred_encoded, _ = _predict_encoded_with_threshold(model, X_cal_np, threshold)
                score = _score_predictions(y_encoded[cal_idx], cal_pred_encoded, config.tuning_selection_metric)
                if score > best_threshold_score:
                    best_threshold_score = score
                    best_threshold = threshold
                    best_threshold_note = f"{threshold_note};selection_{config.tuning_selection_metric}:{score:.6f}"
        else:
            cal_pred_encoded, _ = _predict_encoded_with_threshold(model, X_cal_np, None)
            best_threshold_score = _score_predictions(y_encoded[cal_idx], cal_pred_encoded, config.tuning_selection_metric)
        threshold = best_threshold
        threshold_note = best_threshold_note
        if not config.tune_threshold or n_classes != 2:
            threshold_note = "none"
        cal_pred_encoded, _ = _predict_encoded_with_threshold(model, X_cal_np, threshold)
        candidate_score = _score_predictions(y_encoded[cal_idx], cal_pred_encoded, config.tuning_selection_metric)
    else:
        calibration_note = "skipped_no_calibration_split" if config.calibrate else "none"
        threshold, threshold_note = (None, "skipped_no_calibration_split")
        candidate_score = float("nan")

    return {
        "model_name": spec.name,
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "selected_feature_indices": selected_indices.tolist() if selected_indices is not None else None,
        "feature_selection": feature_note,
        "n_selected_features": int(len(selected_indices)) if selected_indices is not None else int(X.shape[1]),
        "resampling": resample_note,
        "calibration": calibration_note,
        "class_cap": cap_note,
        "source_balance": source_note,
        "decision_threshold": threshold,
        "threshold_tuning": threshold_note,
        "tuning_selection": f"{config.tuning_selection_metric}:{candidate_score:.6f}" if np.isfinite(candidate_score) else "none",
        "candidate_score": candidate_score,
    }


def _select_best_candidate(
    spec: ModelSpec,
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    model_idx: np.ndarray,
    cal_idx: Optional[np.ndarray],
    config: BenchmarkConfig,
    groups: Optional[np.ndarray],
    sources: Optional[np.ndarray],
    n_classes: int,
) -> Dict[str, object]:
    class_cap_values = _candidate_values(config.class_cap_per_group, config.class_cap_grid)
    feature_k_values = _candidate_values(config.feature_selection_k, config.feature_selection_k_grid)
    best: Optional[Dict[str, object]] = None
    best_score = -np.inf
    for class_cap in class_cap_values:
        for feature_k in feature_k_values:
            candidate = _fit_candidate_artifact(
                spec,
                X,
                y_encoded,
                model_idx,
                cal_idx,
                config,
                groups,
                sources,
                n_classes,
                class_cap if n_classes == 2 else None,
                feature_k,
            )
            score = candidate.get("candidate_score", float("nan"))
            score_value = float(score) if np.isfinite(score) else -np.inf
            if best is None or score_value > best_score:
                best = candidate
                best_score = score_value
    if best is None:
        raise RuntimeError("No model candidate could be fitted.")
    return best


def _fit_predict(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    config: BenchmarkConfig,
    groups_train: Optional[np.ndarray] = None,
    sources_train: Optional[np.ndarray] = None,
):
    _require_sklearn()
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    if len(encoder.classes_) < 2:
        only_class = encoder.classes_[0]
        pred = np.full(len(X_test), only_class)
        proba = np.ones((len(X_test), 1), dtype=float)
        return {
            "pred": pred,
            "proba": proba,
            "proba_classes": encoder.classes_,
            "resampling": "skipped_single_class_train",
            "calibration": "none",
            "class_cap": "none",
            "source_balance": "none",
            "feature_selection": "none",
            "n_selected_features": int(X_train.shape[1]),
            "decision_threshold": None,
            "threshold_tuning": "skipped_single_class_train",
            "tuning_selection": "none",
        }
    split = _calibration_split(y_train_encoded, groups_train, config.random_state) if config.calibrate else None
    if split is None:
        model_idx = np.arange(len(y_train_encoded))
        cal_idx = None
    else:
        model_idx, cal_idx = split

    fitted = _select_best_candidate(
        spec,
        X_train,
        y_train_encoded,
        np.asarray(model_idx),
        None if cal_idx is None else np.asarray(cal_idx),
        config,
        groups_train,
        sources_train,
        len(encoder.classes_),
    )

    X_test_np = _transform_features_for_artifact(fitted, X_test)
    pred_encoded, proba = _predict_encoded_with_threshold(fitted["model"], X_test_np, fitted.get("decision_threshold"))
    pred = encoder.inverse_transform(pred_encoded.astype(int))
    return {
        "pred": pred,
        "proba": proba,
        "proba_classes": encoder.classes_,
        "resampling": fitted["resampling"],
        "calibration": fitted["calibration"],
        "class_cap": fitted["class_cap"],
        "source_balance": fitted["source_balance"],
        "feature_selection": fitted["feature_selection"],
        "n_selected_features": fitted["n_selected_features"],
        "decision_threshold": fitted["decision_threshold"],
        "threshold_tuning": fitted["threshold_tuning"],
        "tuning_selection": fitted["tuning_selection"],
    }

def _fit_deploy_artifact(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    config: BenchmarkConfig,
    groups: Optional[np.ndarray] = None,
    sources: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    _require_sklearn()
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    if len(encoder.classes_) < 2:
        raise ValueError("Final training requires at least two target classes.")
    split = _calibration_split(y_encoded, None, config.random_state) if config.calibrate else None
    if split is None:
        model_idx = np.arange(len(y_encoded))
        cal_idx = None
    else:
        model_idx, cal_idx = split
    fitted = _select_best_candidate(
        spec,
        X,
        y_encoded,
        np.asarray(model_idx),
        None if cal_idx is None else np.asarray(cal_idx),
        config,
        groups,
        sources,
        len(encoder.classes_),
    )
    fitted["label_encoder"] = encoder
    return fitted


def _predict_artifact(artifact: Dict[str, object], X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model = artifact["model"]
    encoder = artifact["label_encoder"]
    X_np = _transform_features_for_artifact(artifact, X)
    pred_encoded, proba = _predict_encoded_with_threshold(model, X_np, artifact.get("decision_threshold"))
    pred = encoder.inverse_transform(pred_encoded.astype(int))
    return pred, proba


def evaluate_models(df: pd.DataFrame, config: BenchmarkConfig, out_dir: Path) -> Dict[str, object]:
    _require_sklearn()
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    task_col = "target3" if config.task == "stress3" else "target_binary"
    if task_col not in df:
        raise ValueError(f"Missing target column: {task_col}")
    work = df[df[task_col].notna()].copy()
    work["_target"] = work[task_col].astype(int)
    cols = feature_columns(work, include_time_features=config.include_time_features)
    if not cols:
        raise ValueError("No numeric feature columns available for benchmarking.")

    splits = _make_splits(work, config)
    selected = _requested_model_specs(config)
    if not selected:
        raise ValueError("No requested models are available.")

    metric_rows = []
    prediction_frames = []
    labels = sorted(work["_target"].unique().tolist())

    for spec in selected:
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train = work.iloc[train_idx]
            test = work.iloc[test_idx].copy()
            X_train = train[cols]
            X_test = test[cols]
            y_train = train["_target"].to_numpy()
            y_test = test["_target"].to_numpy()
            groups_train = train["group_id"].astype(str).to_numpy() if "group_id" in train else None
            sources_train = train["source"].astype(str).to_numpy() if "source" in train else None
            fitted = _fit_predict(
                spec, X_train, y_train, X_test, config, groups_train=groups_train, sources_train=sources_train
            )
            pred = fitted["pred"]
            proba = fitted["proba"]
            proba_classes = fitted["proba_classes"]

            test["model"] = spec.name
            test["fold"] = fold_idx
            test["resampling"] = fitted["resampling"]
            test["calibration"] = fitted["calibration"]
            test["class_cap"] = fitted["class_cap"]
            test["source_balance"] = fitted["source_balance"]
            test["feature_selection"] = fitted["feature_selection"]
            test["n_selected_features"] = fitted["n_selected_features"]
            test["decision_threshold"] = fitted["decision_threshold"]
            test["threshold_tuning"] = fitted["threshold_tuning"]
            test["tuning_selection"] = fitted["tuning_selection"]
            test["pred_label"] = pred
            if proba is not None:
                test["confidence"] = _confidence_for_predictions(pred, proba, np.asarray(proba_classes))
                for class_idx, class_label in enumerate(proba_classes):
                    if class_idx < proba.shape[1]:
                        test[f"proba_{class_label}"] = proba[:, class_idx]
            else:
                test["confidence"] = np.nan
            test = apply_decision_support(test)
            prediction_frames.append(test)

            report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, pred, labels=labels)
            metric_rows.append(
                {
                    "model": spec.name,
                    "fold": fold_idx,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "resampling": fitted["resampling"],
                    "calibration": fitted["calibration"],
                    "class_cap": fitted["class_cap"],
                    "source_balance": fitted["source_balance"],
                    "feature_selection": fitted["feature_selection"],
                    "n_selected_features": fitted["n_selected_features"],
                    "decision_threshold": fitted["decision_threshold"],
                    "threshold_tuning": fitted["threshold_tuning"],
                    "tuning_selection": fitted["tuning_selection"],
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                    "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                    "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
                    "classification_report": json.dumps(report),
                    "confusion_matrix": json.dumps(cm.tolist()),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_{config.task}.csv"
    pred_path = out_dir / f"predictions_{config.task}.csv.gz"
    summary_path = out_dir / f"summary_{config.task}.json"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(pred_path, index=False, compression="gzip")

    aggregate = metrics.groupby("model")[["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    summary = {
        "task": config.task,
        "protocol": config.protocol,
        "n_rows": int(len(work)),
        "n_features": int(len(cols)),
        "feature_columns": cols,
        "models": [spec.name for spec in selected],
        "dependency_versions": _dependency_versions(),
        "tuning": {
            "class_cap_per_group": config.class_cap_per_group,
            "class_cap_grid": list(config.class_cap_grid),
            "feature_selection_k": config.feature_selection_k,
            "feature_selection_k_grid": list(config.feature_selection_k_grid),
            "threshold_metric": config.threshold_metric,
            "threshold_metric_grid": list(config.threshold_metric_grid),
            "positive_recall_floor": config.positive_recall_floor,
            "tuning_selection_metric": config.tuning_selection_metric,
            "source_balance": config.source_balance,
        },
        "metrics_by_model": aggregate.round(6).to_dict(orient="index"),
        "paths": {
            "metrics": str(metrics_path),
            "predictions": str(pred_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_deploy_models(df: pd.DataFrame, config: BenchmarkConfig, out_dir: Path) -> Dict[str, object]:
    _require_sklearn()
    import joblib

    task_col = "target3" if config.task == "stress3" else "target_binary"
    if task_col not in df:
        raise ValueError(f"Missing target column: {task_col}")
    work = df[df[task_col].notna()].copy()
    work["_target"] = work[task_col].astype(int)
    cols = feature_columns(work, include_time_features=config.include_time_features)
    if not cols:
        raise ValueError("No numeric feature columns available for final training.")

    selected = _requested_model_specs(config)
    if not selected:
        raise ValueError("No requested models are available.")

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    X = work[cols]
    y = work["_target"].to_numpy()
    class_distribution = {str(k): int(v) for k, v in work["_target"].value_counts().sort_index().items()}
    saved = []

    for spec in selected:
        groups = work["group_id"].astype(str).to_numpy() if "group_id" in work else None
        sources = work["source"].astype(str).to_numpy() if "source" in work else None
        fitted = _fit_deploy_artifact(spec, X, y, config, groups=groups, sources=sources)
        pred, proba = _predict_artifact(fitted, X)
        selected_indices = fitted.get("selected_feature_indices")
        selected_feature_columns = (
            [cols[int(idx)] for idx in selected_indices]
            if selected_indices is not None
            else cols
        )
        artifact = {
            "artifact_type": "stress_benchmark_deploy_bundle",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": config.task,
            "target_column": task_col,
            "model_name": spec.name,
            "dependency_versions": _dependency_versions(),
            "feature_columns": cols,
            "selected_feature_indices": selected_indices,
            "selected_feature_columns": selected_feature_columns,
            "feature_selection": fitted["feature_selection"],
            "n_selected_features": fitted["n_selected_features"],
            "class_cap": fitted["class_cap"],
            "source_balance": fitted["source_balance"],
            "decision_threshold": fitted["decision_threshold"],
            "threshold_tuning": fitted["threshold_tuning"],
            "tuning_selection": fitted["tuning_selection"],
            "calibration": fitted["calibration"],
            "label_classes": [int(x) for x in fitted["label_encoder"].classes_.tolist()],
            "label_semantics": {
                "0": "normal_or_low_stress",
                "1": "moderate_or_suspected_stress",
                "2": "high_stress",
            }
            if config.task == "stress3"
            else {"0": "non_stress", "1": "stress"},
            "baseline_policy": {
                "minutes": 10,
                "features": ["delta_base", "ratio_base", "z_personal"],
                "backend_note": "Collect a per-user normal baseline window before inference, then create the same personalized feature columns.",
            },
            "decision_layer": {
                "module": "stress_benchmark.decision_support.apply_decision_support",
                "states": [
                    "normal",
                    "low_confidence_normal",
                    "monitor_more",
                    "rising_stress",
                    "persistent_stress",
                    "critical_alert",
                    "physical_activity_delay",
                ],
            },
            "training_info": {
                "n_rows": int(len(work)),
                "n_features": int(len(cols)),
                "class_distribution": class_distribution,
                "resampling": fitted["resampling"],
                "calibration": fitted["calibration"],
                "class_cap": fitted["class_cap"],
                "source_balance": fitted["source_balance"],
                "feature_selection": fitted["feature_selection"],
                "n_selected_features": fitted["n_selected_features"],
                "decision_threshold": fitted["decision_threshold"],
                "threshold_tuning": fitted["threshold_tuning"],
                "tuning_selection": fitted["tuning_selection"],
                "use_smote": bool(config.use_smote),
                "class_cap_grid": list(config.class_cap_grid),
                "feature_selection_k_grid": list(config.feature_selection_k_grid),
                "threshold_metric_grid": list(config.threshold_metric_grid),
                "positive_recall_floor": config.positive_recall_floor,
                "tuning_selection_metric": config.tuning_selection_metric,
            },
            "imputer": fitted["imputer"],
            "scaler": fitted["scaler"],
            "label_encoder": fitted["label_encoder"],
            "model": fitted["model"],
        }
        artifact_path = model_dir / f"{config.task}_{spec.name}.joblib"
        joblib.dump(artifact, artifact_path)
        confidence = _confidence_for_predictions(pred, proba, np.asarray(fitted["model"].classes_, dtype=int))
        saved.append(
            {
                "model": spec.name,
                "path": str(artifact_path),
                "training_accuracy": float(np.mean(pred == y)),
                "mean_confidence": float(np.nanmean(confidence)) if proba is not None else None,
                "resampling": fitted["resampling"],
                "calibration": fitted["calibration"],
                "class_cap": fitted["class_cap"],
                "source_balance": fitted["source_balance"],
                "feature_selection": fitted["feature_selection"],
                "n_selected_features": fitted["n_selected_features"],
                "decision_threshold": fitted["decision_threshold"],
                "threshold_tuning": fitted["threshold_tuning"],
                "tuning_selection": fitted["tuning_selection"],
            }
        )

    best_model = None
    metrics_path = out_dir / f"metrics_{config.task}.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if not metrics.empty and "weighted_f1" in metrics:
            ranking = metrics.groupby("model")["weighted_f1"].mean().sort_values(ascending=False)
            for model_name in ranking.index.tolist():
                candidate = next((item for item in saved if item["model"] == model_name), None)
                if candidate:
                    best_model = candidate
                    break
    if best_model is None and saved:
        best_model = sorted(saved, key=lambda item: item["training_accuracy"], reverse=True)[0]

    if best_model:
        best_path = model_dir / f"{config.task}_best_model.joblib"
        shutil.copy2(best_model["path"], best_path)
        best_model = {**best_model, "best_model_path": str(best_path)}

    manifest = {
        "task": config.task,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(work)),
        "n_features": int(len(cols)),
        "dependency_versions": _dependency_versions(),
        "tuning": {
            "class_cap_per_group": config.class_cap_per_group,
            "class_cap_grid": list(config.class_cap_grid),
            "feature_selection_k": config.feature_selection_k,
            "feature_selection_k_grid": list(config.feature_selection_k_grid),
            "threshold_metric": config.threshold_metric,
            "threshold_metric_grid": list(config.threshold_metric_grid),
            "positive_recall_floor": config.positive_recall_floor,
            "tuning_selection_metric": config.tuning_selection_metric,
            "source_balance": config.source_balance,
        },
        "models": saved,
        "best_model": best_model,
    }
    manifest_path = model_dir / f"manifest_{config.task}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
