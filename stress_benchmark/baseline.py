from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


METADATA_PREFIXES = (
    "target",
    "label",
    "source",
    "subject",
    "session",
    "group",
    "window_start_utc",
    "window_end_utc",
)


def _is_feature_column(name: str, series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    lowered = name.lower()
    if lowered in {"window_start_ts", "window_end_ts", "window_start_sec", "window_end_sec"}:
        return False
    if any(lowered.startswith(prefix) for prefix in METADATA_PREFIXES):
        return False
    if "label" in lowered:
        return False
    return True


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if _is_feature_column(col, df[col])]


def _sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["source", "subject_id", "window_start_ts", "window_start_sec"] if c in df.columns]
    return df.sort_values(sort_cols) if sort_cols else df


def _baseline_rows(group: pd.DataFrame, baseline_minutes: int) -> pd.DataFrame:
    group = _sort_columns(group)
    max_rows = max(1, int(np.ceil(baseline_minutes * 60.0 / max(float(group["window_duration_sec"].median()), 1.0)))) if "window_duration_sec" in group else 10

    source = str(group["source"].iloc[0]) if "source" in group else ""
    if source == "wesad" and "wesad_protocol_label" in group:
        candidates = group[group["wesad_protocol_label"] == 1]
        if not candidates.empty:
            return _sort_columns(candidates).head(max_rows)

    # For field data there is no explicit calibration segment. Do not use labels to
    # select a low-stress baseline; that would not be available at deployment time.
    return group.head(max_rows)


def add_personal_baseline_features(df: pd.DataFrame, baseline_minutes: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    feature_cols = numeric_feature_columns(out)
    if not feature_cols:
        out["baseline_quality_n"] = 0
        return out

    keys = ["source", "subject_id"] if "source" in out and "subject_id" in out else ["subject_id"]
    frames = []
    for _, group in out.groupby(keys, dropna=False, sort=False):
        group = group.copy()
        base = _baseline_rows(group, baseline_minutes)
        med = base[feature_cols].median(numeric_only=True)
        q25 = base[feature_cols].quantile(0.25, numeric_only=True)
        q75 = base[feature_cols].quantile(0.75, numeric_only=True)
        iqr = (q75 - q25).replace(0, np.nan)
        std = base[feature_cols].std(numeric_only=True).replace(0, np.nan)
        scale = iqr.fillna(std).fillna(1.0)

        derived = {}
        for col in feature_cols:
            base_value = med.get(col, np.nan)
            scale_value = scale.get(col, 1.0)
            derived[f"{col}_delta_base"] = group[col] - base_value
            derived[f"{col}_ratio_base"] = (
                (group[col] + 1e-8) / (base_value + 1e-8) if np.isfinite(base_value) else np.nan
            )
            derived[f"{col}_z_personal"] = (group[col] - base_value) / (scale_value + 1e-8)
        derived["baseline_quality_n"] = int(len(base))
        derived_frame = pd.DataFrame(derived, index=group.index)
        frames.append(pd.concat([group, derived_frame], axis=1))
    return pd.concat(frames, ignore_index=True)
