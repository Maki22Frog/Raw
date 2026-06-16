from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def apply_decision_support(df: pd.DataFrame, pred_col: str = "pred_label", confidence_col: str = "confidence") -> pd.DataFrame:
    if df.empty or pred_col not in df:
        return df
    out = df.copy()
    hr_col = _first_existing(
        out,
        [
            "hr_mean_delta_base",
            "bvp_derived_hr_mean_delta_base",
            "hr_median_delta_base",
            "bvp_derived_hr_median_delta_base",
        ],
    )
    hr_z_col = _first_existing(out, ["hr_mean_z_personal", "bvp_derived_hr_mean_z_personal"])
    eda_col = _first_existing(out, ["eda_mean_delta_base", "eda_median_delta_base"])
    eda_z_col = _first_existing(out, ["eda_mean_z_personal", "eda_median_z_personal"])
    acc_z_col = _first_existing(out, ["acc_activity_z_personal", "acc_dyn_mag_mean_z_personal", "acc_mag_std_z_personal"])
    acc_col = _first_existing(out, ["acc_activity", "acc_dyn_mag_mean", "acc_mag_std"])

    if confidence_col not in out:
        out[confidence_col] = np.nan

    is_binary_task = "proba_1" in out.columns and "proba_2" not in out.columns
    acc_threshold = out[acc_col].quantile(0.75) if acc_col else np.nan
    states = []
    actions = []
    adjusted = []

    sort_cols = [c for c in ["source", "subject_id", "window_start_ts", "window_start_sec"] if c in out.columns]
    ordered = out.sort_values(sort_cols).copy() if sort_cols else out.copy()
    persistent_counter = {}

    for idx, row in ordered.iterrows():
        pred = int(row[pred_col]) if pd.notna(row[pred_col]) else 0
        confidence = float(row[confidence_col]) if pd.notna(row[confidence_col]) else 0.0
        group_key = (row.get("source", ""), row.get("subject_id", ""))

        hr_high = False
        if hr_col and pd.notna(row.get(hr_col)):
            hr_high = hr_high or float(row[hr_col]) >= 10.0
        if hr_z_col and pd.notna(row.get(hr_z_col)):
            hr_high = hr_high or float(row[hr_z_col]) >= 1.0

        eda_high = False
        if eda_col and pd.notna(row.get(eda_col)):
            eda_high = eda_high or float(row[eda_col]) > 0.05
        if eda_z_col and pd.notna(row.get(eda_z_col)):
            eda_high = eda_high or float(row[eda_z_col]) >= 1.0

        active = False
        if acc_z_col and pd.notna(row.get(acc_z_col)):
            active = active or float(row[acc_z_col]) >= 1.5
        if acc_col and pd.notna(row.get(acc_col)) and pd.notna(acc_threshold):
            active = active or float(row[acc_col]) >= float(acc_threshold)

        high_stress_prediction = pred >= 2 or (is_binary_task and pred >= 1)
        stress_like = high_stress_prediction or (pred >= 1 and confidence >= 0.65)
        if stress_like:
            persistent_counter[group_key] = persistent_counter.get(group_key, 0) + 1
        else:
            persistent_counter[group_key] = 0

        if stress_like and active and hr_high and not eda_high:
            state = "physical_activity_delay"
            action = "delay_alert_keep_monitoring"
            adjusted_label = min(pred, 1)
        elif high_stress_prediction and persistent_counter[group_key] >= 3 and confidence >= 0.70:
            state = "critical_alert"
            action = "stop_task_short_break_breathing_support"
            adjusted_label = 2
        elif high_stress_prediction and confidence >= 0.55:
            state = "persistent_stress" if persistent_counter[group_key] >= 2 else "rising_stress"
            action = "short_break_water_slow_breathing"
            adjusted_label = 2
        elif pred >= 1:
            state = "monitor_more"
            action = "measure_more_before_alert"
            adjusted_label = 1
        elif confidence < 0.45:
            state = "low_confidence_normal"
            action = "continue_monitoring"
            adjusted_label = 0
        else:
            state = "normal"
            action = "no_action"
            adjusted_label = 0

        states.append((idx, state))
        actions.append((idx, action))
        adjusted.append((idx, adjusted_label))

    out["alert_state"] = pd.Series(dict(states))
    out["recommendation"] = pd.Series(dict(actions))
    out["decision_adjusted_label"] = pd.Series(dict(adjusted))
    return out
