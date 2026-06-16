from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd


EPS = 1e-8


def as_1d(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def as_2d(values: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr[np.all(np.isfinite(arr), axis=1)]


def basic_stats(prefix: str, values: Iterable[float]) -> Dict[str, float]:
    arr = as_1d(values)
    if arr.size == 0:
        return {}
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_range": float(np.max(arr) - np.min(arr)),
        f"{prefix}_energy": float(np.mean(arr * arr)),
    }


def slope(values: Iterable[float], fs: float) -> float:
    arr = as_1d(values)
    if arr.size < 2 or fs <= 0:
        return np.nan
    duration = max((arr.size - 1) / fs, EPS)
    return float((arr[-1] - arr[0]) / duration)


def peak_count(values: Iterable[float], min_prominence_std: float = 0.5) -> int:
    arr = as_1d(values)
    if arr.size < 3:
        return 0
    threshold = float(np.mean(arr) + min_prominence_std * np.std(arr))
    middle = arr[1:-1]
    peaks = (middle > arr[:-2]) & (middle >= arr[2:]) & (middle > threshold)
    return int(np.sum(peaks))


def _bvp_peak_indices(values: Iterable[float], fs: float) -> np.ndarray:
    arr = as_1d(values)
    if arr.size < int(fs * 10) or fs <= 0:
        return np.array([], dtype=int)
    threshold = float(np.mean(arr) + 0.35 * np.std(arr))
    middle = arr[1:-1]
    peak_mask = (middle > arr[:-2]) & (middle >= arr[2:]) & (middle > threshold)
    candidates = np.flatnonzero(peak_mask) + 1
    if candidates.size == 0:
        return candidates

    # Keep the strongest candidate in a short refractory interval.
    refractory = max(1, int(round(0.30 * fs)))
    kept = []
    last = -10**9
    for idx in candidates:
        if idx - last >= refractory:
            kept.append(idx)
            last = idx
        elif kept and arr[idx] > arr[kept[-1]]:
            kept[-1] = idx
            last = idx
    return np.asarray(kept, dtype=int)


def hrv_features_from_intervals(prefix: str, intervals_sec: Iterable[float]) -> Dict[str, float]:
    rr = as_1d(intervals_sec)
    rr = rr[(rr >= 0.30) & (rr <= 2.00)]
    if rr.size < 2:
        return {}
    rr_ms = rr * 1000.0
    diff_ms = np.diff(rr_ms)
    out: Dict[str, float] = {}
    out.update(basic_stats(f"{prefix}_rr_ms", rr_ms))
    out[f"{prefix}_sdnn_ms"] = float(np.std(rr_ms, ddof=1)) if rr_ms.size > 1 else np.nan
    out[f"{prefix}_rmssd_ms"] = float(np.sqrt(np.mean(diff_ms * diff_ms))) if diff_ms.size else np.nan
    out[f"{prefix}_pnn50"] = float(np.mean(np.abs(diff_ms) > 50.0)) if diff_ms.size else np.nan
    out[f"{prefix}_cvnn"] = float(np.std(rr_ms) / (np.mean(rr_ms) + EPS))
    out[f"{prefix}_hr_mean"] = float(np.mean(60000.0 / rr_ms))
    out[f"{prefix}_hr_std"] = float(np.std(60000.0 / rr_ms))

    if rr.size >= 8:
        t = np.cumsum(np.r_[0.0, rr[:-1]])
        duration = float(t[-1] - t[0]) if t.size > 1 else 0.0
        if duration >= 20.0:
            interp_fs = 4.0
            uniform_t = np.arange(t[0], t[-1], 1.0 / interp_fs)
            if uniform_t.size >= 16:
                interp_rr = np.interp(uniform_t, t, rr_ms)
                interp_rr = interp_rr - np.mean(interp_rr)
                freqs = np.fft.rfftfreq(interp_rr.size, d=1.0 / interp_fs)
                power = (np.abs(np.fft.rfft(interp_rr)) ** 2) / max(interp_rr.size, 1)

                def band_power(low: float, high: float) -> float:
                    mask = (freqs >= low) & (freqs < high)
                    return float(np.trapz(power[mask], freqs[mask])) if np.any(mask) else 0.0

                lf = band_power(0.04, 0.15)
                hf = band_power(0.15, 0.40)
                out[f"{prefix}_lf_power"] = lf
                out[f"{prefix}_hf_power"] = hf
                out[f"{prefix}_lf_hf_ratio"] = float(lf / (hf + EPS))
                out[f"{prefix}_hf_norm"] = float(hf / (lf + hf + EPS))
    return out


def derived_hr_from_peaks(values: Iterable[float], fs: float) -> Dict[str, float]:
    arr = as_1d(values)
    peak_idx = _bvp_peak_indices(arr, fs)
    if peak_idx.size < 3:
        return {}
    intervals = np.diff(peak_idx) / fs
    intervals = intervals[(intervals >= 0.30) & (intervals <= 2.00)]
    if intervals.size < 2:
        return {}
    bpm = 60.0 / intervals
    out = basic_stats("bvp_derived_hr", bpm)
    out["bvp_peak_count"] = int(peak_idx.size)
    out["bvp_peak_rate"] = float(peak_idx.size / max(arr.size / fs, EPS))
    out.update(hrv_features_from_intervals("bvp_hrv", intervals))
    return out


def ibi_hrv_features(values: Iterable[float]) -> Dict[str, float]:
    intervals = as_1d(values)
    return hrv_features_from_intervals("ibi_hrv", intervals)


def eda_decomposition_features(values: Iterable[float], fs: float) -> Dict[str, float]:
    arr = as_1d(values)
    if arr.size < 4 or fs <= 0:
        return {}
    tonic_window = max(3, int(round(min(20.0, max(4.0, arr.size / fs / 3.0)) * fs)))
    tonic = pd.Series(arr).rolling(window=tonic_window, min_periods=1, center=True).median().to_numpy(dtype=float)
    phasic = arr - tonic
    positive = np.clip(phasic, 0.0, None)
    peaks = _simple_peak_values(positive, min_prominence_std=0.25)
    out: Dict[str, float] = {}
    out.update(basic_stats("eda_tonic", tonic))
    out.update(basic_stats("eda_phasic", phasic))
    out["eda_tonic_slope"] = slope(tonic, fs)
    out["eda_phasic_auc"] = float(np.trapz(positive, dx=1.0 / fs))
    out["eda_phasic_positive_ratio"] = float(np.mean(phasic > 0.0))
    out["eda_scr_peak_count"] = int(peaks.size)
    out["eda_scr_peak_rate"] = float(peaks.size / max(arr.size / fs, EPS))
    out["eda_scr_amp_mean"] = float(np.mean(peaks)) if peaks.size else 0.0
    out["eda_scr_amp_max"] = float(np.max(peaks)) if peaks.size else 0.0
    return out


def _simple_peak_values(values: Iterable[float], min_prominence_std: float = 0.5) -> np.ndarray:
    arr = as_1d(values)
    if arr.size < 3:
        return np.array([])
    threshold = float(np.mean(arr) + min_prominence_std * np.std(arr))
    middle = arr[1:-1]
    mask = (middle > arr[:-2]) & (middle >= arr[2:]) & (middle > threshold)
    return middle[mask]


def normalize_acc_units(acc: np.ndarray) -> np.ndarray:
    arr = as_2d(acc)
    if arr.size == 0:
        return arr
    # Empatica E4 ACC files use 1/64 g. Some processed arrays are already in g.
    if np.nanmax(np.abs(arr)) > 8.0:
        arr = arr / 64.0
    return arr


def acc_features(acc: Iterable[Iterable[float]], fs: float) -> Dict[str, float]:
    arr = normalize_acc_units(np.asarray(acc, dtype=float))
    if arr.size == 0:
        return {}
    if arr.shape[1] >= 3:
        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
    else:
        x = arr[:, 0]
        y = np.zeros_like(x)
        z = np.zeros_like(x)
    mag = np.sqrt(x * x + y * y + z * z)
    centered = arr[:, :3] - np.median(arr[:, :3], axis=0)
    dyn_mag = np.sqrt(np.sum(centered * centered, axis=1))
    jerk = np.diff(mag) * fs if mag.size > 1 else np.array([])
    dyn_abs = np.abs(dyn_mag)
    stationary_threshold = max(0.03, float(np.percentile(dyn_abs, 25) + 0.5 * np.std(dyn_abs))) if dyn_abs.size else 0.03
    out: Dict[str, float] = {}
    out.update(basic_stats("acc_x", x))
    out.update(basic_stats("acc_y", y))
    out.update(basic_stats("acc_z", z))
    out.update(basic_stats("acc_mag", mag))
    out.update(basic_stats("acc_dyn_mag", dyn_mag))
    out.update(basic_stats("acc_jerk", jerk))
    out["acc_activity"] = float(np.std(dyn_mag) + np.mean(np.abs(jerk)) if jerk.size else np.std(dyn_mag))
    out["acc_posture_tilt"] = float(np.mean(z) / (np.mean(mag) + EPS))
    out["acc_stationary_ratio"] = float(np.mean(dyn_abs < stationary_threshold)) if dyn_abs.size else np.nan
    out["acc_active_ratio"] = float(np.mean(dyn_abs > 0.15)) if dyn_abs.size else np.nan
    out["acc_signal_magnitude_area"] = float(np.mean(np.abs(x) + np.abs(y) + np.abs(z)))
    out["acc_vector_magnitude_area"] = float(np.mean(np.abs(mag - np.median(mag))))
    if jerk.size:
        out["acc_jerk_zero_cross_rate"] = float(np.mean(np.diff(np.signbit(jerk)) != 0)) if jerk.size > 1 else 0.0
    return out


def signal_window(signals: Mapping[str, Tuple[np.ndarray, float]], name: str, start_sec: float, end_sec: float) -> Tuple[np.ndarray, float]:
    if name not in signals:
        return np.array([]), np.nan
    values, fs = signals[name]
    if fs <= 0:
        return np.array([]), fs
    start_idx = max(0, int(round(start_sec * fs)))
    end_idx = min(len(values), int(round(end_sec * fs)))
    if end_idx <= start_idx:
        return np.array([]), fs
    return np.asarray(values[start_idx:end_idx]), fs


def window_features(signals: Mapping[str, Tuple[np.ndarray, float]], start_sec: float, end_sec: float) -> Dict[str, float]:
    out: Dict[str, float] = {
        "window_start_sec": float(start_sec),
        "window_end_sec": float(end_sec),
        "window_duration_sec": float(end_sec - start_sec),
    }

    acc, acc_fs = signal_window(signals, "acc", start_sec, end_sec)
    if acc.size:
        out.update(acc_features(acc, float(acc_fs)))

    for name, prefix in [("eda", "eda"), ("temp", "temp"), ("bvp", "bvp"), ("hr", "hr"), ("ibi", "ibi")]:
        values, fs = signal_window(signals, name, start_sec, end_sec)
        if values.size == 0:
            continue
        flat = as_1d(values)
        out.update(basic_stats(prefix, flat))
        out[f"{prefix}_slope"] = slope(flat, float(fs)) if np.isfinite(fs) else np.nan
        if prefix == "eda":
            diffs = np.diff(flat)
            out["eda_rise_rate"] = float(np.mean(diffs > 0)) if diffs.size else np.nan
            out["eda_abs_diff_mean"] = float(np.mean(np.abs(diffs))) if diffs.size else np.nan
            peaks = peak_count(flat, min_prominence_std=0.25)
            out["eda_peak_count"] = peaks
            out["eda_peak_rate"] = float(peaks / max((flat.size / fs), EPS)) if fs and fs > 0 else np.nan
            out.update(eda_decomposition_features(flat, float(fs)))
        if prefix == "bvp":
            out.update(derived_hr_from_peaks(flat, float(fs)))
        if prefix == "ibi":
            out.update(ibi_hrv_features(flat))
    return out
