from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .features import window_features


@dataclass
class E4Record:
    subject_id: str
    session_id: str
    start_ts_utc: float
    signals: Dict[str, Tuple[np.ndarray, float]]


def _first_float(line: bytes) -> float:
    text = line.decode("utf-8", errors="replace").strip()
    if not text:
        return float("nan")
    return float(text.split(",")[0].strip())


def _read_regular_signal(inner: zipfile.ZipFile, filename: str, columns: int) -> Optional[Tuple[float, float, np.ndarray]]:
    try:
        with inner.open(filename) as fh:
            first = fh.readline()
            if not first:
                return None
            second = fh.readline()
            start_ts = _first_float(first)
            fs = _first_float(second)
            if not np.isfinite(start_ts) or not np.isfinite(fs):
                return None
            df = pd.read_csv(fh, header=None)
    except KeyError:
        return None
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    arr = df.iloc[:, :columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return start_ts, fs, arr


def _read_ibi(inner: zipfile.ZipFile) -> Optional[Tuple[float, float, np.ndarray]]:
    try:
        with inner.open("IBI.csv") as fh:
            first = fh.readline()
            if not first:
                return None
            start_ts = _first_float(first)
            df = pd.read_csv(fh, header=None)
    except KeyError:
        return None
    except pd.errors.EmptyDataError:
        return None
    if df.empty or df.shape[1] < 2:
        return None
    # IBI is event based. Use an approximate 1 Hz array containing IBI values.
    rel_t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    ibi = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(rel_t) & np.isfinite(ibi)
    if not np.any(mask):
        return None
    rel_t = rel_t[mask]
    ibi = ibi[mask]
    duration = int(np.nanmax(rel_t)) + 1
    values = np.full(max(duration, 1), np.nan, dtype=float)
    idx = np.clip(rel_t.astype(int), 0, len(values) - 1)
    values[idx] = ibi
    values = pd.Series(values).ffill().bfill().to_numpy(dtype=float)
    return start_ts, 1.0, values


def read_e4_zip_bytes(subject_id: str, session_id: str, data: bytes) -> Optional[E4Record]:
    with zipfile.ZipFile(io.BytesIO(data)) as inner:
        parsed_signals: Dict[str, Tuple[float, float, np.ndarray]] = {}
        starts: List[float] = []
        for filename, name, columns in [
            ("ACC.csv", "acc", 3),
            ("EDA.csv", "eda", 1),
            ("TEMP.csv", "temp", 1),
            ("BVP.csv", "bvp", 1),
            ("HR.csv", "hr", 1),
        ]:
            parsed = _read_regular_signal(inner, filename, columns)
            if parsed is None:
                continue
            start_ts, fs, arr = parsed
            parsed_signals[name] = (start_ts, fs, arr)
            starts.append(start_ts)
        parsed_ibi = _read_ibi(inner)
        if parsed_ibi is not None:
            start_ts, fs, arr = parsed_ibi
            parsed_signals["ibi"] = (start_ts, fs, arr)
            starts.append(start_ts)
        if not parsed_signals or not starts:
            return None
        global_start = float(min(starts))
        signals: Dict[str, Tuple[np.ndarray, float]] = {}
        for name, (start_ts, fs, arr) in parsed_signals.items():
            offset_samples = int(round((float(start_ts) - global_start) * float(fs)))
            if offset_samples > 0:
                pad_shape = (offset_samples,) + arr.shape[1:]
                arr = np.concatenate([np.full(pad_shape, np.nan), arr], axis=0)
            signals[name] = (arr, fs)
        return E4Record(subject_id=subject_id, session_id=session_id, start_ts_utc=global_start, signals=signals)


def signal_duration(values: np.ndarray, fs: float) -> float:
    if fs <= 0:
        return 0.0
    return float(len(values) / fs)


def extract_e4_features(record: E4Record, window_sec: int, step_sec: int) -> pd.DataFrame:
    durations = [signal_duration(values, fs) for values, fs in record.signals.values() if fs > 0]
    if not durations:
        return pd.DataFrame()
    duration = max(durations)
    rows = []
    start = 0.0
    while start + window_sec <= duration + 1e-6:
        end = start + window_sec
        row = window_features(record.signals, start, end)
        row["subject_id"] = record.subject_id
        row["session_id"] = record.session_id
        row["source"] = "nurse"
        row["group_id"] = f"nurse:{record.subject_id}"
        row["window_start_ts"] = record.start_ts_utc + start
        row["window_end_ts"] = record.start_ts_utc + end
        row["window_start_utc"] = datetime.fromtimestamp(row["window_start_ts"], tz=timezone.utc).isoformat()
        row["window_end_utc"] = datetime.fromtimestamp(row["window_end_ts"], tz=timezone.utc).isoformat()
        rows.append(row)
        start += step_sec
    return pd.DataFrame(rows)
