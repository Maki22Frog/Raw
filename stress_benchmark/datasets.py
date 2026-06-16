from __future__ import annotations

import io
import pickle
import re
import zipfile
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .baseline import add_personal_baseline_features
from .config import ExtractionConfig
from .e4 import extract_e4_features, read_e4_zip_bytes
from .features import window_features


WESAD_LABEL_FS = 700.0
WESAD_WRIST_FS = {
    "ACC": 32.0,
    "BVP": 64.0,
    "EDA": 4.0,
    "TEMP": 4.0,
}


def normalize_subject_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def mode_and_purity(values: np.ndarray) -> Tuple[Optional[int], float]:
    arr = np.asarray(values).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, 0.0
    counts = Counter(arr.astype(int).tolist())
    label, count = counts.most_common(1)[0]
    return int(label), float(count / arr.size)


def build_wesad_features(config: ExtractionConfig) -> pd.DataFrame:
    if not config.wesad_zip.exists():
        return pd.DataFrame()
    rows: List[Dict[str, float]] = []
    with zipfile.ZipFile(config.wesad_zip) as zf:
        pkl_names = sorted(
            [name for name in zf.namelist() if re.match(r"WESAD/S\d+/S\d+\.pkl$", name)],
            key=lambda n: int(re.search(r"S(\d+)\.pkl$", n).group(1)),
        )
        if config.max_wesad_subjects:
            pkl_names = pkl_names[: config.max_wesad_subjects]
        for pkl_name in pkl_names:
            subject_id = re.search(r"/(S\d+)/", pkl_name).group(1)
            with zf.open(pkl_name) as fh:
                data = pickle.load(fh, encoding="latin1")
            labels = np.asarray(data["label"]).reshape(-1)
            wrist = data["signal"]["wrist"]
            signals = {
                "acc": (np.asarray(wrist["ACC"], dtype=float), WESAD_WRIST_FS["ACC"]),
                "bvp": (np.asarray(wrist["BVP"], dtype=float), WESAD_WRIST_FS["BVP"]),
                "eda": (np.asarray(wrist["EDA"], dtype=float), WESAD_WRIST_FS["EDA"]),
                "temp": (np.asarray(wrist["TEMP"], dtype=float), WESAD_WRIST_FS["TEMP"]),
            }
            duration = len(labels) / WESAD_LABEL_FS
            start = 0.0
            while start + config.window_sec <= duration + 1e-6:
                end = start + config.window_sec
                label_start = int(round(start * WESAD_LABEL_FS))
                label_end = int(round(end * WESAD_LABEL_FS))
                protocol_label, purity = mode_and_purity(labels[label_start:label_end])
                if protocol_label not in {1, 2, 3} or purity < config.label_purity:
                    start += config.step_sec
                    continue
                row = window_features(signals, start, end)
                row["source"] = "wesad"
                row["subject_id"] = subject_id
                row["session_id"] = subject_id
                row["group_id"] = f"wesad:{subject_id}"
                row["wesad_protocol_label"] = protocol_label
                row["label_purity"] = purity
                # Unified task labels: baseline/amusement are non-stress; TSST is high stress.
                row["target3"] = 2 if protocol_label == 2 else 0
                row["target_binary"] = 1 if protocol_label == 2 else 0
                row["window_start_ts"] = np.nan
                row["window_end_ts"] = np.nan
                rows.append(row)
                start += config.step_sec
    return pd.DataFrame(rows)


def _parse_excel_time(value: object) -> Optional[time]:
    if value is None or value == "":
        return None
    if isinstance(value, time):
        return value
    if isinstance(value, datetime):
        return value.time()
    text = str(value).strip()
    try:
        parts = [int(float(part)) for part in text.split(":")]
        while len(parts) < 3:
            parts.append(0)
        return time(parts[0], parts[1], parts[2])
    except Exception:
        return None


def _parse_excel_date(value: object) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return datetime(value.year, value.month, value.day)
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return datetime(parsed.year, parsed.month, parsed.day)


def load_survey_intervals(path: Path, survey_offset_hours: float) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="in")
    rows = []
    for _, row in raw.iterrows():
        subject_id = normalize_subject_id(row.get("ID"))
        if not subject_id:
            continue
        label = row.get("Stress level")
        if pd.isna(label) or str(label).strip().lower() == "na":
            continue
        try:
            label_int = int(float(label))
        except Exception:
            continue
        if label_int not in {0, 1, 2}:
            continue
        date_value = _parse_excel_date(row.get("date"))
        start_time = _parse_excel_time(row.get("Start time"))
        end_time = _parse_excel_time(row.get("End time"))
        if date_value is None or start_time is None or end_time is None:
            continue
        local_start = datetime.combine(date_value.date(), start_time)
        local_end = datetime.combine(date_value.date(), end_time)
        if local_end < local_start:
            local_end += timedelta(days=1)
        start_utc = (local_start - timedelta(hours=survey_offset_hours)).replace(tzinfo=timezone.utc)
        end_utc = (local_end - timedelta(hours=survey_offset_hours)).replace(tzinfo=timezone.utc)
        rows.append(
            {
                "subject_id": subject_id,
                "survey_start_ts": start_utc.timestamp(),
                "survey_end_ts": end_utc.timestamp(),
                "nurse_label": label_int,
            }
        )
    return pd.DataFrame(rows)


def _session_entries(nurse_zip: Path) -> List[zipfile.ZipInfo]:
    with zipfile.ZipFile(nurse_zip) as zf:
        entries = [
            info
            for info in zf.infolist()
            if (not info.is_dir()) and re.match(r"[^/]+/[^/]+_\d+\.zip$", info.filename)
        ]
    return sorted(entries, key=lambda info: info.filename)


def scan_survey_offsets(data_dir: Path, offsets: Sequence[float] = tuple(range(-12, 15))) -> pd.DataFrame:
    nurse_zip = data_dir / "Stress_dataset.zip"
    survey_path = data_dir / "SurveyResults.xlsx"
    entries = _session_entries(nurse_zip)
    session_rows = []
    for info in entries:
        match = re.match(r"([^/]+)/[^_]+_(\d+)\.zip$", info.filename)
        if not match:
            continue
        session_rows.append((match.group(1), float(match.group(2))))
    rows = []
    for offset in offsets:
        intervals = load_survey_intervals(survey_path, float(offset))
        by_subject = {sid: grp for sid, grp in intervals.groupby("subject_id")}
        matched = 0
        subjects = Counter()
        for subject_id, start_ts in session_rows:
            grp = by_subject.get(normalize_subject_id(subject_id))
            if grp is None:
                continue
            hit = grp[(grp["survey_start_ts"] <= start_ts) & (start_ts <= grp["survey_end_ts"])]
            if not hit.empty:
                matched += 1
                subjects[subject_id] += 1
        rows.append(
            {
                "survey_offset_hours": float(offset),
                "matched_session_starts": int(matched),
                "matched_subjects": int(len(subjects)),
            }
        )
    return pd.DataFrame(rows).sort_values(["matched_session_starts", "matched_subjects"], ascending=False)


def choose_best_survey_offset(data_dir: Path) -> float:
    scan = scan_survey_offsets(data_dir)
    if scan.empty:
        return 0.0
    return float(scan.iloc[0]["survey_offset_hours"])


def _label_nurse_windows(features: pd.DataFrame, intervals: pd.DataFrame, min_overlap: float) -> pd.DataFrame:
    if features.empty or intervals.empty:
        return features
    out = features.copy()
    out["nurse_label"] = np.nan
    out["label_overlap_ratio"] = 0.0
    interval_groups = {sid: grp.sort_values("survey_start_ts") for sid, grp in intervals.groupby("subject_id")}
    labels = []
    overlaps = []
    for _, row in out.iterrows():
        subject_id = normalize_subject_id(row["subject_id"])
        grp = interval_groups.get(subject_id)
        if grp is None:
            labels.append(np.nan)
            overlaps.append(0.0)
            continue
        start_ts = float(row["window_start_ts"])
        end_ts = float(row["window_end_ts"])
        candidates = grp[(grp["survey_end_ts"] > start_ts) & (grp["survey_start_ts"] < end_ts)]
        if candidates.empty:
            labels.append(np.nan)
            overlaps.append(0.0)
            continue
        best_label = np.nan
        best_overlap = 0.0
        for _, interval in candidates.iterrows():
            overlap = max(0.0, min(end_ts, interval["survey_end_ts"]) - max(start_ts, interval["survey_start_ts"]))
            ratio = overlap / max(end_ts - start_ts, 1e-8)
            if ratio > best_overlap:
                best_overlap = ratio
                best_label = int(interval["nurse_label"])
        labels.append(best_label if best_overlap >= min_overlap else np.nan)
        overlaps.append(best_overlap)
    out["nurse_label"] = labels
    out["label_overlap_ratio"] = overlaps
    out["target3"] = out["nurse_label"]
    out["target_binary"] = out["nurse_label"].apply(lambda x: 1 if pd.notna(x) and int(x) >= 1 else (0 if pd.notna(x) else np.nan))
    return out


def build_nurse_features(config: ExtractionConfig) -> pd.DataFrame:
    if not config.nurse_zip.exists() or not config.survey_xlsx.exists():
        return pd.DataFrame()
    if config.survey_offset_hours is None:
        offset = choose_best_survey_offset(config.data_dir)
    else:
        offset = float(config.survey_offset_hours)
    intervals = load_survey_intervals(config.survey_xlsx, offset)
    frames = []
    count = 0
    with zipfile.ZipFile(config.nurse_zip) as zf:
        entries = _session_entries(config.nurse_zip)
        if config.max_nurse_sessions:
            entries = entries[: config.max_nurse_sessions]
        for info in entries:
            match = re.match(r"([^/]+)/([^/]+)\.zip$", info.filename)
            if not match:
                continue
            subject_id = normalize_subject_id(match.group(1))
            session_id = match.group(2)
            record = read_e4_zip_bytes(subject_id, session_id, zf.read(info.filename))
            if record is None:
                continue
            frame = extract_e4_features(record, config.window_sec, config.step_sec)
            if frame.empty:
                continue
            frames.append(frame)
            count += 1
            if config.max_nurse_sessions and count >= config.max_nurse_sessions:
                break
    if not frames:
        return pd.DataFrame()
    features = pd.concat(frames, ignore_index=True)
    features = _label_nurse_windows(features, intervals, config.min_label_overlap)
    features["survey_offset_hours"] = offset
    if not config.keep_unlabeled:
        features = features[features["target3"].notna()].copy()
    return features


def build_combined_features(config: ExtractionConfig) -> pd.DataFrame:
    frames = []
    sources = set(config.sources)
    if "wesad" in sources:
        frames.append(build_wesad_features(config))
    if "nurse" in sources:
        frames.append(build_nurse_features(config))
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = add_personal_baseline_features(combined, baseline_minutes=config.baseline_minutes)
    return combined

