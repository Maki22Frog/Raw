from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class ExtractionConfig:
    data_dir: Path
    out_dir: Path
    sources: Sequence[str] = ("wesad", "nurse")
    window_sec: int = 60
    step_sec: int = 60
    baseline_minutes: int = 10
    label_purity: float = 0.80
    min_label_overlap: float = 0.50
    survey_offset_hours: Optional[float] = None
    max_wesad_subjects: Optional[int] = None
    max_nurse_sessions: Optional[int] = None
    keep_unlabeled: bool = False

    @property
    def wesad_zip(self) -> Path:
        return self.data_dir / "WESAD.zip"

    @property
    def nurse_zip(self) -> Path:
        return self.data_dir / "Stress_dataset.zip"

    @property
    def survey_xlsx(self) -> Path:
        return self.data_dir / "SurveyResults.xlsx"

    @property
    def feature_path(self) -> Path:
        return self.out_dir / "features_combined.csv.gz"


@dataclass(frozen=True)
class BenchmarkConfig:
    task: str = "stress3"
    protocol: str = "groupkfold"
    n_splits: int = 5
    use_smote: bool = True
    calibrate: bool = True
    calibration_method: str = "sigmoid"
    class_cap_per_group: Optional[int] = 150
    class_cap_grid: Sequence[Optional[int]] = ()
    feature_selection_k: Optional[int] = None
    feature_selection_k_grid: Sequence[Optional[int]] = ()
    tune_threshold: bool = True
    threshold_metric: str = "balanced_accuracy"
    threshold_metric_grid: Sequence[str] = ()
    positive_recall_floor: float = 0.70
    tuning_selection_metric: str = "balanced_accuracy"
    source_balance: str = "none"
    random_state: int = 42
    include_time_features: bool = False
    tune: bool = False
    models: Sequence[str] = ("brf", "rf", "extratrees", "knn", "gnb", "gb", "xgb", "lgbm")
