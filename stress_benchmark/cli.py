from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import BenchmarkConfig, ExtractionConfig
from .datasets import build_combined_features, choose_best_survey_offset, scan_survey_offsets
from .models import evaluate_models, train_deploy_models


def _parse_sources(value: str) -> Sequence[str]:
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


def _parse_optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "all", "none", "null"}:
        return None
    return int(text)


def _parse_optional_int_grid(value: str | None) -> tuple[int | None, ...]:
    if not value:
        return ()
    return tuple(_parse_optional_int(part) for part in value.split(",") if part.strip())


def _parse_str_grid(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    class_cap_grid = _parse_optional_int_grid(args.class_cap_grid)
    feature_k_grid = _parse_optional_int_grid(args.feature_k_grid)
    threshold_metric_grid = _parse_str_grid(args.threshold_metric_grid)
    if args.tune:
        if not class_cap_grid:
            class_cap_grid = (100, 150, 250, None)
        if not feature_k_grid:
            feature_k_grid = (100, 200, 300, None)
        if not threshold_metric_grid:
            threshold_metric_grid = ("balanced_accuracy", "macro_f1", "balanced_accuracy_recall_floor")

    return BenchmarkConfig(
        task=args.task,
        protocol=getattr(args, "protocol", "groupkfold"),
        n_splits=args.n_splits,
        use_smote=not args.no_smote,
        calibrate=not args.no_calibrate,
        calibration_method=args.calibration_method,
        class_cap_per_group=None if args.no_class_cap else args.class_cap_per_group,
        class_cap_grid=() if args.no_class_cap else class_cap_grid,
        feature_selection_k=_parse_optional_int(args.feature_selection_k),
        feature_selection_k_grid=feature_k_grid,
        tune_threshold=not args.no_threshold_tuning,
        threshold_metric=args.threshold_metric,
        threshold_metric_grid=threshold_metric_grid,
        positive_recall_floor=args.positive_recall_floor,
        tuning_selection_metric=args.tuning_selection_metric,
        source_balance=args.source_balance,
        random_state=args.random_state,
        include_time_features=args.include_time_features,
        tune=args.tune,
        models=tuple(args.models.split(",")),
    )


def _extraction_config(args: argparse.Namespace) -> ExtractionConfig:
    survey_offset = None
    if getattr(args, "survey_offset", None) not in (None, "auto"):
        survey_offset = float(args.survey_offset)
    return ExtractionConfig(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        sources=_parse_sources(args.sources),
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        baseline_minutes=args.baseline_minutes,
        label_purity=args.label_purity,
        min_label_overlap=args.min_label_overlap,
        survey_offset_hours=survey_offset,
        max_wesad_subjects=args.max_wesad_subjects,
        max_nurse_sessions=args.max_nurse_sessions,
        keep_unlabeled=args.keep_unlabeled,
    )


def cmd_scan_offsets(args: argparse.Namespace) -> None:
    scan = scan_survey_offsets(Path(args.data_dir))
    print(scan.to_string(index=False))


def cmd_extract(args: argparse.Namespace) -> None:
    config = _extraction_config(args)
    if args.survey_offset == "auto":
        offset = choose_best_survey_offset(config.data_dir)
        config = ExtractionConfig(**{**config.__dict__, "survey_offset_hours": offset})
        print(f"Using auto survey offset: {offset}")
    config.out_dir.mkdir(parents=True, exist_ok=True)
    features = build_combined_features(config)
    features.to_csv(config.feature_path, index=False, compression="gzip")
    print(f"Wrote {len(features)} feature rows to {config.feature_path}")
    if "source" in features:
        print(features.groupby("source").size().to_string())
    if "target3" in features:
        print("target3 distribution:")
        print(features["target3"].value_counts(dropna=False).sort_index().to_string())


def cmd_benchmark(args: argparse.Namespace) -> None:
    config = _extraction_config(args)
    if args.survey_offset == "auto":
        offset = choose_best_survey_offset(config.data_dir)
        config = ExtractionConfig(**{**config.__dict__, "survey_offset_hours": offset})
        print(f"Using auto survey offset: {offset}")
    config.out_dir.mkdir(parents=True, exist_ok=True)
    if args.rebuild_features or not config.feature_path.exists():
        features = build_combined_features(config)
        features.to_csv(config.feature_path, index=False, compression="gzip")
        print(f"Wrote {len(features)} feature rows to {config.feature_path}")
    else:
        features = pd.read_csv(config.feature_path)
        print(f"Loaded {len(features)} feature rows from {config.feature_path}")
    effective_out_dir = config.out_dir
    if args.filter_source != "all":
        features = features[features["source"] == args.filter_source].copy()
        effective_out_dir = config.out_dir / f"{args.filter_source}_{args.task}"
        effective_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Filtered source={args.filter_source}: {len(features)} rows")

    bench_config = _benchmark_config(args)
    summary = evaluate_models(features, bench_config, effective_out_dir)
    print(json.dumps(summary["metrics_by_model"], indent=2))


def cmd_train_final(args: argparse.Namespace) -> None:
    config = _extraction_config(args)
    if args.survey_offset == "auto":
        offset = choose_best_survey_offset(config.data_dir)
        config = ExtractionConfig(**{**config.__dict__, "survey_offset_hours": offset})
        print(f"Using auto survey offset: {offset}")
    config.out_dir.mkdir(parents=True, exist_ok=True)
    if args.rebuild_features or not config.feature_path.exists():
        features = build_combined_features(config)
        features.to_csv(config.feature_path, index=False, compression="gzip")
        print(f"Wrote {len(features)} feature rows to {config.feature_path}")
    else:
        features = pd.read_csv(config.feature_path)
        print(f"Loaded {len(features)} feature rows from {config.feature_path}")
    effective_out_dir = config.out_dir
    if args.filter_source != "all":
        features = features[features["source"] == args.filter_source].copy()
        effective_out_dir = config.out_dir / f"{args.filter_source}_{args.task}"
        effective_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Filtered source={args.filter_source}: {len(features)} rows")

    bench_config = _benchmark_config(args)
    manifest = train_deploy_models(features, bench_config, effective_out_dir)
    print(json.dumps(manifest["best_model"], indent=2))


def add_common_extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=".", help="Directory containing WESAD.zip, Stress_dataset.zip, SurveyResults.xlsx")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument("--sources", default="wesad,nurse", help="Comma-separated sources: wesad,nurse")
    parser.add_argument("--window-sec", type=int, default=60)
    parser.add_argument("--step-sec", type=int, default=60)
    parser.add_argument("--baseline-minutes", type=int, default=10)
    parser.add_argument("--label-purity", type=float, default=0.80)
    parser.add_argument("--min-label-overlap", type=float, default=0.50)
    parser.add_argument("--survey-offset", default="auto", help="Survey timezone offset from UTC, or auto")
    parser.add_argument("--max-wesad-subjects", type=int, default=None)
    parser.add_argument("--max-nurse-sessions", type=int, default=None)
    parser.add_argument("--keep-unlabeled", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wearable stress ML benchmark pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan-offsets", help="Scan nurse survey timezone offsets")
    scan.add_argument("--data-dir", default=".")
    scan.set_defaults(func=cmd_scan_offsets)

    extract = sub.add_parser("extract", help="Extract combined feature table")
    add_common_extract_args(extract)
    extract.set_defaults(func=cmd_extract)

    bench = sub.add_parser("benchmark", help="Extract features if needed and run ML benchmark")
    add_common_extract_args(bench)
    bench.add_argument("--task", choices=["stress3", "binary"], default="stress3")
    bench.add_argument("--protocol", choices=["loso", "groupkfold", "stratified"], default="groupkfold")
    bench.add_argument("--n-splits", type=int, default=5)
    bench.add_argument("--models", default="brf,rf,extratrees,knn,gnb,gb,xgb,lgbm")
    bench.add_argument("--no-smote", action="store_true")
    bench.add_argument("--no-calibrate", action="store_true")
    bench.add_argument("--calibration-method", choices=["sigmoid", "isotonic"], default="sigmoid")
    bench.add_argument("--class-cap-per-group", type=int, default=150)
    bench.add_argument("--class-cap-grid", default="", help="Comma grid, e.g. 100,150,250,none")
    bench.add_argument("--no-class-cap", action="store_true")
    bench.add_argument("--feature-selection-k", default="all", help="Top-k features, or all")
    bench.add_argument("--feature-k-grid", default="", help="Comma grid, e.g. 100,200,300,all")
    bench.add_argument("--no-threshold-tuning", action="store_true")
    bench.add_argument(
        "--threshold-metric",
        choices=["balanced_accuracy", "macro_f1", "balanced_accuracy_recall_floor"],
        default="balanced_accuracy",
    )
    bench.add_argument("--threshold-metric-grid", default="", help="Comma grid of threshold metrics")
    bench.add_argument("--positive-recall-floor", type=float, default=0.70)
    bench.add_argument("--tuning-selection-metric", choices=["balanced_accuracy", "macro_f1"], default="balanced_accuracy")
    bench.add_argument("--source-balance", choices=["none", "source", "source_class"], default="none")
    bench.add_argument("--include-time-features", action="store_true")
    bench.add_argument("--tune", action="store_true")
    bench.add_argument("--random-state", type=int, default=42)
    bench.add_argument("--rebuild-features", action="store_true")
    bench.add_argument("--filter-source", choices=["all", "wesad", "nurse"], default="all")
    bench.set_defaults(func=cmd_benchmark)

    train_final = sub.add_parser("train-final", help="Train final deployable model bundles on all labeled features")
    add_common_extract_args(train_final)
    train_final.add_argument("--task", choices=["stress3", "binary"], default="stress3")
    train_final.add_argument("--n-splits", type=int, default=5)
    train_final.add_argument("--models", default="brf,rf,extratrees,knn,gnb,xgb,lgbm")
    train_final.add_argument("--no-smote", action="store_true")
    train_final.add_argument("--no-calibrate", action="store_true")
    train_final.add_argument("--calibration-method", choices=["sigmoid", "isotonic"], default="sigmoid")
    train_final.add_argument("--class-cap-per-group", type=int, default=150)
    train_final.add_argument("--class-cap-grid", default="", help="Comma grid, e.g. 100,150,250,none")
    train_final.add_argument("--no-class-cap", action="store_true")
    train_final.add_argument("--feature-selection-k", default="all", help="Top-k features, or all")
    train_final.add_argument("--feature-k-grid", default="", help="Comma grid, e.g. 100,200,300,all")
    train_final.add_argument("--no-threshold-tuning", action="store_true")
    train_final.add_argument(
        "--threshold-metric",
        choices=["balanced_accuracy", "macro_f1", "balanced_accuracy_recall_floor"],
        default="balanced_accuracy",
    )
    train_final.add_argument("--threshold-metric-grid", default="", help="Comma grid of threshold metrics")
    train_final.add_argument("--positive-recall-floor", type=float, default=0.70)
    train_final.add_argument("--tuning-selection-metric", choices=["balanced_accuracy", "macro_f1"], default="balanced_accuracy")
    train_final.add_argument("--source-balance", choices=["none", "source", "source_class"], default="none")
    train_final.add_argument("--include-time-features", action="store_true")
    train_final.add_argument("--tune", action="store_true")
    train_final.add_argument("--random-state", type=int, default=42)
    train_final.add_argument("--rebuild-features", action="store_true")
    train_final.add_argument("--filter-source", choices=["all", "wesad", "nurse"], default="all")
    train_final.set_defaults(func=cmd_train_final)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
