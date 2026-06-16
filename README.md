# Stress ML Benchmark Pipeline

This project builds a reproducible ML benchmark for two wearable stress datasets:

- `WESAD.zip`: WESAD laboratory stress/affect dataset.
- `Stress_dataset.zip` + `SurveyResults.xlsx`: nurse wearable stress dataset with survey labels.

The code treats them as one benchmark family while preserving dataset provenance,
subject-level splits, and source-specific label rules.

## What Is Implemented

- Streaming readers for zipped Empatica E4 sessions.
- WESAD subject-by-subject feature extraction from synchronized `.pkl` files.
- Nurse dataset label joining from `SurveyResults.xlsx`.
- 60-second physiological feature windows inspired by WESAD.
- One-minute aggregation and class-imbalance handling inspired by the nurse stress paper.
- Personal baseline normalization per subject.
- ACC-based physical-activity context features.
- Leakage-safe subject-level benchmark splits.
- Classical ML benchmark: Random Forest, Balanced Random Forest, k-NN, Gaussian Naive Bayes, Extra Trees, Gradient Boosting, and optional XGBoost/LightGBM.
- Binary imbalance handling: per-subject per-class cap, SMOTE/internal balancing, sigmoid calibration, and threshold tuning.
- Optional ablations: top-k feature selection, class-cap grids, threshold-metric grids, and source-balanced training.
- Decision-support post-processing with alert levels and simple recommendations.

## Setup

Install dependencies in a Python environment:

```powershell
pip install -r requirements.txt
```

The full benchmark expects `scikit-learn`, `imbalanced-learn`, `xgboost`, and
`lightgbm` in addition to the data-processing packages listed in `requirements.txt`.
If a requested model is unavailable, the pipeline stops instead of silently
running a reduced benchmark.
For deployment, keep the Python package versions consistent with the training
environment. The current saved bundles were trained with `scikit-learn==1.9.0`;
loading them with a different scikit-learn version can produce compatibility
warnings or different behavior.

## Quick Run

Scan likely timezone offsets for the nurse survey labels:

```powershell
python -m stress_benchmark.cli scan-offsets --data-dir D:\IntroAI
```

Extract combined features:

```powershell
python -m stress_benchmark.cli extract --data-dir D:\IntroAI --out-dir D:\IntroAI\outputs --survey-offset auto
```

Run the benchmark:

```powershell
python -m stress_benchmark.cli benchmark --data-dir D:\IntroAI --out-dir D:\IntroAI\outputs --survey-offset auto --task binary --protocol groupkfold --models brf,rf,extratrees,knn,gnb,xgb,lgbm
```

For binary tasks, class cap `150` and threshold tuning by balanced accuracy are enabled
by default. Use `--no-class-cap` or `--no-threshold-tuning` only for ablation.

Run the optional tuning/ablation grid:

```powershell
python -m stress_benchmark.cli benchmark --data-dir D:\IntroAI --out-dir D:\IntroAI\outputs --survey-offset auto --task binary --protocol groupkfold --models brf,rf,extratrees,knn,gnb,xgb,lgbm --class-cap-grid 100,150,250 --feature-k-grid 200,300,all --threshold-metric-grid balanced_accuracy,macro_f1,balanced_accuracy_recall_floor --source-balance source_class
```

For the stricter scientific setting, use leave-one-subject-out:

```powershell
python -m stress_benchmark.cli benchmark --data-dir D:\IntroAI --out-dir D:\IntroAI\outputs --survey-offset auto --task stress3 --protocol loso
```

Train deployable backend bundles after feature extraction/benchmarking:

```powershell
python -m stress_benchmark.cli train-final --data-dir D:\IntroAI --out-dir D:\IntroAI\outputs --task binary --models brf,rf,extratrees,knn,gnb,xgb,lgbm
```

Backend loading example:

```python
import pandas as pd
from stress_benchmark.deploy import load_bundle, predict_feature_frame

bundle = load_bundle(r"D:\IntroAI\outputs\models\binary_best_model.joblib")
features = pd.read_csv(r"D:\IntroAI\outputs\features_combined.csv.gz").head(10)
predictions = predict_feature_frame(bundle, features)
```

## Outputs

The benchmark writes:

- `outputs/features_combined.csv.gz`: extracted feature table.
- `outputs/metrics_<task>.csv`: fold/model metrics.
- `outputs/predictions_<task>.csv.gz`: row-level predictions and alert states.
- `outputs/summary_<task>.json`: aggregate metrics and settings.
- `outputs/models/<task>_<model>.joblib`: deployable model bundles.
- `outputs/models/<task>_best_model.joblib`: copy of the best bundle selected from benchmark weighted F1 when available.
- `outputs/models/manifest_<task>.json`: deployment manifest.

Current binary deploy bundles:

- Combined WESAD+nurse binary: `D:\IntroAI\outputs\models\binary_best_model.joblib`
  uses ExtraTrees with threshold `0.70`.
- Nurse-only binary: `D:\IntroAI\outputs\nurse_binary\models\binary_best_model.joblib`
  uses ExtraTrees with threshold `0.83`.

## Methodology

See `docs/methodology.md` for the paper-derived design choices and the added
personalized decision-support layer.
