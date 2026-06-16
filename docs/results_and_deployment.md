# Results And Deployment Notes

## Extracted Dataset

Feature extraction was run on both datasets without row subsampling:

- WESAD: 526 labeled windows.
- Nurse wearable dataset: 4,323 labeled windows.
- Total: 4,849 windows.
- Feature table columns: 717.
- Model input features after excluding metadata/labels/timing columns: 693.

Unified `stress3` label distribution:

- `0`: 1,320 windows.
- `1`: 244 windows.
- `2`: 3,285 windows.

Nurse survey offset auto-selection selected `-4` hours as the best match between
sensor session timestamps and survey intervals.

## Benchmark Protocol

The benchmark uses subject-grouped folds to reduce subject leakage. For binary tasks,
imbalance handling is applied inside each training fold only:

- per-subject per-class cap: 150 windows,
- SMOTE for regular models,
- Balanced Random Forest with internal balancing instead of external SMOTE,
- sigmoid holdout calibration,
- threshold tuning on the calibration split using balanced accuracy.

The test fold is not capped, resampled, calibrated on, or used for threshold selection.

Best current combined 3-class model by weighted F1 after preprocessing and feature updates:

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| ExtraTrees | 0.7377 | 0.4997 | 0.4773 | 0.7051 |
| Random Forest | 0.7163 | 0.4673 | 0.4619 | 0.6864 |
| XGBoost | 0.6886 | 0.4595 | 0.4254 | 0.6670 |
| LightGBM | 0.6790 | 0.4404 | 0.4383 | 0.6534 |

Current combined binary benchmark after imbalance handling:

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| ExtraTrees | 0.7032 | 0.6945 | 0.6549 | 0.7026 |
| LightGBM | 0.6680 | 0.5903 | 0.5981 | 0.6712 |
| XGBoost | 0.6280 | 0.5557 | 0.5376 | 0.6203 |
| k-NN | 0.5959 | 0.6074 | 0.5446 | 0.5789 |
| Balanced Random Forest | 0.5728 | 0.5946 | 0.5252 | 0.5704 |
| Random Forest | 0.5855 | 0.5864 | 0.5163 | 0.5665 |
| Gaussian Naive Bayes | 0.5586 | 0.5039 | 0.4397 | 0.5213 |

Current nurse-only binary benchmark after imbalance handling:

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| ExtraTrees | 0.7017 | 0.6327 | 0.5778 | 0.7004 |
| Random Forest | 0.6227 | 0.5329 | 0.4848 | 0.6235 |
| LightGBM | 0.5693 | 0.5143 | 0.4691 | 0.5832 |
| XGBoost | 0.5890 | 0.5233 | 0.4665 | 0.5822 |
| Balanced Random Forest | 0.5698 | 0.5577 | 0.4615 | 0.5574 |
| k-NN | 0.5039 | 0.5825 | 0.4564 | 0.4773 |
| Gaussian Naive Bayes | 0.4043 | 0.4942 | 0.3161 | 0.3464 |

Source-specific reference binary benchmark:

| Source | Best Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|
| WESAD | ExtraTrees | 0.9388 | 0.9115 | 0.9175 | 0.9344 |
| Nurse, before binary imbalance update | ExtraTrees | 0.7487 | 0.5589 | 0.5140 | 0.6958 |
| Nurse, after binary imbalance update | ExtraTrees | 0.7017 | 0.6327 | 0.5778 | 0.7004 |

The nurse-only update improves balanced accuracy and macro F1 substantially. Combined
binary balanced accuracy changes slightly upward, while weighted F1 drops because the
tuned threshold is less biased toward the majority stress class.

Additional ablations were implemented and tested:

| Ablation | Scope | Best Observed Effect |
|---|---|---|
| ExtraTrees top-k feature selection | Nurse-only ExtraTrees | Accuracy rose to 0.7121, but balanced accuracy dropped to 0.5880 and macro F1 to 0.5420. Not used for deploy. |
| Class-cap grid `50,100,150,250,none` | Nurse-only ExtraTrees | Balanced accuracy dropped to 0.6106 versus fixed cap 150 at 0.6327. Not used for deploy. |
| Threshold-metric grid | Nurse-only ExtraTrees | Same result as balanced-accuracy threshold tuning. Safe, but no gain. |
| Source-balanced source/class training | Combined ExtraTrees | Balanced accuracy dropped to 0.6678 versus 0.6945 without source balancing. Kept as ablation option, not default deploy. |

The deployable binary bundles therefore keep the empirically best conservative setting:
all 693 features, per-subject class cap 150, SMOTE/internal balancing, sigmoid holdout
calibration, and balanced-accuracy threshold tuning.

## Preprocessing Audit Fixes

The following issues were found and fixed:

- Removed `window_start_sec`, `window_end_sec`, `window_duration_sec`, and derived timing columns from model input. These can leak WESAD protocol order.
- Removed survey-offset derived columns from model input.
- Corrected Empatica E4 HR alignment in the nurse dataset. `HR.csv` starts 10 seconds after ACC/EDA/BVP/TEMP in all 609 sessions.
- Removed label-dependent nurse baseline selection. Nurse baseline now uses the first available calibration windows, not low-stress labels.
- Added unit audit: ACC is normalized from Empatica `1/64 g`; EDA/TEMP/BVP are compatible across the common E4 modalities; HR/IBI are nurse-specific and WESAD-compatible HRV is derived from BVP.
- Added calibrated probabilities using sigmoid holdout calibration. SMOTE is applied to model-training data only, not calibration data.
- Added binary per-subject per-class cap, Balanced Random Forest, and threshold tuning. The prediction and deploy code now use the tuned threshold instead of raw probability argmax.
- Added optional train-fold feature-selection grids, class-cap grids, threshold-metric grids, and source-balanced training ablations.
- Fixed binary decision-support handling so binary `pred_label = 1` is treated as stress, not as only a moderate 3-class state.

Checks that passed:

- WESAD `.pkl` wrist signal durations match label duration for sampled subjects.
- Nurse ACC/EDA/BVP/TEMP starts align at session start; HR offset is consistently 10 seconds and is now padded/aligned.
- Nurse survey label overlap is high after matching: mean overlap around 0.993 on labeled windows.
- Model feature list contains no `target`, `label`, `source`, `subject`, `session`, `group`, `window`, or `offset` columns.
- Prediction CSV threshold checks pass: binary `pred_label` matches stored fold threshold for every model/fold.
- Deploy bundles load and apply stored selected-feature indices, threshold, and decision-support logic.

## Saved Model Bundles

The deployable bundles are saved in `D:\IntroAI\outputs\models`:

- `stress3_best_model.joblib`: current best combined 3-class model, ExtraTrees.
- `binary_best_model.joblib`: current best combined binary model, ExtraTrees, deploy threshold `0.70`.
- `D:\IntroAI\outputs\nurse_binary\models\binary_best_model.joblib`: nurse-only binary model, ExtraTrees, deploy threshold `0.83`.
- `D:\IntroAI\outputs\wesad_binary\models\binary_best_model.joblib`: WESAD-only lab/reference binary model.
- `stress3_extratrees.joblib`
- `stress3_rf.joblib`
- `stress3_xgb.joblib`
- `stress3_lgbm.joblib`
- `manifest_stress3.json`
- `manifest_binary.json`
- `D:\IntroAI\outputs\nurse_binary\models\manifest_binary.json`

Each `.joblib` bundle contains:

- fitted model,
- median imputer,
- optional scaler,
- label encoder,
- exact feature column list,
- selected feature indices/columns when feature selection is enabled,
- label semantics,
- baseline policy metadata,
- class-cap/source-balance/resampling/calibration metadata,
- tuned binary decision threshold when applicable,
- decision-support states.

## Backend Usage

```python
import pandas as pd
from stress_benchmark.deploy import load_bundle, predict_feature_frame

bundle = load_bundle(r"D:\IntroAI\outputs\models\binary_best_model.joblib")
features = pd.read_csv(r"D:\IntroAI\outputs\features_combined.csv.gz").head(10)
predictions = predict_feature_frame(bundle, features)
```

Backend input must contain the same 693 model feature columns. In production, generate
those columns from a 60-second sensor window and the user's 5-10 minute personal
baseline before calling `predict_feature_frame`.

Deployment environment note: the current saved bundles were trained with
`scikit-learn==1.9.0`. Use the same scikit-learn version in the backend, or retrain
the bundles inside the backend environment before deploying.

The output includes:

- `pred_label`
- `confidence`
- `proba_0`, `proba_1` for binary models; `proba_2` is also present for 3-class models
- `alert_state`
- `recommendation`
