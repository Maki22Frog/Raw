# Methodology

This benchmark combines the strongest practical ideas from the two local papers:

1. Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection".
2. Korkmaz et al., "Predicting Nurse Stress Levels Using Time-Series Sensor Data and Comparative Evaluation of Classification Algorithms".

## What Is Taken From WESAD

- Use multimodal wearable physiology plus motion data.
- Use 60-second windows for physiological signals.
- Use ACC as a context signal, not only as a classifier input.
- Evaluate with subject-independent splits, especially leave-one-subject-out (LOSO).
- Report F1-score, not only accuracy, because affect/stress labels are imbalanced.
- Keep baseline and stress protocol labels explicit.

For the combined task, WESAD labels are mapped as:

- baseline -> `target3 = 0`
- amusement -> `target3 = 0`
- stress/TSST -> `target3 = 2`
- meditation/transient/ignored labels are excluded

This makes WESAD compatible with the nurse dataset's low/moderate/high stress labels.

## What Is Taken From The Nurse Stress Paper

- Aggregate raw time-series data into one-minute windows.
- Use E4 modalities: ACC, EDA, HR/BVP-derived heart features, and TEMP.
- Handle class imbalance using SMOTE inside each training fold only.
- Add Balanced Random Forest as an imbalance-aware tree baseline.
- Compare Random Forest, Balanced Random Forest, k-NN, Gaussian Naive Bayes,
  ExtraTrees, XGBoost, and LightGBM when installed.
- Report accuracy, balanced accuracy, macro F1, weighted F1, classification reports, and confusion matrices.
- Keep temporal and source analysis available, but do not include time/source as default features.

## Added Improvements

### 1. Personalized Baseline

The pipeline computes a per-subject baseline and adds:

- `feature_delta_base`
- `feature_ratio_base`
- `feature_z_personal`

Baseline source:

- WESAD: the protocol baseline condition.
- Nurse data: earliest available calibration windows, without using labels.

This turns absolute physiological features into personalized changes, reducing the error
caused by different resting HR, EDA, temperature, and movement patterns.

### 2. Stress Versus Physical Activity

The pipeline extracts ACC magnitude, dynamic magnitude, jerk, and activity features.
The decision layer then delays stress alerts when:

- HR/heart-like features are elevated,
- ACC activity is high,
- EDA does not rise enough.

This is intended to reduce false alarms from walking, stairs, or general movement.

### 3. Multi-Level Decision Support

The classifier output is post-processed into:

- `normal`
- `low_confidence_normal`
- `monitor_more`
- `rising_stress`
- `persistent_stress`
- `critical_alert`
- `physical_activity_delay`

The raw ML metrics are kept separate from the decision-support output.

### 4. Simple Constrained Recommendations

Each alert state has a lightweight recommendation:

- no action
- continue monitoring
- measure more before alerting
- delay alert during likely physical activity
- short break, water, slow breathing
- stop task, short break, breathing, support

This keeps the system interpretable and avoids pretending that the classifier alone is a
medical decision system.

### 5. Imbalance Handling For Binary Tasks

For nurse-only binary and combined binary benchmarks, imbalance handling now has three
core layers:

- Per-subject per-class cap: at most 150 windows per subject/class are used in the
  model-training subset. This prevents one long or heavily represented subject/class from
  dominating the fitted model.
- Resampling: regular models use SMOTE only after the train/calibration split and only
  on the capped model-training subset. Balanced Random Forest uses its own internal
  balancing instead of external SMOTE.
- Threshold tuning: binary decision thresholds are tuned on a held-out calibration split
  using balanced accuracy by default. Test folds are never used for threshold selection.

This affects binary prediction and deployment: `pred_label` is decided by the stored
threshold, not by raw probability argmax. The deploy bundle stores `decision_threshold`,
`threshold_tuning`, `class_cap`, `resampling`, and `calibration`.

The code also supports scientific ablations for:

- train-fold feature selection with ExtraTrees top-k importances,
- class-cap grids such as `100,150,250,none`,
- threshold-metric grids including a recall-floor variant,
- source-balanced combined training by downsampling each source/class inside the model-training subset.

These ablations are selected using the train-fold calibration split only. If an ablation
does not improve held-out fold performance, it is reported but not used for the default
deploy bundle.

## Scientific Evaluation Protocol

Recommended order:

1. Run `scan-offsets` for the nurse survey.
2. Extract features with fixed window and step settings.
3. Run `groupkfold` for fast development.
4. Run `loso` for final subject-independent reporting.
5. Report metrics both overall and by source.
6. Apply class caps, SMOTE/internal balancing, calibration, and threshold tuning inside training folds only.
7. Never split randomly by rows for final claims because adjacent windows from the same subject/session leak information.

## Known Limitations

- WESAD is a lab dataset; the nurse dataset is field data. They are not identical distributions.
- WESAD has no moderate stress class in the unified mapping.
- Nurse survey timestamps may need timezone-offset validation.
- The decision-support layer is rule-based and should be reported separately from raw model performance.
- Full WESAD `.pkl` extraction is memory intensive because each subject file is large.
