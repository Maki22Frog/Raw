# Unit Audit

Sources checked:

- WESAD official page: `https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html`
- Nurse Dryad dataset page: `https://datadryad.org/dataset/doi:10.5061/dryad.5hqbzkh6f`
- Local WESAD readme PDF inside `WESAD.zip`.
- Local Empatica E4 `info.txt` files inside nurse session zip files.

## Common E4 Modalities

The shared wrist-worn Empatica E4 signals are:

- ACC
- BVP
- EDA
- TEMP

The nurse dataset also includes E4-derived:

- HR
- IBI

WESAD `.pkl` wrist data does not expose HR/IBI directly, so the common bridge is
BVP-derived HR/HRV.

## Unit Compatibility

| Signal | WESAD local check | Nurse local check | Pipeline handling |
|---|---|---|---|
| ACC | values in `[-128, 127]` | values in `[-128, 127]` | both converted from Empatica `1/64 g` to `g` |
| EDA | microsiemens-scale values | Dryad says electrodermal activity/electrical conductivity; local values match microsiemens scale | used as microsiemens |
| TEMP | Celsius-like skin temperature | Dryad explicitly says Celsius | used as Celsius |
| BVP | Empatica PPG amplitude | Dryad says blood volume pulse; local scale matches Empatica amplitude | used as raw amplitude plus BVP-derived HR/HRV |
| HR | not directly available in WESAD wrist `.pkl` | bpm, starts 10 seconds after other E4 files | aligned and used as nurse-specific feature |
| IBI | not directly available in WESAD wrist `.pkl` | seconds between beats | used for nurse-specific HRV |

## Important Fixes

- ACC is normalized for both datasets. Earlier raw checks confirmed both WESAD `.pkl`
  wrist ACC and nurse E4 ACC use the same `[-128, 127]` Empatica scale.
- HR in nurse E4 starts 10 seconds after ACC/EDA/BVP/TEMP in all checked sessions.
  The reader now pads HR to align all signals to the same session start.
- WESAD HR/IBI are not assumed to exist. HRV features common to both datasets are
  derived from BVP peaks instead.
- Model input excludes time/order columns and survey-offset columns.

## Advanced Features Added

Common features:

- BVP-derived HR.
- BVP-derived HRV: RR mean/std/median/range, SDNN, RMSSD, pNN50, CVNN, LF/HF approximations.
- EDA tonic/phasic approximation.
- EDA SCR peak count/rate/amplitude.
- ACC magnitude/dynamic magnitude/jerk.
- ACC stationary ratio and active ratio.
- Personalized baseline deltas/ratios/z-scores for all numeric features.

Nurse-specific additional features:

- HR statistics.
- IBI statistics.
- IBI-derived HRV.

