# Features (Phase 4)

Code: `ml/features/` (`clean.py`, `windows.py`, `scores.py`, `build_features.py`). Tests: `tests/test_features.py`.
Build the deterioration table with `python -m ml.features.build_features` (run `python -m ml.data.build_labels` first).

## Part A: deterioration features (done)
Output: `data/processed/deterioration_features.parquet`, 1,986 time points x 244 columns.

### Cleaning (`clean.py`)
- Vitals: heart rate, respiratory rate, SpO2, systolic and mean blood pressure (non-invasive and arterial merged), temperature (Fahrenheit converted to Celsius, then merged), GCS total (eye + verbal + motor, only when all three were charted together), and oxygen device.
- Values outside plausible ranges are removed (for example temperature 99 degrees C, arterial pressure 801 mmHg). Duplicates at the same time are averaged.
- Labs: 12 core (sodium, potassium, urea nitrogen, creatinine, bicarbonate, hematocrit, glucose, white blood cells, platelets, hemoglobin, phosphate, calcium) and 2 add-ons for the ablation (lactate, INR).
- Every measurement has an **availability time** (when the result was stored). The median delay is about 23 minutes for vitals and 66 minutes for labs. Features only see results whose availability time is at or before the prediction time.

### Feature groups (`feature_groups()` in `windows.py`)
| Group | Content | Columns |
|---|---|---|
| context | age, male, care unit, hours since ICU admission, on ventilator | 5 |
| vitals_snapshot | last value, hours since last measurement, measured in last 24 h (per vital), plus supplemental oxygen | 22 |
| vitals_trend | mean, min, max, std, slope per hour, last-minus-first, over 6, 12 and 24 h | 126 |
| labs_snapshot / labs_trend | core labs: last value, hours since, measured flag; min, max, change over 24 h | 36 / 36 |
| labs_extra_snapshot / labs_extra_trend | lactate and INR, same layout | 6 / 6 |
| scores | NEWS2 and MEWS (baselines, not model inputs by default) | 2 |

Labs get only a 24-hour window because they are drawn about every 12 to 24 hours; shorter windows are mostly empty.

### NEWS2 and MEWS (`scores.py`)
Computed from the latest value of each input, accepted only if it is at most 12 hours old. If an input is missing the score is empty (available for 93.8% of time points).
Simplifications: NEWS2 uses SpO2 scale 1 only; consciousness comes from GCS (below 15 counts as not alert), so ventilated patients score as impaired; MEWS maps GCS to AVPU (15 Alert, 13-14 Voice, 9-12 Pain, 8 or less Unresponsive); supplemental oxygen means an oxygen device was charted in the last 12 hours, because "room air" is never charted.

### Planned ablations (from `scope.md`)
- snapshot only (`*_snapshot` + context) versus snapshot + trend (adds `*_trend`)
- vitals only versus vitals + core labs versus + lactate and INR

## Leakage checks
1. Unit tests: values after the prediction time are ignored; a result drawn before the time but stored after it is ignored; the full feature table is identical when future data is added.
2. Real-data check: for 200 random time points, all information recorded after the time point was deleted and the features rebuilt. 0 of 200 differed from the full table.

## Honest notes
- 239 feature columns against about 20 positive patients is far too many for a flexible model. Phase 5 must use compact feature sets, regularisation or feature selection inside cross-validation, and must report uncertainty.
- Many trend columns are empty when fewer than two measurements fall in a window (for example GCS over 6 hours, which is charted about every 4 hours). Tree models handle missing values directly; the LSTM will use value plus a "measured" mask. Imputation and scaling happen inside the cross-validation folds (Phase 5), never on the whole table.
- NEWS2 and MEWS are higher on average before deterioration (NEWS2 7.8 vs 6.3, MEWS 4.1 vs 3.3), a plausible sign that the baselines work as intended.

## Part B (next): readmission features and LSTM arrays
Same lookbacks counted back from ICU discharge plus whole-stay summaries, and 24 hourly steps with a "measured" mask for the LSTM.
