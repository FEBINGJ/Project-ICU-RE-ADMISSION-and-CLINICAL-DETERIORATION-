# Cohort and Label Definitions (Phase 3)

Code: `ml/data/mimic_demo.py` (logic) and `ml/data/build_labels.py` (writes the tables).
Tests: `tests/test_labels.py`. Rebuild with `python -m ml.data.build_labels`.

## Cohort
- Source: MIMIC-IV Clinical Database Demo 2.2 (100 adult patients, 140 ICU stays).
- All patients are adults, so no age exclusion was needed.
- Readmission task: ICU stays that ended in death are excluded, because those patients cannot be readmitted. 12 stays were removed, leaving 128 index stays from 93 patients.
- Deterioration task: all 140 ICU stays are used. Stays shorter than 6 hours produce no time points, so 137 stays from 99 patients appear.

## Task 1: ICU readmission
| Item | Definition |
|---|---|
| Unit of prediction | One ICU stay (the "index stay") |
| Prediction time | ICU discharge time (`outtime`) |
| Primary label `readmit_30d` | 1 if the same patient is admitted to an ICU again within 30 days (720 hours) after this discharge |
| Sensitivity labels | `readmit_7d`, `readmit_48h` (kept for robustness checks only; 48 h has too few events to model) |
| Readmission counts | A later ICU stay counts whether it is in the same or a later hospital admission |
| Result | 21 positive stays from 15 patients (16.4% of 128) |

Allowed inputs: anything recorded during the index ICU stay and before it, up to `index_time`. Nothing recorded after ICU discharge may be used.

Known simplifications:
- Patients who die on the ward after leaving the ICU stay in the data as negatives (the flag `died_after_icu_in_hospital` allows a sensitivity check). The iREAD study also only excluded ICU deaths.
- Discharge to hospice is not excluded; the column `discharge_location` is kept for a sensitivity check.

## Task 2: clinical deterioration
| Item | Definition |
|---|---|
| Unit of prediction | One prediction time point |
| Prediction times | Every 6 hours after ICU admission (hour 6, 12, 18, ...) until ICU discharge |
| Label `y` | 1 if the patient **starts invasive mechanical ventilation** or **dies** in the next 24 hours |
| Components | `vent_next` and `death_next` are stored separately |
| Already ventilated | `on_vent` marks time points where the patient is already ventilated. They are kept (ventilation status is a feature); for them only death can create a positive label |
| Result | 88 positive time points (4.43%) from 20 patients, out of 1,986 |

Allowed inputs: data recorded at or before the time point `t`, looking back at most 24 hours. Nothing after `t` may enter a feature.

Why not "ICU transfer": the dataset has vital signs only inside the ICU, so a ward-to-ICU transfer label would have no vitals to predict from.

Known simplifications:
- Each event appears in about four consecutive time points, so rows are strongly correlated. The effective sample size is close to the number of patients (20).
- Ventilation after ICU discharge is not observed (it is an ICU-only record). Death after ICU discharge is observed because death time comes from the hospital admission record.

## Rules that apply to every later phase
1. **Split by patient.** All stays and time points of one patient stay in the same fold.
2. **No leakage.** Features use only data up to the prediction time. Scalers and imputers are fitted on training folds only.
3. **Primary choices are fixed here.** The 6-hour step, 24-hour horizon and 30-day window are the primary settings; other values are reported only as sensitivity checks.
4. **Baselines.** NEWS2 and MEWS are computed at the same prediction times and evaluated the same way.
