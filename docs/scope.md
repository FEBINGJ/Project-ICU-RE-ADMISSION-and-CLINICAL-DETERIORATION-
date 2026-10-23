# Project Scope Note

Status: Phase 1 closed on 2026-10-21. Items marked "to confirm" get checked against the data in Phase 2 and Phase 3.

## Purpose
A clinical decision-support prototype that estimates two separate risks from longitudinal EHR data and explains them on a dashboard.

## Research question
On a small public ICU cohort, do temporal-trend features improve prediction of clinical deterioration and ICU readmission over single-time-point features and over NEWS2/MEWS, and at what alert burden?

How each part is tested:
- **Trend vs snapshot:** an ablation using last-value features only, then last-value plus trend features (slope, variability, rate of change), on the same models and folds.
- **Versus NEWS2/MEWS:** both scores are computed on our own cohort and evaluated the same way as the ML models.
- **Alert burden:** alerts per patient-day and precision at a fixed sensitivity, in the style of the alert analysis in the deterioration literature.

## In scope
- Two prediction tasks: clinical deterioration and ICU readmission
- Model comparison: XGBoost, Random Forest, LSTM; the best performer is chosen per task
- NEWS2 and MEWS computed as clinical baselines
- SHAP-based explanations of each prediction
- Web dashboard: login, patient list, patient details with trends, alerts, add new patient
- Replay-based simulation of ongoing monitoring using the dataset

## Out of scope
- Autonomous diagnosis or treatment recommendations
- Live hospital system integration
- Regulatory approval, prospective clinical validation or real clinical deployment
- External validation (no independent dataset is available)

## Decisions

| Decision | Status | Answer |
|---|---|---|
| Dataset and version | Decided | MIMIC-IV Clinical Database Demo 2.2 (100 patients, 140 ICU stays). Full MIMIC-IV needs credentialed PhysioNet access, which is not available. The schema matches full MIMIC-IV. |
| Candidate models | Decided | XGBoost, Random Forest, LSTM |
| Research gap | Decided | Trend features vs snapshot features vs NEWS2/MEWS, with alert burden (see research question) |
| Readmission definition | Decided | ICU readmission within 30 days of ICU discharge, predicted at ICU discharge. Window to confirm against demo event counts. |
| Deterioration definition | Decided (adapted in Phase 2, confirmed 2026-10-24) | Within the ICU: start of invasive mechanical ventilation or death within 24 hours of a 6-hourly prediction time point. "ICU transfer" is dropped because the dataset has no ward vital signs (see Phase 2 findings). |
| Cohort | Proposed | Adults only; exclude patients who died in the ICU (as in the iREAD study) |
| Prediction time point | Proposed | Readmission: at ICU discharge. Deterioration: every 6 hours (sliding windows). |
| Validation | Proposed | Stratified cross-validation split by patient, never by row |
| Evaluation metrics | Proposed | Accuracy, Precision, F1, AUROC as requested; AUROC and F1 decide the winner. Recall, AUPRC and calibration added because the classes are imbalanced. |
| Final submission | Flexible | Around February 2027; no strict deadline |

## Literature basis (Phase 1)
Four papers reviewed: a 2023 systematic review of ML for clinical deterioration (29 studies), a 2026 systematic review of ML for ICU readmission (Zheng et al., 31 studies, pooled AUC 0.82), a MIMIC-IV deterioration model with a 6-hour sliding window (XGBoost AUROC 0.758 vs modified NEWS2 0.568 and MEWS 0.533), and the iREAD multi-centre readmission model (internal AUROC 0.820, external 0.768 and 0.725).

Recurring limitations in the literature that this project addresses or states openly: limited external validation, heterogeneous outcome definitions and windows, missing data, alert burden, and lack of prospective evaluation.

Literature values for NEWS2/MEWS vary widely between studies because outcomes and populations differ, so this project computes its own baselines.

## Phase 2 findings (data understanding)
Computed in `notebooks/01_eda_mimic_demo.ipynb`; label logic in `ml/data/mimic_demo.py`.

- 100 adult patients, 275 admissions, 140 ICU stays, about 669k charted rows and 108k lab rows.
- Vital signs exist only while the patient is in the ICU, so there are no ward vital signs. Deterioration is therefore defined inside the ICU.
- Readmission (128 index stays after excluding 12 ICU deaths): 30 days has 21 events from 15 patients, 7 days has 14, and 48 hours has only 8. The 30-day window stays primary; 48 hours is not usable.
- Deterioration (1,986 six-hourly time points): 88 positives (4.4%) from only 20 patients. Each event appears in about four consecutive time points, so rows are strongly correlated and the effective sample size is closer to 20 patients than to 88 rows.
- Heart rate, respiratory rate, SpO2 and blood pressure are charted about hourly; temperature and GCS about every 4 hours.
- NEWS2/MEWS inputs are present for 86% of time points (94% if GCS is ignored), so both scores can be computed as baselines.
- Consequences: split by patient, use repeated stratified patient-level cross-validation, report wide uncertainty, and expect the LSTM may not beat the tree models.

## Known limitations of this project
- Small cohort (100 patients); results are a proof of concept, not clinically validated
- Few readmission events, so estimates will have wide uncertainty
- Internal validation only
- Research prototype; not for clinical use

## Success criteria
- Models evaluated with patient-level cross-validation, with uncertainty reported
- Both tasks reported with Accuracy, Precision, F1 and AUROC, plus Recall, AUPRC and calibration
- Trend-feature ablation and NEWS2/MEWS comparison reported for both tasks
- Every dashboard prediction shows its top contributing factors
- Limitations documented honestly