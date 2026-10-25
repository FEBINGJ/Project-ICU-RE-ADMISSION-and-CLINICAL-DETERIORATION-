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
| Cohort | Decided | Adults only; exclude patients who died in the ICU (as in the iREAD study) |
| Prediction time point | Decided | Readmission: at ICU discharge. Deterioration: every 6 hours (sliding windows). |
| Validation | Decided | Repeated 5-fold cross-validation (10 repeats), grouped and stratified by patient, never split by row |
| Evaluation metrics | Decided | Accuracy, Precision, F1, AUROC as requested; AUROC and F1 decide the winner. Recall, AUPRC and calibration added because the classes are imbalanced. |
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

## Feature decisions for Phase 4 (locked 2026-10-24)

| Decision | Choice |
|---|---|
| Lookback windows, deterioration | 6, 12 and 24 hours before each 6-hourly prediction time point |
| Lookback windows, readmission | The same 6, 12 and 24 hours counted back from ICU discharge, plus whole-stay summaries. Stays without enough history keep their rows, with missing trend values marked |
| Snapshot features | Last value, time since last measurement, and a "was measured" flag for each vital and lab |
| Trend features | Slope, mean, minimum, maximum, variability and change (last minus first) per lookback |
| Context features | Age, sex, care unit, hours since ICU admission, on-ventilator flag |
| Clinical baselines | NEWS2 and MEWS computed at the same time points |
| Primary lab set (12) | Sodium, potassium, urea nitrogen, creatinine, bicarbonate, hematocrit, glucose, white blood cells, platelets, hemoglobin, phosphate, calcium (all at least 92% complete) |
| Lab add-on (ablation only) | Lactate (48% complete) and INR (79%): compare "core 12" against "core 12 + lactate + INR" |
| Labs dropped | Albumin, troponin, bilirubin (too sparse); stated as a limitation |
| Feature-set ablation | "Snapshot only" versus "snapshot + trend" (the research question), plus "vitals only" versus "vitals + labs" |
| LSTM input | 24 hourly steps with the values plus a "measured" mask |
| Leakage test | A test that changes data after the prediction time and checks that the features do not change |

## Modeling decisions for Phase 5 (locked 2026-10-24)

| Decision | Choice |
|---|---|
| Structure | Part A: shared test setup, XGBoost, Random Forest, snapshot-vs-trend comparison. Part B: LSTM, reusing the Part A setup unchanged. Part C: compare all three models plus the NEWS2/MEWS baselines, choose the winners |
| One winner per task | Deterioration and readmission are compared and decided separately; the two winners may be the same algorithm or different ones |
| Validation | Repeated 5-fold cross-validation, 10 repeats, all rows of a patient in the same fold; identical folds, seeds and threshold rule for all three models |
| Tuning | None; fixed conservative settings (shallow trees, strong regularisation, class weights), because tuning on this little data mostly fits noise |
| Feature sets | A snapshot + context; B = A + trends; C = B + core labs; D = C + lactate and INR; readmission also adds whole-stay summaries |
| Baselines | NEWS2 and MEWS on exactly the same rows, plus a "predict the average rate" reference |
| Threshold | For Accuracy, Precision, Recall and F1 the cut-off is the F1-best value found inside the training folds, never from the test fold |
| Metrics reported | Accuracy, Precision, Recall, F1, AUROC, AUPRC, calibration (Brier score); alerts per patient-day for deterioration; every number with a confidence range |
| Winner rule (fixed in advance) | Highest average AUROC, F1 as tiebreaker. If the confidence ranges overlap: "no clear winner", and the simpler tree model is chosen (SHAP explanations are easier) |
| After the choice | The winner of each task is refitted on all the data for the dashboard; the cross-validation models are not reused |
| Caveat to report | The winner's score is slightly optimistic because it was picked from the same results that are reported |

## Phase 5 outcome (completed 2026-10-25)

| Item | Result |
|---|---|
| Candidates compared | XGBoost and Random Forest on feature set C, LSTM on vitals + labs; NEWS2 and MEWS as comparators (fixed before the Part C numbers) |
| Winner rule as applied | Highest mean AUROC; "clearly worse" = paired 95% range of the difference excludes 0 (official) or separate ranges do not overlap (locked wording); if the two tests disagree, the locked wording decides. If nothing is clearly worse, the tree model with the highest AUROC wins |
| Deterioration | No clear winner; XGBoost chosen (AUROC 0.59, range 0.47 to 0.71; NEWS2 0.65) |
| Readmission | No clear winner; XGBoost chosen (AUROC 0.59, range 0.43 to 0.73; MEWS 0.66) |
| Trend features | No reliable benefit (paired ranges include zero) |
| Versus NEWS2/MEWS | No evidence of a difference either way; alert burden at matched sensitivity is the same |
| Calibration | Platt scaling inside the folds; calibrated risks stay near the average event rate (deterioration 3 to 8%, readmission 6 to 25%) |
| Saved models | models/deterioration_xgboost.joblib and models/readmission_xgboost.joblib (git-ignored) |
| Dashboard consequence | Show the calibrated risk with NEWS2/MEWS beside it and a research-prototype notice; never show raw model probabilities as "% risk" |
| Details | docs/results_part_a.md, results_part_b.md, results_part_c.md, model_card.md |

## Phase 6 outcome (completed 2026-10-25)

| Item | Result |
|---|---|
| Risk bands | High = at or above the alert threshold; Moderate = below it but at or above the cut-off that catches 80% of events; Low = below. Supported for deterioration only (observed rates 3.3%, 3.8%, 5.3%, gap not confirmed); not shown for readmission (rates do not rise) |
| NEWS2 reference bands | Standard clinical bands (0-4, 5-6, 7+) separate observed risk better than our bands; shown on the dashboard as the reference |
| Explanations | SHAP on the two saved XGBoost models, 5 factors per prediction (adjustable), plain-language names, direction and relative size only, written explanation and warnings |
| Stability | Top inputs change between refits (rank correlation about 0.5); only breathing-rate variability is stable; types of information (trends) are stable |
| Demo caveat | Replay of dataset patients must use cross-validated predictions; saved models are for new patients |
| Details | docs/explainability.md |

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