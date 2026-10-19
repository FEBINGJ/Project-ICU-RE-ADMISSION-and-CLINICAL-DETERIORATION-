# Project Scope Note

Fill the blanks as decisions are made in Phase 1 to Phase 3. Anything marked TBD is still open.

## Purpose
A clinical decision-support prototype that estimates two separate risks from longitudinal EHR data and explains them on a dashboard.

## In scope
- Two prediction tasks: clinical deterioration and ICU readmission
- Model comparison: Logistic Regression, Random Forest, XGBoost, LSTM
- SHAP-based explanations of each prediction
- Web dashboard: login, patient list, patient details with trends, alerts, add new patient
- Replay-based simulation of ongoing monitoring using the dataset

## Out of scope
- Autonomous diagnosis or treatment recommendations
- Live hospital system integration
- Regulatory approval or real clinical deployment

## Open decisions

| Decision | Status | Answer |
|---|---|---|
| Dataset and version | TBD | |
| Cohort (adults only? minimum ICU stay length?) | TBD | |
| Deterioration definition (event and prediction window) | TBD | |
| Readmission definition (follow-up window) | TBD | |
| Prediction time point | TBD | |
| Train / validation / test split (by patient) | TBD | |
| Primary evaluation metrics | TBD | AUROC, AUPRC proposed |
| Research gap / novelty (for example adaptive learning) | TBD | |
| Final submission deadline | TBD | |

## Success criteria
- Models evaluated on a held-out, patient-level test set
- Both tasks reported with AUROC, AUPRC, sensitivity, specificity and calibration
- Every dashboard prediction shows its top contributing factors
- Limitations documented honestly
