# Model card: ICU deterioration and readmission risk models (research prototype)

**Not for clinical use.** A final-year research project built on the public MIMIC-IV Clinical Database Demo. It must not be used to diagnose patients, make treatment decisions or replace clinical judgement.

## Models
| | Deterioration model | Readmission model |
|---|---|---|
| Question | Will the patient start invasive ventilation or die within the next 24 hours? | Will the patient be readmitted to the ICU within 30 days of ICU discharge? |
| When it predicts | Every 6 hours during the ICU stay | At ICU discharge |
| Algorithm | XGBoost (fixed conservative settings, no tuning) | XGBoost (same settings) |
| Inputs | Vital-sign snapshots and 6/12/24-hour trends, 12 core labs, context (age, sex, care unit, hours since admission, ventilator) | The same, counted back from ICU discharge, plus whole-stay vital summaries and stay context |
| Never used as inputs | NEWS2/MEWS, hospital discharge location, death after the ICU, time to the next ICU stay, anything recorded after the prediction time | same |
| Training data | 1,986 time points, 88 positive, 99 patients | 128 ICU stays, 21 positive, 93 patients |

## Performance (10 x repeated 5-fold cross-validation, split by patient; 95% ranges from patient resampling)
| | Deterioration | Readmission |
|---|---|---|
| AUROC | 0.59 (0.47 to 0.71) | 0.59 (0.43 to 0.73) |
| NEWS2 on the same rows | 0.65 (0.56 to 0.73) | 0.66 (0.51 to 0.77) |
| MEWS on the same rows | 0.63 (0.55 to 0.72) | 0.66 (0.51 to 0.81) |
| Difference to the best bedside score | not distinguishable from zero | not distinguishable from zero |
| Alerts to catch 80% of events | 2.7 per patient-day (about two thirds of all time points) | 70% of stays flagged |
| Calibrated risk range (5th to 95th percentile) | 3% to 8% | 6% to 25% |

The model choice was "no clear winner" among XGBoost, Random Forest and LSTM; XGBoost was selected as the tree model with the highest AUROC, because tree models are easier to explain with SHAP. The selected model's score is slightly optimistic for that reason.

## Intended use
Demonstrating an end-to-end risk-prediction pipeline: patient-level validation, leakage control, baseline comparison, calibration and explanation on a public demo dataset.

## Known limitations
- Tiny cohort (100 patients); 20 and 15 patients with events. Performance is close to chance and the uncertainty is wide.
- No external validation; one hospital system.
- Deterioration is defined inside the ICU (ventilation or death) because the dataset has no ward vital signs; "ICU transfer" could not be used.
- Predicted risks are calibrated to the average event rate and carry little individual information.
- Model explanations show what the model used, not causes.

## Files and reproducibility
Trained files are in `models/` (git-ignored). Code: `ml/`. Results: `docs/results_part_a.md`, `results_part_b.md`, `results_part_c.md`. Fixed seed 42; the folds have fingerprints `361a81892d94` (deterioration) and `3bc222d83a36` (readmission).
