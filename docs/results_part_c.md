# Phase 5, Part C: model comparison and the two chosen models

Code: `ml/evaluation/part_c.py` (comparison and winner rule), `ml/evaluation/calibrated.py` (calibration), `ml/models/final.py` (refit and save), `ml/models/factory.py`. Tests: `tests/test_part_c.py`.
Re-run in this order: `python -m ml.evaluation.part_c`, `python -m ml.evaluation.calibrated`, `python -m ml.models.final` (about 5 minutes in total).
Raw numbers: `reports/tables/part_c_results.json` and `part_c_calibration.json`. Figures: `reports/figures/part_c_comparison.png`, `part_c_calibration.png`.

## Headline

1. **Chosen model: XGBoost for both tasks.** For both deterioration and readmission the verdict was **"no clear winner"**: no model's AUROC range separates it from the others, so the locked rule picked the tree model with the highest AUROC, which is XGBoost (0.59 for both tasks). The LSTM did not win either task.
2. **No evidence that the ML models beat NEWS2/MEWS, and no evidence that they are worse.** NEWS2 scored 0.65 (deterioration) and MEWS 0.66 (readmission), and XGBoost trails by 0.06 to 0.08 AUROC, but the paired ranges of the differences (-0.21 to +0.08 and -0.19 to +0.06) include zero.
3. **At the same sensitivity the alert burden is the same.** To catch 80% of deterioration events, NEWS2 and XGBoost both alert on about two thirds of all time points (2.6 and 2.7 alerts per patient-day). At 50% sensitivity it is 1.4 and 1.5 alerts per patient-day. Neither is usable as a real alarm.
4. **After calibration the risk numbers stay close to the average event rate.** Deterioration risk runs between 3% and 8% (average rate 4.4%); readmission between 6% and 25% (average rate 16%). Calibrated Brier scores (0.043 and 0.145) are no better than always predicting the average rate (0.042 and 0.137). This is the honest face of a model with near-chance discrimination.

## What was fixed before looking at the Part C numbers
- **Candidates.** XGBoost and Random Forest on feature set C (vitals snapshot and trends, 12 core labs, plus whole-stay vitals for readmission); LSTM on vitals + labs. NEWS2 and MEWS are comparators and can never be the winner. Set C also matches what the dashboard shows (trends and labs).
- **Winner rule.** Highest mean AUROC; models within 0.005 AUROC are tied and the higher F1 decides. A model is "clearly worse" when the 95% range of the paired difference excludes zero (official test) or when the two separate 95% ranges do not overlap (the locked wording). If nothing is clearly worse than the best, the best wins. Otherwise all models that are not clearly worse stay in, and the tree model with the highest AUROC among them wins. If the two tests choose different models, the separate-ranges test decides. Both tests agreed for both tasks.
- **Sensitivity targets for alert burden:** 50% and 80%.

## Results (mean of 10 repeats, 1,000 patient-level bootstrap draws, same patients for every model)

### Deterioration

| Model | AUROC (95% range) | F1 | Alerts per patient-day at 50% sensitivity | Alerts per patient-day at 80% sensitivity | Precision at 80% sensitivity |
|---|---|---|---|---|---|
| NEWS2 | 0.646 (0.56 to 0.73) | 0.121 | 1.38 (0.9 to 2.3) | 2.61 (2.3 to 3.2) | 0.058 |
| MEWS | 0.629 (0.55 to 0.72) | 0.079 | 1.71 (1.0 to 2.6) | 2.69 (2.3 to 3.5) | 0.054 |
| XGBoost (set C) (winner) | 0.586 (0.47 to 0.71) | 0.096 | 1.48 (1.1 to 2.0) | 2.74 (1.9 to 3.4) | 0.053 |
| Random Forest (set C) | 0.563 (0.46 to 0.68) | 0.084 | 1.65 (1.1 to 2.3) | 2.88 (2.0 to 3.3) | 0.050 |
| LSTM (vitals + labs) | 0.572 (0.46 to 0.68) | 0.096 | 1.52 (1.0 to 2.2) | 2.94 (2.2 to 3.4) | 0.049 |

Paired differences in AUROC (first minus second, 95% range):

| Comparison | Change in AUROC | Change in F1 |
|---|---|---|
| XGBoost vs Random Forest | +0.023 (-0.03 to +0.08) | +0.011 (-0.01 to +0.03) |
| XGBoost vs LSTM | +0.014 (-0.07 to +0.11) | -0.000 (-0.03 to +0.03) |
| XGBoost vs NEWS2 | -0.060 (-0.21 to +0.08) | -0.025 (-0.07 to +0.02) |
| XGBoost vs MEWS | -0.043 (-0.18 to +0.10) | +0.017 (-0.02 to +0.06) |
| NEWS2 vs MEWS | +0.017 (-0.04 to +0.06) | +0.042 (-0.00 to +0.10) |

### Readmission

| Model | AUROC (95% range) | F1 | Precision at 50% sensitivity | Precision at 80% sensitivity | Share of stays flagged at 80% sensitivity |
|---|---|---|---|---|---|
| NEWS2 | 0.658 (0.51 to 0.77) | 0.348 | 0.279 (0.12 to 0.41) | 0.192 (0.10 to 0.31) | 77% |
| MEWS | 0.663 (0.51 to 0.81) | 0.355 | 0.275 (0.10 to 0.58) | 0.179 (0.10 to 0.29) | 91% |
| XGBoost (set C) (winner) | 0.586 (0.43 to 0.73) | 0.254 | 0.198 (0.11 to 0.53) | 0.190 (0.11 to 0.29) | 70% |
| Random Forest (set C) | 0.571 (0.39 to 0.75) | 0.241 | 0.186 (0.10 to 0.60) | 0.180 (0.10 to 0.28) | 74% |
| LSTM (vitals + labs) | 0.568 (0.45 to 0.67) | 0.292 | 0.209 (0.11 to 0.36) | 0.188 (0.11 to 0.29) | 72% |

Paired differences in AUROC (first minus second, 95% range):

| Comparison | Change in AUROC | Change in F1 |
|---|---|---|
| XGBoost vs Random Forest | +0.016 (-0.05 to +0.09) | +0.013 (-0.02 to +0.05) |
| XGBoost vs LSTM | +0.018 (-0.10 to +0.13) | -0.038 (-0.11 to +0.03) |
| XGBoost vs NEWS2 | -0.071 (-0.22 to +0.08) | -0.094 (-0.20 to +0.02) |
| XGBoost vs MEWS | -0.077 (-0.19 to +0.06) | -0.101 (-0.23 to +0.06) |
| NEWS2 vs MEWS | -0.005 (-0.08 to +0.07) | -0.007 (-0.14 to +0.14) |
Alert-burden columns are descriptive: the threshold is read off the pooled out-of-fold predictions to hit the stated sensitivity, so they show what each model could achieve, not a threshold that could be chosen in advance. Alerts per patient-day count every alerting 6-hour time point.

## What the numbers mean
- **Why "no clear winner" is a legitimate result.** With 20 and 15 positive patients, differences of 0.02 to 0.08 AUROC are inside the noise (every paired range spans zero). Choosing XGBoost is a tie-break for explainability and for being the highest of near-equal scores, not a claim that it is better.
- **The winner's score is slightly optimistic.** XGBoost was picked from the same results that are reported. The honest figure for any quoted performance is the cross-validated AUROC with its range (0.59, range 0.47 to 0.71 for deterioration and 0.43 to 0.73 for readmission).
- **Readmission F1 does not rank the models fairly.** NEWS2/MEWS reach F1 0.35 and the LSTM 0.29, but alerting for every patient already scores 0.28. This is why AUROC, with its range, is the deciding metric.
- **Calibration.** Class weighting makes raw probabilities far too high (for example 42% predicted against 5.5% observed in the top bin for deterioration). Platt scaling learned inside the folds fixes that: predicted and observed risk now agree on average, and the Brier score drops from 0.077 to 0.043 and 0.148 to 0.145. It cannot add information, so calibrated risks cluster near the average rate. The pooled AUROC of calibrated scores (0.55 and 0.50) is lower than the raw 0.59 only because each fold learns slightly different calibration curves; inside a fold the ranking of patients is unchanged (tested).
- **Interview line.** "On a 100-patient demo cohort, an untuned XGBoost, a Random Forest and an LSTM could not be told apart from each other or from NEWS2/MEWS, and I report that with confidence ranges instead of picking the best-looking number."

## The saved models (for SHAP and the dashboard)
Each winner is refitted on all rows and saved to `models/<task>_xgboost.joblib` (git-ignored) together with its columns, alert threshold (F1-best from repeated 5-fold patient-grouped cross-validation on all rows), calibration curve and cross-validated scores. `ml.models.final.predict(artifact, rows)` returns the raw score, the calibrated risk and the alert flag.

| | Deterioration | Readmission |
|---|---|---|
| Training rows / positives | 1,986 / 88 | 128 / 21 |
| Input columns | 233 | 275 |
| Alert threshold (raw score) | 0.108 | 0.092 |
| Calibration (intercept, slope) | -2.69, 0.19 | -1.01, 0.48 |
| Cross-validated AUROC | 0.586 (0.47 to 0.71) | 0.586 (0.43 to 0.73) |

The models' accuracy on their own training data is near perfect (AUROC 1.0) and must never be quoted; only the cross-validated figures count. A model card is in `docs/model_card.md`.

## Limitations of this Part
- Positive patients: 20 for deterioration and 15 for readmission; every conclusion is low-power.
- One fixed setting per model and no tuning, on purpose.
- Calibration is learned from few events, so the calibration curve itself is uncertain (the readmission reliability curve is not monotone).
- The saved model's threshold and calibration come from models trained on 80% of the rows, so they are slightly mismatched to the model trained on 100%.
- Internal validation on a 100-patient demo only; no external validation.

## What this means for the next phases
- Phase 6 explains the two saved XGBoost models with SHAP. Because discrimination is weak, the explanation page must say that the factors describe what the model used, not proven causes.
- The dashboard shows the calibrated risk, the NEWS2/MEWS value next to it as the bedside reference, and a visible "research prototype, not for clinical use" notice.
