# Phase 5, Part B: results (LSTM)

Code: `ml/models/lstm.py` (model and in-fold preprocessing), `ml/evaluation/run_part_b.py` (runner), `ml/evaluation/permutation_check.py` (shuffled-label sanity check). Tests: `tests/test_lstm.py`.
Re-run: `python -m ml.evaluation.run_part_b` (about 8 minutes). Raw numbers: `reports/tables/part_b_results.json` and `part_b_paired.json`.

## Headline

1. **The LSTM did not do better than the tree models or the bedside scores.** Deterioration AUROC 0.57 (range 0.46 to 0.68), readmission 0.57 (0.45 to 0.67). The tree models scored 0.56 to 0.59 and 0.55 to 0.60, NEWS2/MEWS 0.65 and 0.66. All ranges overlap.
2. **Adding the 12 labs to the 7 vitals changed little.** AUROC +0.02 for deterioration (range -0.10 to +0.13) and +0.03 for readmission (-0.05 to +0.10).
3. **A single readmission result is hard to tell from luck at this size.** 50 runs with shuffled labels gave AUROC between 0.37 and 0.76 (average 0.50, SD about 0.07). One shuffle even scored 0.598, equal to the best real readmission result in Parts A and B. Readmission differences smaller than about 0.1 AUROC cannot be trusted.

## Setup (identical to Part A)
- Same folds: the runner stops if the fold fingerprints differ from Part A (deterioration `361a81892d94`, readmission `3bc222d83a36`; both matched).
- Same threshold rule (F1-best cut-off inside the training fold, inner 3-fold patient-grouped), same metrics and 95% patient-level bootstrap.
- Input: 24 hourly steps; per channel the scaled value, a "newly measured" flag and the scaled hours since the last measurement (57 inputs per step for all channels, 21 for vitals only), plus the static context features (13 columns for deterioration and 27 for readmission, after one-hot encoding of the care units).
- Learned only from training rows: median fill, mean and standard deviation scaling, and early stopping (a patient-grouped fifth of the training rows). A test confirms that flipping the test-fold labels does not change predictions.
- Fixed settings, no tuning: 1 LSTM layer, 32 units, dropout 0.3, AdamW (learning rate 0.003, weight decay 0.01), class-weighted loss, at most 30 epochs with patience 5, 3 seeds averaged.

## Results (mean of 10 repeats)

| Task | Input | AUROC (95% range) | AUPRC | Accuracy | Precision | Recall | F1 | Brier | Alerts per patient-day |
|---|---|---|---|---|---|---|---|---|---|
| Deterioration | 7 vitals + 12 labs | 0.572 (0.46 to 0.68) | 0.059 | 0.73 | 0.06 | 0.32 | 0.10 | 0.202 | 1.04 |
| Deterioration | 7 vitals only | 0.554 (0.48 to 0.64) | 0.060 | 0.69 | 0.05 | 0.35 | 0.09 | 0.189 | 1.20 |
| Readmission | 7 vitals + 12 labs | 0.568 (0.45 to 0.67) | 0.208 | 0.44 | 0.19 | 0.71 | 0.29 | 0.207 | n/a |
| Readmission | 7 vitals only | 0.537 (0.41 to 0.64) | 0.194 | 0.41 | 0.17 | 0.68 | 0.27 | 0.212 | n/a |

For comparison (Part A, same folds): deterioration NEWS2 0.646, best tree model XGBoost + labs 0.586; readmission MEWS 0.663, best tree model XGBoost snapshot 0.598.

## What the numbers mean
- **The LSTM barely trains in a useful way.** In a look at one training fold (deterioration), training loss kept falling while loss on the held-out patients was lowest after 1 to 2 epochs and then rose steeply. The network memorises the training patients quickly; early stopping then keeps an almost untrained model. This is expected with 15 to 20 positive patients and confirms the earlier warning that the LSTM may not beat the trees.
- **Readmission F1 is flattering and should not be read at face value.** F1 0.29 is only just above 0.28, the F1 of alerting for every patient, and accuracy (0.44) is lower than a model that never alerts. The LSTM alerts for about 70% of stays.
- **Probabilities are not calibrated** (Brier 0.19 to 0.21, versus 0.042 and 0.137 for always predicting the average rate), because of class weighting. The same rule applies as in Part A: no "% risk" on the dashboard before calibration.
- **Shuffled-label checks.** One full run with shuffled labels: deterioration 0.54 (range 0.48 to 0.60), readmission 0.60 (0.51 to 0.69). Because one shuffle can be lucky, `permutation_check.py` ran many: readmission LSTM 50 shuffles (two batches of 20 and 30) with average 0.50 and SD about 0.07, range 0.37 to 0.76; readmission XGBoost 20 shuffles, average 0.48, SD 0.09, range 0.28 to 0.66; deterioration LSTM 6 shuffles, average 0.50, SD 0.03, range 0.47 to 0.54. So nothing in the pipeline gives above-chance scores, and the readmission task is too small for fine comparisons.

## Limitations of this Part
- One fixed LSTM size and training recipe; deliberately not tuned.
- Early stopping uses roughly 3 positive patients per validation slice for readmission, which makes the stopping point noisy.
- 20 deterioration and 15 readmission positive patients: conclusions are low-power.

## Next: Part C
Compare XGBoost, Random Forest, LSTM and NEWS2/MEWS per task with paired ranges, apply the locked winner rule, and refit the winners on all data.
