# Phase 5, Part A: results (XGBoost, Random Forest, baselines)

Code: `ml/evaluation/` (folds, metrics, runner, paired comparison, figure) and `ml/models/` (feature sets, trees, baselines).
Re-run: `python -m ml.evaluation.run_part_a` (about 25 minutes), then `python -m ml.evaluation.paired` and `python -m ml.evaluation.plot_part_a`.
Raw numbers: `reports/tables/part_a_results.json` and `reports/tables/part_a_paired.json`. Figure: `reports/figures/part_a_auroc.png`.

## Headline

1. **On this small cohort neither tree model beat NEWS2/MEWS, for either task.** NEWS2 reached AUROC 0.65 for deterioration (best tree model 0.59) and MEWS 0.66 for readmission (best tree model 0.60). The ranges overlap, so the honest statement is "no evidence that the ML models are better", not "the ML models are worse".
2. **Trend features did not show a reliable benefit.** Adding trends to the snapshot moved deterioration AUROC by +0.03 to +0.05 (range includes zero) and readmission by -0.03 to -0.04 (range includes zero). The direction is plausible for deterioration but cannot be confirmed with 20 positive patients.
3. **Every ML configuration is statistically close to chance for readmission** (lower end of every range is 0.35 to 0.43) and only weakly above it for deterioration (lower end 0.39 to 0.47).
4. **The pipeline is not leaking and not broken.** With shuffled labels, AUROC fell to 0.42 to 0.49 with ranges that include 0.5, and NEWS2/MEWS, which need no training, do score above chance on the same rows.

## How it was evaluated (frozen setup, reused unchanged by Part B and Part C)

- Repeated 5-fold cross-validation, 10 repeats, split by patient (no patient in both train and test), stratified on the label. Fold fingerprints: deterioration `361a81892d94`, readmission `3bc222d83a36`. Part B must print the same fingerprints.
- Fixed settings, no tuning. XGBoost: 150 trees, depth 2, learning rate 0.05, row sample 0.7, column sample 0.5, minimum child weight 2, L2 penalty 10, positive class weight = negatives / positives in the training fold. Random Forest: 300 trees, depth 5, at least 5 rows per leaf, square-root feature sampling, balanced class weights.
- Threshold for Precision, Recall, F1, Accuracy and alerts: the F1-best cut-off found inside the training fold (inner 3-fold patient-grouped split of the training rows), never from the test fold. A test checks that the inner step never sees test rows.
- Confidence ranges: each metric is computed on pooled out-of-fold predictions per repeat and averaged over the 10 repeats; the 95% range comes from 500 patient-level bootstrap draws (patients resampled with replacement, all metrics recomputed). The spread over repeats is smaller than the bootstrap range and is only stored in the JSON.
- Baselines on exactly the same rows and folds: NEWS2 and MEWS (the score is the prediction; missing score filled with the training-fold median) and an "average rate" reference that predicts the overall event rate and alerts for everyone.
- Alerts per patient-day (deterioration): every 6-hourly time point that triggers an alert counts as one alert, so the rate = alert share x 4. Repeated alerts for the same event are not merged.
- Feature sets: A snapshot + context; B = A + vitals trends; S = B + whole-stay vitals (readmission only); C = core 12 labs added; D = C + lactate and INR. Missing values stay missing (both models handle them). Care units are one-hot encoded.

## Results (mean of 10 repeats)

### Deterioration

| Model / features | AUROC (95% range) | AUPRC | Accuracy | Precision | Recall | F1 | Brier | Alerts per patient-day |
|---|---|---|---|---|---|---|---|---|
| Average rate (reference) | 0.500 (0.50 to 0.50) | 0.044 | 0.04 | 0.04 | 1.00 | 0.08 | 0.042 | 4.00 |
| NEWS2 | 0.646 (0.57 to 0.74) | 0.071 | 0.78 | 0.07 | 0.35 | 0.12 | n/a | 0.84 |
| MEWS | 0.629 (0.55 to 0.72) | 0.067 | 0.80 | 0.05 | 0.20 | 0.08 | n/a | 0.71 |
| XGBoost A snapshot | 0.510 (0.39 to 0.63) | 0.066 | 0.65 | 0.05 | 0.35 | 0.08 | 0.129 | 1.33 |
| XGBoost B + trends | 0.558 (0.45 to 0.68) | 0.053 | 0.57 | 0.05 | 0.49 | 0.09 | 0.083 | 1.72 |
| XGBoost C + core labs | 0.586 (0.47 to 0.70) | 0.056 | 0.55 | 0.05 | 0.54 | 0.10 | 0.077 | 1.80 |
| XGBoost D + lactate, INR | 0.569 (0.46 to 0.69) | 0.054 | 0.61 | 0.05 | 0.45 | 0.09 | 0.077 | 1.55 |
| Random Forest A snapshot | 0.531 (0.43 to 0.65) | 0.063 | 0.66 | 0.05 | 0.34 | 0.08 | 0.110 | 1.29 |
| Random Forest B + trends | 0.562 (0.47 to 0.66) | 0.051 | 0.55 | 0.05 | 0.51 | 0.09 | 0.075 | 1.82 |
| Random Forest C + core labs | 0.563 (0.45 to 0.68) | 0.055 | 0.58 | 0.05 | 0.44 | 0.08 | 0.068 | 1.68 |
| Random Forest D + lactate, INR | 0.559 (0.44 to 0.68) | 0.054 | 0.57 | 0.05 | 0.46 | 0.09 | 0.068 | 1.72 |

### Readmission

| Model / features | AUROC (95% range) | AUPRC | Accuracy | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|---|
| Average rate (reference) | 0.500 (0.50 to 0.50) | 0.164 | 0.16 | 0.16 | 1.00 | 0.28 | 0.137 |
| NEWS2 | 0.658 (0.50 to 0.76) | 0.292 | 0.68 | 0.26 | 0.52 | 0.35 | n/a |
| MEWS | 0.663 (0.51 to 0.80) | 0.352 | 0.77 | 0.33 | 0.38 | 0.36 | n/a |
| XGBoost A snapshot | 0.598 (0.42 to 0.74) | 0.244 | 0.53 | 0.19 | 0.56 | 0.28 | 0.172 |
| XGBoost B + trends | 0.559 (0.40 to 0.72) | 0.225 | 0.57 | 0.18 | 0.44 | 0.25 | 0.157 |
| XGBoost S + whole stay | 0.564 (0.40 to 0.73) | 0.261 | 0.53 | 0.18 | 0.50 | 0.26 | 0.154 |
| XGBoost C + core labs | 0.586 (0.43 to 0.73) | 0.257 | 0.52 | 0.17 | 0.50 | 0.25 | 0.148 |
| XGBoost D + lactate, INR | 0.579 (0.42 to 0.73) | 0.249 | 0.51 | 0.17 | 0.52 | 0.26 | 0.149 |
| Random Forest A snapshot | 0.585 (0.39 to 0.73) | 0.240 | 0.51 | 0.18 | 0.55 | 0.27 | 0.155 |
| Random Forest B + trends | 0.551 (0.35 to 0.74) | 0.265 | 0.53 | 0.18 | 0.49 | 0.26 | 0.147 |
| Random Forest S + whole stay | 0.554 (0.36 to 0.76) | 0.279 | 0.51 | 0.16 | 0.47 | 0.24 | 0.145 |
| Random Forest C + core labs | 0.571 (0.39 to 0.75) | 0.269 | 0.53 | 0.17 | 0.46 | 0.24 | 0.143 |
| Random Forest D + lactate, INR | 0.573 (0.39 to 0.75) | 0.271 | 0.51 | 0.17 | 0.50 | 0.25 | 0.143 |
AUROC range = 95% patient-level bootstrap. Brier is not defined for NEWS2/MEWS (raw scores are not probabilities).

### Does adding feature groups help? (paired comparison, change in AUROC with 95% range)

The same patients are scored under both configurations, so these ranges are tighter than comparing two separate ranges.

| Step | Deterioration XGBoost | Deterioration Random Forest | Readmission XGBoost | Readmission Random Forest |
|---|---|---|---|---|
| Add trends (A to B) | +0.049 (-0.042 to +0.142) | +0.031 (-0.054 to +0.111) | -0.040 (-0.148 to +0.076) | -0.034 (-0.181 to +0.118) |
| Add whole-stay summaries (B to S) | n/a | n/a | +0.005 (-0.049 to +0.057) | +0.004 (-0.032 to +0.036) |
| Add core labs (to C) | +0.028 (-0.021 to +0.074) | +0.000 (-0.067 to +0.057) | +0.022 (-0.024 to +0.063) | +0.016 (-0.032 to +0.054) |
| Add lactate and INR (C to D) | -0.016 (-0.036 to -0.002) | -0.004 (-0.013 to +0.006) | -0.008 (-0.017 to +0.001) | +0.003 (-0.011 to +0.017) |

Only one range excludes zero: lactate and INR slightly lowered XGBoost deterioration AUROC (a very small effect, consistent with those two labs being mostly missing). Everything else is "not distinguishable from no change".

## What the numbers mean

- **Alert burden (deterioration).** At the in-fold F1-best threshold NEWS2 raises 0.84 alerts per patient-day (catches 35% of events, 7% of alerts are followed by an event). XGBoost with trends raises 1.72 per day (catches 49%, 5% precision). The reference "alert for everyone" raises 4.0 per day. Precision at 80% sensitivity is 4 to 6% for every model, barely above the 4.4% event rate. No model here would be usable as an alarm; this is expected at 20 positive patients.
- **Accuracy is misleading** for rare events. "Never alert" would score 96% accuracy for deterioration; the F1-best threshold deliberately alerts more often, so the tree models show 55 to 66%. Read F1, recall and AUROC instead.
- **Brier score.** Because the models use class weights, their probabilities are inflated (Brier 0.07 to 0.13 against 0.042 for the constant average-rate reference). The ranking (AUROC) is unaffected, but the raw numbers must NOT be shown as "% risk" on the dashboard. Probability calibration is a Phase 6/7 task.
- **Why the bedside scores win here.** NEWS2/MEWS encode expert knowledge and need no training data. A tree model has to learn the same relationships from about 16 training patients with events per fold, with rows from one patient strongly correlated (about four consecutive time points per event), so it overfits patient-specific patterns. That is a data-size finding, not proof that ML cannot help; the published studies that report AUROC around 0.76 used thousands of patients.
- **Sanity check (labels shuffled).** XGBoost and Random Forest on feature set C: deterioration AUROC 0.47 and 0.49, readmission 0.42 and 0.46, all with ranges that include 0.5. Nothing in the pipeline produces above-chance scores by itself.

## Consequences for Part B and Part C

- Part B (LSTM) reuses `make_splits`, `evaluate` and `summarise` unchanged. Expect it to be no better than the tree models; the result will be reported either way.
- Part C applies the locked winner rule. With overlapping ranges the answer per task is likely "no clear winner, choose the simpler tree model". NEWS2/MEWS stay as baselines: the project can honestly say that on the demo cohort ML did not show a gain over the bedside scores, and why.
- The winner of each task is refitted on all data for SHAP and the dashboard (Phase 6 onwards). Because the winner is picked from the same results that are reported, its score is slightly optimistic.

## Limitations of this Part
- 20 deterioration and 15 readmission positive patients: all conclusions are low-power.
- One fixed hyperparameter setting per model; a different setting could score higher or lower (not explored on purpose, to avoid fitting noise).
- Alerts are not merged per event, and F1-best thresholds tend to over-alert on rare events.
- Internal validation only on the 100-patient demo.
