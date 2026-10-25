# Phase 6: risk bands and explanations

Code: `ml/evaluation/bands.py`, `ml/evaluation/risk_bands.py` (bands), `ml/explain/names.py` (plain-language names), `ml/explain/explain.py` (SHAP explanations), `ml/explain/report.py` (global summary, stability, examples). Tests: `tests/test_explain.py`.
Re-run in this order: `python -m ml.evaluation.risk_bands`, `python -m ml.models.final`, `python -m ml.explain.report`.
Raw numbers: `reports/tables/phase6_bands.json`, `phase6_global.json`, `phase6_examples.json`. Figures: `reports/figures/phase6_global.png`, `phase6_examples.png`.

## Headline
1. **Risk bands are supported for deterioration only, and only weakly.** Observed deterioration rate: Low 3.3%, Moderate 3.8%, High 5.3% (average 4.4%). The order is right but the High-minus-Low gap (+2.0 points, range -3.6 to +6.2) is not confirmed. For readmission the model's bands do not rise in order (14.6%, 17.5%, 17.1%), so they are **not shown** for readmission.
2. **NEWS2's own standard bands separate real risk better than ours.** Deterioration: 1.5% (score 0 to 4), 4.2% (5 to 6), 6.0% (7 and above), a gap of +4.5 points with a range of +1.5 to +8.0. Readmission: 10.6%, 19.4%, 28.0% (gap +17 points, range 0.0 to +36). The dashboard should show NEWS2's bands as the clinical reference.
3. **The explanations are real but unstable.** For every patient the SHAP factors add up exactly to the model's score (tested). But when the model is refitted on resampled patients, the ranking of inputs changes a lot: rank correlation 0.51 (deterioration) and 0.47 (readmission), and only 46% and 32% of the top-10 inputs stay in the top 10. Only one input is stable: the variability of breathing rate over 24 hours (top 10 in 93% of deterioration refits). No readmission input is stable.
4. **What is stable is the type of information.** Both models rely mostly on trend features: vital-sign trends 55% (deterioration) and 53% (readmission) of the total attribution, lab trends 26% and 13%, whole-stay vitals 18% for readmission. Latest single values, measurement pattern and context are small. This agrees with the research question's design, but the use of trends did not translate into a better score than snapshot features or NEWS2 (see `results_part_a.md`).

## Risk bands
- **Rule (fixed before looking):** High = at or above the alert threshold; Moderate = below it but at or above the highest cut-off that still catches 80% of events; Low = below that. Both cut-offs are found on training-fold predictions only. "Useful" means the observed event rate rises from Low to Moderate to High.
- **Cross-validated result:**

| Task | Band | Share of rows | Observed event rate (95% range) |
|---|---|---|---|
| Deterioration | Low | 12% | 3.3% (0.3% to 8.5%) |
| | Moderate | 43% | 3.8% (2.2% to 5.4%) |
| | High | 45% | 5.3% (3.2% to 7.6%) |
| Readmission | Low | 29% | 14.6% (3.8% to 26.1%) |
| | Moderate | 23% | 17.5% (8.6% to 26.3%) |
| | High | 48% | 17.1% (6.8% to 26.6%) |

- **Consequence.** The saved deterioration model has `bands_supported = True`; readmission has `False`. Each explanation carries the observed rate of its band, so the dashboard never shows a colour without the real rate next to it. Almost half of all time points land in High, so the High band is a weak alert, not a rare one.

## How an explanation is built (`explain(artifact, rows, top_k=5)`)
Each prediction returns: the calibrated risk, the band (or none) with its observed rate, the top factors (default 5, any number allowed), the net push of each type of information, the combined push of all other inputs, a written explanation and a list of warnings. Factors carry a plain-language label, the patient's value, whether it raises or lowers the score, a relative size (large, medium or small, based on its share of everything that moves the score) and a note when relevant. Contributions are shown as direction and relative size, never as percentage points of risk.

Example (deterioration, patient with the highest score, in-sample): "The model estimates a 10.2% chance of starting invasive ventilation or dying within the next 24 hours (the average in the training data is 4.4%). This falls in the High band. In cross-validation, 5.3% of cases in the High band had the event (95% range 3.2% to 7.6%). Factors that pushed the score UP: variability of breathing rate over the last 24 h (5 breaths/min), lowest temperature in the last 24 h (35.2 °C), latest platelet count (33 K/µL), lowest systolic blood pressure in the last 24 h (80 mmHg) ... Factors that pushed the score DOWN: trend of heart rate over the last 24 h (-1.3 bpm per hour)."

## How to read an explanation
- The model starts from an average score. Each factor moves it up or down; all inputs together, including the many small ones, give the patient's score.
- The top factors usually make up only 30% to 40% of the total movement. The rest is spread over hundreds of small contributions. This is what a model with little signal looks like.
- These are the things the **model used**. They are not causes, and not proven risk factors.

## Measurement pattern and missing values
Inputs about how often something was recorded (hours since the last measurement, measured in the last 24 hours) are kept in their own group and carry a note, because they mostly reflect how closely staff watch a patient. A value that was never recorded also contributes (the model treats "not recorded" as information); such factors carry the same kind of note. Overall, 5% of the deterioration attribution and 3% of the readmission attribution comes from inputs with no recorded value, and the measurement-pattern group is 1% and 3%.

## Stability check (30 refits on resampled patients)
| | Deterioration | Readmission |
|---|---|---|
| Top 10 inputs share of all attribution | 40% | 33% |
| Rank correlation with the saved model (average, lowest) | 0.51, 0.37 | 0.47, 0.32 |
| Top-10 overlap with the saved model (average, lowest) | 46%, 20% | 32%, 10% |
| Inputs in the top 10 in at least 70% of refits | variability of breathing rate over 24 h | none |

Reading: report types of information (trends matter more than single values) with some confidence, individual inputs with little.

## Known issue for the demo (Phases 7 and 9)
The saved models were fitted on all 100 patients. If the dashboard replays those same patients, their scores are optimistic and overconfident (for example the highest readmission score shows 56.8% risk, well outside the cross-validated 6% to 25%). For the replay simulation we should show predictions from models that never saw that patient (cross-validated predictions, with explanations from the matching fold model) and use the saved models for new patients added through the "add new patient" flow.

## Limitations
- Explanations describe the model, not the disease; the model's own discrimination is close to chance.
- SHAP values here are computed on the data the models were fitted on.
- Plain-language units assume MIMIC's standard units.
- Interpretation of the stability check is limited by 20 and 15 positive patients.
