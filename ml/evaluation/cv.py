"""The cross-validation runner. Model-agnostic: XGBoost, Random Forest, LSTM and the baselines all plug in
through one function with this signature:

    fit_predict(train_idx, eval_idx, seed) -> predicted probabilities (or scores) for the rows eval_idx

where train_idx / eval_idx are row positions in the table. The function must fit everything (imputer, scaler,
model) using ONLY the rows in train_idx.

For every outer fold the runner
  1. picks the alert threshold inside the training fold: an inner grouped 3-fold split of the TRAINING rows
     gives out-of-fold predictions, and the F1-best cut-off on those is the threshold;
  2. fits on the whole training fold and predicts the test fold;
  3. turns the test-fold probabilities into alerts with the threshold from step 1.
The test fold is never used for the threshold. A baseline that must flag everyone sets
`fit_predict.fixed_threshold = 0.0`.
"""
import numpy as np
from joblib import Parallel, delayed

from ml.evaluation.folds import Split, inner_splits
from ml.evaluation.metrics import best_f1_threshold


def choose_threshold(fit_predict, y, groups, train_idx, seed: int) -> float:
    fixed = getattr(fit_predict, "fixed_threshold", None)
    if fixed is not None:
        return float(fixed)
    pooled_idx, pooled_p = [], []
    for k, (a, b) in enumerate(inner_splits(y, groups, train_idx, seed=seed)):
        pooled_idx.append(b)
        pooled_p.append(np.asarray(fit_predict(a, b, seed + k)))
    idx, p = np.concatenate(pooled_idx), np.concatenate(pooled_p)
    return best_f1_threshold(np.asarray(y)[idx], p)


def _one_split(fit_predict, y, groups, s: Split):
    seed = 10_000 + 100 * s.repeat + s.fold
    thr = choose_threshold(fit_predict, y, groups, s.train, seed)
    p = np.asarray(fit_predict(s.train, s.test, seed), dtype=float)
    return s, thr, p


def evaluate(fit_predict, y, groups, splits: list[Split], n_jobs: int = 2) -> dict:
    """Run every split. Returns out-of-fold probabilities and alerts, shape (n_repeats, n_rows)."""
    y, groups = np.asarray(y), np.asarray(groups)
    results = Parallel(n_jobs=n_jobs)(delayed(_one_split)(fit_predict, y, groups, s) for s in splits)
    n_rep = max(s.repeat for s in splits) + 1
    oof_p = np.full((n_rep, len(y)), np.nan)
    oof_pred = np.zeros((n_rep, len(y)), dtype=int)
    thresholds = np.full((n_rep, max(s.fold for s in splits) + 1), np.nan)
    for s, thr, p in results:
        oof_p[s.repeat, s.test] = p
        oof_pred[s.repeat, s.test] = (p >= thr).astype(int)
        thresholds[s.repeat, s.fold] = thr
    assert not np.isnan(oof_p).any(), "some rows were never in a test fold"
    return dict(oof_p=oof_p, oof_pred=oof_pred, thresholds=thresholds)
