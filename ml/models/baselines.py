"""Reference predictors evaluated with exactly the same folds, threshold rule and metrics as the ML models.

  news2 / mews : the clinical early-warning score itself is the "prediction" (higher = riskier).
                 The only thing learned is the fill value for a missing score (training-fold median)
                 and the alert threshold (F1-best on the training fold, like every other model).
  base_rate    : predicts the overall event rate for everyone, and alerts for everyone. This is a fixed reference
                 constant, not a trained model, so AUROC is exactly 0.5 and Brier is rate x (1 - rate). (A per-fold
                 training-fold rate would differ between folds and create a fake signal when predictions are pooled.)
"""
import numpy as np


def score_fit_predict(scores: np.ndarray):
    scores = np.asarray(scores, dtype=float)

    def fit_predict(train_idx, eval_idx, seed):
        fill = np.nanmedian(scores[train_idx])
        return np.where(np.isnan(scores[eval_idx]), fill, scores[eval_idx])

    return fit_predict


def base_rate_fit_predict(y: np.ndarray):
    rate = float(np.asarray(y).astype(float).mean())

    def fit_predict(train_idx, eval_idx, seed):
        return np.full(len(eval_idx), rate)

    fit_predict.fixed_threshold = 0.0          # alert for everyone
    return fit_predict
