"""Metrics, the in-fold threshold rule and confidence ranges. Shared by every model.

Two kinds of metrics:
  - threshold-free (AUROC, AUPRC, Brier): use the predicted probability only
  - at a threshold (Accuracy, Precision, Recall, F1, alerts per patient-day): use the 0/1 alert decision
    made with the threshold that was chosen INSIDE the training fold

Confidence ranges: every metric is computed once per repeat on the pooled out-of-fold predictions
(each row is predicted once per repeat). We report
  - mean over the repeats and the spread (SD) of the repeats  (variation caused by how folds were drawn)
  - a 95% patient-level bootstrap range (variation caused by WHICH patients are in the data):
    patients are resampled with replacement, every metric is recomputed (repeat-averaged), 2.5th to 97.5th percentile.
The bootstrap range is the honest one to quote; it is wide on 20 positive patients, and that is the point.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve

METRICS = ["accuracy", "precision", "recall", "f1", "auroc", "auprc", "brier"]
HOURS_PER_ROW = 6          # deterioration: one prediction every 6 hours -> one row covers 6 patient-hours


# --------------------------------------------------------------------------- threshold rule
def best_f1_threshold(y, p) -> float:
    """Threshold t (alert when p >= t) with the highest F1 on (y, p). Ties go to the higher t (fewer alerts).

    This is called ONLY on training-fold predictions, never on the test fold.
    """
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.sum() == 0:
        return float("inf")                       # nothing to find: never alert
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]               # the last pair has no threshold
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    best = np.flatnonzero(f1 >= f1.max() - 1e-12)
    return float(thr[best[-1]])


# --------------------------------------------------------------------------- weighted metric engine
def _point(y, pred, w):
    tp = float(np.sum(w * (pred == 1) * (y == 1)))
    fp = float(np.sum(w * (pred == 1) * (y == 0)))
    fn = float(np.sum(w * (pred == 0) * (y == 1)))
    tn = float(np.sum(w * (pred == 0) * (y == 0)))
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
    return dict(accuracy=(tp + tn) / n, precision=precision, recall=recall, f1=f1,
                alert_rate=(tp + fp) / n)


class ScoreView:
    """All metrics of one set of predictions (one repeat). The sorting is done once, so the
    bootstrap (which only changes the row weights) is fast. Matches sklearn (tests/test_evaluation.py)."""

    def __init__(self, y, p, pred):
        self.y, self.p, self.pred = np.asarray(y).astype(int), np.asarray(p, dtype=float), np.asarray(pred).astype(int)
        order = np.argsort(-self.p, kind="mergesort")           # descending score
        ps = self.p[order]
        new_group = np.r_[True, ps[1:] != ps[:-1]]               # ties form one threshold step
        self.order = order
        self.group = np.cumsum(new_group) - 1
        self.n_groups = int(self.group[-1]) + 1

    def metrics(self, w=None) -> dict:
        w = np.ones(len(self.y)) if w is None else np.asarray(w, dtype=float)
        out = _point(self.y, self.pred, w)
        yw = w * self.y
        nw = w * (1 - self.y)
        out["brier"] = float(np.sum(w * (self.p - self.y) ** 2) / w.sum())
        pos_tot, neg_tot = yw.sum(), nw.sum()
        if pos_tot == 0 or neg_tot == 0:
            out.update(auroc=float("nan"), auprc=float("nan"), p_at_recall80=float("nan"))
            return out
        pos_g = np.bincount(self.group, weights=yw[self.order], minlength=self.n_groups)
        neg_g = np.bincount(self.group, weights=nw[self.order], minlength=self.n_groups)
        # AUROC: positives ranked above negatives, ties count half (scan from the LOWEST score upwards)
        neg_below = np.cumsum(neg_g[::-1])[::-1] - neg_g          # negatives with a lower score than the group
        out["auroc"] = float(np.sum(pos_g * (neg_below + 0.5 * neg_g)) / (pos_tot * neg_tot))
        # AUPRC (average precision) and precision at 80% recall, scanning thresholds from high to low
        tp, fp = np.cumsum(pos_g), np.cumsum(neg_g)
        denom = tp + fp                                              # 0 for leading groups a bootstrap gave zero weight
        precision = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
        recall = tp / pos_tot
        out["auprc"] = float(np.sum(pos_g / pos_tot * precision))
        ok = recall >= 0.8 - 1e-12
        out["p_at_recall80"] = float(precision[ok].max()) if ok.any() else float("nan")
        return out


# --------------------------------------------------------------------------- summary with confidence ranges
def summarise(y, groups, oof_p, oof_pred, n_boot: int = 500, seed: int = 0, hours_per_row=None) -> dict:
    """oof_p, oof_pred: arrays (n_repeats, n_rows). Returns {metric: {mean, sd_repeats, ci_low, ci_high}}.

    hours_per_row: if given (deterioration = 6), adds alerts_per_patient_day = alert rate x 24 / hours_per_row.
    Also adds p_at_recall80 (precision when 80% of events are caught; descriptive, read off the pooled predictions).
    """
    y, groups = np.asarray(y), np.asarray(groups)
    views = [ScoreView(y, p, d) for p, d in zip(oof_p, oof_pred)]
    names = METRICS + ["p_at_recall80"] + (["alerts_per_patient_day"] if hours_per_row else [])

    def run(w):
        per_repeat = [v.metrics(w) for v in views]
        for m in per_repeat:
            if hours_per_row:
                m["alerts_per_patient_day"] = m["alert_rate"] * 24.0 / hours_per_row
        return {k: np.array([m[k] for m in per_repeat], dtype=float) for k in names}

    full = run(None)
    patients, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in names}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))
        res = run(counts[inverse].astype(float))
        for k in names:
            boot[k].append(np.nanmean(res[k]))
    out = {}
    for k in names:
        b = np.array(boot[k])
        out[k] = dict(mean=float(np.nanmean(full[k])), sd_repeats=float(np.nanstd(full[k], ddof=1)),
                      ci_low=float(np.nanpercentile(b, 2.5)), ci_high=float(np.nanpercentile(b, 97.5)))
    return out


def ranges_overlap(a: dict, b: dict, metric: str = "auroc") -> bool:
    """True when the 95% ranges of two summaries overlap (used by the winner rule: overlap = no clear winner)."""
    return not (a[metric]["ci_high"] < b[metric]["ci_low"] or b[metric]["ci_high"] < a[metric]["ci_low"])
