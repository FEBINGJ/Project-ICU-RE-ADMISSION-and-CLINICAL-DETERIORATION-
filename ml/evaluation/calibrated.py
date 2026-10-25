"""Probability calibration inside the cross-validation folds (Phase 5 Part C).

The models use class weights, so their raw probabilities are far too high (the dashboard must not show them as "% risk").
Platt scaling (a 2-parameter logistic curve on the log-odds of the score) is learned on the INNER out-of-fold predictions
of the training fold, the same predictions that already choose the alert threshold, so the test fold is never used.
The slope is kept >= 0.001 so calibration can never reverse the ranking of patients.
Alerts are identical to the uncalibrated run (same threshold step, calibration is a monotone transform).

Run:  python -m ml.evaluation.calibrated        (winners are read from reports/tables/part_c_results.json)
Outputs: reports/tables/part_c_calibration.json, reports/figures/part_c_calibration.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize

from ml.config import FIGURES_DIR, PROJECT_ROOT
from ml.evaluation.folds import inner_splits, make_splits
from ml.evaluation.metrics import ScoreView, best_f1_threshold
from ml.models.factory import build

TABLES = PROJECT_ROOT / "reports" / "tables"


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p, y, min_slope=0.001):
    """Returns (intercept a, slope b): calibrated = sigmoid(a + b * logit(p)), b >= min_slope."""
    z, y = logit(p), np.asarray(y, dtype=float)

    def nll(ab):
        s = ab[0] + ab[1] * z
        return float(np.sum(np.logaddexp(0, s) - y * s)) + 1e-3 * ab[1] ** 2

    res = minimize(nll, x0=[0.0, 1.0], bounds=[(None, None), (min_slope, None)], method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def apply_platt(p, a, b):
    return 1.0 / (1.0 + np.exp(-(a + b * logit(p))))


def _one(fit_predict, y, groups, s):
    seed = 10_000 + 100 * s.repeat + s.fold                                   # same seeds as ml/evaluation/cv.py
    idx, ps = [], []
    for k, (a, b) in enumerate(inner_splits(y, groups, s.train, seed=seed)):
        idx.append(b)
        ps.append(np.asarray(fit_predict(a, b, seed + k)))
    idx, ps = np.concatenate(idx), np.concatenate(ps)
    thr = best_f1_threshold(y[idx], ps)
    a, b = fit_platt(ps, y[idx])
    p = np.asarray(fit_predict(s.train, s.test, seed), dtype=float)
    return s, thr, a, b, p


def evaluate_calibrated(fit_predict, y, groups, splits, n_jobs=2):
    y, groups = np.asarray(y), np.asarray(groups)
    res = Parallel(n_jobs=n_jobs)(delayed(_one)(fit_predict, y, groups, s) for s in splits)
    n_rep = max(s.repeat for s in splits) + 1
    raw, cal = np.full((n_rep, len(y)), np.nan), np.full((n_rep, len(y)), np.nan)
    pred = np.zeros((n_rep, len(y)), dtype=int)
    for s, thr, a, b, p in res:
        raw[s.repeat, s.test], cal[s.repeat, s.test] = p, apply_platt(p, a, b)
        pred[s.repeat, s.test] = (p >= thr).astype(int)
    return dict(oof_raw=raw, oof_cal=cal, oof_pred=pred, platt=[(a, b) for _, _, a, b, _ in res])


def reliability(y, p, bins=5):
    """Pooled over repeats: mean predicted risk vs observed event rate in equal-count bins."""
    y = np.tile(np.asarray(y), (p.shape[0], 1)).ravel()
    p = p.ravel()
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return [(float(p[which == k].mean()), float(y[which == k].mean())) for k in range(bins) if (which == k).any()]


def brier_range(y, groups, p, n_boot=1000, seed=0):
    patients, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    se = (p - np.asarray(y)[None, :]) ** 2                                      # (repeats, rows)
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))[inverse]
        draws.append(float((se * w).sum(axis=1).mean() / w.sum()))
    return float(se.mean()), [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main():
    winners = {t: r["final"]["winner"] for t, r in json.loads((TABLES / "part_c_results.json").read_text()).items()}
    out, curves = {}, {}
    for task, model in winners.items():
        b = build(task, model)
        y, groups = b["y"], b["groups"]
        r = evaluate_calibrated(b["fit_predict"], y, groups, make_splits(y, groups))
        prev = float(y.mean())
        braw, rraw = brier_range(y, groups, r["oof_raw"])
        bcal, rcal = brier_range(y, groups, r["oof_cal"])
        aur = lambda p: float(np.mean([ScoreView(y, a, d).metrics()["auroc"] for a, d in zip(p, r["oof_pred"])]))
        out[task] = dict(model=model, prevalence=prev, brier_reference=prev * (1 - prev),
                         brier_raw=braw, brier_raw_ci=rraw, brier_calibrated=bcal, brier_calibrated_ci=rcal,
                         auroc_raw=aur(r["oof_raw"]), auroc_calibrated=aur(r["oof_cal"]),
                         calibrated_risk_range=[float(np.percentile(r["oof_cal"], 5)), float(np.percentile(r["oof_cal"], 95))],
                         mean_platt_slope=float(np.mean([s[1] for s in r["platt"]])))
        curves[task] = dict(raw=reliability(y, r["oof_raw"]), cal=reliability(y, r["oof_cal"]))
        print(f"{task}/{model}: Brier raw {braw:.3f} -> calibrated {bcal:.3f} (always-average reference {prev * (1 - prev):.3f}); "
              f"AUROC {out[task]['auroc_raw']:.3f} -> {out[task]['auroc_calibrated']:.3f}; "
              f"calibrated risk 5th-95th percentile {out[task]['calibrated_risk_range'][0]:.3f} to {out[task]['calibrated_risk_range'][1]:.3f}", flush=True)
    (TABLES / "part_c_calibration.json").write_text(json.dumps(dict(summary=out, curves=curves), indent=1))

    fig, axes = plt.subplots(1, len(curves), figsize=(10, 4.2))
    for ax, (task, c) in zip(axes, curves.items()):
        hi = max(max(x for x, _ in c["raw"]), max(y for _, y in c["raw"]), 0.1) * 1.1
        ax.plot([0, hi], [0, hi], color="#9ca3af", ls="--", lw=1, label="perfect")
        ax.plot(*zip(*c["raw"]), "o-", color="#c2670f", label="raw (class-weighted)")
        ax.plot(*zip(*c["cal"]), "s-", color="#2563a8", label="calibrated")
        ax.set_xlabel("predicted risk")
        ax.set_ylabel("observed event rate")
        ax.set_title(f"{task.capitalize()} ({out[task]['model']})", loc="left", fontsize=10.5)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(color="#e5e7eb", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_c_calibration.png", dpi=160)
    print("Saved", TABLES / "part_c_calibration.json")


if __name__ == "__main__":
    main()
