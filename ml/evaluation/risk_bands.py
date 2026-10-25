"""How well do the Low / Moderate / High bands separate real risk? (Phase 6, cross-validated)

For every outer fold, both cut-offs come from the inner out-of-fold predictions of the TRAINING rows, then the
test rows are placed in a band. We report, per band, the share of rows and the observed event rate with a 95%
patient-level range. NEWS2's own standard bands (0-4 low, 5-6 medium, 7+ high) are shown on the same rows.

A set of bands is called useful when the observed event rate rises from Low to Moderate to High.

Run:  python -m ml.evaluation.risk_bands      Output: reports/tables/phase6_bands.json
"""
import json

import numpy as np
from joblib import Parallel, delayed

from ml.config import PROJECT_ROOT
from ml.evaluation.bands import BAND_NAMES, assign_band, band_cutoffs
from ml.evaluation.folds import inner_splits, make_splits
from ml.models.factory import build

TABLES = PROJECT_ROOT / "reports" / "tables"
N_BOOT = 1000


def _one(fit_predict, y, groups, s):
    seed = 10_000 + 100 * s.repeat + s.fold                    # same seeds as cv.py and calibrated.py
    idx, ps = [], []
    for k, (a, b) in enumerate(inner_splits(y, groups, s.train, seed=seed)):
        idx.append(b)
        ps.append(np.asarray(fit_predict(a, b, seed + k)))
    moderate, high = band_cutoffs(y[np.concatenate(idx)], np.concatenate(ps))
    p = np.asarray(fit_predict(s.train, s.test, seed), dtype=float)
    return s, assign_band(p, moderate, high)


def rate_table(y, bands, w, n_levels=3):
    """Per level: share of rows and event rate, averaged over repeats (weights w = patient resampling)."""
    share, rate = np.zeros(n_levels), np.full(n_levels, np.nan)
    reps_rate = np.full((bands.shape[0], n_levels), np.nan)
    for r in range(bands.shape[0]):
        for k in range(n_levels):
            m = bands[r] == k
            if (w * m).sum() > 0:
                reps_rate[r, k] = (w * m * y).sum() / (w * m).sum()
        share += np.array([(w * (bands[r] == k)).sum() / w.sum() for k in range(n_levels)]) / bands.shape[0]
    return share, np.nanmean(reps_rate, axis=0)


def summarise_bands(y, groups, bands, names, n_boot=N_BOOT, seed=0):
    patients, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    share0, rate0 = rate_table(y, bands, np.ones(len(y)), len(names))
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))[inverse].astype(float)
        draws.append(rate_table(y, bands, w, len(names))[1])
    draws = np.array(draws)
    out = {}
    for k, n in enumerate(names):
        col = draws[:, k][~np.isnan(draws[:, k])]
        out[n] = dict(share_of_rows=float(share0[k]), event_rate=None if np.isnan(rate0[k]) else float(rate0[k]),
                      event_rate_ci=[float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))] if len(col) else None,
                      events=float(np.mean([(y[bands[r] == k]).sum() for r in range(bands.shape[0])])))
    diff = draws[:, -1] - draws[:, 0]
    diff = diff[~np.isnan(diff)]
    rates = [out[n]["event_rate"] for n in names if out[n]["event_rate"] is not None and out[n]["share_of_rows"] > 0]
    return out, dict(high_minus_low=float(rate0[-1] - rate0[0]), high_minus_low_ci=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                     rises_low_to_high=bool(all(a <= b + 1e-12 for a, b in zip(rates, rates[1:]))))


def news2_bands(df, y, groups):
    """NEWS2's standard clinical bands on the same rows (rows without a NEWS2 score are left out)."""
    s = df["news2"].to_numpy(float)
    ok = ~np.isnan(s)
    lvl = np.where(s >= 7, 2, np.where(s >= 5, 1, 0))[ok]
    out, extra = summarise_bands(y[ok], groups[ok], lvl[None, :], ["Low (0-4)", "Medium (5-6)", "High (7+)"])
    out["_rows_with_score"] = float(ok.mean())
    return out, extra


def main():
    winners = {t: r["final"]["winner"] for t, r in json.loads((TABLES / "part_c_results.json").read_text()).items()}
    result = {}
    for task, model in winners.items():
        b = build(task, model)
        y, groups = b["y"], b["groups"]
        splits = make_splits(y, groups)
        res = Parallel(n_jobs=2)(delayed(_one)(b["fit_predict"], y, groups, s) for s in splits)
        bands = np.zeros((max(s.repeat for s in splits) + 1, len(y)), dtype=int)
        for s, band in res:
            bands[s.repeat, s.test] = band
        table, extra = summarise_bands(y, groups, bands, BAND_NAMES)
        n2, n2extra = news2_bands(b["df"], y, groups)
        result[task] = dict(model=model, prevalence=float(y.mean()), bands=table, **extra, news2_standard_bands=n2,
                            news2_rises_low_to_high=n2extra["rises_low_to_high"], news2_high_minus_low=n2extra["high_minus_low"],
                            news2_high_minus_low_ci=n2extra["high_minus_low_ci"])
        print(f"\n{task} ({model}), average event rate {y.mean():.3f}")
        for n in BAND_NAMES:
            t = table[n]
            ci = t["event_rate_ci"]
            print(f"  {n:<9} {t['share_of_rows']:>5.0%} of rows, event rate "
                  f"{'n/a' if t['event_rate'] is None else format(t['event_rate'], '.3f')}"
                  f"{'' if ci is None else f' ({ci[0]:.3f} to {ci[1]:.3f})'}")
        print(f"  High minus Low: {extra['high_minus_low']:+.3f} ({extra['high_minus_low_ci'][0]:+.3f} to {extra['high_minus_low_ci'][1]:+.3f}); rises Low->High: {extra['rises_low_to_high']}")
        for n, t in n2.items():
            if n.startswith("_"):
                continue
            ci = t["event_rate_ci"]
            print(f"  NEWS2 {n:<13} {t['share_of_rows']:>5.0%} of rows, event rate "
                  f"{'n/a' if t['event_rate'] is None else format(t['event_rate'], '.3f')}{'' if ci is None else f' ({ci[0]:.3f} to {ci[1]:.3f})'}")
    (TABLES / "phase6_bands.json").write_text(json.dumps(result, indent=1))
    print("\nSaved", TABLES / "phase6_bands.json")


if __name__ == "__main__":
    main()
