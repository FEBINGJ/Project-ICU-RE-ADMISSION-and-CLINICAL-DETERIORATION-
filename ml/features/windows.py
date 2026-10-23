"""Snapshot and trend features for one prediction time point.

Core rule (no leakage): a feature at time t may only use measurements whose availability time
is <= t. Everything measured or stored after t is invisible to the feature code.

Feature names look like  <variable>__<name>  for example  hr__last, hr__slope_12h, creatinine__delta_24h.
    snapshot features: last, tsl_h (hours since last measurement), measured_24h (1 if measured in last 24 h)
    trend features:    mean / min / max / std / slope / delta over a lookback window
                       (slope = change per hour from a straight-line fit, delta = last minus first value)
"""
import numpy as np
import pandas as pd

from ml.features import scores
from ml.features.clean import LABS_CORE, LABS_EXTRA

VITAL_VARS = ["hr", "rr", "spo2", "sbp", "map", "temp", "gcs"]
LOOKBACKS = (6, 12, 24)                       # hours, for vitals
VITAL_STATS = ("mean", "min", "max", "std", "slope", "delta")
LAB_LOOKBACKS = (24,)                         # labs are drawn about every 12-24 h, shorter windows are mostly empty
LAB_STATS = ("min", "max", "delta")
SNAPSHOT_NAMES = {"last", "tsl_h", "measured_24h"}
MAX_SCORE_AGE_H = 12                          # a value older than this is not used for NEWS2 / MEWS
HOUR = np.timedelta64(1, "h")


def window_stats(x: np.ndarray, v: np.ndarray) -> dict:
    """Summary of values v observed at hours x (x relative to the prediction time, both time-sorted)."""
    nan = float("nan")
    out = dict(mean=nan, min=nan, max=nan, std=nan, slope=nan, delta=nan)
    n = len(v)
    if n == 0:
        return out
    out.update(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
    if n >= 2:
        out["std"] = float(v.std(ddof=1))
        out["delta"] = float(v[-1] - v[0])
        dx = x - x.mean()
        den = float((dx ** 2).sum())
        if den > 0:
            out["slope"] = float((dx * (v - v.mean())).sum() / den)
    return out


def var_features(ch, av, val, t, lookbacks, stat_names) -> dict:
    """Features of one variable at time t.

    ch, av, val: charttime, availability time and value arrays, sorted by charttime.
    t: np.datetime64.
    """
    ok = (av <= t) & (ch <= t)                # only what was known at t
    c, v = ch[ok], val[ok]
    out = {"last": float(v[-1]) if len(v) else float("nan"),
           "tsl_h": float((t - c[-1]) / HOUR) if len(v) else float("nan")}
    out["measured_24h"] = float(bool(np.any(c > t - 24 * HOUR)))
    for hours in lookbacks:
        m = c > t - hours * HOUR
        stats = window_stats((c[m] - t) / HOUR, v[m])
        for name in stat_names:
            out[f"{name}_{hours}h"] = stats[name]
    return out


def _series_lookup(df: pd.DataFrame, key: str) -> dict:
    """(key, var) -> (charttime, avail_time, value) arrays sorted by charttime."""
    lookup = {}
    for (k, var), g in df.groupby([key, "var"]):
        g = g.sort_values("charttime")
        lookup[(k, var)] = (g["charttime"].to_numpy(), g["avail_time"].to_numpy(), g["value"].to_numpy(dtype=float))
    return lookup


_EMPTY = (np.array([], dtype="datetime64[ns]"), np.array([], dtype="datetime64[ns]"), np.array([], dtype=float))


def build_deterioration_features(windows: pd.DataFrame, vitals: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    """One row of features per prediction time point in `windows`."""
    vit = _series_lookup(vitals, "stay_id")
    lab = _series_lookup(labs, "hadm_id")
    lab_vars = list(LABS_CORE) + list(LABS_EXTRA)

    rows = []
    for w in windows.itertuples():
        t = pd.Timestamp(w.t).to_datetime64()
        row = {"subject_id": w.subject_id, "hadm_id": w.hadm_id, "stay_id": w.stay_id, "t": w.t, "y": w.y,
               "age": w.anchor_age, "male": float(w.gender == "M"), "first_careunit": w.first_careunit,
               "hours_since_icu_admit": w.hours_since_icu_admit, "on_vent": float(w.on_vent)}
        latest = {}
        for var in VITAL_VARS:
            f = var_features(*vit.get((w.stay_id, var), _EMPTY), t, LOOKBACKS, VITAL_STATS)
            row.update({f"{var}__{k}": x for k, x in f.items()})
            latest[var] = f["last"] if f["tsl_h"] <= MAX_SCORE_AGE_H else np.nan
        o2 = var_features(*vit.get((w.stay_id, "o2_device"), _EMPTY), t, (), ())
        row["supp_o2"] = float(o2["tsl_h"] <= MAX_SCORE_AGE_H)      # NaN comparison is False, so no device = 0
        for var in lab_vars:
            f = var_features(*lab.get((w.hadm_id, var), _EMPTY), t, LAB_LOOKBACKS, LAB_STATS)
            row.update({f"{var}__{k}": x for k, x in f.items()})
        row["news2"] = float(scores.news2(latest["rr"], latest["spo2"], row["supp_o2"], latest["sbp"],
                                          latest["hr"], latest["gcs"], latest["temp"]))
        row["mews"] = float(scores.mews(latest["sbp"], latest["hr"], latest["rr"], latest["temp"], latest["gcs"]))
        rows.append(row)
    return pd.DataFrame(rows)


def feature_groups(columns) -> dict:
    """Split feature columns into named groups so ablations can pick them (for example snapshot vs trend)."""
    groups = {k: [] for k in ["context", "vitals_snapshot", "vitals_trend", "labs_snapshot", "labs_trend",
                              "labs_extra_snapshot", "labs_extra_trend", "scores"]}
    core, extra = set(LABS_CORE), set(LABS_EXTRA)
    for col in columns:
        if col in ("age", "male", "first_careunit", "hours_since_icu_admit", "on_vent"):
            groups["context"].append(col)
        elif col in ("news2", "mews"):
            groups["scores"].append(col)
        elif col == "supp_o2":
            groups["vitals_snapshot"].append(col)
        elif "__" in col:
            var, name = col.split("__", 1)
            kind = "snapshot" if name in SNAPSHOT_NAMES else "trend"
            if var in VITAL_VARS:
                groups[f"vitals_{kind}"].append(col)
            elif var in core:
                groups[f"labs_{kind}"].append(col)
            elif var in extra:
                groups[f"labs_extra_{kind}"].append(col)
    return groups
