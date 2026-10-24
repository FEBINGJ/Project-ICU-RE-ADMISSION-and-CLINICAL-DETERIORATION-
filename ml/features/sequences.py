"""Input arrays for the LSTM: the 24 hours before a prediction time, as 24 hourly steps.

For every variable and every hourly step there are three numbers:
    X  the last known value up to the end of that hour (carried forward; NaN if never measured)
    M  1 if a new measurement arrived during that hour, else 0 (the "measured" mask)
    D  hours since the last measurement, capped at 48

Only measurements with availability time <= the prediction time are used, so nothing from the future
can enter. Missing values in X are filled later, inside the cross-validation folds, with training-fold medians.
"""
import numpy as np
import pandas as pd

from ml.features.clean import LABS_CORE
from ml.features.windows import _EMPTY, HOUR, VITAL_VARS, _series_lookup

SEQ_VARS = VITAL_VARS + list(LABS_CORE)      # 7 vitals + 12 core labs = 19 channels
STEPS = 24
MAX_GAP_H = 48.0


def sequence_for(ch, av, val, t, steps: int = STEPS):
    """X, M, D arrays of length `steps` for one variable at time t (t is np.datetime64)."""
    edges = t - (steps - np.arange(steps + 1)) * HOUR      # edges[0] = t - 24 h ... edges[steps] = t
    ok = (av <= t) & (ch <= t)
    c, v = ch[ok], val[ok]
    if len(c) == 0:
        return np.full(steps, np.nan), np.zeros(steps), np.full(steps, MAX_GAP_H)
    idx = np.searchsorted(c, edges[1:], side="right") - 1  # last measurement at or before each hour's end
    valid = idx >= 0
    safe = np.where(valid, idx, 0)
    last_time = c[safe]
    x = np.where(valid, v[safe], np.nan)
    d = np.where(valid, np.minimum((edges[1:] - last_time) / HOUR, MAX_GAP_H), MAX_GAP_H)
    m = (valid & (last_time > edges[:-1])).astype(float)    # the last measurement fell inside this hour
    return x, m, d


def build_sequences(anchors: pd.DataFrame, vitals: pd.DataFrame, labs: pd.DataFrame, time_col: str) -> dict:
    """Arrays for every anchor row (columns stay_id, hadm_id and the prediction time in `time_col`)."""
    vit = _series_lookup(vitals, "stay_id")
    lab = _series_lookup(labs, "hadm_id")
    n, k = len(anchors), len(SEQ_VARS)
    X = np.full((n, STEPS, k), np.nan, dtype=np.float32)
    M = np.zeros((n, STEPS, k), dtype=np.uint8)
    D = np.full((n, STEPS, k), MAX_GAP_H, dtype=np.float32)
    for i, a in enumerate(anchors.itertuples(index=False)):
        t = pd.Timestamp(getattr(a, time_col)).to_datetime64()
        for j, var in enumerate(SEQ_VARS):
            src = vit.get((a.stay_id, var)) if var in VITAL_VARS else lab.get((a.hadm_id, var))
            X[i, :, j], M[i, :, j], D[i, :, j] = sequence_for(*(src or _EMPTY), t)
    return {"X": X, "M": M, "D": D, "var_names": np.array(SEQ_VARS)}
