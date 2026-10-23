"""NEWS2 and MEWS early-warning scores, used as clinical baselines.

Both functions take numpy arrays (or scalars) and return NaN when a required input is missing.

Simplifications, stated openly:
- NEWS2 uses SpO2 scale 1 only (scale 2 for patients with chronic hypercapnic failure is not available).
- Consciousness is derived from the Glasgow Coma Scale: GCS below 15 counts as "not alert".
  Ventilated patients usually have a low verbal score, so they score as impaired consciousness.
- MEWS maps GCS to AVPU as: 15 = Alert, 13-14 = Voice, 9-12 = Pain, 8 or less = Unresponsive.
"""
import numpy as np


def _score(x, conditions, values, default):
    """Band lookup: first matching condition wins; NaN stays NaN."""
    x = np.asarray(x, dtype=float)
    out = np.select([c(x) for c in conditions], values, default=default).astype(float)
    return np.where(np.isnan(x), np.nan, out)


def news2(rr, spo2, supp_o2, sbp, hr, gcs, temp):
    """National Early Warning Score 2 (Royal College of Physicians, 2017). Range 0-20."""
    parts = [
        _score(rr, [lambda x: x <= 8, lambda x: x <= 11, lambda x: x <= 20, lambda x: x <= 24], [3, 1, 0, 2], 3),
        _score(spo2, [lambda x: x <= 91, lambda x: x <= 93, lambda x: x <= 95], [3, 2, 1], 0),
        2.0 * np.asarray(supp_o2, dtype=float),
        _score(sbp, [lambda x: x <= 90, lambda x: x <= 100, lambda x: x <= 110, lambda x: x <= 219], [3, 2, 1, 0], 3),
        _score(hr, [lambda x: x <= 40, lambda x: x <= 50, lambda x: x <= 90, lambda x: x <= 110, lambda x: x <= 130],
               [3, 1, 0, 1, 2], 3),
        _score(gcs, [lambda x: x < 15], [3], 0),
        _score(temp, [lambda x: x <= 35.0, lambda x: x <= 36.0, lambda x: x <= 38.0, lambda x: x <= 39.0],
               [3, 1, 0, 1], 2),
    ]
    return np.sum(parts, axis=0)


def mews(sbp, hr, rr, temp, gcs):
    """Modified Early Warning Score (Subbe et al., 2001). Range 0-14."""
    parts = [
        _score(sbp, [lambda x: x <= 70, lambda x: x <= 80, lambda x: x <= 100, lambda x: x <= 199], [3, 2, 1, 0], 2),
        _score(hr, [lambda x: x < 40, lambda x: x <= 50, lambda x: x <= 100, lambda x: x <= 110, lambda x: x <= 129],
               [2, 1, 0, 1, 2], 3),
        _score(rr, [lambda x: x < 9, lambda x: x <= 14, lambda x: x <= 20, lambda x: x <= 29], [2, 0, 1, 2], 3),
        _score(temp, [lambda x: x < 35.0, lambda x: x <= 38.4], [2, 0], 2),
        _score(gcs, [lambda x: x >= 15, lambda x: x >= 13, lambda x: x >= 9], [0, 1, 2], 3),
    ]
    return np.sum(parts, axis=0)
