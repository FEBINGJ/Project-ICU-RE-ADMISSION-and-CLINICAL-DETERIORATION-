"""Low / Moderate / High risk bands (Phase 6).

Two cut-offs on the RAW model score, both found on training-fold predictions only:
    High      score >= the alert threshold (the F1-best cut-off used for alerts since Part A)
    Moderate  below the alert threshold, but at or above the highest cut-off that still catches 80% of the events
    Low       below that 80%-sensitivity cut-off
If the alert threshold is already below the 80%-sensitivity cut-off (the model must alert on a large share to find
the events), the Moderate band is empty.
"""
import numpy as np

from ml.evaluation.metrics import best_f1_threshold

BAND_NAMES = ["Low", "Moderate", "High"]
TARGET_SENSITIVITY = 0.8


def sensitivity_cutoff(y, p, target: float = TARGET_SENSITIVITY) -> float:
    """Highest cut-off t such that alerting at p >= t catches at least `target` of the events."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.sum() == 0:
        return float("inf")
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    recall = np.cumsum(ys) / ys.sum()
    last_of_tie = np.r_[ps[1:] != ps[:-1], True]               # a cut-off cannot split tied scores
    ok = np.flatnonzero((recall >= target - 1e-12) & last_of_tie)
    return float(ps[ok[0]])


def band_cutoffs(y, p) -> tuple[float, float]:
    """Returns (moderate_cutoff, high_cutoff) with moderate_cutoff <= high_cutoff."""
    high = best_f1_threshold(y, p)
    return min(sensitivity_cutoff(y, p), high), high


def assign_band(p, moderate: float, high: float) -> np.ndarray:
    """0 = Low, 1 = Moderate, 2 = High."""
    p = np.asarray(p, dtype=float)
    return np.where(p >= high, 2, np.where(p >= moderate, 1, 0))
