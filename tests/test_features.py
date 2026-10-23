"""Tests for window features, NEWS2/MEWS scores and, most importantly, the no-leakage rule."""
import numpy as np
import pandas as pd
import pytest

from ml.features.scores import mews, news2
from ml.features.windows import (LABS_CORE, build_deterioration_features, feature_groups,
                                 var_features, window_stats)

T0 = pd.Timestamp("2100-01-01 12:00")


def ts(hours):
    """Timestamp `hours` after T0 (negative = before)."""
    return T0 + pd.Timedelta(hours=hours)


def arrays(points):
    """points: (charttime_hour, avail_hour, value) -> sorted numpy arrays."""
    points = sorted(points)
    ch = np.array([ts(p[0]).to_datetime64() for p in points], dtype="datetime64[ns]")
    av = np.array([ts(p[1]).to_datetime64() for p in points], dtype="datetime64[ns]")
    return ch, av, np.array([p[2] for p in points], dtype=float)


# --------------------------------------------------------------- window stats
def test_window_stats_on_a_known_straight_line():
    s = window_stats(np.array([-3.0, -2.0, -1.0]), np.array([1.0, 2.0, 3.0]))
    assert s["slope"] == pytest.approx(1.0)          # +1 per hour
    assert s["delta"] == 2.0 and s["mean"] == 2.0 and s["min"] == 1.0 and s["max"] == 3.0
    assert s["std"] == pytest.approx(1.0)


def test_window_stats_with_one_or_zero_values():
    one = window_stats(np.array([-1.0]), np.array([5.0]))
    assert one["mean"] == 5.0 and np.isnan(one["slope"]) and np.isnan(one["std"]) and np.isnan(one["delta"])
    none = window_stats(np.array([]), np.array([]))
    assert all(np.isnan(x) for x in none.values())


def test_snapshot_features_and_time_since_last():
    ch, av, val = arrays([(-5, -5, 80.0), (-2, -2, 90.0)])
    f = var_features(ch, av, val, ts(0).to_datetime64(), (6,), ("mean",))
    assert f["last"] == 90.0 and f["tsl_h"] == pytest.approx(2.0)
    assert f["measured_24h"] == 1.0 and f["mean_6h"] == pytest.approx(85.0)


def test_variable_never_measured_gives_nan_and_zero_flag():
    ch, av, val = arrays([])
    f = var_features(ch, av, val, ts(0).to_datetime64(), (6,), ("mean",))
    assert np.isnan(f["last"]) and np.isnan(f["tsl_h"]) and f["measured_24h"] == 0.0


# ------------------------------------------------------------------- leakage
def test_values_after_the_prediction_time_are_ignored():
    base = [(-4, -4, 80.0), (-2, -2, 85.0), (-1, -1, 90.0)]
    future = [(1, 1, 999.0), (5, 5, -50.0)]
    t = ts(0).to_datetime64()
    a = var_features(*arrays(base), t, (6, 12), ("mean", "slope", "delta"))
    b = var_features(*arrays(base + future), t, (6, 12), ("mean", "slope", "delta"))
    assert a == b


def test_a_result_taken_before_t_but_stored_after_t_is_ignored():
    # drawn 2 h before t, but only stored (visible) 3 h after t, so it was unknown at t
    visible = [(-6, -5, 100.0)]
    late = [(-2, 3, 500.0)]
    t = ts(0).to_datetime64()
    f = var_features(*arrays(visible + late), t, (24,), ("max",))
    assert f["last"] == 100.0 and f["max_24h"] == 100.0


def _tiny_inputs(extra_vitals=(), extra_labs=()):
    windows = pd.DataFrame([dict(subject_id=1, hadm_id=10, stay_id=100, t=ts(0), y=0, anchor_age=70, gender="M",
                                 first_careunit="MICU", hours_since_icu_admit=6.0, on_vent=False)])
    rows = [(100, "hr", ts(-3), ts(-3), 80.0), (100, "hr", ts(-1), ts(-1), 90.0)] + list(extra_vitals)
    vitals = pd.DataFrame(rows, columns=["stay_id", "var", "charttime", "avail_time", "value"])
    lab_rows = [(10, "creatinine", ts(-10), ts(-9), 1.0)] + list(extra_labs)
    labs = pd.DataFrame(lab_rows, columns=["hadm_id", "var", "charttime", "avail_time", "value"])
    return windows, vitals, labs


def test_full_feature_table_does_not_change_when_future_data_is_added():
    w, v, l = _tiny_inputs()
    base = build_deterioration_features(w, v, l)
    w2, v2, l2 = _tiny_inputs(
        extra_vitals=[(100, "hr", ts(2), ts(2), 250.0), (100, "rr", ts(3), ts(3), 60.0)],
        extra_labs=[(10, "creatinine", ts(4), ts(5), 15.0), (10, "creatinine", ts(-1), ts(2), 9.0)])
    changed = build_deterioration_features(w2, v2, l2)
    pd.testing.assert_frame_equal(base, changed)


def test_feature_columns_are_grouped_for_ablations():
    w, v, l = _tiny_inputs()
    groups = feature_groups(build_deterioration_features(w, v, l).columns)
    assert "hr__last" in groups["vitals_snapshot"] and "hr__slope_6h" in groups["vitals_trend"]
    assert "creatinine__last" in groups["labs_snapshot"] and "lactate__last" in groups["labs_extra_snapshot"]
    assert set(groups["scores"]) == {"news2", "mews"}
    assert "creatinine" in LABS_CORE


# --------------------------------------------------------------------- scores
def test_news2_bands_at_their_boundaries():
    rr = np.array([8, 9, 11, 12, 20, 21, 24, 25])
    expected = [3, 1, 1, 0, 0, 2, 2, 3]
    others = dict(spo2=98, supp_o2=0, sbp=120, hr=70, gcs=15, temp=37.0)
    got = [news2(r, **others) for r in rr]
    assert got == expected
    sbp = np.array([90, 91, 100, 101, 110, 111, 219, 220])
    got = [news2(16, 98, 0, s, 70, 15, 37.0) for s in sbp]
    assert got == [3, 2, 2, 1, 1, 0, 0, 3]


def test_news2_and_mews_extreme_and_normal_patients():
    assert news2(16, 98, 0, 120, 70, 15, 37.0) == 0
    assert news2(26, 90, 1, 85, 135, 10, 34.5) == 20          # every component at its maximum
    assert mews(120, 70, 16, 37.0, 15) == 1                   # only RR 15-20 scores 1


def test_mews_maximum_is_14():
    assert mews(65, 135, 32, 39.0, 6) == 14


def test_scores_are_nan_when_an_input_is_missing():
    assert np.isnan(news2(np.nan, 98, 0, 120, 70, 15, 37.0))
    assert np.isnan(mews(120, 70, 16, np.nan, 15))
