"""Tests for the readmission features (at ICU discharge) and the LSTM sequence arrays."""
import numpy as np
import pandas as pd
import pytest

from ml.data.mimic_demo import INVASIVE_VENTILATION_ITEMID
from ml.features.readmission import build_readmission_features
from ml.features.sequences import MAX_GAP_H, STEPS, build_sequences, sequence_for

T0 = pd.Timestamp("2100-01-01 00:00")


def ts(hours):
    return T0 + pd.Timedelta(hours=hours)


def arrays(points):
    """points: (charttime_hour, avail_hour, value)"""
    points = sorted(points)
    return (np.array([ts(p[0]).to_datetime64() for p in points], dtype="datetime64[ns]"),
            np.array([ts(p[1]).to_datetime64() for p in points], dtype="datetime64[ns]"),
            np.array([p[2] for p in points], dtype=float))


# ------------------------------------------------------------------ sequences
def test_sequence_values_mask_and_time_since_last():
    # one measurement before the 24 h window, one at -5.5 h, one at -0.5 h
    x, m, d = sequence_for(*arrays([(-30, -30, 10.0), (-5.5, -5.5, 20.0), (-0.5, -0.5, 30.0)]), ts(0).to_datetime64())
    assert len(x) == STEPS
    assert x[0] == 10.0 and m[0] == 0 and d[0] == pytest.approx(7.0)     # carried in from before the window
    assert x[17] == 10.0                                                  # the -5.5 h value is not visible yet
    assert x[18] == 20.0 and m[18] == 1 and d[18] == pytest.approx(0.5)   # new measurement in this hour
    assert x[19] == 20.0 and m[19] == 0 and d[19] == pytest.approx(1.5)
    assert x[22] == 20.0 and d[22] == pytest.approx(4.5)
    assert x[23] == 30.0 and m[23] == 1 and d[23] == pytest.approx(0.5)


def test_sequence_for_a_variable_that_was_never_measured():
    x, m, d = sequence_for(*arrays([]), ts(0).to_datetime64())
    assert np.isnan(x).all() and (m == 0).all() and (d == MAX_GAP_H).all()


def test_sequence_ignores_the_future_and_results_stored_after_t():
    base = [(-10, -10, 5.0), (-3, -3, 6.0)]
    extra = [(2, 2, 999.0),          # measured after t
             (-1, 4, 888.0)]         # measured before t but only stored after t
    t = ts(0).to_datetime64()
    for a, b in zip(sequence_for(*arrays(base), t), sequence_for(*arrays(base + extra), t)):
        np.testing.assert_array_equal(a, b)


def test_build_sequences_shapes_and_separate_variables():
    anchors = pd.DataFrame([dict(stay_id=1, hadm_id=10, t=ts(0))])
    vitals = pd.DataFrame([(1, "hr", ts(-1), ts(-1), 80.0)], columns=["stay_id", "var", "charttime", "avail_time", "value"])
    labs = pd.DataFrame([(10, "sodium", ts(-5), ts(-4), 140.0)], columns=["hadm_id", "var", "charttime", "avail_time", "value"])
    out = build_sequences(anchors, vitals, labs, time_col="t")
    names = list(out["var_names"])
    assert out["X"].shape == (1, STEPS, len(names)) and out["M"].shape == out["X"].shape
    assert out["X"][0, -1, names.index("hr")] == 80.0
    assert out["X"][0, -1, names.index("sodium")] == 140.0
    assert np.isnan(out["X"][0, -1, names.index("rr")])             # never measured


# ------------------------------------------------------------------ readmission
def _readmission_inputs(extra_vitals=(), extra_labs=()):
    icu = pd.DataFrame([dict(subject_id=1, hadm_id=10, stay_id=1, intime=ts(0), outtime=ts(48)),
                        dict(subject_id=1, hadm_id=11, stay_id=2, intime=ts(100), outtime=ts(120))])
    adm = pd.DataFrame({"hadm_id": [10, 11], "admittime": [ts(-24), ts(90)]})
    proc = pd.DataFrame([dict(stay_id=1, itemid=INVASIVE_VENTILATION_ITEMID, starttime=ts(10), endtime=ts(30))])
    index = icu.assign(index_time=icu.outtime, icu_los_h=[48.0, 20.0], first_careunit="MICU", last_careunit="MICU",
                       gender="F", anchor_age=70, readmit_30d=[1, 0], readmit_7d=[1, 0], readmit_48h=[0, 0])
    vitals = pd.DataFrame([(1, "hr", ts(40), ts(40), 90.0), (1, "hr", ts(47), ts(47), 100.0)] + list(extra_vitals),
                          columns=["stay_id", "var", "charttime", "avail_time", "value"])
    labs = pd.DataFrame([(10, "creatinine", ts(20), ts(21), 1.0)] + list(extra_labs),
                        columns=["hadm_id", "var", "charttime", "avail_time", "value"])
    return index, icu, adm, proc, vitals, labs


def test_readmission_features_at_icu_discharge():
    feats = build_readmission_features(*_readmission_inputs()).set_index("stay_id")
    first, second = feats.loc[1], feats.loc[2]
    assert first["hr__last"] == 100.0 and first["hr__stay_max"] == 100.0 and first["hr__stay_mean"] == 95.0
    assert first["hours_hosp_admit_to_icu_discharge"] == pytest.approx(72.0)
    assert first["n_prior_icu_stays"] == 0 and np.isnan(first["hours_since_prev_icu_discharge"])
    assert second["n_prior_icu_stays"] == 1 and second["hours_since_prev_icu_discharge"] == pytest.approx(52.0)
    assert first["ever_ventilated"] == 1 and first["vent_hours"] == pytest.approx(20.0)
    assert first["on_vent_at_discharge"] == 0 and first["hours_since_extubation"] == pytest.approx(18.0)
    assert second["ever_ventilated"] == 0 and second["vent_hours"] == 0


def test_readmission_features_exclude_information_known_only_later():
    cols = build_readmission_features(*_readmission_inputs()).columns
    for leaky in ["discharge_location", "died_after_icu_in_hospital", "gap_h", "next_icu_in"]:
        assert leaky not in cols


def test_readmission_features_ignore_data_after_icu_discharge():
    base = build_readmission_features(*_readmission_inputs())
    later = build_readmission_features(*_readmission_inputs(
        extra_vitals=[(1, "hr", ts(60), ts(60), 999.0), (1, "hr", ts(47.5), ts(52), 777.0)],   # after / stored after
        extra_labs=[(10, "creatinine", ts(70), ts(71), 9.0)]))
    pd.testing.assert_frame_equal(base, later)
