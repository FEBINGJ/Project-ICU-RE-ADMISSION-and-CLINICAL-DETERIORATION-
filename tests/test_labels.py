"""Tests for the label logic using tiny hand-made patients where the right answer is known."""
import pandas as pd

from ml.data.mimic_demo import (INVASIVE_VENTILATION_ITEMID, deterioration_windows,
                                readmission_index)

T0 = pd.Timestamp("2100-01-01 00:00")


def hours(h):
    return T0 + pd.Timedelta(hours=h)


def make_icu(rows):
    """rows: (subject_id, hadm_id, stay_id, in_hour, out_hour)"""
    return pd.DataFrame(
        [dict(subject_id=s, hadm_id=h, stay_id=st, intime=hours(i), outtime=hours(o))
         for s, h, st, i, o in rows])


def make_adm(hadm_ids, deaths=None):
    deaths = deaths or {}
    return pd.DataFrame({"hadm_id": hadm_ids, "deathtime": [deaths.get(h, pd.NaT) for h in hadm_ids]})


# ---------------------------------------------------------------- readmission
def test_readmission_windows_follow_the_gap_between_icu_stays():
    icu = make_icu([
        (1, 10, 1, 0, 24),      # discharged at hour 24 ...
        (1, 10, 2, 48, 72),     # ... back in the ICU 24 h later -> within 48 h
        (4, 40, 5, 0, 24),      # discharged at hour 24 ...
        (4, 41, 6, 124, 150),   # ... back 100 h later -> within 7 d and 30 d, not 48 h
        (3, 30, 4, 0, 24),      # never returns
    ])
    out = readmission_index(make_adm([10, 30, 40, 41]), icu).set_index("stay_id")
    assert out.loc[1, ["readmit_48h", "readmit_7d", "readmit_30d"]].tolist() == [1, 1, 1]
    assert out.loc[5, ["readmit_48h", "readmit_7d", "readmit_30d"]].tolist() == [0, 1, 1]
    assert out.loc[2, ["readmit_48h", "readmit_7d", "readmit_30d"]].tolist() == [0, 0, 0]
    assert out.loc[4, ["readmit_48h", "readmit_7d", "readmit_30d"]].tolist() == [0, 0, 0]


def test_readmission_excludes_stays_that_end_in_death():
    icu = make_icu([(2, 20, 3, 0, 24)])
    out = readmission_index(make_adm([20], deaths={20: hours(24)}), icu)
    assert out.empty


def test_readmission_gap_beyond_30_days_is_not_positive():
    icu = make_icu([(1, 10, 1, 0, 24), (1, 11, 2, 24 + 31 * 24, 24 + 32 * 24)])
    out = readmission_index(make_adm([10, 11]), icu).set_index("stay_id")
    assert out.loc[1, "readmit_30d"] == 0


# -------------------------------------------------------------- deterioration
def vent_events(start_hour, end_hour, stay_id=1):
    return pd.DataFrame([dict(stay_id=stay_id, itemid=INVASIVE_VENTILATION_ITEMID,
                              starttime=hours(start_hour), endtime=hours(end_hour))])


def test_deterioration_label_uses_the_next_24_hours_only():
    icu = make_icu([(1, 10, 1, 0, 48)])
    w = deterioration_windows(make_adm([10]), icu, vent_events(20, 44))
    w["h"] = w.t.sub(T0).dt.total_seconds().div(3600)   # hours since the ICU admission
    w = w.set_index("h")
    # time points at hours 6, 12, 18 see the ventilation start (hour 20) inside their 24 h horizon
    assert w.loc[[6, 12, 18], "y"].tolist() == [1, 1, 1]
    assert w.loc[[6, 12, 18], "on_vent"].tolist() == [False, False, False]
    # from hour 24 the patient is already ventilated and no NEW start follows, so y = 0
    assert w.loc[[24, 30, 36, 42], "y"].tolist() == [0, 0, 0, 0]
    assert w.loc[[24, 30, 36], "on_vent"].tolist() == [True, True, True]


def test_deterioration_death_counts_inside_horizon_but_not_at_the_time_point():
    icu = make_icu([(1, 10, 1, 0, 48)])
    adm = make_adm([10], deaths={10: hours(30)})
    no_vent = pd.DataFrame(columns=["stay_id", "itemid", "starttime", "endtime"])
    w = deterioration_windows(adm, icu, no_vent)
    w["h"] = w.t.sub(T0).dt.total_seconds().div(3600)
    w = w.set_index("h")
    assert w.loc[[6, 12, 18, 24], "y"].tolist() == [1, 1, 1, 1]  # death within next 24 h
    assert w.loc[30, "y"] == 0                                    # death exactly at t is not "after" t
    assert w.loc[30, "death_next"] == False                       # noqa: E712
