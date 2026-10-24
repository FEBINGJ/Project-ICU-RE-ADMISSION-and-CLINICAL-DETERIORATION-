"""Features for the ICU readmission task, measured at ICU discharge (the index time).

Same no-leakage rule as the deterioration features: only measurements whose availability time is
at or before the ICU discharge time are used. Columns that are only known later (hospital discharge
location, whether the patient died on the ward, the gap to the next ICU stay) are never features.
"""
import numpy as np
import pandas as pd

from ml.data.mimic_demo import INVASIVE_VENTILATION_ITEMID
from ml.features import scores
from ml.features.clean import LABS_CORE, LABS_EXTRA
from ml.features.windows import (_EMPTY, HOUR, LAB_LOOKBACKS, LAB_STATS, LOOKBACKS, MAX_SCORE_AGE_H,
                                 VITAL_STATS, VITAL_VARS, _series_lookup, var_features, window_stats)

STAY_STATS = ("mean", "min", "max", "std")   # whole-stay summaries of each vital


def _icu_history(icu: pd.DataFrame) -> pd.DataFrame:
    """Earlier ICU stays of the same patient (known at the time of this discharge)."""
    d = icu.sort_values(["subject_id", "intime"]).copy()
    d["n_prior_icu_stays"] = d.groupby("subject_id").cumcount().astype(float)
    prev_out = d.groupby("subject_id")["outtime"].shift()
    d["hours_since_prev_icu_discharge"] = (d["intime"] - prev_out).dt.total_seconds() / 3600
    return d.set_index("stay_id")[["n_prior_icu_stays", "hours_since_prev_icu_discharge"]]


def _ventilation(proc: pd.DataFrame, intime, outtime) -> dict:
    """Invasive ventilation during the stay, using only episodes up to ICU discharge."""
    nan = float("nan")
    if proc.empty:
        return dict(ever_ventilated=0.0, vent_hours=0.0, on_vent_at_discharge=0.0, hours_since_extubation=nan)
    start = proc["starttime"].clip(lower=intime)
    end = proc["endtime"].clip(upper=outtime)
    hours = ((end - start).dt.total_seconds() / 3600).clip(lower=0).sum()
    on_vent = bool(((proc["starttime"] <= outtime) & (proc["endtime"] > outtime)).any())
    ended = proc.loc[proc["endtime"] <= outtime, "endtime"]
    since = (outtime - ended.max()).total_seconds() / 3600 if len(ended) and not on_vent else nan
    return dict(ever_ventilated=float((proc["starttime"] <= outtime).any()), vent_hours=float(hours),
                on_vent_at_discharge=float(on_vent), hours_since_extubation=since)


def build_readmission_features(index: pd.DataFrame, icu: pd.DataFrame, adm: pd.DataFrame, proc: pd.DataFrame,
                               vitals: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    """One row of features per index ICU stay.

    index: the readmission table from build_labels (one row per ICU stay, with index_time and labels).
    """
    vit = _series_lookup(vitals, "stay_id")
    lab = _series_lookup(labs, "hadm_id")
    history = _icu_history(icu)
    admit = adm.set_index("hadm_id")["admittime"]
    vent_all = proc[proc["itemid"] == INVASIVE_VENTILATION_ITEMID]
    vent_by_stay = {k: g for k, g in vent_all.groupby("stay_id")}
    lab_vars = list(LABS_CORE) + list(LABS_EXTRA)

    rows = []
    for s in index.itertuples():
        t = pd.Timestamp(s.index_time).to_datetime64()
        row = {"subject_id": s.subject_id, "hadm_id": s.hadm_id, "stay_id": s.stay_id,
               "index_time": s.index_time, "readmit_30d": s.readmit_30d,
               "readmit_7d": s.readmit_7d, "readmit_48h": s.readmit_48h,
               "age": s.anchor_age, "male": float(s.gender == "M"),
               "first_careunit": s.first_careunit, "last_careunit": s.last_careunit,
               "icu_los_h": s.icu_los_h,
               "hours_hosp_admit_to_icu_discharge": (s.outtime - admit[s.hadm_id]).total_seconds() / 3600}
        row.update(history.loc[s.stay_id].to_dict())
        row.update(_ventilation(vent_by_stay.get(s.stay_id, vent_all.iloc[0:0]), s.intime, s.outtime))

        latest = {}
        for var in VITAL_VARS:
            ch, av, val = vit.get((s.stay_id, var), _EMPTY)
            f = var_features(ch, av, val, t, LOOKBACKS, VITAL_STATS)
            row.update({f"{var}__{k}": x for k, x in f.items()})
            latest[var] = f["last"] if f["tsl_h"] <= MAX_SCORE_AGE_H else np.nan
            known = (av <= t) & (ch <= t)                      # whole-stay summary, still only what was known
            stay = window_stats((ch[known] - t) / HOUR, val[known])
            row.update({f"{var}__stay_{k}": stay[k] for k in STAY_STATS})
        o2 = var_features(*vit.get((s.stay_id, "o2_device"), _EMPTY), t, (), ())
        row["supp_o2"] = float(o2["tsl_h"] <= MAX_SCORE_AGE_H)
        for var in lab_vars:
            f = var_features(*lab.get((s.hadm_id, var), _EMPTY), t, LAB_LOOKBACKS, LAB_STATS)
            row.update({f"{var}__{k}": x for k, x in f.items()})
        row["news2"] = float(scores.news2(latest["rr"], latest["spo2"], row["supp_o2"], latest["sbp"],
                                          latest["hr"], latest["gcs"], latest["temp"]))
        row["mews"] = float(scores.mews(latest["sbp"], latest["hr"], latest["rr"], latest["temp"], latest["gcs"]))
        rows.append(row)
    return pd.DataFrame(rows)
