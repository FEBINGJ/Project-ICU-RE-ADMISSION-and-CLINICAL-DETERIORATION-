"""Loading helpers and label construction for the MIMIC-IV Clinical Database Demo.

These functions are reused by the EDA notebook now and by the cohort/label phase later,
so the labels are defined in exactly one place.
"""
from pathlib import Path

import pandas as pd

from ml.config import DATA_RAW

# chartevents item ids used in this project (MIMIC-IV d_items)
VITAL_ITEMS = {
    220045: "Heart Rate",
    220210: "Respiratory Rate",
    220277: "SpO2",
    220179: "NBP systolic",
    220050: "ABP systolic",
    223761: "Temperature F",
    223762: "Temperature C",
    223900: "GCS verbal",
    223901: "GCS motor",
    220739: "GCS eye",
    226732: "O2 delivery device",
}
INVASIVE_VENTILATION_ITEMID = 225792  # procedureevents


def find_root(search_dir: Path = DATA_RAW) -> Path:
    """Locate the folder that contains hosp/ and icu/, wherever the zip was extracted."""
    for path in search_dir.rglob("admissions.csv.gz"):
        return path.parent.parent
    raise FileNotFoundError(f"No MIMIC demo found under {search_dir}. Extract the zip into data/raw/.")


def load_core(root: Path) -> dict:
    """Load the small tables used for the cohort and labels."""
    adm = pd.read_csv(root / "hosp/admissions.csv.gz", parse_dates=["admittime", "dischtime", "deathtime"])
    pat = pd.read_csv(root / "hosp/patients.csv.gz")
    icu = pd.read_csv(root / "icu/icustays.csv.gz", parse_dates=["intime", "outtime"])
    proc = pd.read_csv(root / "icu/procedureevents.csv.gz", parse_dates=["starttime", "endtime"])
    return {"adm": adm, "pat": pat, "icu": icu, "proc": proc}


def readmission_index(adm: pd.DataFrame, icu: pd.DataFrame) -> pd.DataFrame:
    """One row per ICU stay that can be an 'index' stay for readmission.

    A stay is excluded if the patient died in the ICU (they cannot be readmitted).
    gap_h = hours from this ICU discharge to the same patient's next ICU admission (NaN if none).
    """
    df = icu.merge(adm[["hadm_id", "deathtime"]], on="hadm_id")
    df["died_in_icu"] = df.deathtime.notna() & (df.deathtime <= df.outtime + pd.Timedelta(hours=1))
    df = df.sort_values(["subject_id", "intime"]).reset_index(drop=True)
    df["next_icu_in"] = df.groupby("subject_id")["intime"].shift(-1)
    df["gap_h"] = (df.next_icu_in - df.outtime).dt.total_seconds() / 3600
    df = df[~df.died_in_icu].copy()
    for name, hours in [("readmit_48h", 48), ("readmit_7d", 7 * 24), ("readmit_30d", 30 * 24)]:
        df[name] = (df.gap_h <= hours).astype(int)
    return df


def deterioration_windows(adm: pd.DataFrame, icu: pd.DataFrame, proc: pd.DataFrame,
                          step_h: int = 6, horizon_h: int = 24) -> pd.DataFrame:
    """Prediction time points every `step_h` hours during each ICU stay.

    Label y = 1 if the patient starts invasive ventilation or dies within `horizon_h` hours
    after the time point. `on_vent` marks time points where the patient is already ventilated.
    """
    icu = icu.merge(adm[["hadm_id", "deathtime"]], on="hadm_id")
    vent = proc[proc.itemid == INVASIVE_VENTILATION_ITEMID]
    vent_by_stay = {k: v for k, v in vent.groupby("stay_id")}
    empty = vent.iloc[0:0]
    horizon = pd.Timedelta(hours=horizon_h)
    rows = []
    for s in icu.itertuples():
        v = vent_by_stay.get(s.stay_id, empty)
        for t in pd.date_range(s.intime + pd.Timedelta(hours=step_h), s.outtime, freq=f"{step_h}h"):
            on_vent = bool(((v.starttime <= t) & (v.endtime > t)).any())
            vent_next = bool(((v.starttime > t) & (v.starttime <= t + horizon)).any())
            death_next = bool(pd.notna(s.deathtime) and (t < s.deathtime <= t + horizon))
            rows.append((s.subject_id, s.hadm_id, s.stay_id, t, on_vent, vent_next, death_next))
    w = pd.DataFrame(rows, columns=["subject_id", "hadm_id", "stay_id", "t", "on_vent", "vent_next", "death_next"])
    w["y"] = (w.vent_next | w.death_next).astype(int)
    return w
