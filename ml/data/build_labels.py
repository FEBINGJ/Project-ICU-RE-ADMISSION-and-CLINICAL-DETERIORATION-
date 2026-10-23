"""Phase 3: build the two labeled datasets and save them to data/processed/.

Run from the project root:
    python -m ml.data.build_labels

Outputs
    data/processed/readmission_index.parquet   one row per ICU stay (task: ICU readmission)
    data/processed/deterioration_windows.parquet   one row per 6-hourly time point (task: deterioration)
"""
import pandas as pd

from ml.config import DATA_PROCESSED
from ml.data.mimic_demo import (deterioration_windows, find_root, load_core,
                                readmission_index)


def build_readmission_table(adm: pd.DataFrame, icu: pd.DataFrame, pat: pd.DataFrame) -> pd.DataFrame:
    """Index ICU stays for the readmission task.

    Prediction time = ICU discharge (outtime). Primary label = readmit_30d.
    readmit_7d and readmit_48h are kept only for the sensitivity analysis.
    """
    df = readmission_index(adm, icu)
    df = df.merge(adm[["hadm_id", "discharge_location", "hospital_expire_flag"]], on="hadm_id", how="left")
    df = df.merge(pat[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
    df["index_time"] = df["outtime"]
    df["icu_los_h"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600
    # Patients who die on the ward after leaving the ICU cannot be readmitted. We keep them
    # (the iREAD study only excluded ICU deaths) but flag them for a sensitivity check.
    df["died_after_icu_in_hospital"] = (df["hospital_expire_flag"] == 1).astype(int)
    cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "index_time", "icu_los_h",
            "first_careunit", "last_careunit", "gender", "anchor_age", "discharge_location",
            "died_after_icu_in_hospital", "gap_h", "readmit_30d", "readmit_7d", "readmit_48h"]
    return df[cols].reset_index(drop=True)


def build_deterioration_table(adm: pd.DataFrame, icu: pd.DataFrame, proc: pd.DataFrame,
                              pat: pd.DataFrame) -> pd.DataFrame:
    """Prediction time points every 6 hours inside each ICU stay.

    Label y = starts invasive ventilation or dies within the next 24 hours.
    Features for a time point t may only use data recorded at or before t.
    """
    w = deterioration_windows(adm, icu, proc, step_h=6, horizon_h=24)
    w = w.merge(icu[["stay_id", "intime", "outtime", "first_careunit"]], on="stay_id", how="left")
    w = w.merge(pat[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
    w["hours_since_icu_admit"] = (w["t"] - w["intime"]).dt.total_seconds() / 3600
    cols = ["subject_id", "hadm_id", "stay_id", "t", "hours_since_icu_admit", "first_careunit",
            "gender", "anchor_age", "on_vent", "vent_next", "death_next", "y"]
    return w[cols].reset_index(drop=True)


def main() -> None:
    root = find_root()
    core = load_core(root)
    adm, pat, icu, proc = core["adm"], core["pat"], core["icu"], core["proc"]

    readmit = build_readmission_table(adm, icu, pat)
    deter = build_deterioration_table(adm, icu, proc, pat)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    readmit.to_parquet(DATA_PROCESSED / "readmission_index.parquet", index=False)
    deter.to_parquet(DATA_PROCESSED / "deterioration_windows.parquet", index=False)

    print("Readmission table:", len(readmit), "index stays,", readmit.subject_id.nunique(), "patients")
    for col in ["readmit_30d", "readmit_7d", "readmit_48h"]:
        pos = readmit[readmit[col] == 1]
        print(f"  {col}: {len(pos)} positive stays, {pos.subject_id.nunique()} patients")
    print("Deterioration table:", len(deter), "time points,", deter.subject_id.nunique(), "patients")
    pos = deter[deter.y == 1]
    print(f"  y=1: {len(pos)} time points ({deter.y.mean():.2%}), {pos.subject_id.nunique()} patients")
    print("Saved to", DATA_PROCESSED)


if __name__ == "__main__":
    main()
