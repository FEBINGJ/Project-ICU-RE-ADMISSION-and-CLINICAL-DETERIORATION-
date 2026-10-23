"""Clean vitals and labs from MIMIC-IV into tidy long tables.

Both tables have the columns: <key>, var, charttime, avail_time, value
    key        stay_id for vitals, hadm_id for labs
    charttime  when the measurement was taken
    avail_time when the result was stored and could first be seen by a clinician.
               Features must use avail_time (not charttime) to decide what was known at time t.
"""
from pathlib import Path

import pandas as pd

# ---- vitals (chartevents item ids) and plausible ranges (values outside are removed)
VITALS_NUMERIC = {
    "hr": ([220045], (20, 250)),
    "rr": ([220210], (4, 60)),
    "spo2": ([220277], (50, 100)),
    "sbp": ([220179, 220050], (40, 300)),   # non-invasive and arterial systolic pressure
    "map": ([220181, 220052], (20, 200)),   # non-invasive and arterial mean pressure
}
TEMP_F_ITEM, TEMP_C_ITEM = 223761, 223762
TEMP_RANGE_C = (30.0, 43.5)
GCS_ITEMS = {220739: "eye", 223900: "verbal", 223901: "motor"}
GCS_RANGE = (3, 15)
O2_DEVICE_ITEM = 226732   # any charted device (cannula, mask, tube ...) means supplemental oxygen

# ---- labs (labevents item ids) and plausible ranges
LABS_CORE = {
    "sodium": (50983, (100, 180)),
    "potassium": (50971, (1.5, 9)),
    "urea_nitrogen": (51006, (1, 200)),
    "creatinine": (50912, (0.1, 20)),
    "bicarbonate": (50882, (2, 50)),
    "hematocrit": (51221, (5, 70)),
    "glucose": (50931, (20, 1500)),
    "wbc": (51301, (0.1, 200)),
    "platelets": (51265, (1, 1500)),
    "hemoglobin": (51222, (2, 25)),
    "phosphate": (50970, (0.1, 20)),
    "calcium": (50893, (3, 20)),
}
LABS_EXTRA = {   # tested only as an add-on in the ablation
    "lactate": (50813, (0.2, 30)),
    "inr": (51237, (0.5, 15)),
}


def _tidy(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """One row per (key, var, charttime): average duplicates, keep the latest availability time."""
    out = (df.groupby([key, "var", "charttime"], as_index=False)
             .agg(avail_time=("avail_time", "max"), value=("value", "mean")))
    return out.sort_values([key, "var", "charttime"]).reset_index(drop=True)


def _in_range(df: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    lo = df["var"].map({v: r[0] for v, r in ranges.items()})
    hi = df["var"].map({v: r[1] for v, r in ranges.items()})
    return df[(df["value"] >= lo) & (df["value"] <= hi)]


def load_vitals(root: Path, stay_ids) -> pd.DataFrame:
    """Vital signs in long format, keyed by stay_id."""
    ce = pd.read_csv(root / "icu/chartevents.csv.gz",
                     usecols=["stay_id", "charttime", "storetime", "itemid", "valuenum", "value"],
                     parse_dates=["charttime", "storetime"])
    ce = ce[ce["stay_id"].isin(set(stay_ids))].copy()
    ce["avail_time"] = ce[["charttime", "storetime"]].max(axis=1)   # a result is never available before it was taken

    parts, ranges = [], {}
    # plain numeric vitals
    for var, (items, rng) in VITALS_NUMERIC.items():
        sub = ce[ce["itemid"].isin(items)]
        parts.append(pd.DataFrame({"stay_id": sub.stay_id, "var": var, "charttime": sub.charttime,
                                   "avail_time": sub.avail_time, "value": sub.valuenum}))
        ranges[var] = rng
    # temperature: convert Fahrenheit to Celsius, then merge
    for item, convert in [(TEMP_F_ITEM, lambda x: (x - 32) * 5 / 9), (TEMP_C_ITEM, lambda x: x)]:
        sub = ce[ce["itemid"] == item]
        parts.append(pd.DataFrame({"stay_id": sub.stay_id, "var": "temp", "charttime": sub.charttime,
                                   "avail_time": sub.avail_time, "value": convert(sub.valuenum)}))
    ranges["temp"] = TEMP_RANGE_C
    # GCS total = eye + verbal + motor, only when all three were charted at the same time
    g = ce[ce["itemid"].isin(GCS_ITEMS)].copy()
    g["part"] = g["itemid"].map(GCS_ITEMS)
    wide = g.pivot_table(index=["stay_id", "charttime"], columns="part", values="valuenum", aggfunc="mean")
    wide = wide.dropna(subset=["eye", "verbal", "motor"])
    avail = g.groupby(["stay_id", "charttime"])["avail_time"].max()
    gcs = pd.DataFrame({"value": wide[["eye", "verbal", "motor"]].sum(axis=1)}).join(avail).reset_index()
    gcs["var"] = "gcs"
    parts.append(gcs[["stay_id", "var", "charttime", "avail_time", "value"]])
    ranges["gcs"] = GCS_RANGE
    # oxygen device: a charted device means supplemental oxygen
    o2 = ce[(ce["itemid"] == O2_DEVICE_ITEM) & ce["value"].notna()]
    o2_df = pd.DataFrame({"stay_id": o2.stay_id, "var": "o2_device", "charttime": o2.charttime,
                          "avail_time": o2.avail_time, "value": 1.0})

    vit = pd.concat(parts, ignore_index=True).dropna(subset=["value"])
    vit = _in_range(vit, ranges)
    vit = pd.concat([vit, o2_df], ignore_index=True)
    return _tidy(vit, "stay_id")


def load_labs(root: Path, hadm_ids, include_extra: bool = True) -> pd.DataFrame:
    """Lab results in long format, keyed by hadm_id (labs belong to the hospital admission)."""
    labs = dict(LABS_CORE)
    if include_extra:
        labs.update(LABS_EXTRA)
    item_to_var = {item: var for var, (item, _) in labs.items()}
    ranges = {var: rng for var, (_, rng) in labs.items()}

    lab = pd.read_csv(root / "hosp/labevents.csv.gz",
                      usecols=["hadm_id", "itemid", "charttime", "storetime", "valuenum"],
                      parse_dates=["charttime", "storetime"])
    lab = lab[lab["hadm_id"].isin(set(hadm_ids)) & lab["itemid"].isin(item_to_var) & lab["valuenum"].notna()].copy()
    lab["avail_time"] = lab[["charttime", "storetime"]].max(axis=1)
    out = pd.DataFrame({"hadm_id": lab.hadm_id.astype("int64"), "var": lab.itemid.map(item_to_var),
                        "charttime": lab.charttime, "avail_time": lab.avail_time, "value": lab.valuenum})
    return _tidy(_in_range(out, ranges), "hadm_id")
