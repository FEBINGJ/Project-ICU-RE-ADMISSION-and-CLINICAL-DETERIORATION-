"""Plain-language names, units and groups for every model input (Phase 6).

describe("hr__slope_12h", 2.1) -> {label, value_text, group, note}

Groups (used for the "which kind of information mattered" summary):
    Latest vital signs      Vital-sign trends       Latest lab results      Lab trends (24 h)
    Measurement pattern     Whole-stay vitals       Patient and stay context
"Measurement pattern" (time since a value was last recorded, was it recorded) mostly reflects how closely staff
watch a patient, not the patient's physiology, so it is kept apart and carries a warning note.
"""
import math

VITALS = {"hr": ("heart rate", "bpm", 0), "rr": ("breathing rate", "breaths/min", 0), "spo2": ("oxygen saturation (SpO2)", "%", 0),
          "sbp": ("systolic blood pressure", "mmHg", 0), "map": ("mean blood pressure", "mmHg", 0),
          "temp": ("temperature", "°C", 1), "gcs": ("Glasgow Coma Scale score", "points", 0)}
LABS = {"sodium": ("sodium", "mmol/L", 0), "potassium": ("potassium", "mmol/L", 1), "urea_nitrogen": ("urea nitrogen", "mg/dL", 0),
        "creatinine": ("creatinine", "mg/dL", 1), "bicarbonate": ("bicarbonate", "mmol/L", 0), "hematocrit": ("hematocrit", "%", 0),
        "glucose": ("glucose", "mg/dL", 0), "wbc": ("white blood cell count", "K/µL", 1), "platelets": ("platelet count", "K/µL", 0),
        "hemoglobin": ("hemoglobin", "g/dL", 1), "phosphate": ("phosphate", "mg/dL", 1), "calcium": ("calcium", "mg/dL", 1),
        "lactate": ("lactate", "mmol/L", 1), "inr": ("INR", "", 1)}

G_LATEST_V, G_TREND_V, G_LATEST_L, G_TREND_L = "Latest vital signs", "Vital-sign trends", "Latest lab results", "Lab trends (24 h)"
G_PATTERN, G_STAY, G_CONTEXT = "Measurement pattern", "Whole-stay vitals", "Patient and stay context"
GROUPS = [G_LATEST_V, G_TREND_V, G_LATEST_L, G_TREND_L, G_PATTERN, G_STAY, G_CONTEXT]
PATTERN_NOTE = "Reflects how often staff recorded this, not the patient's physiology."
MISSING_NOTE = "No value was recorded. The model treats 'not recorded' as information, which mostly reflects how closely staff monitor the patient, not physiology."

CONTEXT = {  # column: (label, unit/kind, decimals)
    "age": ("Age", "years", 0), "male": ("Sex", "sex", 0), "hours_since_icu_admit": ("Time since ICU admission", "h", 0),
    "on_vent": ("On a ventilator now", "yesno", 0), "supp_o2": ("Supplemental oxygen in use (device charted in last 12 h)", "yesno", 0),
    "icu_los_h": ("Length of the ICU stay", "h", 0), "hours_hosp_admit_to_icu_discharge": ("Time from hospital admission to ICU discharge", "h", 0),
    "n_prior_icu_stays": ("Earlier ICU stays in this patient's records", "count", 0),
    "hours_since_prev_icu_discharge": ("Time since the previous ICU discharge", "h", 0), "ever_ventilated": ("Was on a ventilator during the ICU stay", "yesno", 0),
    "vent_hours": ("Hours on a ventilator during the ICU stay", "h", 0), "on_vent_at_discharge": ("On a ventilator at ICU discharge", "yesno", 0),
    "hours_since_extubation": ("Time since the breathing tube was removed", "h", 0)}
UNIT_SHORT = {"Cardiac Vascular Intensive Care Unit (CVICU)": "cardiac-vascular ICU (CVICU)", "Coronary Care Unit (CCU)": "coronary care unit (CCU)",
               "Medical Intensive Care Unit (MICU)": "medical ICU (MICU)", "Medical/Surgical Intensive Care Unit (MICU/SICU)": "medical/surgical ICU",
               "Neuro Intermediate": "neuro intermediate unit", "Neuro Stepdown": "neuro step-down unit",
               "Neuro Surgical Intensive Care Unit (Neuro SICU)": "neuro surgical ICU", "Surgical Intensive Care Unit (SICU)": "surgical ICU (SICU)",
               "Trauma SICU (TSICU)": "trauma surgical ICU (TSICU)"}


def _num(v, decimals=0, signed=False):
    s = f"{v:+.{decimals}f}" if signed else f"{v:.{decimals}f}"
    return s


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def describe(column: str, value=None) -> dict:
    """Plain-language label, formatted value, group and (optional) note for one model input."""
    out = _describe(column, value)
    if _missing(value) and not out["note"] and column not in CONTEXT and not column.startswith(("first_careunit_", "last_careunit_")):
        out["note"] = MISSING_NOTE
    return out


def _describe(column: str, value=None) -> dict:
    out = dict(column=column, note="")
    # --- care units (one-hot columns)
    for prefix, word in (("first_careunit_", "First ICU care unit"), ("last_careunit_", "Last ICU care unit before discharge")):
        if column.startswith(prefix):
            unit = column[len(prefix):]
            out.update(label=f"{word}: {UNIT_SHORT.get(unit, unit)}", group=G_CONTEXT,
                       value_text="not available" if _missing(value) else ("yes" if value >= 0.5 else "no"))
            return out
    # --- context columns
    if column in CONTEXT:
        label, kind, dec = CONTEXT[column]
        out.update(label=label, group=G_CONTEXT)
        if _missing(value):
            out["value_text"] = "not available" if column != "hours_since_prev_icu_discharge" else "no earlier ICU stay"
        elif kind == "yesno":
            out["value_text"] = "yes" if value >= 0.5 else "no"
        elif kind == "sex":
            out["value_text"] = "male" if value >= 0.5 else "female"
        elif kind == "count":
            out["value_text"] = _num(value, 0)
        else:
            out["value_text"] = f"{_num(value, dec)} {kind}"
        return out
    # --- <variable>__<statistic>
    var, stat = column.split("__", 1)
    if var in VITALS:
        name, unit, dec = VITALS[var]
        lab = False
    elif var in LABS:
        name, unit, dec = LABS[var]
        lab = True
    else:
        raise KeyError(f"no plain-language name for input column '{column}'")
    u = f" {unit}" if unit else ""
    latest, trend = (G_LATEST_L, G_TREND_L) if lab else (G_LATEST_V, G_TREND_V)
    miss = _missing(value)
    if stat == "last":
        out.update(label=f"Latest {name}", group=latest, value_text="never measured so far" if miss else f"{_num(value, dec)}{u}")
    elif stat == "tsl_h":
        out.update(label=f"Time since {name} was last measured", group=G_PATTERN, note=PATTERN_NOTE,
                   value_text="never measured so far" if miss else f"{_num(value, 1)} h")
    elif stat == "measured_24h":
        out.update(label=f"{name.capitalize()} measured in the last 24 h", group=G_PATTERN, note=PATTERN_NOTE,
                   value_text="not available" if miss else ("yes" if value >= 0.5 else "no"))
    elif stat.startswith("stay_"):
        kind = {"mean": "Average", "min": "Lowest", "max": "Highest", "std": "Variability (standard deviation) of"}[stat[5:]]
        out.update(label=f"{kind} {name} over the whole ICU stay", group=G_STAY,
                   value_text="not available" if miss else f"{_num(value, 2 if stat == 'stay_std' else dec)}{u}")
    else:
        what, window = stat.rsplit("_", 1)
        w = window.replace("h", " h")
        dd = dec if what != "slope" else dec + 1
        text = {"mean": f"Average {name} over the last {w}", "min": f"Lowest {name} in the last {w}",
                "max": f"Highest {name} in the last {w}", "std": f"Variability (standard deviation) of {name} over the last {w}",
                "slope": f"Trend of {name} over the last {w}", "delta": f"Change in {name} over the last {w} (last value minus first)"}[what]
        if miss:
            vt = "no value: too few measurements in this window"
        elif what == "slope":
            vt = f"{_num(value, dd, True)}{u} per hour"
        elif what == "delta":
            vt = f"{_num(value, dd, True)}{u}"
        else:
            vt = f"{_num(value, dd)}{u}"
        out.update(label=text, group=trend, value_text=vt)
    return out
