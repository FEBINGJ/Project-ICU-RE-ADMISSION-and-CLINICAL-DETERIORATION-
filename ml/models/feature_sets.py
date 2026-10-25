"""Loading the Phase 4 tables and choosing the feature set for an experiment.

Feature sets (locked in docs/scope.md). Each set contains the one before it:
    A  snapshot + context          (last value, hours since last measurement, measured flag, care context)
    B  A + vitals trends           (mean / min / max / std / slope / change over 6, 12 and 24 h)
    S  B + whole-stay vitals       (readmission only: summaries of the entire ICU stay)
    C  B (or S) + 12 core labs     (last value + 24 h min / max / change)
    D  C + lactate and INR         (ablation: these two are only 48% / 79% complete)
NEWS2 and MEWS are never model inputs; they are separate baselines.
"""
import numpy as np
import pandas as pd

from ml.config import DATA_PROCESSED
from ml.features.windows import feature_groups

TASKS = {
    "deterioration": dict(file="deterioration_features.parquet", label="y", hours_per_row=6,
                          sets=["A", "B", "C", "D"]),
    "readmission": dict(file="readmission_features.parquet", label="readmit_30d", hours_per_row=None,
                        sets=["A", "B", "S", "C", "D"]),
}

SET_DESCRIPTION = {
    "A": "snapshot + context",
    "B": "A + vitals trends",
    "S": "B + whole-stay vitals (readmission only)",
    "C": "core labs added (12 labs: last value + 24 h trend)",
    "D": "C + lactate and INR",
}

CATEGORICAL = ["first_careunit", "last_careunit"]


def load_task(task: str):
    """Returns (table, y, groups). groups = patient id (subject_id); y = the task's label."""
    cfg = TASKS[task]
    path = DATA_PROCESSED / cfg["file"]
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python -m ml.data.build_labels  then  python -m ml.features.build_features")
    df = pd.read_parquet(path)
    return df, df[cfg["label"]].to_numpy().astype(int), df["subject_id"].to_numpy()


def columns_for(task: str, name: str, columns) -> list[str]:
    """Column names (before one-hot encoding) of feature set `name`."""
    g = feature_groups(columns)
    cols = g["context"] + g["vitals_snapshot"]                     # A
    if name in ("B", "S", "C", "D"):
        cols += g["vitals_trend"]                                   # B
    if name in ("S", "C", "D") and task == "readmission":
        cols += g["vitals_stay"]                                    # S (readmission only)
    if name in ("C", "D"):
        cols += g["labs_snapshot"] + g["labs_trend"]                # C
    if name == "D":
        cols += g["labs_extra_snapshot"] + g["labs_extra_trend"]    # D
    return cols


def design_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Numeric matrix for the models. Care units are one-hot encoded; missing values stay as NaN
    (both XGBoost and scikit-learn's Random Forest handle them directly, so nothing is imputed here)."""
    X = df[cols].copy()
    cats = [c for c in CATEGORICAL if c in X.columns]
    if cats:
        X = pd.get_dummies(X, columns=cats, dtype=float)
    return X.astype(float)
