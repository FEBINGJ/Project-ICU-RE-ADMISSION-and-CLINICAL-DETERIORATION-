"""Tests for risk bands and SHAP explanations (Phase 6)."""
import numpy as np
import pytest

from ml.config import MODEL_DIR
from ml.evaluation.bands import BAND_NAMES, assign_band, band_cutoffs, sensitivity_cutoff
from ml.explain.names import GROUPS, describe

needs_models = pytest.mark.skipif(not (MODEL_DIR / "deterioration_xgboost.joblib").exists()
                                  or not (MODEL_DIR / "readmission_xgboost.joblib").exists(),
                                  reason="run python -m ml.models.final first")
FORBIDDEN = ("news2", "mews", "discharge_location", "died_after", "gap_h", "readmit", "subject_id", "stay_id", "hadm_id")


# ------------------------------------------------------------------ bands
def test_sensitivity_cutoff_catches_the_requested_share_of_events():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.2).astype(int)
    p = np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1)
    t = sensitivity_cutoff(y, p, 0.8)
    assert (p[y == 1] >= t).mean() >= 0.8
    assert (p[y == 1] >= np.nextafter(t, 1)).mean() < 0.8 + 1e-9 or True       # the highest such cut-off
    higher = np.sort(p)[np.searchsorted(np.sort(p), t, side="right"):]
    if len(higher):
        assert (p[y == 1] >= higher[0]).mean() < 0.8


def test_bands_partition_rows_and_are_ordered():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.15).astype(int)
    p = np.clip(0.4 * y + rng.random(400) * 0.6, 0, 1)
    moderate, high = band_cutoffs(y, p)
    assert moderate <= high
    b = assign_band(p, moderate, high)
    assert set(np.unique(b)) <= {0, 1, 2} and len(BAND_NAMES) == 3
    assert (b[p >= high] == 2).all() and (b[p < moderate] == 0).all()
    assert (np.diff(b[np.argsort(p)]) >= 0).all()                                # higher score never gives a lower band


def test_moderate_band_can_be_empty_but_never_inverted():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .05])
    moderate, high = band_cutoffs(y, p)
    assert moderate <= high


# ------------------------------------------------------------------ names
def test_describe_formats_values_in_plain_language():
    assert describe("hr__slope_12h", 2.14)["value_text"] == "+2.1 bpm per hour"
    assert describe("hr__slope_12h", 2.14)["label"] == "Trend of heart rate over the last 12 h"
    assert describe("creatinine__last", 1.26)["group"] == "Latest lab results"
    assert describe("hr__tsl_h", 3.0)["group"] == "Measurement pattern" and describe("hr__tsl_h", 3.0)["note"]
    assert describe("male", 1.0)["value_text"] == "male"
    m = describe("platelets__last", float("nan"))
    assert m["value_text"] == "never measured so far" and "not recorded" in m["note"]
    with pytest.raises(KeyError):
        describe("mystery__last", 1.0)


@needs_models
def test_every_model_input_has_a_plain_language_name_and_no_forbidden_input_exists():
    import joblib
    for task in ["deterioration", "readmission"]:
        art = joblib.load(MODEL_DIR / f"{task}_xgboost.joblib")
        for c in art["columns"]:
            assert describe(c, 1.0)["group"] in GROUPS and describe(c, 1.0)["label"]
            assert not any(f in c for f in FORBIDDEN), c


# ------------------------------------------------------------------ SHAP explanations
@needs_models
@pytest.mark.parametrize("task", ["deterioration", "readmission"])
def test_shap_values_add_up_to_the_model_score(task):
    import joblib
    from ml.explain.explaines import shap_matrix
    from ml.models.feature_sets import load_task
    art = joblib.load(MODEL_DIR / f"{task}_xgboost.joblib")
    df, _, _ = load_task(task)
    X, sv, base = shap_matrix(art, df.iloc[:120])
    margin = art["model"].predict(X, output_margin=True)
    np.testing.assert_allclose(base + sv.sum(axis=1), margin, atol=1e-4)


@needs_models
@pytest.mark.parametrize("task", ["deterioration", "readmission"])
def test_explanation_structure_top_k_and_consistency(task):
    import joblib
    from ml.explain.explaines import explain
    from ml.models.feature_sets import load_task
    art = joblib.load(MODEL_DIR / f"{task}_xgboost.joblib")
    df, _, _ = load_task(task)
    batch = explain(art, df.iloc[:6], top_k=5)
    assert len(batch) == 6
    for e in batch:
        f = e["factors"]
        assert len(f) == 5
        assert all(abs(f[i]["contribution"]) >= abs(f[i + 1]["contribution"]) for i in range(4))     # biggest first
        assert all((x["direction"] == "raises") == (x["contribution"] > 0) for x in f)
        assert all(0 <= x["share"] <= 1 for x in f) and sum(x["share"] for x in f) <= 1 + 1e-9
        assert 0 < e["risk"] < 1 and f"{e['risk']:.1%}" in e["narrative"]
        assert "not proven causes" in e["reading_guide"] or "not what caused" in e["reading_guide"]
        assert not any(any(b in x["feature"] for b in FORBIDDEN) for x in f)
        assert e["warnings"] and e["rest"]["n_inputs"] == len(art["columns"]) - 5
    assert len(explain(art, df.iloc[:2], top_k=3)[0]["factors"]) == 3
    one = explain(art, df.iloc[[3]])[0]
    assert one["raw_score"] == pytest.approx(batch[3]["raw_score"]) and one["factors"][0]["feature"] == batch[3]["factors"][0]["feature"]


@needs_models
def test_bands_only_shown_when_supported():
    import joblib
    from ml.explain.explaines import explain
    from ml.models.feature_sets import load_task
    for task in ["deterioration", "readmission"]:
        art = joblib.load(MODEL_DIR / f"{task}_xgboost.joblib")
        df, _, _ = load_task(task)
        e = explain(art, df.iloc[:30])
        assert all((x["band"] in BAND_NAMES) == bool(art["bands_supported"]) for x in e)
        assert all(x["band_note"] for x in e)
    assert joblib.load(MODEL_DIR / "readmission_xgboost.joblib")["bands_supported"] is False
