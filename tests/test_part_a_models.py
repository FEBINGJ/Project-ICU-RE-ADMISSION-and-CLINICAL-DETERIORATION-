"""Tests for feature-set definitions, the tree models and the baselines (need data/processed/ to exist)."""
import numpy as np
import pytest

from ml.config import DATA_PROCESSED
from ml.models.baselines import base_rate_fit_predict, score_fit_predict
from ml.models.feature_sets import TASKS, columns_for, design_matrix, load_task
from ml.models.trees import tree_fit_predict

needs_data = pytest.mark.skipif(not (DATA_PROCESSED / "deterioration_features.parquet").exists(),
                                reason="run the Phase 4 build first")
FORBIDDEN = {"news2", "mews", "subject_id", "hadm_id", "stay_id", "t", "index_time", "y",
             "readmit_30d", "readmit_7d", "readmit_48h", "discharge_location", "died_after_icu_in_hospital", "gap_h"}


@needs_data
@pytest.mark.parametrize("task", list(TASKS))
def test_feature_sets_are_nested_and_never_contain_labels_ids_or_scores(task):
    df, y, groups = load_task(task)
    prev = set()
    for name in TASKS[task]["sets"]:
        cols = set(columns_for(task, name, df.columns))
        assert prev <= cols and len(cols) > len(prev)             # each set contains the one before
        assert not cols & FORBIDDEN
        prev = cols
    assert not any(c.startswith("lactate") or c.startswith("inr") for c in columns_for(task, "C", df.columns))
    assert any(c.startswith("lactate") for c in columns_for(task, "D", df.columns))


@needs_data
def test_whole_stay_summaries_only_for_readmission():
    det, _, _ = load_task("deterioration")
    rea, _, _ = load_task("readmission")
    assert not any("__stay_" in c for c in columns_for("deterioration", "D", det.columns))
    assert not any("__stay_" in c for c in columns_for("readmission", "B", rea.columns))
    assert any("__stay_" in c for c in columns_for("readmission", "S", rea.columns))


@needs_data
def test_design_matrix_is_numeric_and_keeps_missing_values():
    df, _, _ = load_task("deterioration")
    X = design_matrix(df, columns_for("deterioration", "C", df.columns))
    assert X.dtypes.map(lambda t: t.kind == "f").all() and X.isna().any().any()
    assert not any(c == "first_careunit" for c in X.columns) and any(c.startswith("first_careunit_") for c in X.columns)


def test_tree_models_learn_a_signal_and_use_only_training_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] + 0.3 * rng.normal(size=300) > 1).astype(int)
    train, test = np.arange(200), np.arange(200, 300)
    for name in ["xgboost", "random_forest"]:
        p = tree_fit_predict(name, X, y)(train, test, 1)
        assert p.shape == (100,) and ((p >= 0) & (p <= 1)).all()
        order = np.argsort(p)
        assert y[test][order[-20:]].mean() > y[test][order[:20]].mean()
        # changing the labels of the TEST rows must not change the predictions
        y2 = y.copy()
        y2[test] = 1 - y2[test]
        np.testing.assert_allclose(p, tree_fit_predict(name, X, y2)(train, test, 1))


def test_baselines():
    s = np.array([1.0, np.nan, 3.0, 5.0, 7.0])
    out = score_fit_predict(s)(np.array([0, 2, 3]), np.array([1, 4]), 0)
    assert out[0] == 3.0 and out[1] == 7.0                          # missing -> training median (3.0)
    fp = base_rate_fit_predict(np.array([0, 0, 0, 1]))
    assert (fp(np.array([0, 1]), np.array([2, 3]), 0) == 0.25).all() and fp.fixed_threshold == 0.0
