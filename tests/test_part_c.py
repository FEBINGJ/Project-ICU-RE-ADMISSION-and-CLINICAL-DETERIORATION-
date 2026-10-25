"""Tests for Part C: winner rule, matched-sensitivity alert burden, calibration, final model."""
import numpy as np
import pytest

from ml.config import DATA_PROCESSED, PROJECT_ROOT
from ml.evaluation.calibrated import apply_platt, evaluate_calibrated, fit_platt
from ml.evaluation.cv import evaluate
from ml.evaluation.folds import make_splits
from ml.evaluation.metrics import ScoreView
from ml.evaluation.part_c import choose_winner, operating_point

needs_results = pytest.mark.skipif(
    not ((DATA_PROCESSED / "readmission_features.parquet").exists()
         and (PROJECT_ROOT / "reports" / "tables" / "part_c_results.json").exists()),
    reason="run Phase 4 and Part C first")


# ------------------------------------------------------------------ winner rule
def never(best, other):
    return False


def test_clear_winner_when_everything_else_is_clearly_worse():
    out = choose_winner(dict(xgboost=0.70, random_forest=0.55, lstm=0.54), dict(xgboost=.3, random_forest=.2, lstm=.2),
                        lambda best, other: True)
    assert out["verdict"] == "clear winner" and out["winner"] == "xgboost"


def test_no_clear_winner_chooses_the_best_tree_even_if_the_lstm_scores_highest():
    out = choose_winner(dict(xgboost=0.58, random_forest=0.57, lstm=0.60), dict(xgboost=.3, random_forest=.2, lstm=.2), never)
    assert out["verdict"] == "no clear winner" and out["best"] == "lstm" and out["winner"] == "xgboost"


def test_clearly_worse_models_are_dropped_from_the_contenders():
    worse = lambda best, other: other == "random_forest"
    out = choose_winner(dict(xgboost=0.62, random_forest=0.50, lstm=0.60), dict(xgboost=.3, random_forest=.2, lstm=.2), worse)
    assert out["contenders"] == ["xgboost", "lstm"] and out["winner"] == "xgboost"


def test_lstm_wins_only_when_it_is_clearly_better_than_the_trees():
    out = choose_winner(dict(xgboost=0.55, random_forest=0.54, lstm=0.75), dict(xgboost=.3, random_forest=.2, lstm=.4),
                        lambda best, other: True)
    assert out["winner"] == "lstm"


def test_f1_breaks_a_tie_in_auroc():
    out = choose_winner(dict(xgboost=0.600, random_forest=0.602, lstm=0.50), dict(xgboost=.30, random_forest=.20, lstm=.1),
                        lambda best, other: other == "lstm")
    assert out["best"] == "xgboost"                       # within 0.005 AUROC: higher F1 decides


# ------------------------------------------------------------------ operating points
def test_operating_point_fewest_alerts_for_a_target_sensitivity():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    v = ScoreView(y, p, (p > 0.5).astype(int))
    share, precision = operating_point(v, np.ones(8), 2 / 3)       # need 2 of 3 events: top 3 scores
    assert share == pytest.approx(3 / 8) and precision == pytest.approx(2 / 3)
    share, precision = operating_point(v, np.ones(8), 1.0)         # all 3 events: top 6 scores
    assert share == pytest.approx(6 / 8) and precision == pytest.approx(3 / 6)


# ------------------------------------------------------------------ calibration
def test_platt_recovers_a_known_calibration_and_never_reverses_the_ranking():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 4000)
    true = 0.2 * p                                            # the score is 5x too high
    y = (rng.random(4000) < true).astype(int)
    a, b = fit_platt(p, y)
    cal = apply_platt(p, a, b)
    assert abs(cal.mean() - y.mean()) < 0.01 and b > 0
    assert np.all(np.diff(cal[np.argsort(p)]) >= 0)
    noise = rng.random(4000)                                  # no signal at all: the slope stays >= the floor
    _, b0 = fit_platt(noise, (rng.random(4000) < 0.1).astype(int))
    assert b0 >= 0.001


def test_calibrated_run_gives_the_same_alerts_as_the_normal_run():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(60), 4)
    y = (rng.random(240) < 0.2).astype(int)
    score = np.clip(0.3 * y + rng.random(240) * 0.7, 0, 1)

    def fit_predict(train_idx, eval_idx, seed):
        return score[eval_idx]

    splits = make_splits(y, groups, n_repeats=2)
    base = evaluate(fit_predict, y, groups, splits, n_jobs=1)
    cal = evaluate_calibrated(fit_predict, y, groups, splits, n_jobs=1)
    np.testing.assert_array_equal(base["oof_pred"], cal["oof_pred"])
    np.testing.assert_allclose(base["oof_p"], cal["oof_raw"])
    for s in splits:                                          # calibration keeps the order of patients inside a fold
        r, t = cal["oof_raw"][s.repeat, s.test], cal["oof_cal"][s.repeat, s.test]
        assert np.all(np.diff(t[np.argsort(r, kind="stable")]) >= -1e-12)


# ------------------------------------------------------------------ final model
@needs_results
def test_final_model_round_trip_and_row_by_row_prediction(tmp_path):
    import joblib
    from ml.models.final import fit_final, predict
    from ml.models.feature_sets import load_task
    art = fit_final("readmission", "xgboost")
    path = tmp_path / "m.joblib"
    joblib.dump(art, path)
    loaded = joblib.load(path)
    df, y, _ = load_task("readmission")
    full = predict(loaded, df)
    one = predict(loaded, df.iloc[[7]])                       # one patient alone: care-unit dummies must still line up
    assert full["risk"].shape == (len(df),) and ((full["risk"] > 0) & (full["risk"] < 1)).all()
    assert one["raw_score"][0] == pytest.approx(full["raw_score"][7])
    assert loaded["columns"] == art["columns"] and "news2" not in "".join(loaded["columns"]) and "mews" not in loaded["columns"]
    assert set(np.unique(full["alert"])) <= {0, 1} and 0 < full["alert"].mean() < 1
