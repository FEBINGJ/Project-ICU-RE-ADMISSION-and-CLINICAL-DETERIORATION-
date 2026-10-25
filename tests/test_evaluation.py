"""Tests for the shared evaluation setup: folds, threshold rule, metrics, runner."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ml.evaluation.cv import evaluate
from ml.evaluation.folds import fingerprint, inner_splits, make_splits
from ml.evaluation.metrics import ScoreView, best_f1_threshold, ranges_overlap, summarise


def toy(n_pat=40, rows=5, pos_pat=10, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_pat), rows)
    pos = set(rng.choice(n_pat, pos_pat, replace=False))
    y = np.array([1 if g in pos and rng.random() < 0.6 else 0 for g in groups])
    return y, groups


# ------------------------------------------------------------------ folds
def test_no_patient_in_both_train_and_test_and_every_row_tested_once_per_repeat():
    y, g = toy()
    splits = make_splits(y, g, n_splits=5, n_repeats=3)
    assert len(splits) == 15
    for s in splits:
        assert not set(g[s.train]) & set(g[s.test])
    for r in range(3):
        tested = np.concatenate([s.test for s in splits if s.repeat == r])
        assert sorted(tested) == list(range(len(y)))


def test_folds_are_reproducible_and_depend_on_seed():
    y, g = toy()
    assert fingerprint(make_splits(y, g, n_repeats=2)) == fingerprint(make_splits(y, g, n_repeats=2))
    assert fingerprint(make_splits(y, g, n_repeats=2)) != fingerprint(make_splits(y, g, n_repeats=2, seed=7))


def test_every_test_fold_has_positives():
    y, g = toy()
    for s in make_splits(y, g, n_repeats=3):
        assert y[s.test].sum() > 0


def test_inner_split_only_uses_training_rows():
    y, g = toy()
    s = make_splits(y, g, n_repeats=1)[0]
    used = set()
    for a, b in inner_splits(y, g, s.train, seed=1):
        used |= set(a) | set(b)
        assert not set(g[a]) & set(g[b])
    assert used == set(s.train) and not used & set(s.test)


# ------------------------------------------------------------------ threshold rule
def test_best_f1_threshold_simple_case():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.9])
    assert best_f1_threshold(y, p) == pytest.approx(0.6)          # catches both positives, no false alarm
    assert best_f1_threshold(np.zeros(4, int), p[:4]) == float("inf")


# ------------------------------------------------------------------ metrics match sklearn
def test_scoreview_matches_sklearn_including_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.round(rng.random(300) * 0.7 + 0.3 * y, 1)               # many ties
    m = ScoreView(y, p, (p >= 0.5).astype(int)).metrics()
    assert m["auroc"] == pytest.approx(roc_auc_score(y, p))
    assert m["auprc"] == pytest.approx(average_precision_score(y, p))
    assert m["brier"] == pytest.approx(brier_score_loss(y, p))
    pred = (p >= 0.5)
    assert m["accuracy"] == pytest.approx((pred == y).mean())
    assert m["precision"] == pytest.approx(y[pred].mean())
    assert m["recall"] == pytest.approx(pred[y == 1].mean())


def test_integer_weights_equal_repeating_rows():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 60)
    p = np.round(rng.random(60), 2)
    pred = (p > 0.5).astype(int)
    w = rng.integers(0, 4, 60)
    weighted = ScoreView(y, p, pred).metrics(w)
    rep = np.repeat(np.arange(60), w)
    plain = ScoreView(y[rep], p[rep], pred[rep]).metrics()
    for k in ["auroc", "auprc", "brier", "accuracy", "f1", "recall", "precision"]:
        assert weighted[k] == pytest.approx(plain[k])


def test_summarise_ranges_and_alert_rate():
    y, g = toy(n_pat=60, pos_pat=20)
    rng = np.random.default_rng(3)
    p = np.clip(0.3 * y + rng.random(len(y)) * 0.7, 0, 1)
    oof_p = np.vstack([p, p])
    out = summarise(y, g, oof_p, (oof_p > 0.7).astype(int), n_boot=100, hours_per_row=6)
    a = out["auroc"]
    assert a["ci_low"] <= a["mean"] <= a["ci_high"] and a["mean"] == pytest.approx(roc_auc_score(y, p))
    assert out["alerts_per_patient_day"]["mean"] == pytest.approx((p > 0.7).mean() * 4)
    assert ranges_overlap(out, out)


# ------------------------------------------------------------------ runner
def test_runner_never_shows_the_test_fold_to_the_threshold_step():
    y, g = toy()
    splits = make_splits(y, g, n_repeats=1)
    calls = []

    def fit_predict(train_idx, eval_idx, seed):
        calls.append((set(train_idx), set(eval_idx)))
        return np.random.default_rng(seed).random(len(eval_idx))

    evaluate(fit_predict, y, g, splits, n_jobs=1)
    # 5 folds x (3 inner calls + 1 outer call). Every inner call must stay inside ONE training fold.
    assert len(calls) == 5 * 4
    for k, s in enumerate(splits):
        inner, outer = calls[4 * k:4 * k + 3], calls[4 * k + 3]
        assert outer == (set(s.train), set(s.test))
        for tr, ev in inner:
            assert (tr | ev) <= set(s.train)


def test_runner_perfect_predictor_scores_perfectly_and_fixed_threshold_flags_everyone():
    y, g = toy()
    splits = make_splits(y, g, n_repeats=2)
    out = evaluate(lambda tr, ev, seed: y[ev].astype(float), y, g, splits, n_jobs=1)
    assert (out["oof_pred"] == y).all()
    base = lambda tr, ev, seed: np.full(len(ev), 0.2)
    base.fixed_threshold = 0.0
    assert evaluate(base, y, g, splits, n_jobs=1)["oof_pred"].all()
