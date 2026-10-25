"""Tests for the LSTM: preprocessing is fitted on training rows only, shapes, row alignment, folds match Part A."""
import json

import numpy as np
import pytest

from ml.config import DATA_PROCESSED, PROJECT_ROOT
from ml.evaluation.folds import fingerprint, make_splits
from ml.models.lstm import Net, Prep, load_sequence_data, lstm_fit_predict

needs_data = pytest.mark.skipif(not (DATA_PROCESSED / "readmission_sequences.npz").exists(),
                                reason="run the Phase 4 build first")


def synthetic(n=120, steps=24, ch=4, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n // 3), 3)
    y = (rng.random(n) < 0.25).astype(int)
    X = rng.normal(size=(n, steps, ch)).astype(np.float32)
    X[:, :, 0] += y[:, None] * 1.5                          # a clear signal in channel 0
    X[rng.random(X.shape) < 0.2] = np.nan
    M = (rng.random(X.shape) < 0.3).astype(np.float32)
    D = rng.uniform(0, 48, size=X.shape).astype(np.float32)
    return dict(X=X, M=M, D=D, static=rng.normal(size=(n, 3)).astype(np.float32), y=y, groups=groups)


def test_prep_uses_only_the_rows_it_is_given():
    d = synthetic()
    tr = np.arange(80)
    a = Prep(d["X"][tr], d["static"][tr])
    # a different test fold must not change what was learned
    b = Prep(d["X"][tr], d["static"][tr])
    for k in ["med", "mu", "sd", "s_med", "s_mu", "s_sd"]:
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
    other = Prep(d["X"][40:], d["static"][40:])
    assert not np.allclose(a.mu, other.mu)                    # different rows -> different statistics


def test_prep_output_has_no_missing_values_and_expected_width():
    d = synthetic()
    p = Prep(d["X"], d["static"])
    seq = p.sequence(d["X"], d["M"], d["D"])
    assert seq.shape == (120, 24, 12) and not np.isnan(seq).any()           # 4 channels x (value, measured, time since)
    assert 0 <= seq[:, :, 8:].min() and seq[:, :, 8:].max() <= 1.0 + 1e-6    # scaled hours since last measurement
    assert not np.isnan(p.static_(d["static"])).any()


def test_network_output_shape():
    import torch
    out = Net(12, 3, 8, 0.1)(torch.zeros(5, 24, 12), torch.zeros(5, 3))
    assert out.shape == (5,)


def test_test_fold_labels_never_influence_the_predictions():
    d = synthetic()
    s = make_splits(d["y"], d["groups"], n_repeats=1)[0]
    fast = dict(n_seeds=1, max_epochs=3)
    p1 = lstm_fit_predict(d, **fast)(s.train, s.test, 5)
    d2 = {k: v.copy() for k, v in d.items()}
    d2["y"][s.test] = 1 - d2["y"][s.test]                     # flip the TEST labels
    p2 = lstm_fit_predict(d2, **fast)(s.train, s.test, 5)
    np.testing.assert_allclose(p1, p2)                        # fitting, scaling and early stopping saw only training rows


def test_lstm_learns_a_clear_signal():
    from sklearn.metrics import roc_auc_score
    d = synthetic(n=300)
    splits = make_splits(d["y"], d["groups"], n_repeats=1)
    s = splits[0]
    p = lstm_fit_predict(d, n_seeds=2, max_epochs=15)(s.train, s.test, 1)
    assert roc_auc_score(d["y"][s.test], p) > 0.75


def test_no_positives_in_training_gives_zeros():
    d = synthetic()
    d["y"][:] = 0
    assert (lstm_fit_predict(d)(np.arange(60), np.arange(60, 90), 1) == 0).all()


@needs_data
@pytest.mark.parametrize("task", ["deterioration", "readmission"])
def test_sequence_rows_align_with_the_tabular_table_and_channel_sets(task):
    allc, vit = load_sequence_data(task, "all"), load_sequence_data(task, "vitals")
    assert allc["X"].shape[2] == 19 and vit["X"].shape[2] == 7
    np.testing.assert_array_equal(allc["X"][:, :, :7], vit["X"])
    assert len(allc["y"]) == allc["static"].shape[0] and allc["static"].shape[1] >= 5


@needs_data
def test_same_folds_as_part_a():
    path = PROJECT_ROOT / "reports" / "tables" / "part_a_results.json"
    if not path.exists():
        pytest.skip("Part A results not found")
    part_a = json.loads(path.read_text())
    for task in ["deterioration", "readmission"]:
        d = load_sequence_data(task)
        assert fingerprint(make_splits(d["y"], d["groups"])) == part_a[task]["fingerprint"]
