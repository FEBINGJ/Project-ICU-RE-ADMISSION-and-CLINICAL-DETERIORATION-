"""Frozen cross-validation folds shared by every model (XGBoost, Random Forest, LSTM, baselines).

Rules (locked in docs/scope.md):
  - repeated 5-fold cross-validation, 10 repeats
  - all rows of a patient are in the same fold (split by patient, never by row)
  - folds are stratified on the label, so every test fold contains some positives
  - the folds depend only on (labels, patient ids, seed), so every model sees exactly the same ones

Part B (LSTM) must call make_splits() with the same arguments and use the same fingerprint.
"""
import hashlib
from typing import Iterator, NamedTuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ml.config import RANDOM_SEED

N_SPLITS = 5
N_REPEATS = 10
INNER_SPLITS = 3          # used inside a training fold to choose the threshold


class Split(NamedTuple):
    repeat: int
    fold: int
    train: np.ndarray     # row positions (not patient ids)
    test: np.ndarray


def make_splits(y, groups, n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS,
                seed: int = RANDOM_SEED) -> list[Split]:
    """Patient-grouped, label-stratified repeated K-fold. Repeat r uses random_state = seed + r."""
    y, groups = np.asarray(y), np.asarray(groups)
    out = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            out.append(Split(r, k, tr, te))
    return out


def inner_splits(y, groups, train_idx, seed: int, n_splits: int = INNER_SPLITS) -> Iterator[tuple]:
    """Grouped, stratified split of ONE training fold (used only to pick the threshold).

    Yields (inner_train, inner_val) as row positions in the full table. Only rows of train_idx
    are ever used, so the outer test fold cannot influence the threshold.
    """
    y, groups, train_idx = np.asarray(y), np.asarray(groups), np.asarray(train_idx)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for a, b in cv.split(np.zeros(len(train_idx)), y[train_idx], groups[train_idx]):
        yield train_idx[a], train_idx[b]


def fingerprint(splits: list[Split]) -> str:
    """Short hash of all test-fold memberships. Two runs with the same fingerprint used the same folds."""
    h = hashlib.sha256()
    for s in splits:
        h.update(np.int64([s.repeat, s.fold]).tobytes())
        h.update(np.sort(s.test).astype(np.int64).tobytes())
    return h.hexdigest()[:12]
