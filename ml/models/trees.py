"""XGBoost and Random Forest with FIXED conservative settings (no tuning; see docs/scope.md).

Why fixed: with about 20 positive patients (deterioration) and 15 (readmission), tuning on the same data
mostly fits noise. Settings are chosen once, in advance, to be shallow and strongly regularised, and are the
same for both tasks and every feature set.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

XGB_PARAMS = dict(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.7, colsample_bytree=0.5,
                  min_child_weight=2, reg_lambda=10.0, tree_method="hist", n_jobs=1, verbosity=0)
RF_PARAMS = dict(n_estimators=300, max_depth=5, min_samples_leaf=5, max_features="sqrt",
                 class_weight="balanced_subsample", n_jobs=1)


def make_model(name: str, y_train: np.ndarray, seed: int):
    if name == "xgboost":
        pos = max(int(y_train.sum()), 1)
        neg = len(y_train) - int(y_train.sum())
        return XGBClassifier(scale_pos_weight=neg / pos, random_state=seed, **XGB_PARAMS)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **RF_PARAMS)
    raise ValueError(name)


def tree_fit_predict(name: str, X: np.ndarray, y: np.ndarray):
    """Returns fit_predict(train_idx, eval_idx, seed) for the CV runner. Everything is fitted on train_idx only."""
    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)

    def fit_predict(train_idx, eval_idx, seed):
        if y[train_idx].sum() == 0:                       # no positives to learn from: predict "never"
            return np.zeros(len(eval_idx))
        model = make_model(name, y[train_idx], seed)
        model.fit(X[train_idx], y[train_idx])
        return model.predict_proba(X[eval_idx])[:, 1]

    return fit_predict
