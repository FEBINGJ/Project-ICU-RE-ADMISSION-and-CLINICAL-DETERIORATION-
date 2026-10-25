"""Refit the two winning models on ALL the data and save them for the dashboard (Phase 5 Part C).

Run from the project root:   python -m ml.models.final
(the winners are read from reports/tables/part_c_results.json)

For each task the saved file models/<task>_<model>.joblib holds
    model        the fitted classifier (single model, so SHAP explains exactly what is served)
    columns      the exact input columns, in order (care units one-hot encoded)
    threshold    alert cut-off on the RAW score: the F1-best value from repeated 5-fold patient-grouped CV on all rows
    platt        (intercept, slope) that turns the raw score into a calibrated risk, learned on the same CV predictions
    cv           the cross-validated scores reported in docs/results_part_c.md (the honest estimate of performance)
The cross-validation models are not reused; this model is a fresh fit. Its own score on the training data is
optimistic and must never be quoted as performance.
"""
import datetime
import json

import joblib
import numpy as np
import sklearn
import xgboost

from ml.config import MODEL_DIR, PROJECT_ROOT, RANDOM_SEED
from ml.evaluation.calibrated import apply_platt, fit_platt
from ml.evaluation.folds import inner_splits
from ml.evaluation.metrics import ScoreView, best_f1_threshold
from ml.features.windows import feature_groups
from ml.models.factory import build
from ml.models.feature_sets import design_matrix
from ml.models.trees import RF_PARAMS, XGB_PARAMS, make_model

TABLES = PROJECT_ROOT / "reports" / "tables"
DUMMY_PREFIXES = ("first_careunit_", "last_careunit_")
N_CV_REPEATS = 5


def fit_final(task: str, model: str, seed: int = RANDOM_SEED) -> dict:
    if model not in ("xgboost", "random_forest"):
        raise SystemExit(f"{model} won for {task}; saving is implemented for the tree models only (SHAP in Phase 6 needs them).")
    b = build(task, model)
    X, y, groups = b["X"].to_numpy(), b["y"], b["groups"]
    # 1) threshold and calibration from repeated grouped CV predictions on all rows
    all_rows = np.arange(len(y))
    idx, ps = [], []
    for r in range(N_CV_REPEATS):
        for k, (a, v) in enumerate(inner_splits(y, groups, all_rows, seed=seed + r, n_splits=5)):
            idx.append(v)
            ps.append(b["fit_predict"](a, v, seed + 100 * r + k))
    idx, ps = np.concatenate(idx), np.concatenate(ps)
    threshold = best_f1_threshold(y[idx], ps)
    platt = fit_platt(ps, y[idx])
    # 2) the model that will be served: fitted on every row
    clf = make_model(model, y, seed)
    clf.fit(X, y)
    cv = json.loads((TABLES / "part_c_results.json").read_text())[task]["models"][model]
    return dict(task=task, model_name=model, feature_set="C", model=clf, columns=b["columns"], raw_columns=b["raw_columns"],
                threshold=float(threshold), platt=platt, prevalence=float(y.mean()), n_rows=int(len(y)),
                n_positive=int(y.sum()), n_patients=int(len(np.unique(groups))),
                cv=dict(auroc=cv["auroc"], auroc_ci=cv["auroc_ci"], f1=cv["f1"]),
                settings=XGB_PARAMS if model == "xgboost" else RF_PARAMS, created=datetime.date.today().isoformat(),
                versions=dict(xgboost=xgboost.__version__, sklearn=sklearn.__version__),
                note="Research prototype, not for clinical use. Cross-validated AUROC is close to chance; "
                     "calibrated risks stay near the average event rate.")


def prepare(artifact: dict, df):
    """Rows of the Phase 4 feature table -> the model's input matrix (same columns, same order)."""
    X = design_matrix(df, artifact["raw_columns"])
    out = X.reindex(columns=artifact["columns"])
    dummies = [c for c in artifact["columns"] if c.startswith(DUMMY_PREFIXES)]
    out[dummies] = out[dummies].fillna(0.0)                         # a care unit that is absent = 0, not missing
    return out


def predict(artifact: dict, df) -> dict:
    """Returns the raw score, the calibrated risk (0 to 1) and the alert flag for each row of df."""
    raw = artifact["model"].predict_proba(prepare(artifact, df).to_numpy(dtype=float))[:, 1]
    a, b = artifact["platt"]
    return dict(raw_score=raw, risk=apply_platt(raw, a, b), alert=(raw >= artifact["threshold"]).astype(int))


def main():
    final = {t: r["final"]["winner"] for t, r in json.loads((TABLES / "part_c_results.json").read_text()).items()}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for task, model in final.items():
        art = fit_final(task, model)
        path = MODEL_DIR / f"{task}_{model}.joblib"
        joblib.dump(art, path)
        b = build(task, model)
        out = predict(art, b["df"])
        apparent = ScoreView(b["y"], out["raw_score"], out["alert"]).metrics()["auroc"]
        print(f"{task}: saved {path.name} | {art['n_rows']} rows, {art['n_positive']} positives, {len(art['columns'])} input columns | "
              f"threshold {art['threshold']:.3f} | platt (a={art['platt'][0]:.2f}, b={art['platt'][1]:.2f}) | "
              f"calibrated risk range {out['risk'].min():.3f} to {out['risk'].max():.3f} (average {out['risk'].mean():.3f}) | "
              f"CV AUROC {art['cv']['auroc']:.3f} | training-data AUROC {apparent:.3f} (optimistic, do not quote)")


if __name__ == "__main__":
    main()
