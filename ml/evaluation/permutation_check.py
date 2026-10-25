"""Sanity check with many shuffled-label runs (a single shuffle can look "above chance" by luck).

Labels are shuffled at the patient level (readmission) or row level (deterioration), the whole pipeline is run, and the
AUROC is recorded. Over many shuffles the average must sit near 0.5; the spread shows how far one lucky shuffle can go.

Run from the project root (all options are optional; the defaults are shown):
    python -m ml.evaluation.permutation_check
    python -m ml.evaluation.permutation_check --task readmission --model lstm --shuffles 20
    python -m ml.evaluation.permutation_check --task deterioration --model xgboost --shuffles 6 --offset 5000

    --task      deterioration | readmission                 (default readmission)
    --model     lstm | xgboost | random_forest              (default lstm)
    --shuffles  how many different shuffles to run          (default 20)
    --offset    seed offset, change it to get new shuffles  (default 1000)
    --repeats   cross-validation repeats per shuffle        (default 2)
    --compare   AUROC of the real result to compare with    (default 0.598, the best real readmission result)
"""
import argparse

import numpy as np

from ml.evaluation.cv import evaluate
from ml.evaluation.folds import make_splits
from ml.evaluation.metrics import ScoreView
from ml.models.feature_sets import TASKS, columns_for, design_matrix, load_task
from ml.models.lstm import load_sequence_data, lstm_fit_predict
from ml.models.trees import tree_fit_predict


def shuffled(y, groups, task, rng):
    if task == "readmission":
        patients = np.unique(groups)
        lab = dict(zip(patients, rng.permutation([int(y[groups == p].max()) for p in patients])))
        return np.array([lab[g] for g in groups])
    return rng.permutation(y)


def main(task="readmission", model="lstm", n_shuffles=20, n_repeats=2, offset=1000, compare=0.598):
    df, y0, groups = load_task(task)
    data = load_sequence_data(task) if model == "lstm" else None
    X = None if model == "lstm" else design_matrix(df, columns_for(task, "C", df.columns)).to_numpy()
    aurocs = []
    for k in range(n_shuffles):
        y = shuffled(y0, groups, task, np.random.default_rng(offset + k))
        splits = make_splits(y, groups, n_repeats=n_repeats)
        if model == "lstm":
            data["y"] = y
            fp = lstm_fit_predict(data)
        else:
            fp = tree_fit_predict(model, X, y)
        out = evaluate(fp, y, groups, splits)
        aurocs.append(np.mean([ScoreView(y, p, d).metrics()["auroc"] for p, d in zip(out["oof_p"], out["oof_pred"])]))
        print(f"shuffle {k:2d}: AUROC {aurocs[-1]:.3f}", flush=True)
    a = np.array(aurocs)
    sd = a.std(ddof=1) if len(a) > 1 else float("nan")
    print(f"\n{task} / {model}: mean AUROC over {n_shuffles} shuffles {a.mean():.3f}, SD {sd:.3f}, "
          f"range {a.min():.3f} to {a.max():.3f}; share of shuffles >= {compare}: {(a >= compare).mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the pipeline on many shuffled-label copies of the data.")
    ap.add_argument("--task", choices=list(TASKS), default="readmission")
    ap.add_argument("--model", choices=["lstm", "xgboost", "random_forest"], default="lstm")
    ap.add_argument("--shuffles", type=int, default=30)
    ap.add_argument("--offset", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--compare", type=float, default=0.598)
    a = ap.parse_args()
    main(a.task, a.model, a.shuffles, a.repeats, a.offset, a.compare)