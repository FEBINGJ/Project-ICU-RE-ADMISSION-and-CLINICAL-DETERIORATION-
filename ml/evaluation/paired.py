"""Paired comparison of feature sets (the research question: do trends help?).

For two configurations that were evaluated on the same folds, patients are resampled with replacement
and BOTH configurations are scored on the same resampled patients; the difference (second minus first) is
recorded. The 95% range of that difference is much tighter than comparing two separate ranges by eye.
"If the range of the difference contains 0, there is no evidence that the change helps."

Run:  python -m ml.evaluation.paired      (after run_part_a)
Output: reports/tables/part_a_paired.json
"""
import json

import numpy as np

from ml.config import DATA_PROCESSED, PROJECT_ROOT
from ml.evaluation.metrics import ScoreView
from ml.models.feature_sets import TASKS, load_task

OUT = PROJECT_ROOT / "reports" / "tables" / "part_a_paired.json"
STEPS = {"deterioration": [("A", "B", "trends added to snapshot"), ("B", "C", "core labs added"),
                           ("C", "D", "lactate + INR added")],
         "readmission": [("A", "B", "trends added to snapshot"), ("B", "S", "whole-stay summaries added"),
                         ("S", "C", "core labs added"), ("C", "D", "lactate + INR added")]}


def load_views(task, name, y):
    z = np.load(DATA_PROCESSED / "oof" / f"{task}__{name}.npz")
    return [ScoreView(y, p, d) for p, d in zip(z["p"], z["pred"])]


def paired(task, model, first, second, y, groups, n_boot=1000, seed=0):
    va, vb = load_views(task, f"{model}__{first}", y), load_views(task, f"{model}__{second}", y)
    patients, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    diff = {"auroc": [], "f1": []}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))[inverse].astype(float)
        for k in diff:
            diff[k].append(np.nanmean([b.metrics(w)[k] for b in vb]) - np.nanmean([a.metrics(w)[k] for a in va]))
    out = {}
    for k, d in diff.items():
        d = np.array(d)
        full = np.mean([b.metrics()[k] for b in vb]) - np.mean([a.metrics()[k] for a in va])
        out[k] = dict(diff=float(full), ci_low=float(np.percentile(d, 2.5)), ci_high=float(np.percentile(d, 97.5)),
                      share_of_draws_above_0=float((d > 0).mean()))
    return out


def main():
    res = {}
    for task, steps in STEPS.items():
        _, y, groups = load_task(task)
        res[task] = {}
        print(f"\n{task}")
        for model in ["xgboost", "random_forest"]:
            for a, b, what in steps:
                r = paired(task, model, a, b, y, groups)
                res[task][f"{model}: {a}->{b}"] = dict(what=what, **r)
                x = r["auroc"]
                print(f"  {model:<14} {a}->{b} ({what:<30}) AUROC change {x['diff']:+.3f} "
                      f"[{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]")
    OUT.write_text(json.dumps(res, indent=1))
    print("\nSaved", OUT)


if __name__ == "__main__":
    main()
