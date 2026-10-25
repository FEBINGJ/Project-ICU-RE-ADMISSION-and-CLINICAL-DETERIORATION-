"""Phase 5 Part A: baselines, XGBoost and Random Forest on feature sets A-D, for both tasks.

Run from the project root:
    python -m ml.evaluation.run_part_a                    # full run (10 repeats, 500 bootstrap draws)
    python -m ml.evaluation.run_part_a --quick            # 2 repeats, 50 draws, to check timing
    python -m ml.evaluation.run_part_a --shuffle-check    # sanity check: shuffled labels must give AUROC ~ 0.5

Outputs
    reports/tables/part_a_results.json   every number with its confidence range (committed)
    data/processed/oof/<task>__<model>__<set>.npz   out-of-fold predictions (patient-level, git-ignored)
"""
import argparse
import json
import time

import numpy as np

from ml.config import DATA_PROCESSED, PROJECT_ROOT
from ml.evaluation.cv import evaluate
from ml.evaluation.folds import fingerprint, make_splits
from ml.evaluation.metrics import summarise
from ml.models.baselines import base_rate_fit_predict, score_fit_predict
from ml.models.feature_sets import TASKS, columns_for, design_matrix, load_task
from ml.models.trees import tree_fit_predict

MODELS = ["xgboost", "random_forest"]
OOF_DIR = DATA_PROCESSED / "oof"
RESULTS = PROJECT_ROOT / "reports" / "tables" / "part_a_results.json"


def run_config(name, fit_predict, y, groups, splits, hours, n_boot, task):
    t0 = time.time()
    out = evaluate(fit_predict, y, groups, splits)
    summary = summarise(y, groups, out["oof_p"], out["oof_pred"], n_boot=n_boot, hours_per_row=hours)
    if name in ("news2", "mews"):
        summary["brier"] = None            # a raw score is not a probability, so Brier is undefined
    if task is not None:
        OOF_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OOF_DIR / f"{task}__{name.replace('/', '__')}.npz", p=out["oof_p"], pred=out["oof_pred"],
                            thr=out["thresholds"])
    s = summary
    print(f"  {name:<22} AUROC {s['auroc']['mean']:.3f} [{s['auroc']['ci_low']:.3f}, {s['auroc']['ci_high']:.3f}]  "
          f"F1 {s['f1']['mean']:.3f}  prec {s['precision']['mean']:.3f}  rec {s['recall']['mean']:.3f}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shuffle-check", action="store_true")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    args = ap.parse_args()
    n_rep, n_boot = (2, 50) if args.quick else (10, 500)

    results = {}
    for task in args.tasks:
        cfg = TASKS[task]
        df, y, groups = load_task(task)
        if args.shuffle_check:
            rng = np.random.default_rng(0)
            patients = np.unique(groups)
            lab = dict(zip(patients, rng.permutation([int(y[groups == p].max()) for p in patients])))
            y = np.array([lab[g] for g in groups]) if task == "readmission" else rng.permutation(y)
        splits = make_splits(y, groups, n_repeats=n_rep)
        print(f"\n== {task}: {len(y)} rows, {len(np.unique(groups))} patients, {y.sum()} positives, "
              f"{n_rep} repeats x 5 folds, fold fingerprint {fingerprint(splits)}", flush=True)
        res = {"fingerprint": fingerprint(splits), "n_rows": int(len(y)), "n_pos": int(y.sum()),
               "n_patients": int(len(np.unique(groups))), "n_repeats": n_rep, "configs": {}}
        save = None if (args.quick or args.shuffle_check) else task
        hours = cfg["hours_per_row"]
        if not args.shuffle_check:
            print("baselines")
            for name, fp in [("base_rate", base_rate_fit_predict(y)),
                             ("news2", score_fit_predict(df["news2"].to_numpy())),
                             ("mews", score_fit_predict(df["mews"].to_numpy()))]:
                res["configs"][name] = run_config(name, fp, y, groups, splits, hours, n_boot, save)
        sets = ["C"] if args.shuffle_check else cfg["sets"]
        for model in MODELS:
            print(model)
            for fs in sets:
                X = design_matrix(df, columns_for(task, fs, df.columns))
                name = f"{model}/{fs}"
                res["configs"][name] = run_config(name, tree_fit_predict(model, X.to_numpy(), y), y, groups, splits,
                                                  hours, n_boot, save)
                res["configs"][name]["n_features"] = int(X.shape[1])
        results[task] = res

    if not (args.quick or args.shuffle_check):
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=1))
        print("\nSaved", RESULTS)


if __name__ == "__main__":
    main()
