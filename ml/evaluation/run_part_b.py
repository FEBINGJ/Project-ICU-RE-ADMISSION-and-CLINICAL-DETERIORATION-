"""Phase 5 Part B: LSTM on the sequence arrays, evaluated with the frozen Part A setup.

Run from the project root (after run_part_a, which writes the Part A fold fingerprints):
    python -m ml.evaluation.run_part_b                   # full run (10 repeats, 500 bootstrap draws)
    python -m ml.evaluation.run_part_b --quick           # 2 repeats, 50 draws
    python -m ml.evaluation.run_part_b --shuffle-check   # shuffled labels must give AUROC ~ 0.5

Variants: "all" = 7 vitals + 12 core labs (the full input), "vitals" = 7 vitals only.
Outputs
    reports/tables/part_b_results.json
    data/processed/oof/<task>__lstm__<variant>.npz   (git-ignored)
"""
import argparse
import json

import numpy as np

from ml.evaluation.folds import fingerprint, make_splits
from ml.evaluation.run_part_a import RESULTS as PART_A_RESULTS
from ml.evaluation.run_part_a import run_config
from ml.config import PROJECT_ROOT
from ml.models.feature_sets import TASKS
from ml.models.lstm import SETTINGS, load_sequence_data, lstm_fit_predict

RESULTS = PROJECT_ROOT / "reports" / "tables" / "part_b_results.json"
VARIANTS = ["all", "vitals"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shuffle-check", action="store_true")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    args = ap.parse_args()
    n_rep, n_boot = (2, 50) if args.quick else (10, 500)
    part_a = json.loads(PART_A_RESULTS.read_text()) if PART_A_RESULTS.exists() else {}

    results = {}
    for task in args.tasks:
        cfg = TASKS[task]
        data = load_sequence_data(task, "all")
        y, groups = data["y"], data["groups"]
        if args.shuffle_check:
            rng = np.random.default_rng(0)
            patients = np.unique(groups)
            lab = dict(zip(patients, rng.permutation([int(y[groups == p].max()) for p in patients])))
            y = np.array([lab[g] for g in groups]) if task == "readmission" else rng.permutation(y)
        splits = make_splits(y, groups, n_repeats=n_rep)
        fp = fingerprint(splits)
        if not (args.quick or args.shuffle_check):
            expected = part_a.get(task, {}).get("fingerprint")
            assert expected is None or expected == fp, f"folds differ from Part A ({fp} vs {expected})"
        print(f"\n== {task}: {len(y)} rows, {y.sum()} positives, {n_rep} repeats x 5 folds, fold fingerprint {fp}"
              f"{' (same as Part A)' if part_a.get(task, {}).get('fingerprint') == fp else ''}", flush=True)
        res = {"fingerprint": fp, "n_rows": int(len(y)), "n_pos": int(y.sum()),
               "n_patients": int(len(np.unique(groups))), "n_repeats": n_rep, "settings": SETTINGS, "configs": {}}
        for variant in (["all"] if args.shuffle_check else VARIANTS):
            d = load_sequence_data(task, variant)
            d["y"] = y
            name = f"lstm/{variant}"
            save = None if (args.quick or args.shuffle_check) else task
            res["configs"][name] = run_config(name, lstm_fit_predict(d), y, groups, splits, cfg["hours_per_row"],
                                              n_boot, save)
        results[task] = res

    if not (args.quick or args.shuffle_check):
        RESULTS.write_text(json.dumps(results, indent=1))
        print("\nSaved", RESULTS)


if __name__ == "__main__":
    main()
