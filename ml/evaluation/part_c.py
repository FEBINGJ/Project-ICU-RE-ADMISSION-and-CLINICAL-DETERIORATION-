"""Phase 5 Part C: compare the three models and the bedside scores, apply the winner rule, measure alert burden.

Reads the out-of-fold predictions saved by Parts A and B (nothing is retrained here) and writes
    reports/tables/part_c_results.json
    reports/figures/part_c_comparison.png

Run from the project root:   python -m ml.evaluation.part_c

Candidates (fixed in advance, docs/results_part_c.md):
    XGBoost and Random Forest on feature set C, LSTM on vitals + labs      -> the three models, one winner per task
    NEWS2, MEWS                                                            -> comparators, never winners

Winner rule (locked in docs/scope.md, details fixed before the Part C numbers were seen):
    1. best = highest mean AUROC (models within 0.005 AUROC count as tied; the higher F1 then decides)
    2. a model is "clearly worse" than best when the 95% range does not allow it to be equal
         - OFFICIAL test: paired range of the difference (same resampled patients for both models) excludes 0
         - CHECK test: the two separate 95% ranges do not overlap (the locked wording)
    3. if nothing is clearly worse than best -> winner = best, else all models that are not clearly worse stay as
       contenders and the verdict is "no clear winner": the tree model (XGBoost / Random Forest) with the highest
       AUROC among the contenders is chosen, because SHAP explanations are easier for trees
    4. if the two tests give different winners, the locked (separate-ranges) test decides
"""
import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.config import DATA_PROCESSED, FIGURES_DIR, PROJECT_ROOT
from ml.evaluation.metrics import ScoreView
from ml.models.feature_sets import TASKS, load_task

TABLES = PROJECT_ROOT / "reports" / "tables"
RESULTS = TABLES / "part_c_results.json"
FILES = {"xgboost": "xgboost__C", "random_forest": "random_forest__C", "lstm": "lstm__all",
         "news2": "news2", "mews": "mews"}
MODELS = ["xgboost", "random_forest", "lstm"]          # the candidates
TREES = ("xgboost", "random_forest")
LABEL = {"xgboost": "XGBoost", "random_forest": "Random Forest", "lstm": "LSTM", "news2": "NEWS2", "mews": "MEWS"}
TIE = 0.005                                            # AUROC difference treated as a tie
N_BOOT = 1000
TARGETS = (0.5, 0.8)                                   # sensitivities for the alert-burden table


# --------------------------------------------------------------------------- winner rule (pure function, tested)
def choose_winner(auroc: dict, f1: dict, clearly_worse) -> dict:
    """auroc, f1: mean values per model name. clearly_worse(best, other) -> True when `other` is clearly worse."""
    names = list(auroc)
    top = max(auroc.values())
    tied = [n for n in names if top - auroc[n] < TIE]
    best = max(tied, key=lambda n: (f1[n], auroc[n]))
    contenders = [n for n in names if n == best or not clearly_worse(best, n)]
    if len(contenders) == 1:
        return dict(best=best, winner=best, verdict="clear winner", contenders=contenders)
    trees = [n for n in contenders if n in TREES]
    winner = max(trees, key=lambda n: (round(auroc[n] / TIE), f1[n]))
    return dict(best=best, winner=winner, verdict="no clear winner", contenders=contenders)


# --------------------------------------------------------------------------- operating points at a matched sensitivity
def operating_point(view: ScoreView, w, target: float):
    """Fewest alerts that still catch >= `target` of the events. Returns (alert share of all rows, precision)."""
    yw, nw = w * view.y, w * (1 - view.y)
    pos_g = np.bincount(view.group, weights=yw[view.order], minlength=view.n_groups)
    neg_g = np.bincount(view.group, weights=nw[view.order], minlength=view.n_groups)
    tp, fp = np.cumsum(pos_g), np.cumsum(neg_g)
    ok = np.flatnonzero(tp / yw.sum() >= target - 1e-12)
    g = ok[0]
    return (tp[g] + fp[g]) / w.sum(), tp[g] / (tp[g] + fp[g])


# --------------------------------------------------------------------------- bootstrap over patients, shared by all models
def load_views(task, y):
    out = {}
    for name, key in FILES.items():
        z = np.load(DATA_PROCESSED / "oof" / f"{task}__{key}.npz")
        out[name] = [ScoreView(y, p, d) for p, d in zip(z["p"], z["pred"])]
    return out


def bootstrap_draws(views, groups, n_boot=N_BOOT, seed=0):
    """Same resampled patients for every model. Returns dict: quantity -> array (n_boot, n_models)."""
    names = list(views)
    patients, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    keys = ["auroc", "f1"] + [f"alert_share_{int(t * 100)}" for t in TARGETS] + [f"precision_{int(t * 100)}" for t in TARGETS]
    draws = {k: np.zeros((n_boot, len(names))) for k in keys}

    def one(w):
        res = {k: np.zeros(len(names)) for k in keys}
        for j, n in enumerate(names):
            m = [v.metrics(w) for v in views[n]]
            res["auroc"][j] = np.nanmean([x["auroc"] for x in m])
            res["f1"][j] = np.nanmean([x["f1"] for x in m])
            for t in TARGETS:
                ops = [operating_point(v, w, t) for v in views[n]]
                res[f"alert_share_{int(t * 100)}"][j] = np.mean([o[0] for o in ops])
                res[f"precision_{int(t * 100)}"][j] = np.mean([o[1] for o in ops])
        return res

    full = one(np.ones(len(groups)))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))[inverse].astype(float)
        r = one(w)
        for k in keys:
            draws[k][b] = r[k]
    return names, full, draws


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# --------------------------------------------------------------------------- one task
def analyse_task(task):
    cfg = TASKS[task]
    _, y, groups = load_task(task)
    views = load_views(task, y)
    names, full, draws = bootstrap_draws(views, groups)
    idx = {n: i for i, n in enumerate(names)}
    a_full = {n: float(full["auroc"][idx[n]]) for n in names}
    f_full = {n: float(full["f1"][idx[n]]) for n in names}

    # separate (unpaired) 95% ranges, taken from the bootstrap draws of each model on its own
    own = {n: ci(draws["auroc"][:, idx[n]]) for n in names}

    def worse_paired(best, other):
        return ci(draws["auroc"][:, idx[best]] - draws["auroc"][:, idx[other]])[0] > 0

    def worse_unpaired(best, other):
        return own[best][0] > own[other][1]

    cand_a = {n: a_full[n] for n in MODELS}
    cand_f = {n: f_full[n] for n in MODELS}
    paired_rule = choose_winner(cand_a, cand_f, worse_paired)
    locked_rule = choose_winner(cand_a, cand_f, worse_unpaired)
    final = locked_rule if paired_rule["winner"] != locked_rule["winner"] else paired_rule

    pairs = {}
    for a, b in itertools.combinations(names, 2):
        d, df = draws["auroc"][:, idx[a]] - draws["auroc"][:, idx[b]], draws["f1"][:, idx[a]] - draws["f1"][:, idx[b]]
        pairs[f"{a} minus {b}"] = dict(auroc_diff=float(a_full[a] - a_full[b]), auroc_ci=ci(d),
                                       f1_diff=float(f_full[a] - f_full[b]), f1_ci=ci(df),
                                       share_a_better=float((d > 0).mean()))
    models = {}
    for n in names:
        m = dict(auroc=a_full[n], auroc_ci=own[n], f1=f_full[n], f1_ci=ci(draws["f1"][:, idx[n]]))
        for t in TARGETS:
            k = int(t * 100)
            share = float(full[f"alert_share_{k}"][idx[n]])
            m[f"at_sensitivity_{k}"] = dict(
                alert_share=share, alert_share_ci=ci(draws[f"alert_share_{k}"][:, idx[n]]),
                precision=float(full[f"precision_{k}"][idx[n]]), precision_ci=ci(draws[f"precision_{k}"][:, idx[n]]))
            if cfg["hours_per_row"]:
                m[f"at_sensitivity_{k}"]["alerts_per_patient_day"] = share * 24 / cfg["hours_per_row"]
                m[f"at_sensitivity_{k}"]["alerts_per_patient_day_ci"] = [c * 24 / cfg["hours_per_row"]
                                                                          for c in m[f"at_sensitivity_{k}"]["alert_share_ci"]]
        models[n] = m
    return dict(n_rows=int(len(y)), n_pos=int(y.sum()), n_boot=N_BOOT, models=models, pairs=pairs,
                rule_paired=paired_rule, rule_locked_unpaired=locked_rule, final=final,
                rules_agree=paired_rule["winner"] == locked_rule["winner"])


# --------------------------------------------------------------------------- figure and printout
def figure(results):
    fig, axes = plt.subplots(1, len(results), figsize=(12, 3.8), sharex=True)
    colors = {"xgboost": ("#2563a8", "o"), "random_forest": ("#c2670f", "D"), "lstm": ("#15803d", "^"),
              "news2": ("#6b7280", "s"), "mews": ("#6b7280", "s")}
    for ax, (task, res) in zip(axes, results.items()):
        order = ["news2", "mews", "xgboost", "random_forest", "lstm"]
        for i, n in enumerate(order):
            m = res["models"][n]
            c, mk = colors[n]
            ax.plot(m["auroc_ci"], [i, i], color=c, lw=2, solid_capstyle="round")
            ax.plot(m["auroc"], i, mk, color=c, ms=9, mec="white", mew=1.5)
            if n == res["final"]["winner"]:
                ax.plot(m["auroc"], i, mk, ms=15, mfc="none", mec="#111827", mew=1.5)
            ax.text(m["auroc_ci"][1] + 0.012, i, f"{m['auroc']:.2f}", va="center", fontsize=9, color="#374151")
        ax.axvline(0.5, color="#9ca3af", lw=1, ls="--")
        ax.set_yticks(range(len(order)), [LABEL[n] + (" (winner)" if n == res["final"]["winner"] else "") for n in order])
        ax.invert_yaxis()
        ax.set_xlim(0.3, 0.9)
        ax.set_xlabel("AUROC (mean of 10 repeats, bar = 95% patient-level range)")
        ax.set_title(f"{task.capitalize()}: {res['final']['verdict']}", fontsize=10.5, loc="left")
        ax.grid(axis="x", color="#e5e7eb", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "part_c_comparison.png", dpi=160)


def main():
    results = {task: analyse_task(task) for task in TASKS}
    TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1))
    figure(results)
    for task, res in results.items():
        print(f"\n== {task}")
        for n, m in res["models"].items():
            s80 = m["at_sensitivity_80"]
            extra = f"alerts/day {s80['alerts_per_patient_day']:.2f}, " if "alerts_per_patient_day" in s80 else ""
            print(f"  {LABEL[n]:<14} AUROC {m['auroc']:.3f} ({m['auroc_ci'][0]:.2f} to {m['auroc_ci'][1]:.2f})  F1 {m['f1']:.3f}  "
                  f"at 80% sensitivity: {extra}precision {s80['precision']:.3f}")
        for rule in ("rule_paired", "rule_locked_unpaired"):
            r = res[rule]
            print(f"  {rule}: best {r['best']}, {r['verdict']}, contenders {r['contenders']}, winner {r['winner']}")
        print(f"  FINAL: {res['final']['winner']} ({res['final']['verdict']}); rules agree: {res['rules_agree']}")
    print("\nSaved", RESULTS)


if __name__ == "__main__":
    main()
