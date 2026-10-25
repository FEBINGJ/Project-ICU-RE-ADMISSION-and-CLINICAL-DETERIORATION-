"""Global SHAP summary, stability check and example patients (Phase 6).

Run from the project root:   python -m ml.explain.report
Outputs
    reports/tables/phase6_global.json     top inputs, share by type of information, stability across refits
    reports/tables/phase6_examples.json   written explanations of example patients
    reports/figures/phase6_global.png     most important inputs and share by type of information
    reports/figures/phase6_examples.png   example single-patient explanations

Stability check: the model is refitted on 30 resamples of the PATIENTS (same settings), SHAP importance is computed
for each refit, and compared with the saved model. If the top inputs change a lot between refits, an explanation of one
model must not be over-read.
All SHAP values here are computed on the data the models were fitted on (this describes the model's behaviour, it does
not measure performance).
"""
import json
import textwrap

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from scipy.stats import spearmanr

from ml.config import FIGURES_DIR, MODEL_DIR, PROJECT_ROOT, RANDOM_SEED
from ml.explain.explaines import explain, shap_matrix
from ml.explain.names import GROUPS, describe
from ml.models.feature_sets import load_task
from ml.models.final import prepare
from ml.models.trees import make_model

TABLES = PROJECT_ROOT / "reports" / "tables"
N_REFITS = 30
TOP = 10
STABLE_SHARE = 0.7
GROUP_COLORS = {"Latest vital signs": "#2563a8", "Vital-sign trends": "#0e7490", "Latest lab results": "#c2670f",
                "Lab trends (24 h)": "#a16207", "Measurement pattern": "#7c3aed", "Whole-stay vitals": "#15803d",
                "Patient and stay context": "#6b7280"}


def mean_abs(model, X):
    return np.abs(shap.TreeExplainer(model).shap_values(X)).mean(axis=0)


def group_shares(imp, cols):
    g = {n: 0.0 for n in GROUPS}
    for v, c in zip(imp, cols):
        g[describe(c, 1.0)["group"]] += v
    tot = sum(g.values())
    return {n: v / tot for n, v in g.items()}


def analyse(task):
    art = joblib.load(MODEL_DIR / f"{task}_xgboost.joblib")
    df, y, groups = load_task(task)
    X = prepare(art, df).to_numpy(float)
    cols = art["columns"]
    sv_all = np.abs(shap.TreeExplainer(art["model"]).shap_values(X))
    imp = sv_all.mean(axis=0)
    missing_share = float(sv_all[np.isnan(X)].sum() / sv_all.sum())
    top_idx = np.argsort(-imp)[:TOP]
    # --- stability over refits on resampled patients
    rng = np.random.default_rng(RANDOM_SEED)
    patients = np.unique(groups)
    rows_of = {p: np.flatnonzero(groups == p) for p in patients}
    imps, shares = [], []
    for b in range(N_REFITS):
        pick = rng.choice(patients, len(patients), replace=True)
        idx = np.concatenate([rows_of[p] for p in pick])
        m = make_model("xgboost", y[idx], RANDOM_SEED + b)
        m.fit(X[idx], y[idx])
        i = mean_abs(m, X)
        imps.append(i)
        shares.append(group_shares(i, cols))
    imps = np.array(imps)
    rho = [spearmanr(imp, i)[0] for i in imps]
    overlap = [len(set(top_idx) & set(np.argsort(-i)[:TOP])) / TOP for i in imps]
    freq = np.mean([np.isin(np.arange(len(cols)), np.argsort(-i)[:TOP]) for i in imps], axis=0)
    main_shares = group_shares(imp, cols)
    res = dict(
        n_rows=int(len(y)), n_refits=N_REFITS, share_of_attribution_from_missing_values=missing_share,
        top_inputs=[dict(column=cols[j], label=describe(cols[j], 1.0)["label"], group=describe(cols[j], 1.0)["group"],
                         mean_abs_shap=float(imp[j]), share_of_total=float(imp[j] / imp.sum()),
                         in_top10_of_refits=float(freq[j])) for j in top_idx],
        stable_inputs=[dict(column=cols[j], label=describe(cols[j], 1.0)["label"], in_top10_of_refits=float(freq[j]))
                       for j in np.argsort(-freq) if freq[j] >= STABLE_SHARE],
        group_share={n: dict(share=main_shares[n], refit_mean=float(np.mean([s[n] for s in shares])),
                             refit_sd=float(np.std([s[n] for s in shares], ddof=1))) for n in GROUPS},
        stability=dict(rank_correlation_mean=float(np.mean(rho)), rank_correlation_min=float(np.min(rho)),
                       top10_overlap_mean=float(np.mean(overlap)), top10_overlap_min=float(np.min(overlap))),
        top10_share_of_total=float(imp[top_idx].sum() / imp.sum()))
    return art, df, X, imp, top_idx, res


def examples(task, art, df):
    X, sv, base = shap_matrix(art, df)
    raw = art["model"].predict(X, output_margin=True)
    order = np.argsort(raw)
    picks = {"highest score": order[-1], "median score": order[len(order) // 2], "lowest score": order[0]}
    out = {}
    for name, i in picks.items():
        e = explain(art, df.iloc[[int(i)]], top_k=8)[0]
        out[name] = dict(row=int(i), explanation=e)
    return out


def main():
    results, exs = {}, {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw=dict(height_ratios=[1.5, 1]))
    for k, task in enumerate(["deterioration", "readmission"]):
        art, df, X, imp, top_idx, res = analyse(task)
        results[task] = res
        exs[task] = examples(task, art, df)
        ax = axes[0, k]
        t = list(top_idx)[::-1]
        ax.barh(range(len(t)), imp[t] / imp.sum() * 100, color=[GROUP_COLORS[describe(art["columns"][j], 1.0)["group"]] for j in t])
        ax.set_yticks(range(len(t)), [textwrap.fill(describe(art["columns"][j], 1.0)["label"], 44) for j in t], fontsize=7.5)
        ax.set_xlabel("share of the model's total attribution (%)")
        ax.set_title(f"{task.capitalize()}: ten most used inputs", loc="left", fontsize=11)
        ax2 = axes[1, k]
        names = GROUPS[::-1]
        ax2.barh(range(len(names)), [res["group_share"][n]["share"] * 100 for n in names],
                 xerr=[res["group_share"][n]["refit_sd"] * 100 for n in names], color=[GROUP_COLORS[n] for n in names],
                 error_kw=dict(ecolor="#374151", lw=1, capsize=3))
        ax2.set_yticks(range(len(names)), names, fontsize=9)
        ax2.set_xlabel("share by type of information (%); line = spread over 30 refits")
        for a in (ax, ax2):
            a.grid(axis="x", color="#e5e7eb", lw=0.8)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        s = res["stability"]
        print(f"\n{task}: top-10 inputs carry {res['top10_share_of_total']:.0%} of total attribution; across {N_REFITS} refits "
              f"rank correlation {s['rank_correlation_mean']:.2f} (min {s['rank_correlation_min']:.2f}), "
              f"top-10 overlap {s['top10_overlap_mean']:.0%} (min {s['top10_overlap_min']:.0%})")
        for r in res["top_inputs"]:
            print(f"   {r['share_of_total']:.1%}  [{r['group']}] {r['label']}  (in top 10 of {r['in_top10_of_refits']:.0%} of refits)")
        print(f"   {res['share_of_attribution_from_missing_values']:.0%} of the attribution comes from inputs with no recorded value")
        print("   share by type:", {n: f"{v['share']:.0%}" for n, v in res["group_share"].items()})
        print("   stable inputs (top 10 in >=70% of refits):", [r["label"] for r in res["stable_inputs"]])
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "phase6_global.png", dpi=150)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for r, task in enumerate(exs):
        for c, (name, ex) in enumerate(exs[task].items()):
            e = ex["explanation"]
            f = e["factors"][::-1]
            ax = axes[r, c]
            ax.barh(range(len(f)), [x["contribution"] for x in f], color=["#b91c1c" if x["contribution"] > 0 else "#1d4ed8" for x in f])
            ax.set_yticks(range(len(f)), [textwrap.fill(x["label"], 40) + "\n= " + textwrap.shorten(x["value_text"], 34) for x in f], fontsize=6.5)
            ax.axvline(0, color="#374151", lw=0.8)
            ax.set_title(f"{task.capitalize()}, {name}: risk {e['risk_percent']}%" + (f", {e['band']} band" if e["band"] else ""), loc="left", fontsize=9.5)
            ax.set_xlabel("pushes score down (blue) / up (red)", fontsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase6_examples.png", dpi=150)
    (TABLES / "phase6_global.json").write_text(json.dumps(results, indent=1, default=float))
    (TABLES / "phase6_examples.json").write_text(json.dumps(exs, indent=1, default=float))
    print("\nSaved phase6_global.json, phase6_examples.json and two figures")


if __name__ == "__main__":
    main()
