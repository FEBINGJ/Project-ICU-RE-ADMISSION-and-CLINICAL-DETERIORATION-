"""Figure for Part A: AUROC with 95% patient-level bootstrap range for every configuration.

Run:  python -m ml.evaluation.plot_part_a      (needs reports/tables/part_a_results.json)
Output: reports/figures/part_a_auroc.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.config import FIGURES_DIR, PROJECT_ROOT

RESULTS = PROJECT_ROOT / "reports" / "tables" / "part_a_results.json"
STYLE = {"baseline": ("#6b7280", "s"), "xgboost": ("#2563a8", "o"), "random_forest": ("#c2670f", "D")}
LABEL = {"base_rate": "average rate (reference)", "news2": "NEWS2", "mews": "MEWS"}
SET_NAME = {"A": "A snapshot", "B": "B + trends", "S": "S + whole stay", "C": "C + labs", "D": "D + lactate, INR"}


def pretty(cfg: str) -> str:
    if cfg in LABEL:
        return LABEL[cfg]
    model, fs = cfg.split("/")
    return f"{'XGBoost' if model == 'xgboost' else 'Random Forest'}  {SET_NAME[fs]}"


def main():
    results = json.loads(RESULTS.read_text())
    fig, axes = plt.subplots(1, len(results), figsize=(13, 6.2), sharex=True)
    for ax, (task, res) in zip(axes, results.items()):
        cfgs = list(res["configs"])
        for i, cfg in enumerate(cfgs):
            s = res["configs"][cfg]["auroc"]
            kind = "baseline" if "/" not in cfg else cfg.split("/")[0]
            color, marker = STYLE[kind]
            ax.plot([s["ci_low"], s["ci_high"]], [i, i], color=color, lw=2, solid_capstyle="round")
            ax.plot(s["mean"], i, marker, color=color, ms=8, mec="white", mew=1.5)
            ax.text(s["ci_high"] + 0.012, i, f"{s['mean']:.2f}", va="center", fontsize=9, color="#374151")
        ax.axvline(0.5, color="#9ca3af", lw=1, ls="--")
        ax.set_yticks(range(len(cfgs)), [pretty(c) for c in cfgs], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0.3, 0.95)
        ax.set_xlabel("AUROC (mean of 10 repeats, bar = 95% patient-level range)")
        ax.set_title(f"{task.capitalize()}: {res['n_pos']} positives of {res['n_rows']} rows, {res['n_patients']} patients",
                     fontsize=10.5, loc="left")
        ax.grid(axis="x", color="#e5e7eb", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.text(0.01, 0.005, "Dashed line = chance (0.5). Ranges that cross it mean the result cannot be told apart from guessing.",
             fontsize=9, color="#4b5563")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "part_a_auroc.png"
    fig.savefig(out, dpi=160)
    print("Saved", out)


if __name__ == "__main__":
    main()
