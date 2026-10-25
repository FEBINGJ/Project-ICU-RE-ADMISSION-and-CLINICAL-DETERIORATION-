"""SHAP explanations for the two saved models, in plain language (Phase 6).

explain(artifact, rows, top_k=5) returns, for every row, a dictionary the backend can send to the dashboard:
    risk, risk_percent     calibrated chance of the event (see docs/results_part_c.md: close to the average rate)
    band, band_note        Low / Moderate / High with the observed event rate, or None when the bands are not supported
    factors                the top_k inputs that moved the score most (label, value, direction, size, group, note)
    groups                 net push of each type of information (vitals, labs, trends, context, measurement pattern)
    rest                   the combined push of all the other inputs
    narrative              a written explanation, several short paragraphs
    reading_guide          how to read the explanation
    warnings               things the reader must keep in mind for this particular patient

How it works: SHAP gives every input a contribution (in log-odds units). Starting from the model's average score
("base score"), the contributions add up exactly to this patient's raw score (checked by a test). Large positive
contributions raise the risk, large negative ones lower it. These are about what the MODEL used; they are not causes.
"""
import numpy as np
import shap

from ml.evaluation.bands import BAND_NAMES, assign_band
from ml.explain.names import describe
from ml.models.final import predict, prepare

EVENT = {"deterioration": "starting invasive ventilation or dying within the next 24 hours",
         "readmission": "being readmitted to the ICU within 30 days of ICU discharge"}
READING_GUIDE = ("The model starts from an average score. Each factor below moves the score up (raises the risk) or down "
                 "(lowers it); all factors together, including the many small ones, give this patient's score. "
                 "Sizes are relative to each other. They show what the model used, not what caused the outcome.")
_CACHE = {}


def _explainer(artifact):
    model = artifact["model"]
    entry = _CACHE.get(id(model))
    if entry is None or entry[0] is not model:                  # keep the model alive so its id cannot be reused by another model
        entry = _CACHE[id(model)] = (model, shap.TreeExplainer(model))
    return entry[1]


def shap_matrix(artifact, df):
    """Returns (X as a numeric array, SHAP values, base score). base + sum(SHAP) = the model's raw log-odds."""
    X = prepare(artifact, df).to_numpy(dtype=float)
    ex = _explainer(artifact)
    return X, np.asarray(ex.shap_values(X)), float(np.ravel(ex.expected_value)[0])


def _strength(share):
    return "large" if share >= 0.10 else "medium" if share >= 0.04 else "small"


def _phrase(f):
    return f"{f['label']} ({f['value_text']}), {f['strength']} influence"


def _band(artifact, raw):
    if not artifact.get("bands_supported") or artifact.get("threshold_moderate") is None:
        return None, ("Risk bands are not shown for this model: in cross-validation the observed event rate did not rise "
                      "reliably from low to high scores.")
    k = int(assign_band([raw], artifact["threshold_moderate"], artifact["threshold"])[0])
    name = BAND_NAMES[k]
    t = artifact["bands_cv"]["bands"][name]
    if t["event_rate"] is None:
        return name, ""
    ci = t["event_rate_ci"]
    return name, (f"In cross-validation, {t['event_rate']:.1%} of cases in the {name} band had the event "
                  f"(95% range {ci[0]:.1%} to {ci[1]:.1%}); the average for everyone is {artifact['prevalence']:.1%}.")


def explain(artifact: dict, df, top_k: int = 5) -> list[dict]:
    Xn, sv, base = shap_matrix(artifact, df)
    pred = predict(artifact, df)
    cols = artifact["columns"]
    out = []
    for i in range(len(Xn)):
        c = sv[i]
        total_abs = float(np.abs(c).sum()) or 1.0
        order = np.argsort(-np.abs(c))
        factors = []
        for j in order[:top_k]:
            d = describe(cols[j], float(Xn[i, j]) if not np.isnan(Xn[i, j]) else None)
            share = float(abs(c[j]) / total_abs)
            factors.append(dict(feature=cols[j], label=d["label"], value=None if np.isnan(Xn[i, j]) else float(Xn[i, j]),
                                value_text=d["value_text"], group=d["group"], note=d["note"], contribution=float(c[j]),
                                direction="raises" if c[j] > 0 else "lowers", share=share, strength=_strength(share)))
        group_net = {}
        for j, name in enumerate(cols):
            g = describe(name, 1.0)["group"]
            group_net[g] = group_net.get(g, 0.0) + float(c[j])
        groups = sorted(({"group": g, "net_contribution": v, "direction": "raises" if v > 0 else "lowers"} for g, v in group_net.items()),
                        key=lambda x: -abs(x["net_contribution"]))
        shown = {f["feature"] for f in factors}
        rest_net = float(sum(c[j] for j, n in enumerate(cols) if n not in shown))
        risk, raw = float(pred["risk"][i]), float(pred["raw_score"][i])
        band, band_note = _band(artifact, raw)
        warnings = [f"Model discrimination is close to chance (cross-validated AUROC {artifact['cv']['auroc']:.2f}, "
                    f"95% range {artifact['cv']['auroc_ci'][0]:.2f} to {artifact['cv']['auroc_ci'][1]:.2f}): use as context, not as a decision."]
        if any(f["note"] for f in factors):
            warnings.append("One or more top factors reflect how often or whether something was recorded (shown with a note), "
                            "which mostly describes how closely staff monitor the patient, not the patient's physiology.")
        up = [f for f in factors if f["direction"] == "raises"]
        down = [f for f in factors if f["direction"] == "lowers"]
        paras = [f"The model estimates a {risk:.1%} chance of {EVENT[artifact['task']]} (the average in the training data is "
                 f"{artifact['prevalence']:.1%}). " + (f"This falls in the {band} band. " if band else "") + band_note]
        paras.append("Factors that pushed the score UP: " + ("; ".join(_phrase(f) for f in up) + "." if up else "none among the top factors."))
        paras.append("Factors that pushed the score DOWN: " + ("; ".join(_phrase(f) for f in down) + "." if down else "none among the top factors."))
        n_rest = len(cols) - len(factors)
        top_share = sum(f["share"] for f in factors)
        paras.append(f"The {len(factors)} factors above make up {top_share:.0%} of everything that moves this patient's score; the other "
                     f"{n_rest} inputs together {'raise' if rest_net > 0 else 'lower'} it. "
                     + ("The score is spread over many small contributions rather than a few strong ones, which is typical when a model "
                        "has no strong signal to rely on." if top_share < 0.35 else ""))
        paras.append("By type of information, the biggest net pushes are: " + "; ".join(
            f"{g['group'].lower()} pushed the score {'up' if g['direction'] == 'raises' else 'down'}" for g in groups[:3]) + ".")
        paras.append("Remember: " + " ".join(warnings))
        out.append(dict(task=artifact["task"], risk=risk, risk_percent=round(100 * risk, 1), raw_score=raw, base_score=base,
                        alert=int(pred["alert"][i]), band=band, band_note=band_note, factors=factors, groups=groups,
                        rest=dict(n_inputs=n_rest, net_contribution=rest_net), narrative="\n\n".join(paras),
                        reading_guide=READING_GUIDE, warnings=warnings))
    return out


if __name__ == "__main__":
    import joblib
    from ml.config import MODEL_DIR
    from ml.models.feature_sets import load_task

    for task in ("deterioration", "readmission"):
        model_path = MODEL_DIR / f"{task}_xgboost.joblib"
        if not model_path.exists():
            print(f"Artifact {model_path} not found.")
            continue
        artifact = joblib.load(model_path)
        df, _, _ = load_task(task)
        explanations = explain(artifact, df.iloc[:1])
        print(f"\n{'='*20} {task.upper()} EXAMPLE EXPLANATION {'='*20}")
        print(explanations[0]["narrative"])

