"""
Per-die interpretability for the Model A and Model B THREE-ENGINE STACKS.

`model_ab_attribution.py` explains the LightGBM booster of each official model. This
module extends that to all three engines (LightGBM, CatBoost, XGBoost) so the
attribution covers the stacks that are quoted as the Official Model A and Official
Model B deliverables.

The honest limitation, stated up front
--------------------------------------
The official stacks blend in PROBABILITY space:

    p_stack = 0.50 * p_lightgbm + 0.30 * p_catboost + 0.20 * p_xgboost

Shapley values from TreeSHAP are exactly additive in each engine's own raw
log-odds MARGIN, not in probability. Because the sigmoid linking margin to
probability is non-linear, the three engines' margin-space SHAP values cannot be
weight-averaged to obtain a decomposition of ``p_stack``. Any report that sums them
is wrong.

What this module therefore does:

1. Computes EXACT TreeSHAP separately for each of the six boosters and verifies
   additivity against that booster's own raw margin.
2. Reports each engine's domain evidence shares independently.
3. Reports a blend-weighted average of those per-engine SHARES, labelled as an
   aggregate of three separate decompositions rather than a decomposition of the
   stack. Shares are scale-free ratios, so averaging them is meaningful even though
   averaging the raw SHAP values is not.
4. Provides per-die drivers from LightGBM, which carries the largest blend weight
   (0.50), and records that choice on the artifact.

This is the same reasoning the project applies to Model F, where the rank-space
blend forces attribution onto a single dominant engine. It is applied consistently
here rather than quietly summing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402
from catboost import CatBoostClassifier, Pool  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from src.interpretability.model_ab_attribution import DOMAIN_OF, gloss  # noqa: E402
from src.models.common import MODEL_A_FEATURES, MODEL_B_FEATURES  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"

#: Documented stack weights, identical for Model A and Model B by design so the
#: A-vs-B comparison isolates the inputs.
WEIGHTS = {"lightgbm": 0.50, "catboost": 0.30, "xgboost": 0.20}
N_CONTROLS = 5000
TOP_K = 10
SEED = 42


def load_engines(tag: str):
    t = tag.lower()
    lb = lgb.Booster(model_file=str(MODELS_DIR / f"model_{t}.txt"))
    cb = CatBoostClassifier()
    cb.load_model(str(MODELS_DIR / f"catboost_{t}.cbm"))
    xb = xgb.Booster()
    xb.load_model(str(MODELS_DIR / f"xgb_{t}.json"))
    return {"lightgbm": lb, "catboost": cb, "xgboost": xb}


def engine_prob(name: str, model, X: np.ndarray, feats) -> np.ndarray:
    if name == "lightgbm":
        return np.asarray(model.predict(X), dtype=np.float64)
    if name == "catboost":
        return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
    return np.asarray(model.predict(xgb.DMatrix(X, feature_names=list(feats))), dtype=np.float64)


def engine_shap(name: str, model, X: np.ndarray, feats):
    """Return (shap, base, margin) — all exact TreeSHAP, all in log-odds margin space."""
    if name == "lightgbm":
        c = np.asarray(model.predict(X, pred_contrib=True), dtype=np.float64)
        margin = np.asarray(model.predict(X, raw_score=True), dtype=np.float64)
        return c[:, :-1], c[:, -1], margin
    if name == "catboost":
        sv = np.asarray(model.get_feature_importance(Pool(X), type="ShapValues"), dtype=np.float64)
        margin = np.asarray(model.predict(X, prediction_type="RawFormulaVal"), dtype=np.float64)
        return sv[:, :-1], sv[:, -1], margin
    d = xgb.DMatrix(X, feature_names=list(feats))
    c = np.asarray(model.predict(d, pred_contribs=True), dtype=np.float64)
    margin = np.asarray(model.predict(d, output_margin=True), dtype=np.float64)
    return c[:, :-1], c[:, -1], margin


def main() -> None:
    print("=" * 78)
    print("PER-DIE INTERPRETABILITY FOR THE OFFICIAL MODEL A / MODEL B 3-ENGINE STACKS")
    print("=" * 78)

    need = ["wafer_id", "die_row", "die_col", "old_label", "label"] + MODEL_B_FEATURES
    df = pq.read_table(PROCESSED_DIR / "test_features.parquet", columns=need).to_pandas()
    df = df.loc[df["old_label"].to_numpy() == 0].reset_index(drop=True)
    y = df["label"].to_numpy().astype(np.int8)
    print(f"{len(df):,} eligible test dies, {int(y.sum()):,} true failures ({100*y.mean():.3f}%)")

    feats = {"A": MODEL_A_FEATURES, "B": MODEL_B_FEATURES}
    out = {
        "deliverable": "per-die interpretability for the Official Model A and Model B 3-engine stacks",
        "stack_weights": WEIGHTS,
        "blend_space": "probability",
        "shap_method": "exact TreeSHAP, computed independently per engine in its own log-odds margin space",
        "why_shap_is_not_summed_across_engines": (
            "The stack averages PROBABILITIES. TreeSHAP is exactly additive in each "
            "engine's raw margin, and the sigmoid linking margin to probability is "
            "non-linear, so the three engines' margin-space SHAP values do not sum to a "
            "decomposition of the stack probability. Per-engine decompositions are "
            "reported separately; the blend-weighted figure below averages per-engine "
            "SHARES (scale-free ratios), not raw SHAP values."
        ),
        "per_die_driver_engine": "lightgbm (largest blend weight, 0.50)",
        "models": {},
    }

    for tag in ("A", "B"):
        F = feats[tag]
        X = df.loc[:, F].to_numpy(np.float32)
        eng = load_engines(tag)
        probs = {n: engine_prob(n, m, X, F) for n, m in eng.items()}
        p_stack = sum(WEIGHTS[n] * probs[n] for n in WEIGHTS)

        pr, rc, th = precision_recall_curve(y, p_stack)
        f1 = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(pr), where=(pr + rc) > 0)
        thr = float(th[int(np.nanargmax(f1[:-1]))])
        flagged = p_stack >= thr

        rng = np.random.default_rng(SEED)
        pool = np.flatnonzero(~flagged)
        ctrl = rng.choice(pool, size=int(min(N_CONTROLS, pool.size)), replace=False)
        sel = np.union1d(np.flatnonzero(flagged), ctrl)
        print(f"\n########## MODEL {tag} STACK ({len(F)} features) ##########")
        print(f"  stack T* = {thr:.4f}, flagged {int(flagged.sum()):,}; "
              f"explaining {sel.size:,} dies ({ctrl.size:,} random controls, no labels used)")

        per_engine = {}
        shares_by_engine = {}
        keep_shap = None
        for n in ("lightgbm", "catboost", "xgboost"):
            sv, base, margin = engine_shap(n, eng[n], X[sel], F)
            err = float(np.max(np.abs(sv.sum(axis=1) + base - margin)))
            mean_abs = np.abs(sv).mean(axis=0)
            imp = pd.DataFrame({"feature": F, "mean_abs_shap": mean_abs,
                                "domain": [DOMAIN_OF[c] for c in F]})
            dom = imp.groupby("domain")["mean_abs_shap"].sum()
            share = (100 * dom / dom.sum()).round(2)
            shares_by_engine[n] = share
            top = imp.sort_values("mean_abs_shap", ascending=False).head(6)
            print(f"  {n:9s} additivity err {err:.2e} | " +
                  " ".join(f"{d}={share[d]:.1f}%" for d in share.index))
            print(f"            top: " + ", ".join(f"{r.feature}({r.mean_abs_shap:.3f})"
                                                   for r in top.itertuples()))
            per_engine[n] = {
                "weight": WEIGHTS[n],
                "additivity_max_abs_error_margin_space": err,
                "domain_share_pct": share.to_dict(),
                "top_features": top.to_dict(orient="records"),
            }
            if n == "lightgbm":
                keep_shap = sv

        doms = sorted({DOMAIN_OF[c] for c in F})
        blended = {d: round(float(sum(WEIGHTS[n] * shares_by_engine[n].get(d, 0.0)
                                      for n in WEIGHTS)), 2) for d in doms}
        print(f"  BLEND-WEIGHTED domain shares (aggregate of 3 decompositions): "
              + " ".join(f"{d}={blended[d]:.1f}%" for d in doms))

        # per-die table from the dominant engine
        order = np.argsort(-np.abs(keep_shap), axis=1)[:, :TOP_K]
        dom_idx = {d: np.array([i for i, c in enumerate(F) if DOMAIN_OF[c] == d]) for d in doms}
        rows = []
        for i, gi in enumerate(sel):
            drivers = [{"feature": F[j], "domain": DOMAIN_OF[F[j]], "meaning": gloss(F[j]),
                        "value": float(X[gi, j]), "shap_margin": float(keep_shap[i, j])}
                       for j in order[i]]
            tot = float(np.abs(keep_shap[i]).sum()) or 1.0
            sh = {d: round(100 * float(np.abs(keep_shap[i, ix]).sum()) / tot, 2)
                  for d, ix in dom_idx.items()}
            rows.append({
                "wafer_id": str(df["wafer_id"].iloc[gi]),
                "die_row": int(df["die_row"].iloc[gi]),
                "die_col": int(df["die_col"].iloc[gi]),
                "stack_probability": float(p_stack[gi]),
                "p_lightgbm": float(probs["lightgbm"][gi]),
                "p_catboost": float(probs["catboost"][gi]),
                "p_xgboost": float(probs["xgboost"][gi]),
                "flagged": bool(flagged[gi]),
                "top_drivers_json": json.dumps(drivers),
                "domain_share_json": json.dumps(sh),
            })
        p_out = REPORTS_DIR / f"model_{tag.lower()}_stack_per_die_attribution.parquet"
        pd.DataFrame(rows).to_parquet(p_out, index=False)
        print(f"  per-die attribution -> {p_out.name} ({len(rows):,} dies)")

        out["models"][tag] = {
            "n_features": len(F),
            "stack_threshold": thr,
            "n_flagged": int(flagged.sum()),
            "n_explained": int(sel.size),
            "per_engine": per_engine,
            "blend_weighted_domain_share_pct": blended,
            "per_die_parquet": p_out.name,
        }

    with open(REPORTS_DIR / "model_ab_stack_interpretability.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\nwritten: reports/model_ab_stack_interpretability.json")
    print("DONE")


if __name__ == "__main__":
    main()
