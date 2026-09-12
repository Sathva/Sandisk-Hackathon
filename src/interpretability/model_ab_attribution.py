"""
Per-die interpretability for the two OFFICIAL deliverables: Model A and Model B.

The hackathon problem statement asks specifically for:

    "Interpretability output: per-die feature importance + spatial contribution
     visualization (Model A), plus block reading pattern analysis (Model B)"

The existing suite in this directory explains Model F, which is the project's
best-performing architecture but is not one of the two mandated deliverables. This
module closes that gap by attributing Model A and Model B directly.

Method. Both official models are LightGBM boosters, so exact TreeSHAP is available
natively through ``predict(..., pred_contrib=True)``. No sampling, no surrogate, and
no ``shap`` package. LightGBM returns ``n_features + 1`` columns where the final
column is the base value, and the row sums to the model's raw margin, which is
verified here rather than assumed.

Outputs
-------
reports/model_a_per_die_attribution.parquet   per-die drivers + domain shares
reports/model_b_per_die_attribution.parquet
reports/model_ab_interpretability.json        global hierarchy, domain shares, checks
reports/model_ab_comparison_with_accuracy.json  the A vs B table including accuracy
reports/figures/40_model_a_spatial_contribution.png   Model A spatial deliverable
reports/figures/41_model_b_block_attribution.png      Model B block deliverable
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lightgbm as lgb  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.models.common import (  # noqa: E402
    BLOCK_FEATURES,
    MODEL_A_FEATURES,
    MODEL_B_FEATURES,
    PARAMETRIC_FEATURES,
    SPATIAL_FEATURES,
)

PROCESSED_DIR = REPO_ROOT / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TEST_PARQUET = PROCESSED_DIR / "test_features.parquet"
N_CONTROLS = 5000
TOP_K = 10
SEED = 42

DOMAIN_OF = {}
for _c in PARAMETRIC_FEATURES:
    DOMAIN_OF[_c] = "Die Parametric"
for _c in SPATIAL_FEATURES:
    DOMAIN_OF[_c] = "Spatial Context"
for _c in BLOCK_FEATURES:
    DOMAIN_OF[_c] = "Sub-Die Block"

#: Plain-English gloss so a process engineer reads meaning, not column names.
GLOSS = {
    "max_rolling_mean_400": "strongest sustained sub-die burst over a 400-reading window",
    "max_rolling_mean_200": "strongest sustained sub-die burst over a 200-reading window",
    "max_rolling_mean_100": "strongest sustained sub-die burst over a 100-reading window",
    "max_rolling_mean_50": "strongest sustained sub-die burst over a 50-reading window",
    "max_rolling_mean_100_start_idx": "position along the trace where the strongest burst begins",
    "block_mean": "mean amplitude across all 2,000 sub-die readings",
    "block_mean_top200": "mean of the 200 most extreme sub-die readings",
    "block_mean_top100": "mean of the 100 most extreme sub-die readings",
    "block_mean_top50": "mean of the 50 most extreme sub-die readings",
    "block_mean_top10": "mean of the 10 most extreme sub-die readings",
    "block_q75": "75th percentile of this die's sub-die readings",
    "block_q95": "95th percentile of this die's sub-die readings",
    "block_q99": "99th percentile of this die's sub-die readings",
    "block_std": "spread of this die's sub-die readings",
    "block_max_z": "most extreme standardised sub-die reading",
    "largest_contiguous_anomaly_run": "longest unbroken run of abnormal sub-die readings",
    "wafer_old_fail_rate": "share of this wafer already failing before test",
    "wafer_old_fail_count": "number of dies on this wafer already failing before test",
    "wafer_die_count": "how many dies this wafer carries",
    "distance_to_edge": "distance from the wafer edge",
    "distance_to_nearest_old_failure": "distance to the nearest pre-existing failure",
    "radius": "normalised distance from the wafer centre",
    "radius_squared": "squared normalised distance from the wafer centre",
    "normalized_row": "vertical position on the wafer",
    "normalized_col": "horizontal position on the wafer",
}
for _w in (3, 5, 7, 9, 11):
    GLOSS[f"old_fail_density_{_w}x{_w}"] = f"density of pre-existing failures in the {_w}x{_w} neighbourhood"
    GLOSS[f"old_fail_count_{_w}x{_w}"] = f"count of pre-existing failures in the {_w}x{_w} neighbourhood"


def gloss(name: str) -> str:
    if name in GLOSS:
        return GLOSS[name]
    if name.startswith("feature_"):
        return f"die-level parametric test {name.split('_')[-1]}"
    return name


def f1_optimal(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]), float(f1[i])


def metrics_at(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = (p >= t).astype(np.int8)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(t),
        "auc_pr": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_score": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def main() -> None:
    print("=" * 78)
    print("PER-DIE INTERPRETABILITY FOR OFFICIAL MODEL A AND MODEL B")
    print("=" * 78)

    need = ["wafer_id", "die_row", "die_col", "old_label", "label"] + MODEL_B_FEATURES
    print(f"loading {TEST_PARQUET.name} ...")
    import pyarrow.parquet as pq

    df_all = pq.read_table(TEST_PARQUET, columns=need).to_pandas()
    elig = df_all["old_label"].to_numpy() == 0
    df = df_all.loc[elig].reset_index(drop=True)
    y = df["label"].to_numpy().astype(np.int8)
    print(f"  {len(df_all):,} total dies -> {len(df):,} eligible, {int(y.sum()):,} true failures "
          f"({100 * y.mean():.3f}%)")

    bst = {
        "A": lgb.Booster(model_file=str(MODELS_DIR / "model_a.txt")),
        "B": lgb.Booster(model_file=str(MODELS_DIR / "model_b.txt")),
    }
    feats = {"A": MODEL_A_FEATURES, "B": MODEL_B_FEATURES}

    X = {k: df.loc[:, v].to_numpy(np.float32) for k, v in feats.items()}
    prob = {k: np.asarray(bst[k].predict(X[k]), dtype=np.float64) for k in ("A", "B")}

    # ---------------- comparison table, including accuracy ----------------
    print("\n--- MODEL A vs MODEL B (LightGBM single boosters, unseen test) ---")
    comp = {}
    for k in ("A", "B"):
        t, _ = f1_optimal(y, prob[k])
        comp[k] = metrics_at(y, prob[k], t)
        m = comp[k]
        print(f"  Model {k}: AUC-PR {m['auc_pr']:.5f}  ROC-AUC {m['roc_auc']:.5f}  "
              f"acc {m['accuracy']:.5f}  F1 {m['f1_score']:.5f}  "
              f"P {100*m['precision']:.2f}%  R {100*m['recall']:.2f}%")
    trivial = metrics_at(y, np.zeros_like(prob["A"]), 0.5)
    print(f"  predict-all-pass baseline: accuracy {trivial['accuracy']:.5f} with "
          f"{trivial['true_positives']} defects found")
    comp["predict_all_pass_baseline"] = trivial
    comp["lift_a_to_b"] = {
        "auc_pr_delta": comp["B"]["auc_pr"] - comp["A"]["auc_pr"],
        "auc_pr_relative_pct": 100 * (comp["B"]["auc_pr"] - comp["A"]["auc_pr"]) / comp["A"]["auc_pr"],
        "f1_delta_at_own_optima": comp["B"]["f1_score"] - comp["A"]["f1_score"],
        "recall_delta_points": 100 * (comp["B"]["recall"] - comp["A"]["recall"]),
        "extra_true_positives": comp["B"]["true_positives"] - comp["A"]["true_positives"],
    }
    comp["note_on_accuracy"] = (
        "Accuracy is reported because the problem statement asks for it, but it is "
        "close to uninformative at 3.556% prevalence: predicting PASS for every die "
        f"scores {trivial['accuracy']:.4f} accuracy while finding zero defects. AUC-PR "
        "and positive-class F1 are the metrics that separate these models."
    )
    print(f"  A->B: AUC-PR {comp['lift_a_to_b']['auc_pr_delta']:+.5f} "
          f"({comp['lift_a_to_b']['auc_pr_relative_pct']:+.2f}%), "
          f"F1 {comp['lift_a_to_b']['f1_delta_at_own_optima']:+.5f}, "
          f"recall {comp['lift_a_to_b']['recall_delta_points']:+.2f} pts, "
          f"{comp['lift_a_to_b']['extra_true_positives']:+,} defects")

    # ---------------- selection for attribution (no labels used) ----------------
    thr = {k: comp[k]["threshold"] for k in ("A", "B")}
    flagged = (prob["A"] >= thr["A"]) | (prob["B"] >= thr["B"])
    rng = np.random.default_rng(SEED)
    pool = np.flatnonzero(~flagged)
    ctrl = rng.choice(pool, size=int(min(N_CONTROLS, pool.size)), replace=False)
    sel = np.union1d(np.flatnonzero(flagged), ctrl)
    print(f"\nexplaining {sel.size:,} dies = {int(flagged.sum()):,} flagged by either model "
          f"+ {ctrl.size:,} random controls (selection uses no labels)")

    summary = {
        "deliverable": "per-die interpretability for Official Model A and Official Model B",
        "explained_models": {
            "A": "LightGBM Model A (519 features), exact native TreeSHAP",
            "B": "LightGBM Model B (555 features), exact native TreeSHAP",
        },
        "shap_method": "LightGBM predict(pred_contrib=True); exact, no sampling, no surrogate",
        "selection_rule": (
            f"all {int(flagged.sum())} dies flagged by Model A or Model B at their own "
            f"F1-optimal thresholds, plus {ctrl.size} uniformly random unflagged controls. "
            "Ground-truth labels are NOT used to select dies."
        ),
        "n_eligible_test_dies": int(len(df)),
        "n_explained": int(sel.size),
        "comparison": comp,
        "models": {},
    }

    shap_store = {}
    for k in ("A", "B"):
        print(f"\n--- exact TreeSHAP for Model {k} ---")
        contrib = np.asarray(bst[k].predict(X[k][sel], pred_contrib=True), dtype=np.float64)
        sv, base = contrib[:, :-1], contrib[:, -1]
        margin = np.asarray(bst[k].predict(X[k][sel], raw_score=True), dtype=np.float64)
        err = float(np.max(np.abs(sv.sum(axis=1) + base - margin)))
        print(f"  additivity check vs raw margin: max abs error {err:.3e}")

        names = feats[k]
        mean_abs = np.abs(sv).mean(axis=0)
        imp = pd.DataFrame(
            {
                "feature": names,
                "mean_abs_shap": mean_abs,
                "domain": [DOMAIN_OF[c] for c in names],
                "meaning": [gloss(c) for c in names],
            }
        ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
        imp.to_csv(REPORTS_DIR / f"model_{k.lower()}_shap_global.csv", index=False)

        dom = imp.groupby("domain")["mean_abs_shap"].sum()
        share = (100 * dom / dom.sum()).round(2).sort_values(ascending=False)
        print("  domain evidence shares:")
        for d, s in share.items():
            print(f"    {d:18s} {s:6.2f}%")
        print("  top features:")
        for _, r in imp.head(8).iterrows():
            print(f"    {r['feature']:32s} {r['mean_abs_shap']:.5f}  [{r['domain']}]")

        # per-die table
        order = np.argsort(-np.abs(sv), axis=1)[:, :TOP_K]
        dom_idx = {d: np.array([i for i, c in enumerate(names) if DOMAIN_OF[c] == d])
                   for d in set(DOMAIN_OF[c] for c in names)}
        rows = []
        for i, gi in enumerate(sel):
            drivers = [
                {
                    "feature": names[j],
                    "domain": DOMAIN_OF[names[j]],
                    "meaning": gloss(names[j]),
                    "value": float(X[k][gi, j]),
                    "shap_margin": float(sv[i, j]),
                }
                for j in order[i]
            ]
            tot = float(np.abs(sv[i]).sum()) or 1.0
            shares = {d: round(100 * float(np.abs(sv[i, idx]).sum()) / tot, 2)
                      for d, idx in dom_idx.items()}
            rows.append(
                {
                    "wafer_id": str(df["wafer_id"].iloc[gi]),
                    "die_row": int(df["die_row"].iloc[gi]),
                    "die_col": int(df["die_col"].iloc[gi]),
                    "probability": float(prob[k][gi]),
                    "raw_margin": float(margin[i]),
                    "base_margin": float(base[i]),
                    "flagged": bool(prob[k][gi] >= thr[k]),
                    "top_drivers_json": json.dumps(drivers),
                    "domain_share_json": json.dumps(shares),
                }
            )
        out = REPORTS_DIR / f"model_{k.lower()}_per_die_attribution.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"  per-die attribution -> {out.name} ({len(rows):,} dies)")

        summary["models"][k] = {
            "n_features": len(names),
            "additivity_max_abs_error_margin_space": err,
            "domain_share_pct": share.to_dict(),
            "top_features": imp.head(25).to_dict(orient="records"),
            "per_die_parquet": out.name,
        }
        shap_store[k] = (sv, names)

    # ================= FIGURE 1: Model A spatial contribution =================
    print("\n--- Model A spatial contribution visualisation ---")
    sv_a, names_a = shap_store["A"]
    sp_idx = np.array([i for i, c in enumerate(names_a) if DOMAIN_OF[c] == "Spatial Context"])
    spatial_shap_sel = sv_a[:, sp_idx].sum(axis=1)
    spatial_full = np.full(len(df), np.nan)
    spatial_full[sel] = spatial_shap_sel

    counts = pd.Series(df["wafer_id"]).value_counts()
    cand = [w for w in counts.index if np.isfinite(spatial_full[df["wafer_id"].to_numpy() == w]).sum() > 60]
    picks = cand[:3] if len(cand) >= 3 else cand
    fig, axes = plt.subplots(len(picks), 4, figsize=(19, 4.6 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    for r, wid in enumerate(picks):
        m = df["wafer_id"].to_numpy() == wid
        rr = df["die_row"].to_numpy()[m]
        cc = df["die_col"].to_numpy()[m]
        pa = prob["A"][m]
        yy = y[m]
        ss = spatial_full[m]

        axes[r, 0].scatter(cc, rr, c="#2ca02c", s=9, marker="s")
        axes[r, 0].set_title(f"{wid}\nPanel 1: eligible dies (pre-test healthy)", fontsize=9)
        s1 = axes[r, 1].scatter(cc, rr, c=pa, s=9, marker="s", cmap="viridis")
        axes[r, 1].set_title("Panel 2: Model A failure probability", fontsize=9)
        plt.colorbar(s1, ax=axes[r, 1], fraction=0.046)

        cat = np.where((pa >= thr["A"]) & (yy == 1), 0,
              np.where((pa >= thr["A"]) & (yy == 0), 1,
              np.where((pa < thr["A"]) & (yy == 1), 2, 3)))
        for code, col, lab in ((0, "#d62728", "TP"), (1, "#ff7f0e", "FP"),
                               (2, "#1f77b4", "FN"), (3, "#c7c7c7", "TN")):
            q = cat == code
            if q.any():
                axes[r, 2].scatter(cc[q], rr[q], c=col, s=9, marker="s", label=lab)
        axes[r, 2].legend(fontsize=6, loc="upper right", markerscale=0.8)
        axes[r, 2].set_title(f"Panel 3: post-test outcome (T*={thr['A']:.3f})", fontsize=9)

        ok = np.isfinite(ss)
        if ok.any():
            lim = float(np.nanmax(np.abs(ss[ok]))) or 1.0
            s3 = axes[r, 3].scatter(cc[ok], rr[ok], c=ss[ok], s=9, marker="s", cmap="coolwarm",
                                    norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))
            plt.colorbar(s3, ax=axes[r, 3], fraction=0.046)
        axes[r, 3].set_title("Panel 4: SPATIAL-channel TreeSHAP\n(signed log-odds contribution)", fontsize=9)
        for a in axes[r]:
            a.invert_yaxis()
            a.set_aspect("equal")
            a.tick_params(labelsize=6)
    fig.suptitle("Official Model A - Per-Die Spatial Contribution Attribution (exact TreeSHAP, 519 features)",
                 fontsize=13, y=0.998)
    fig.tight_layout()
    f1p = FIGURES_DIR / "40_model_a_spatial_contribution.png"
    fig.savefig(f1p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {f1p.name}")

    # ================= FIGURE 2: Model B block pattern attribution =================
    print("--- Model B block reading pattern attribution ---")
    sv_b, names_b = shap_store["B"]
    bl_idx = np.array([i for i, c in enumerate(names_b) if DOMAIN_OF[c] == "Sub-Die Block"])
    block_shap = sv_b[:, bl_idx].sum(axis=1)
    y_sel = y[sel]
    pa_sel, pb_sel = prob["A"][sel], prob["B"][sel]
    rescued = (pb_sel >= thr["B"]) & (pa_sel < thr["A"]) & (y_sel == 1)

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    bn = [names_b[i] for i in bl_idx]
    bm = np.abs(sv_b[:, bl_idx]).mean(axis=0)
    o = np.argsort(bm)[-12:]
    ax[0, 0].barh([bn[i] for i in o], bm[o], color="#ff7f0e")
    ax[0, 0].set_title("Panel 1: Model B block-feature attribution\n(mean |TreeSHAP|, log-odds)", fontsize=10)
    ax[0, 0].tick_params(labelsize=7)
    ax[0, 0].set_xlabel("mean |SHAP|", fontsize=8)

    ax[0, 1].hist(block_shap[y_sel == 0], bins=60, alpha=0.6, density=True,
                  label="stayed healthy", color="#2ca02c")
    ax[0, 1].hist(block_shap[y_sel == 1], bins=60, alpha=0.6, density=True,
                  label="newly failed", color="#d62728")
    ax[0, 1].axvline(0, color="k", lw=0.8, ls="--")
    ax[0, 1].set_title("Panel 2: total block-channel evidence per die", fontsize=10)
    ax[0, 1].set_xlabel("summed block-channel SHAP (log-odds)", fontsize=8)
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].scatter(pa_sel, pb_sel, s=3, c="#c7c7c7", label="all explained dies")
    if rescued.any():
        ax[1, 0].scatter(pa_sel[rescued], pb_sel[rescued], s=12, c="#d62728",
                         label=f"rescued by B only (n={int(rescued.sum())})")
    ax[1, 0].axhline(thr["B"], color="#1f77b4", lw=0.8, ls="--")
    ax[1, 0].axvline(thr["A"], color="#1f77b4", lw=0.8, ls="--")
    ax[1, 0].set_xlabel("Model A probability", fontsize=8)
    ax[1, 0].set_ylabel("Model B probability", fontsize=8)
    ax[1, 0].set_title("Panel 3: dies the block channel rescues", fontsize=10)
    ax[1, 0].legend(fontsize=8)

    if rescued.any():
        rb = np.abs(sv_b[rescued][:, bl_idx]).mean(axis=0)
        o2 = np.argsort(rb)[-12:]
        ax[1, 1].barh([bn[i] for i in o2], rb[o2], color="#d62728")
        ax[1, 1].set_title(f"Panel 4: what drove the {int(rescued.sum())} rescued dies\n"
                           "(block-feature mean |SHAP|)", fontsize=10)
    else:
        ax[1, 1].text(0.5, 0.5, "no rescued dies in the explained sample",
                      ha="center", va="center")
    ax[1, 1].tick_params(labelsize=7)
    fig.suptitle("Official Model B - Sub-Die Block Reading Pattern Attribution (exact TreeSHAP, 555 features)",
                 fontsize=13)
    fig.tight_layout()
    f2p = FIGURES_DIR / "41_model_b_block_attribution.png"
    fig.savefig(f2p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {f2p.name}")

    summary["model_a_spatial_figure"] = f1p.name
    summary["model_b_block_figure"] = f2p.name
    summary["model_b_rescued_dies_in_sample"] = int(rescued.sum())
    summary["wafers_in_spatial_figure"] = [str(w) for w in picks]

    with open(REPORTS_DIR / "model_ab_interpretability.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(REPORTS_DIR / "model_ab_comparison_with_accuracy.json", "w", encoding="utf-8") as fh:
        json.dump(comp, fh, indent=2, default=float)
    print("\nwritten: reports/model_ab_interpretability.json")
    print("written: reports/model_ab_comparison_with_accuracy.json")
    print("DONE")


if __name__ == "__main__":
    main()
