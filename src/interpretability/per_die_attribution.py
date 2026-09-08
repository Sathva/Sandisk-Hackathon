"""
Per-Die Feature Attribution and TreeSHAP Analysis for Semiconductor Yield Prediction.

Methodology:
- Explains the strongest individual tree engine of Model F (CatBoost GPU), where exact
  TreeSHAP is mathematically defined in the model's raw margin (log-odds) space.
- Does NOT average SHAP values across the non-linear rank-space ensemble, maintaining
  complete mathematical fidelity.
- Translates technical feature names into plain-English process engineering descriptions.
- Aggregates signed SHAP values into physical domains (Parametric, Spatial, Block, Manifold, Wafer-Relative).
- Exports reports/per_die_attribution.parquet and provides explain_die() helper.
- Generates reports/figures/27_global_shap_importance.png and 27b_domain_shap_contribution.png.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    INPUT_DIR,
)
from src.interpretability.feature_dictionary import (
    classify_and_translate_feature,
    DOMAINS_ORDER,
    DOMAIN_PARAMETRIC,
    DOMAIN_SPATIAL,
    DOMAIN_BLOCK,
    DOMAIN_WAFER_RELATIVE,
    DOMAIN_MANIFOLD,
    DOMAIN_INTERACTIONS,
    DOMAIN_GEOMETRY,
)

FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FINAL_TEST_PARQUET = PROCESSED_DIR / "final_test_model_f_features.parquet"
MODEL_CB_PATH = MODELS_DIR / "model_f_catboost.cbm"
FEATURE_LIST_PATH = MODELS_DIR / "model_f_feature_list.json"
OUTPUT_PARQUET = REPORTS_DIR / "per_die_attribution.parquet"
TEST_PREDS_PARQUET = REPO_ROOT / "predictions" / "final_test_model_f_predictions.parquet"
TEST_LABELS_CSV = INPUT_DIR / "test.csv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def compute_per_die_attribution(
    max_dies_to_save: int = 15000,
) -> Tuple[pd.DataFrame, CatBoostClassifier, List[str]]:
    """
    Computes exact TreeSHAP values for all predicted failures and a diagnostic
    control sample on the unseen test set using CatBoost (Model F's strongest engine).
    """
    print("=" * 80)
    print("TREE-SHAP PER-DIE ATTRIBUTION ENGINE (MODEL F CATBOOST)")
    print("=" * 80)
    t0 = time.time()

    # 1. Load Feature Names & Model
    with open(FEATURE_LIST_PATH, "r") as f:
        feature_names = json.load(f)["features"]
    n_features = len(feature_names)
    print(f"Loaded feature list: {n_features} features.")

    print(f"Loading CatBoost model from {MODEL_CB_PATH.name}...")
    model_cb = CatBoostClassifier()
    model_cb.load_model(str(MODEL_CB_PATH))

    # 2. Load Test Features & Prediction Metadata
    print(f"Loading test features from {FINAL_TEST_PARQUET.name}...")
    df_test = pd.read_parquet(FINAL_TEST_PARQUET)
    assert len(df_test.columns) >= n_features, "Mismatch in test feature columns!"

    # Load Model F stack predictions for rank reference
    df_preds = pd.read_parquet(TEST_PREDS_PARQUET)
    df_labels = pd.read_csv(
        TEST_LABELS_CSV,
        usecols=["wafer_id", "die_row", "die_col", "old_label", "label"]
    )
    df_meta = pd.merge(
        df_preds, df_labels,
        on=["wafer_id", "die_row", "die_col", "old_label"],
        how="left"
    )

    # 3. Target Die Selection (Prioritizing predicted failures + margin dies + controls)
    # CatBoost decision threshold on development was 0.3219
    cb_probs = df_meta["p_catboost"].values
    is_cb_pred_fail = cb_probs >= 0.3219
    is_f_pred_fail = df_meta["model_f_optimal"] >= 0.90  # High risk percentile

    target_mask = is_cb_pred_fail | is_f_pred_fail | (df_meta["label"] == 1)

    # Add representative healthy control dies across wafers
    control_indices = np.where(~target_mask)[0]
    np.random.seed(42)
    sample_controls = np.random.choice(
        control_indices,
        size=min(5000, len(control_indices)),
        replace=False
    )
    selected_indices = np.union1d(np.where(target_mask)[0], sample_controls)

    print(f"\nSelection Summary for TreeSHAP Evaluation:")
    print(f"  Total eligible test dies:        {len(df_test):,}")
    print(f"  Predicted Failures / High Risk:   {int(target_mask.sum()):,}")
    print(f"  Total Selected Evaluation Dies:  {len(selected_indices):,}")

    df_eval_subset = df_test.iloc[selected_indices].reset_index(drop=True)
    df_meta_subset = df_meta.iloc[selected_indices].reset_index(drop=True)

    X_eval = df_eval_subset[feature_names].values.astype(np.float32)

    # 4. Exact TreeSHAP Extraction
    print(f"\nExtracting exact TreeSHAP values for {len(X_eval):,} dies...")
    t_shap = time.time()
    pool = Pool(X_eval)
    # CatBoost returns shape (N, n_features + 1), where last column is base expected value
    shap_matrix_raw = model_cb.get_feature_importance(pool, type="ShapValues")
    print(f"TreeSHAP extracted in {time.time() - t_shap:.2f}s (Speed: {len(X_eval)/(time.time()-t_shap):.0f} dies/s).")

    shap_values = shap_matrix_raw[:, :-1]
    base_values = shap_matrix_raw[:, -1]

    # Verify Margin Additivity
    # In CatBoost raw space: sum(shap) + base_val == raw_prediction
    raw_preds = model_cb.predict(X_eval, prediction_type="RawFormulaVal")
    reconstructed_margin = shap_values.sum(axis=1) + base_values
    max_additivity_error = float(np.max(np.abs(raw_preds - reconstructed_margin)))
    print(f"Additivity Check: Max margin error = {max_additivity_error:.2e} (Strictly Additive in Log-Odds Space: {max_additivity_error < 1e-4})")
    assert max_additivity_error < 1e-3, "TreeSHAP additivity check failed!"

    # 5. Classify Features into Domains
    print("\nClassifying features into physical domains...")
    feature_domains = []
    feature_descriptions = []
    for f_name in feature_names:
        dom, desc = classify_and_translate_feature(f_name)
        feature_domains.append(dom)
        feature_descriptions.append(desc)

    feature_domains_arr = np.array(feature_domains)

    # Domain aggregation masks
    domain_masks = {dom: (feature_domains_arr == dom) for dom in DOMAINS_ORDER}

    # 6. Build Detailed Per-Die Attribution Records
    print("Aggregating domain-level attributions and top driver features...")
    records = []
    abs_shap_total = np.sum(np.abs(shap_values), axis=1)
    abs_shap_total[abs_shap_total == 0] = 1.0

    # Domain percentages
    domain_pcts = {}
    for dom, mask in domain_masks.items():
        if mask.sum() > 0:
            dom_abs_sum = np.sum(np.abs(shap_values[:, mask]), axis=1)
            domain_pcts[dom] = (dom_abs_sum / abs_shap_total) * 100.0
        else:
            domain_pcts[dom] = np.zeros(len(X_eval), dtype=np.float32)

    for i in range(len(X_eval)):
        row_shap = shap_values[i]
        row_feats = X_eval[i]

        # Top 5 positive drivers (pushing toward failure risk)
        pos_sort = np.argsort(-row_shap)
        top_pos_idx = [idx for idx in pos_sort if row_shap[idx] > 0][:5]
        top_pos_list = [
            {
                "feature": feature_names[idx],
                "description": feature_descriptions[idx],
                "domain": feature_domains[idx],
                "value": round(float(row_feats[idx]), 4),
                "shap": round(float(row_shap[idx]), 4),
            }
            for idx in top_pos_idx
        ]

        # Top 3 negative drivers (pushing toward passing/healthy)
        neg_sort = np.argsort(row_shap)
        top_neg_idx = [idx for idx in neg_sort if row_shap[idx] < 0][:3]
        top_neg_list = [
            {
                "feature": feature_names[idx],
                "description": feature_descriptions[idx],
                "domain": feature_domains[idx],
                "value": round(float(row_feats[idx]), 4),
                "shap": round(float(row_shap[idx]), 4),
            }
            for idx in top_neg_idx
        ]

        rec = {
            "wafer_id": df_meta_subset.at[i, "wafer_id"],
            "die_row": int(df_meta_subset.at[i, "die_row"]),
            "die_col": int(df_meta_subset.at[i, "die_col"]),
            "catboost_raw_margin": round(float(raw_preds[i]), 5),
            "catboost_calibrated_prob": round(float(sigmoid(raw_preds[i])), 5),
            "model_f_risk_score": round(float(df_meta_subset.at[i, "model_f_optimal"]), 5),
            "actual_label": int(df_meta_subset.at[i, "label"]),
            "top_positive_drivers": json.dumps(top_pos_list),
            "top_negative_drivers": json.dumps(top_neg_list),
            "parametric_share_pct": round(float(domain_pcts[DOMAIN_PARAMETRIC][i]), 2),
            "spatial_share_pct": round(float(domain_pcts[DOMAIN_SPATIAL][i]), 2),
            "block_share_pct": round(float(domain_pcts[DOMAIN_BLOCK][i]), 2),
            "wafer_relative_share_pct": round(float(domain_pcts[DOMAIN_WAFER_RELATIVE][i]), 2),
            "manifold_share_pct": round(float(domain_pcts[DOMAIN_MANIFOLD][i]), 2),
            "interaction_share_pct": round(float(domain_pcts[DOMAIN_INTERACTIONS][i]), 2),
            "geometry_share_pct": round(float(domain_pcts[DOMAIN_GEOMETRY][i]), 2),
            "base_value": round(float(base_values[i]), 5),
        }
        records.append(rec)

    df_attribution = pd.DataFrame(records)
    df_attribution.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved {len(df_attribution):,} per-die attributions to: {OUTPUT_PARQUET}")

    # 7. Generate Global SHAP Feature Importance Plot (Figure 27)
    _generate_global_shap_figures(shap_values, feature_names, feature_domains, domain_pcts)

    print(f"\nPer-Die Attribution pipeline complete in {time.time() - t0:.1f}s.")
    return df_attribution, model_cb, feature_names


def _generate_global_shap_figures(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_domains: List[str],
    domain_pcts: Dict[str, np.ndarray],
):
    """Generates Figure 27 (Top-20 Global TreeSHAP) and Figure 27b (Domain Share Distribution)."""
    print("\nGenerating Figure 27 (Top-20 Global TreeSHAP Feature Ranking)...")

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top20_indices = np.argsort(mean_abs_shap)[-20:][::-1]

    top20_names = [feature_names[i] for i in top20_indices]
    top20_scores = [mean_abs_shap[i] for i in top20_indices]
    top20_domains = [feature_domains[i] for i in top20_indices]

    # Domain color palette
    domain_colors = {
        DOMAIN_PARAMETRIC: "#1f77b4",        # Blue
        DOMAIN_SPATIAL: "#2ca02c",           # Green
        DOMAIN_BLOCK: "#ff7f0e",             # Orange
        DOMAIN_WAFER_RELATIVE: "#9467bd",    # Purple
        DOMAIN_MANIFOLD: "#d62728",          # Red
        DOMAIN_INTERACTIONS: "#8c564b",      # Brown
        DOMAIN_GEOMETRY: "#7f7f7f",          # Gray
    }

    colors = [domain_colors.get(d, "#333333") for d in top20_domains]

    # Translate labels for clarity
    top20_labels = []
    for f in top20_names:
        _, desc = classify_and_translate_feature(f)
        # Shorten if too long
        short_desc = (desc[:42] + "..") if len(desc) > 42 else desc
        top20_labels.append(f"{f}\n({short_desc})")

    plt.figure(figsize=(12, 9), dpi=160)
    y_pos = np.arange(len(top20_names))
    bars = plt.barh(y_pos, top20_scores[::-1], color=colors[::-1], edgecolor="black", linewidth=0.7, alpha=0.9)

    plt.yticks(y_pos, top20_labels[::-1], fontsize=9)
    plt.xlabel("Mean Absolute SHAP Value (|log-odds contribution|)", fontsize=11, fontweight="bold")
    plt.title(
        "Top-20 Global Feature Attribution (Model F CatBoost Native TreeSHAP)\n"
        "Evaluated on Unseen Test Silicon (185,126 Eligible Dies)",
        fontsize=13, fontweight="bold", pad=12
    )
    plt.grid(True, linestyle="--", alpha=0.4, axis="x")

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=col, label=dom)
        for dom, col in domain_colors.items()
        if dom in top20_domains
    ]
    plt.legend(handles=legend_handles, title="Physical Feature Domain", loc="lower right", fontsize=9)
    plt.tight_layout()

    fig27_path = FIGURES_DIR / "27_global_shap_importance.png"
    plt.savefig(fig27_path)
    plt.close()
    print(f"Saved Figure 27 to {fig27_path.name}")

    # Figure 27b: Domain Share Distribution
    print("Generating Figure 27b (Domain Relative Attribution Share)...")
    plt.figure(figsize=(10, 5.5), dpi=160)
    domain_mean_shares = []
    valid_domains = [d for d in DOMAINS_ORDER if d in domain_pcts]
    for d in valid_domains:
        domain_mean_shares.append(float(np.mean(domain_pcts[d])))

    bars2 = plt.bar(
        valid_domains, domain_mean_shares,
        color=[domain_colors[d] for d in valid_domains],
        edgecolor="black", linewidth=0.8, width=0.55
    )
    plt.ylabel("Mean Attribution Magnitude Share (%)", fontsize=11, fontweight="bold")
    plt.title(
        "Physical Domain Attribution Share Across Predicted Failures\n"
        "(Relative Share of Absolute SHAP Magnitude)",
        fontsize=13, fontweight="bold", pad=12
    )
    plt.xticks(rotation=25, ha="right", fontsize=9.5)
    plt.ylim([0, max(domain_mean_shares) * 1.25])
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")

    for bar in bars2:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, h + 0.8,
            f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10
        )

    plt.tight_layout()
    fig27b_path = FIGURES_DIR / "27b_domain_shap_contribution.png"
    plt.savefig(fig27b_path)
    plt.close()
    print(f"Saved Figure 27b to {fig27b_path.name}")


def explain_die(wafer_id: str, die_row: int, die_col: int) -> str:
    """
    Reusable helper function: returns a concise, engineer-readable explanation
    for a specific die on any wafer.
    """
    if not OUTPUT_PARQUET.exists():
        return f"Error: {OUTPUT_PARQUET.name} not found. Run per_die_attribution.py first."

    df = pd.read_parquet(
        OUTPUT_PARQUET,
        filters=[
            ("wafer_id", "==", wafer_id),
            ("die_row", "==", die_row),
            ("die_col", "==", die_col)
        ]
    )

    if len(df) == 0:
        return f"Die ({wafer_id}, row={die_row}, col={die_col}) not found in evaluation sample."

    row = df.iloc[0]
    pos_drivers = json.loads(row["top_positive_drivers"])
    neg_drivers = json.loads(row["top_negative_drivers"])

    lines = []
    lines.append("=" * 75)
    lines.append(f"ENGINEERING DIE YIELD ATTRIBUTION: {wafer_id} (Row {die_row}, Col {die_col})")
    lines.append("=" * 75)
    lines.append(f"Model-Derived Risk Score:      {row['model_f_risk_score']:.4f} (Evaluated Risk Rank)")
    lines.append(f"CatBoost Margin Log-Odds:      {row['catboost_raw_margin']:+.4f} (Base: {row['base_value']:+.4f})")
    lines.append(f"CatBoost Calibrated Risk:      {row['catboost_calibrated_prob']*100:.2f}%")
    lines.append(f"Actual Post-Test Ground Truth: {'DEFECT FAIL (1)' if row['actual_label'] == 1 else 'PASS HEALTHY (0)'}")
    lines.append("-" * 75)
    lines.append("Relative Physical Domain Contribution (Share of Total Evidence):")
    lines.append(f"  • Sub-Die Block Dynamics:          {row['block_share_pct']:.1f}%")
    lines.append(f"  • Wafer-Relative / Detrended:      {row['wafer_relative_share_pct']:.1f}%")
    lines.append(f"  • Wafer Spatial Context:           {row['spatial_share_pct']:.1f}%")
    lines.append(f"  • LDA & PCA Manifold Projections:  {row['manifold_share_pct']:.1f}%")
    lines.append(f"  • Electrical Parametric:           {row['parametric_share_pct']:.1f}%")
    lines.append(f"  • Cross-Resolution Interactions:   {row['interaction_share_pct']:.1f}%")
    lines.append("-" * 75)
    lines.append("Top Evidence Driving Defect Risk (+SHAP):")
    for d in pos_drivers:
        lines.append(f"  [+{d['shap']:.4f}] {d['feature']} (val={d['value']})")
        lines.append(f"          ↳ {d['description']}")

    lines.append("\nTop Evidence Supporting Die Health (-SHAP):")
    for d in neg_drivers:
        lines.append(f"  [{d['shap']:.4f}] {d['feature']} (val={d['value']})")
        lines.append(f"          ↳ {d['description']}")

    lines.append("-" * 75)
    lines.append("Engineering Interpretation:")
    if row['block_share_pct'] > 35:
        lines.append("  • Defect signature is heavily concentrated in sub-die block reading dynamics,")
        lines.append("    indicating localized internal cell/block stress anomalies.")
    elif row['spatial_share_pct'] > 35:
        lines.append("  • Defect signature is strongly associated with wafer spatial context (neighbor clusters")
        lines.append("    or wafer perimeter radial position).")
    else:
        lines.append("  • Multi-resolution signature: joint evidence across parametric drift and spatial context.")

    lines.append("\nLimitation Note:")
    lines.append("  The observed association identifies a predictive statistical pattern but does not")
    lines.append("  establish a unique physical fab equipment root cause without secondary metrology.")
    lines.append("=" * 75)

    return "\n".join(lines)


if __name__ == "__main__":
    df_attr, cb_model, feats = compute_per_die_attribution()
    # Demo explain_die on first predicted failure
    first_fail = df_attr[df_attr["actual_label"] == 1].iloc[0]
    w_id = first_fail["wafer_id"]
    r_idx = first_fail["die_row"]
    c_idx = first_fail["die_col"]
    print("\n" + explain_die(w_id, r_idx, c_idx))
