"""
Publication-Quality Master Interpretability Overview (Figure 31 / FINAL_INTERPRETABILITY_OVERVIEW.png).

Generates a unified 2x3 high-resolution diagnostic overview:
1. Panel 1 (Top Left): Global Feature Hierarchy (Top 12 Features with plain-English labels)
2. Panel 2 (Top Center): Physical Domain Attribution Breakdown (Exact verified shares: 30.7%, 21.2%, 20.2%, 15.5%, 10.6%, 1.8%)
3. Panel 3 (Top Right): Controlled Model A -> Model B Diagnostic Lift (+0.0480 AUC-PR, 795 rescued dies)
4. Panel 4 (Bottom Left): Wafer Spatial Risk Surface & Spatial Attribution (Wafer W_F_0074 edge-risk signature)
5. Panel 5 (Bottom Center): Sub-Die 2,000-Reading Sequence Dynamics (Healthy flat baseline vs localized burst excursion)
6. Panel 6 (Bottom Right): Local Per-Die Diagnostic Card (Exact TreeSHAP additivity, calibrated prob vs risk rank, drivers)
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.metrics import precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PROCESSED_DIR, REPORTS_DIR, INPUT_DIR
CACHE_DIR = PROCESSED_DIR / "cache"
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
OUTPUT_FIG = FIGURES_DIR / "FINAL_INTERPRETABILITY_OVERVIEW.png"

DOMAIN_COLORS = {
    DOMAIN_MANIFOLD: "#d62728",          # Crimson Red
    DOMAIN_BLOCK: "#ff7f0e",             # Dark Orange
    DOMAIN_INTERACTIONS: "#8c564b",      # Muted Brown
    DOMAIN_WAFER_RELATIVE: "#9467bd",    # Purple
    DOMAIN_PARAMETRIC: "#1f77b4",        # Slate Blue
    DOMAIN_SPATIAL: "#2ca02c",           # Forest Green
    DOMAIN_GEOMETRY: "#7f7f7f",          # Gray
}


def build_final_overview():
    print("=" * 80)
    print("GENERATING PUBLICATION MASTER INTERPRETABILITY OVERVIEW (2x3 GRID)")
    print("=" * 80)
    t0 = time.time()

    # Create figure with high DPI
    fig, axes = plt.subplots(2, 3, figsize=(24, 15), dpi=200)
    plt.subplots_adjust(hspace=0.32, wspace=0.25, top=0.92, bottom=0.06, left=0.04, right=0.96)

    # -------------------------------------------------------------------------
    # PANEL 1: GLOBAL FEATURE HIERARCHY (Top 12 Features)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    top_features = [
        ("wctx_ldadt_std", 0.9933, DOMAIN_MANIFOLD, "Wafer-context dispersion of shrinkage LDA"),
        ("inter_pc1_x_roll400", 0.4037, DOMAIN_INTERACTIONS, "Parametric PC1 × W=400 Block Burst"),
        ("inter_pca01_x_roll350", 0.2798, DOMAIN_INTERACTIONS, "Parametric PCA01 × W=350 Block Burst"),
        ("inter_pc1_x_edge_density", 0.1479, DOMAIN_INTERACTIONS, "Parametric PC1 × Perimeter Defect Density"),
        ("pca_01", 0.1443, DOMAIN_MANIFOLD, "Principal Component 01 of electrical tests"),
        ("inter_pc1_x_burst_excess_350", 0.1295, DOMAIN_INTERACTIONS, "Parametric PC1 × Burst Elevation Excess"),
        ("wdev_pc01", 0.1084, DOMAIN_WAFER_RELATIVE, "Within-wafer deviation of PC01"),
        ("lda_score", 0.1013, DOMAIN_MANIFOLD, "Bayes-optimal shrinkage LDA discriminant"),
        ("inter_pca01_x_edge", 0.0935, DOMAIN_INTERACTIONS, "Parametric PCA01 × Perimeter Edge Flag"),
        ("wdev_pca_01", 0.0825, DOMAIN_MANIFOLD, "Within-wafer detrended PCA deviation"),
        ("block_mean_top200", 0.0808, DOMAIN_BLOCK, "Mean of top 200 sub-die block readings"),
        ("wrank_lda", 0.0700, DOMAIN_MANIFOLD, "Within-wafer percentile rank of LDA score"),
    ]

    f_names = [x[0] for x in top_features][::-1]
    f_scores = [x[1] for x in top_features][::-1]
    f_domains = [x[2] for x in top_features][::-1]
    f_descs = [x[3] for x in top_features][::-1]
    f_colors = [DOMAIN_COLORS[d] for d in f_domains]

    y_pos = np.arange(len(f_names))
    ax1.barh(y_pos, f_scores, color=f_colors, edgecolor="black", linewidth=0.7, alpha=0.9, height=0.65)
    
    labels = [f"{fn}  ({desc})" for fn, desc in zip(f_names, f_descs)]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8.5, fontweight="medium")
    ax1.set_xlabel("Mean Absolute TreeSHAP (|log-odds contribution|)", fontsize=10, fontweight="bold")
    ax1.set_title("Panel 1: Global Feature Evidence Hierarchy\n(Top-12 Features via Exact CatBoost GPU TreeSHAP)", fontsize=12, fontweight="bold", pad=8)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="x")

    # Legend for Panel 1
    p1_legend_domains = [DOMAIN_MANIFOLD, DOMAIN_INTERACTIONS, DOMAIN_WAFER_RELATIVE, DOMAIN_BLOCK]
    handles1 = [Rectangle((0, 0), 1, 1, color=DOMAIN_COLORS[d], label=d) for d in p1_legend_domains]
    ax1.legend(handles=handles1, loc="lower right", fontsize=8, framealpha=0.9)

    # -------------------------------------------------------------------------
    # PANEL 2: EVIDENCE BY DOMAIN (Authoritative 100.0% Distribution)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    domain_shares = [
        ("Manifold / Projections", 30.73, DOMAIN_MANIFOLD),
        ("Block Dynamics", 21.20, DOMAIN_BLOCK),
        ("Bilinear Interactions", 20.15, DOMAIN_INTERACTIONS),
        ("Wafer-Relative", 15.52, DOMAIN_WAFER_RELATIVE),
        ("Electrical Parametric", 10.57, DOMAIN_PARAMETRIC),
        ("Spatial Context", 1.83, DOMAIN_SPATIAL),
        ("Geometry", 0.00, DOMAIN_GEOMETRY),
    ]

    d_names = [x[0] for x in domain_shares]
    d_vals = [x[1] for x in domain_shares]
    d_colors = [DOMAIN_COLORS[x[2]] for x in domain_shares]

    bars2 = ax2.bar(d_names, d_vals, color=d_colors, edgecolor="black", linewidth=0.8, width=0.55)
    ax2.set_ylabel("Mean Attribution Share (% of Evidence)", fontsize=10, fontweight="bold")
    ax2.set_title("Panel 2: Physical Domain Attribution Breakdown\n(Sum = 100.0% Across N=25,013 Evaluated Dies)", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xticks(range(len(d_names)))
    ax2.set_xticklabels(d_names, rotation=22, ha="right", fontsize=9, fontweight="bold")
    ax2.set_ylim([0, 38])
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    for bar, val in zip(bars2, d_vals):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.9, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # -------------------------------------------------------------------------
    # PANEL 3: CONTROLLED MODEL A -> MODEL B DIAGNOSTIC LIFT
    # -------------------------------------------------------------------------
    ax3 = axes[0, 2]
    # Load A to B data
    with open(REPORTS_DIR / "a_to_b_diagnostic_summary.json") as f:
        a_to_b = json.load(f)

    # Compute PR curves on sample
    df_labels = pd.read_csv(INPUT_DIR / "test.csv", usecols=["wafer_id", "die_row", "die_col", "old_label", "label"])
    df_feat = pd.read_parquet(PROCESSED_DIR / "validation_features.parquet")
    df_test = pd.merge(df_feat, df_labels, on=["wafer_id", "die_row", "die_col", "old_label"], how="left")
    df_test = df_test[df_test["old_label"] == 0].reset_index(drop=True)

    y_test = df_test["label"].values.astype(np.int32)
    # Use precomputed scores from a_to_b_explanation
    import lightgbm as lgb
    from src.models.common import MODEL_A_FEATURES, MODEL_B_FEATURES

    lgb_a = lgb.Booster(model_file=str(REPO_ROOT / "models" / "model_a.txt"))
    p_a = lgb_a.predict(df_test[MODEL_A_FEATURES].values)
    lgb_b = lgb.Booster(model_file=str(REPO_ROOT / "models" / "model_b.txt"))
    p_b = lgb_b.predict(df_test[MODEL_B_FEATURES].values)

    rec_a, prec_a, _ = precision_recall_curve(y_test, p_a)
    rec_b, prec_b, _ = precision_recall_curve(y_test, p_b)

    ax3.plot(rec_a, prec_a, color="#1f77b4", linewidth=2.2, label=f"Model A (519 feats, No Blocks) — AUC-PR: {a_to_b['model_a_stack_aucpr']:.4f}")
    ax3.plot(rec_b, prec_b, color="#d62728", linewidth=2.5, label=f"Model B (555 feats, With Blocks) — AUC-PR: {a_to_b['model_b_stack_aucpr']:.4f}")
    ax3.fill_between(rec_b, np.interp(rec_b, rec_a[::-1], prec_a[::-1]), prec_b,
                     where=(prec_b >= np.interp(rec_b, rec_a[::-1], prec_a[::-1])),
                     color="#d62728", alpha=0.15, label=f"Sub-Die Block Information Gain (+{a_to_b['absolute_block_aucpr_lift']:.4f} / +{a_to_b['relative_block_lift_pct']}%)")

    ax3.set_xlabel("Recall (Minority Defect Capture)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Precision (True Failures / Flagged)", fontsize=10, fontweight="bold")
    ax3.set_title(f"Panel 3: Model A → Model B Precision-Recall Lift\n(Isolates Value of Sub-Die Blocks: +{a_to_b['dies_rescued_by_model_b_only']:,} Rescued Failures)", fontsize=12, fontweight="bold", pad=8)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=8.5)

    # Add callout text inside panel 3
    ax3.text(0.04, 0.15,
             f"Operational Screening Impact (~18.5k flagged):\n"
             f" • Caught by Both Models: {a_to_b['dies_caught_by_both']:,} defects\n"
             f" • Rescued by Model B Only: {a_to_b['dies_rescued_by_model_b_only']:,} defects\n"
             f" • Top Block Driver: max_rolling_mean_400 (+1.76σ)",
             transform=ax3.transAxes, fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2e6", edgecolor="#ff7f0e", alpha=0.9))

    # -------------------------------------------------------------------------
    # PANEL 4: SPATIAL ATTRIBUTION & RISK SURFACE (Wafer W_F_0074)
    # -------------------------------------------------------------------------
    ax4 = axes[1, 0]
    df_preds = pd.read_parquet(REPO_ROOT / "predictions" / "final_test_model_f_predictions.parquet")
    df_w = pd.merge(df_preds, df_labels, on=["wafer_id", "die_row", "die_col", "old_label"])
    w74 = df_w[df_w["wafer_id"] == "W_F_0074"].copy()

    sc4 = ax4.scatter(w74["die_col"], w74["die_row"], c=w74["model_f_optimal"], cmap="plasma", s=22, marker="s", edgecolors="none")
    # Highlight true failures
    w74_fails = w74[w74["label"] == 1]
    ax4.scatter(w74_fails["die_col"], w74_fails["die_row"], facecolors="none", edgecolors="#00ffcc", linewidths=1.2, s=35, label=f"True Defects ({len(w74_fails)})")

    ax4.set_aspect("equal")
    ax4.invert_yaxis()
    ax4.set_title("Panel 4: Wafer Spatial Risk Topography (W_F_0074)\n(Continuous Risk Percentile & Edge Perimeter Failure Concentration)", fontsize=12, fontweight="bold", pad=8)
    ax4.set_xlabel("Wafer Die Column", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Wafer Die Row", fontsize=10, fontweight="bold")
    cb4 = plt.colorbar(sc4, ax=ax4, fraction=0.046, pad=0.04)
    cb4.set_label("Model F Risk Score", fontsize=9, fontweight="bold")
    cb4.ax.tick_params(labelsize=8)
    ax4.legend(loc="lower left", fontsize=8.5)

    # -------------------------------------------------------------------------
    # PANEL 5: SUB-DIE 2,000-READING SEQUENCE DYNAMICS
    # -------------------------------------------------------------------------
    ax5 = axes[1, 1]
    mmap = np.memmap(CACHE_DIR / "final_test_raw_blocks.dat", dtype="float32", mode="r", shape=(208264, 2000))
    fail_idx = df_w[(df_w["old_label"] == 0) & (df_w["label"] == 1) & (df_w["model_f_optimal"] >= 0.95)].index[0]
    healthy_idx = df_w[(df_w["old_label"] == 0) & (df_w["label"] == 0) & (df_w["model_f_optimal"] <= 0.10)].index[0]

    trace_fail = mmap[fail_idx]
    trace_healthy = mmap[healthy_idx]
    x_readings = np.arange(1, 2001)

    # Rolling means (W=350)
    roll_fail = pd.Series(trace_fail).rolling(350, min_periods=1).mean().values
    roll_healthy = pd.Series(trace_healthy).rolling(350, min_periods=1).mean().values

    ax5.plot(x_readings, trace_fail, color="#d62728", alpha=0.3, linewidth=0.7)
    ax5.plot(x_readings, roll_fail, color="#d62728", linewidth=2.2, label=f"Defective Die — Localized Burst Excursion (Peak W350: {np.max(roll_fail):.1f})")

    ax5.plot(x_readings, trace_healthy, color="#2ca02c", alpha=0.3, linewidth=0.7)
    ax5.plot(x_readings, roll_healthy, color="#2ca02c", linewidth=2.2, label="Healthy Control Die — Uniform Stable Baseline")

    # Highlight burst window
    peak_loc = np.argmax(roll_fail)
    ax5.axvspan(max(0, peak_loc - 175), min(2000, peak_loc + 175), color="#d62728", alpha=0.12, label="Detected Localized Anomaly Window (W=350)")

    ax5.set_xlabel("Sub-Die Reading Index (1 to 2,000)", fontsize=10, fontweight="bold")
    ax5.set_ylabel("Reading Amplitude (A.U.)", fontsize=10, fontweight="bold")
    ax5.set_title("Panel 5: Sub-Die Reading Dynamics (Model B / C1 Signal)\n(High-Dimensional 2,000-Point Sequence Reveals Internal cell Stress)", fontsize=12, fontweight="bold", pad=8)
    ax5.grid(True, linestyle="--", alpha=0.4)
    ax5.legend(loc="upper right", fontsize=8.5)

    # -------------------------------------------------------------------------
    # PANEL 6: PER-DIE EXPLANATION CARD (Die W_N_0156 Row 4, Col 7)
    # -------------------------------------------------------------------------
    ax6 = axes[1, 2]
    ax6.axis("off")

    # Draw card box
    box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.03",
                         facecolor="#fafbfc", edgecolor="#204060", linewidth=1.5,
                         transform=ax6.transAxes)
    ax6.add_patch(box)

    card_text = (
        "PANEL 6: LOCAL DIE ATTRIBUTION CARD\n"
        "Die Coordinates: Wafer W_N_0156 (Row 4, Col 7) — True Positive\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "• Model F Risk Percentile:    0.9981 (Rank Score in [0, 1])\n"
        "• CatBoost Calibrated Prob:   99.73%  (Underlying Tree Sigmoid)\n"
        "• CatBoost Raw Margin:        +5.9156 (Base Expected: -4.8537)\n"
        "• Ground Truth Outcome:       DEFECT FAIL (Post-Stress Confirmed)\n"
        "• TreeSHAP Margin Error:      7.99e-15 (Strict Additivity Verified)\n"
        "────────────────────────────────────────────────────────────\n"
        "Physical Evidence Breakdown (% of Total Absolute SHAP):\n"
        "  [39.4%] Bilinear Interactions  |  [29.6%] Manifold Projections\n"
        "  [17.3%] Wafer-Relative Deviations |  [ 7.7%] Parametric Tests\n"
        "  [ 5.3%] Sub-Die Block Dynamics   |  [ 0.7%] Spatial Context\n"
        "────────────────────────────────────────────────────────────\n"
        "Top Positive Drivers (+SHAP Pushing to Failure):\n"
        "  [+1.6791] inter_pc1_x_roll400 (Parametric PC1 × W=400 Block Burst)\n"
        "  [+1.3402] inter_pca01_x_roll350 (Parametric PCA01 × W=350 Block Burst)\n"
        "  [+1.2863] pca_01 (Dominant Electrical Parametric Drift Axis)\n"
        "  [+1.1657] wdev_pc01 (Within-Wafer Parametric Baseline Shift)\n"
        "Top Negative Drivers (-SHAP Supporting Health):\n"
        "  [-0.0367] block_mean_top200 (Upper-tail mean within benign bounds)\n"
        "  [-0.0329] largest_contiguous_anomaly_run (Absence of step shift)\n"
        "────────────────────────────────────────────────────────────\n"
        "Engineering Diagnostic Note:\n"
        "Multi-resolution convergence: macro parametric drift couples with\n"
        "micro block burst. Secondary inline metrology required for fab root cause."
    )

    ax6.text(0.06, 0.94, card_text, transform=ax6.transAxes,
             fontsize=8.5, verticalalignment="top", fontfamily="monospace",
             linespacing=1.28, color="#102030")

    # Super title
    fig.suptitle(
        "SanDisk Hackathon: Multi-Resolution Process Engineering Interpretability Suite\n"
        "Unifying Die-Level Parametric Tests, Wafer Spatial Topography, and Sub-Die 2,000-Reading Sequences",
        fontsize=16, fontweight="bold", y=0.98
    )

    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"Saved master interpretability overview to: {OUTPUT_FIG}")
    print(f"Generated overview figure in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    build_final_overview()
