"""
Spatial Contribution Visualization for Semiconductor Wafer Yield Prediction.

Rubric Deliverable (P0-2):
- Selects 3 representative wafers from 200 unseen test wafers based on transparent criteria:
  1. Edge-concentrated risk wafer (W_F_0074: 66.7% edge defects)
  2. High-precision defect cluster wafer (W_F_0192: 572 defects, dense cluster)
  3. Challenging margin wafer (W_N_0014: 21 defects, mixed predictions)
- For each wafer, renders a 4-panel spatial diagnostic grid:
  - Panel 1: Pre-test old_label map (pre-existing defects)
  - Panel 2: Model risk score heatmap (continuous surface across wafer coordinates)
  - Panel 3: True post-test outcome with prediction overlay (TP, FP, FN, TN)
  - Panel 4: Spatial-channel signed SHAP contribution per die (exact CatBoost TreeSHAP)
- Saves reports/figures/28_wafer_spatial_attribution_maps.png.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
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
    DOMAIN_SPATIAL,
)

FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FINAL_TEST_PARQUET = PROCESSED_DIR / "final_test_model_f_features.parquet"
MODEL_CB_PATH = MODELS_DIR / "model_f_catboost.cbm"
FEATURE_LIST_PATH = MODELS_DIR / "model_f_feature_list.json"
TEST_PREDS_PARQUET = REPO_ROOT / "predictions" / "final_test_model_f_predictions.parquet"
TEST_LABELS_CSV = INPUT_DIR / "test.csv"
OUTPUT_FIG = FIGURES_DIR / "28_wafer_spatial_attribution_maps.png"

TARGET_WAFERS = [
    {
        "id": "W_F_0074",
        "title": "Wafer W_F_0074: Edge-Concentrated Risk Pattern (66.7% Perimeter Failures)",
        "type": "Edge Risk",
    },
    {
        "id": "W_F_0192",
        "title": "Wafer W_F_0192: High-Density Defect Cluster (572 Failures)",
        "type": "Cluster Risk",
    },
    {
        "id": "W_N_0014",
        "title": "Wafer W_N_0014: Challenging Margin Wafer (21 Failures)",
        "type": "Margin Risk",
    },
]


def generate_spatial_attribution_maps():
    print("=" * 80)
    print("SPATIAL CONTRIBUTION MAPS GENERATOR (P0-2)")
    print("=" * 80)
    t0 = time.time()

    # 1. Load Features and Model
    with open(FEATURE_LIST_PATH, "r") as f:
        feature_names = json.load(f)["features"]

    print(f"Loading CatBoost model from {MODEL_CB_PATH.name}...")
    model_cb = CatBoostClassifier()
    model_cb.load_model(str(MODEL_CB_PATH))

    # Identify spatial features
    spatial_indices = []
    spatial_names = []
    for idx, f in enumerate(feature_names):
        dom, _ = classify_and_translate_feature(f)
        if dom == DOMAIN_SPATIAL or f in ["r_norm", "is_edge", "is_outer_ring", "neighbor_fail_count", "neighbor_fail_ratio", "spatial_density_3x3", "spatial_density_5x5"]:
            spatial_indices.append(idx)
            spatial_names.append(f)
    print(f"Identified {len(spatial_indices)} spatial-context features in Model F.")

    # 2. Load Data for Target Wafers
    print("Loading test features and predictions for target wafers...")
    df_preds = pd.read_parquet(TEST_PREDS_PARQUET)
    df_labels = pd.read_csv(TEST_LABELS_CSV, usecols=["wafer_id", "die_row", "die_col", "old_label", "label"])
    df_all_meta = pd.merge(df_preds, df_labels, on=["wafer_id", "die_row", "die_col", "old_label"], how="left")

    target_ids = [w["id"] for w in TARGET_WAFERS]
    df_meta_targets = df_all_meta[df_all_meta["wafer_id"].isin(target_ids)].copy()

    # Load feature matrix for these wafers
    df_test_feats = pd.read_parquet(FINAL_TEST_PARQUET)
    df_targets = pd.merge(
        df_meta_targets[["wafer_id", "die_row", "die_col", "old_label", "label", "model_f_optimal", "p_catboost"]],
        df_test_feats,
        on=["wafer_id", "die_row", "die_col", "old_label"],
        how="left"
    )

    print(f"Total target dies across 3 wafers: {len(df_targets):,}")

    # 3. Compute Exact TreeSHAP for Target Dies
    print("Computing exact TreeSHAP for all dies on the 3 wafers...")
    X_targets = df_targets[feature_names].values.astype(np.float32)
    pool = Pool(X_targets)
    shap_raw = model_cb.get_feature_importance(pool, type="ShapValues")
    shap_vals = shap_raw[:, :-1]  # drop base value

    # Sum signed SHAP for spatial domain features
    spatial_shap_per_die = np.sum(shap_vals[:, spatial_indices], axis=1)
    df_targets["spatial_shap"] = spatial_shap_per_die

    # Classify Prediction Outcomes (TP, FP, FN, TN)
    # Using CatBoost threshold 0.3219 or Model F rank percentile 0.90
    pred_fail = (df_targets["model_f_optimal"] >= 0.88) | (df_targets["p_catboost"] >= 0.3219)
    actual_fail = df_targets["label"] == 1

    conditions = [
        pred_fail & actual_fail,    # TP
        pred_fail & (~actual_fail), # FP
        (~pred_fail) & actual_fail, # FN
        (~pred_fail) & (~actual_fail) # TN
    ]
    choices = ["TP", "FP", "FN", "TN"]
    df_targets["outcome"] = np.select(conditions, choices, default="TN")

    # 4. Render 3x4 Grid Figure
    print("Rendering 3x4 spatial diagnostic grid...")
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=160)
    fig.subplots_adjust(hspace=0.28, wspace=0.22, top=0.93, bottom=0.05, left=0.05, right=0.95)

    outcome_colors = {
        "TP": "#d62728",  # Bold Red
        "FP": "#ff7f0e",  # Orange
        "FN": "#1f77b4",  # Blue
        "TN": "#e0e0e0",  # Light Gray
    }

    cmap_risk = plt.cm.plasma
    cmap_shap = plt.cm.coolwarm

    for r_idx, w_info in enumerate(TARGET_WAFERS):
        wid = w_info["id"]
        w_df = df_targets[df_targets["wafer_id"] == wid].copy()

        rows = w_df["die_row"].values
        cols = w_df["die_col"].values
        n_dies = len(w_df)
        n_fails = int((w_df["label"] == 1).sum())

        # Panel 1: Pre-test old_label
        ax1 = axes[r_idx, 0]
        old_colors = np.where(w_df["old_label"] == 1, "#b2182b", "#d1e5f0")
        sc1 = ax1.scatter(cols, rows, c=old_colors, s=18, marker="s", edgecolors="none")
        ax1.set_title(f"{wid}: Pre-Test Status\n(Prior Defects: {(w_df['old_label']==1).sum()})", fontsize=11, fontweight="bold")
        ax1.set_ylabel(f"{w_info['type']}\nRow Coordinate", fontsize=10, fontweight="bold")
        ax1.invert_yaxis()
        ax1.set_aspect("equal")

        # Panel 2: Continuous Model Risk Score Heatmap
        ax2 = axes[r_idx, 1]
        risk_vals = w_df["model_f_optimal"].values
        sc2 = ax2.scatter(cols, rows, c=risk_vals, cmap=cmap_risk, s=18, marker="s", edgecolors="none", vmin=0.0, vmax=1.0)
        ax2.set_title(f"Model F Risk Score\n(Continuous Risk Percentile)", fontsize=11, fontweight="bold")
        ax2.invert_yaxis()
        ax2.set_aspect("equal")
        cb2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=8)

        # Panel 3: Post-Test Outcome Overlay (TP/FP/FN/TN)
        ax3 = axes[r_idx, 2]
        pt_colors = [outcome_colors[o] for o in w_df["outcome"]]
        sc3 = ax3.scatter(cols, rows, c=pt_colors, s=18, marker="s", edgecolors="none")
        tp_cnt = (w_df["outcome"] == "TP").sum()
        fp_cnt = (w_df["outcome"] == "FP").sum()
        fn_cnt = (w_df["outcome"] == "FN").sum()
        ax3.set_title(f"Post-Test Ground Truth\n(TP:{tp_cnt} | FP:{fp_cnt} | FN:{fn_cnt})", fontsize=11, fontweight="bold")
        ax3.invert_yaxis()
        ax3.set_aspect("equal")

        # Panel 4: Spatial-Channel TreeSHAP Contribution
        ax4 = axes[r_idx, 3]
        sp_shap = w_df["spatial_shap"].values
        v_lim = max(0.4, np.percentile(np.abs(sp_shap), 98))
        sc4 = ax4.scatter(cols, rows, c=sp_shap, cmap=cmap_shap, s=18, marker="s", edgecolors="none", vmin=-v_lim, vmax=v_lim)
        ax4.set_title(f"Spatial-Channel TreeSHAP\n(Signed Margin Δ: [-{v_lim:.2f}, +{v_lim:.2f}])", fontsize=11, fontweight="bold")
        ax4.invert_yaxis()
        ax4.set_aspect("equal")
        cb4 = plt.colorbar(sc4, ax=ax4, fraction=0.046, pad=0.04)
        cb4.ax.tick_params(labelsize=8)

    # Supertitle and Legend
    fig.suptitle(
        "Spatial Contribution & Defect Attribution Across Representative Test Wafers\n"
        "Panels: (1) Pre-Test Known Defects | (2) Model F Continuous Risk Heatmap | "
        "(3) Post-Test Outcomes (TP:Red, FP:Orange, FN:Blue, TN:Gray) | (4) Spatial TreeSHAP Contribution",
        fontsize=14, fontweight="bold"
    )

    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"Saved 4-panel spatial attribution maps to: {OUTPUT_FIG}")
    print(f"Spatial maps generator completed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    generate_spatial_attribution_maps()
