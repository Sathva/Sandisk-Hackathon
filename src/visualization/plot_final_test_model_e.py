"""
Plotting script for final unseen test evaluation of Model E vs Champion.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import INPUT_DIR, PROCESSED_DIR

PREDS_PATH = REPO_ROOT / "predictions" / "final_test_model_e_predictions.parquet"
TEST_CSV_PATH = INPUT_DIR / "test.csv"
PLOTS_DIR = REPO_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

def generate_test_plots():
    df_preds = pd.read_parquet(PREDS_PATH)
    df_test_labels = pd.read_csv(TEST_CSV_PATH, usecols=["wafer_id", "die_row", "die_col", "old_label", "label"])
    df_test_labels_eligible = df_test_labels[df_test_labels["old_label"] == 0].reset_index(drop=True)
    y_test = df_test_labels_eligible["label"].values

    # 1. Final Unseen Test PR Curves
    plt.figure(figsize=(10, 7), dpi=300)

    models_to_plot = [
        ("cb_deep", "Engine 1: CatBoost-Deep", "#1f77b4", "-"),
        ("lgb_focal", "Engine 3: LightGBM-Focal", "#2ca02c", "-."),
        ("xgb_deep", "Engine 4: XGBoost-Deep", "#9467bd", ":"),
        ("champion_baseline", "Champion Baseline (C1+C2+B)", "#7f7f7f", "--"),
        ("champion_with_model_e", "Champion with Model E (C1+C2+E)", "#ff7f0e", "-"),
        ("model_e_optimal", "Model E: Optimal Convex Blend", "#d62728", "-"),
    ]

    for col, label, color, ls in models_to_plot:
        p = df_preds[col].values
        pr_score = average_precision_score(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        lw = 2.8 if "Model E: Optimal" in label or "Champion Baseline" in label else 1.6
        plt.plot(rec, prec, label=f"{label} (AUC={pr_score:.4f})", color=color, linestyle=ls, linewidth=lw)

    plt.xlabel("Recall (Sensitivity)", fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    plt.title("Final Unseen Test: Precision-Recall Curves (200 Wafers, 185,126 Dies)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.tight_layout()
    pr_fig_path = PLOTS_DIR / "19_final_test_model_e_pr_curves.png"
    plt.savefig(pr_fig_path)
    plt.close()
    print(f"Saved: {pr_fig_path}")

    # 2. Final Test AUC-PR Comparison Bar Chart
    plt.figure(figsize=(11, 6), dpi=300)
    labels = [
        "Model B (LGBM 555)",
        "Model C2 (MS-CNN)",
        "Model C1 (CNN)",
        "Champion Baseline",
        "Champion + Model E",
        "CB-Recall (E5)",
        "LGB-DART (E2)",
        "LGB-Focal (E3)",
        "XGB-Deep (E4)",
        "CB-Deep (E1)",
        "Model E Optimal",
    ]
    scores = [
        0.53528,
        0.55791,
        0.56024,
        0.56283,
        0.57287,
        0.60471,
        0.60562,
        0.61104,
        0.61191,
        0.61333,
        0.61382,
    ]
    colors = [
        "#aec7e8", "#c5b0d5", "#c5b0d5", "#7f7f7f", "#ff7f0e",
        "#8c564b", "#ffbb78", "#2ca02c", "#9467bd", "#1f77b4", "#d62728"
    ]

    bars = plt.bar(range(len(labels)), scores, color=colors, width=0.65, edgecolor="black", linewidth=0.8)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9.5)
    plt.ylabel("Final Unseen Test AUC-PR", fontweight="bold")
    plt.title("Final Unseen Test Performance: Model E vs Prior Milestones (200 Wafers)", fontsize=13, fontweight="bold", pad=12)
    plt.ylim([0.50, 0.65])
    plt.axhline(0.56283, color="gray", linestyle="--", alpha=0.7, label="Champion Baseline (0.5628)")
    plt.axhline(0.61382, color="red", linestyle=":", alpha=0.8, label="Model E Optimal (0.6138)")

    for bar, val in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2.0, val + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.legend(loc="upper left")
    plt.tight_layout()
    bar_fig_path = PLOTS_DIR / "20_final_test_aucpr_comparison.png"
    plt.savefig(bar_fig_path)
    plt.close()
    print(f"Saved: {bar_fig_path}")

if __name__ == "__main__":
    generate_test_plots()
