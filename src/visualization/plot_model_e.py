"""
Generates publication-quality figures for Model E evaluation:
1. PR Curves: All 5 engines vs Model E Optimal vs Champion Baseline
2. Engine Comparison Bar Chart: AUC-PR & F1
3. Pairwise Prediction Correlation Heatmap
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

from src.config import REPORTS_DIR

PREDS_PATH = REPORTS_DIR / "model_e_dev_val_predictions.parquet"
METRICS_PATH = REPORTS_DIR / "model_e_metrics.json"
PLOTS_DIR = REPO_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

def generate_plots():
    df_preds = pd.read_parquet(PREDS_PATH)
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    y_val = df_preds["label"].values

    # -------------------------------------------------------------
    # 1. Precision-Recall Curves
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 7), dpi=300)
    
    models_to_plot = [
        ("cb_deep", "Engine 1: CatBoost-Deep", "#1f77b4", "-"),
        ("lgb_dart", "Engine 2: LightGBM-DART", "#ff7f0e", "--"),
        ("lgb_focal", "Engine 3: LightGBM-Focal", "#2ca02c", "-."),
        ("xgb_deep", "Engine 4: XGBoost-Deep", "#9467bd", ":"),
        ("cb_recall", "Engine 5: CatBoost-Recall", "#8c564b", "-"),
        ("model_e_optimal", "Model E: Optimal Convex Blend", "#d62728", "-"),
        ("champion_c1_c2_b", "Champion Baseline (C1+C2+B)", "#7f7f7f", "--"),
    ]

    for col, label, color, ls in models_to_plot:
        p = df_preds[col].values
        pr_score = average_precision_score(y_val, p)
        prec, rec, _ = precision_recall_curve(y_val, p)
        lw = 2.5 if "Model E: Optimal" in label or "Champion" in label else 1.5
        plt.plot(rec, prec, label=f"{label} (AUC={pr_score:.4f})", color=color, linestyle=ls, linewidth=lw)

    plt.xlabel("Recall (Sensitivity)", fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    plt.title("Model E: Precision-Recall Curves on Dev-Val (160 Wafers, 137,576 Dies)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.tight_layout()
    pr_fig_path = PLOTS_DIR / "16_model_e_pr_curves.png"
    plt.savefig(pr_fig_path)
    plt.close()
    print(f"Saved: {pr_fig_path}")

    # -------------------------------------------------------------
    # 2. AUC-PR Comparison Bar Chart
    # -------------------------------------------------------------
    plt.figure(figsize=(11, 6), dpi=300)
    
    engine_labels = [
        "Model B (LGBM 555)",
        "Model C1 (CNN)",
        "Model C2 (MS-CNN)",
        "Champion (C1+C2+B)",
        "Engine 1: CB-Deep",
        "Engine 2: LGB-DART",
        "Engine 3: LGB-Focal",
        "Engine 4: XGB-Deep",
        "Engine 5: CB-Recall",
        "Model E: Mean",
        "Model E: Optimal",
    ]
    
    pr_values = [
        0.55430,  # Model B
        0.57657,  # Model C1
        0.57170,  # Model C2
        0.57945,  # Champion
        0.61915,  # CB-Deep
        0.61133,  # LGB-DART
        0.61645,  # LGB-Focal
        0.61585,  # XGB-Deep
        0.60951,  # CB-Recall
        0.61771,  # Model E Mean
        0.61931,  # Model E Optimal
    ]

    colors = [
        "#aec7e8", "#c5b0d5", "#c5b0d5", "#7f7f7f",
        "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
        "#e377c2", "#d62728"
    ]

    bars = plt.bar(range(len(engine_labels)), pr_values, color=colors, width=0.65, edgecolor="black", linewidth=0.8)
    plt.xticks(range(len(engine_labels)), engine_labels, rotation=35, ha="right", fontsize=9.5)
    plt.ylabel("Dev-Val AUC-PR", fontweight="bold")
    plt.title("Model E Engine & Ensemble Performance vs Prior Milestones", fontsize=13, fontweight="bold", pad=12)
    plt.ylim([0.50, 0.65])
    plt.axhline(0.57945, color="gray", linestyle="--", alpha=0.7, label="Champion Baseline (0.5795)")
    plt.axhline(0.61931, color="red", linestyle=":", alpha=0.8, label="Model E Optimal (0.6193)")

    for bar, val in zip(bars, pr_values):
        plt.text(bar.get_x() + bar.get_width()/2.0, val + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.legend(loc="upper left")
    plt.tight_layout()
    bar_fig_path = PLOTS_DIR / "17_model_e_aucpr_comparison.png"
    plt.savefig(bar_fig_path)
    plt.close()
    print(f"Saved: {bar_fig_path}")

    # -------------------------------------------------------------
    # 3. Correlation Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    corr_df = pd.DataFrame(metrics["pairwise_pearson_correlation"])
    rename_dict = {
        "cb_deep": "CB-Deep",
        "lgb_dart": "LGB-DART",
        "lgb_focal": "LGB-Focal",
        "xgb_deep": "XGB-Deep",
        "cb_recall": "CB-Recall",
    }
    corr_df = corr_df.rename(index=rename_dict, columns=rename_dict)
    
    mat = corr_df.values
    im = ax.imshow(mat, cmap="Blues", vmin=0.60, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson Correlation", fontweight="bold")
    
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=35, ha="right")
    ax.set_yticklabels(corr_df.index)

    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            val = mat[i, j]
            text_color = "white" if val > 0.85 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontweight="bold")

    plt.title("Model E: Pairwise Prediction Correlation Heatmap", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    heat_fig_path = PLOTS_DIR / "18_model_e_correlation_heatmap.png"
    plt.savefig(heat_fig_path)
    plt.close()
    print(f"Saved: {heat_fig_path}")

if __name__ == "__main__":
    generate_plots()
