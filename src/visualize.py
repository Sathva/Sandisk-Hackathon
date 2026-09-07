"""
Diagnostic Visualizations for SanDisk Die Yield Prediction Hackathon.
Generates 6 publication-grade figures:
1. Target Distribution among eligible dies (Pie & Bar)
2. Distribution of key spatial features (healthy vs new failure)
3. Distribution of key block features (healthy vs new failure)
4. Raw 2000 block readings line plots for new-failure dies
5. Raw 2000 block readings line plots for healthy passing dies
6. Comprehensive 2D Wafer Map showing old failures, eligible passes, and new failures
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import TRAIN_PARQUET, TRAIN_CSV, PLOTS_DIR, SEED

# Set non-interactive backend and clean style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300


def plot_target_distribution(eligible_df):
    print("Generating Plot 1: Target Distribution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    counts = eligible_df["label"].value_counts().sort_index()
    labels = ["Stayed Healthy (0)", "Newly Failed (1)"]
    colors = ["#2ecc71", "#e67e22"]

    # Bar chart
    bars = ax1.bar(labels, counts.values, color=colors, edgecolor="black", width=0.5, alpha=0.9)
    ax1.set_ylabel("Number of Dies", fontsize=12, fontweight="bold")
    ax1.set_title("Eligible Population Counts (old_label = 0)", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:,}\n({h/len(eligible_df)*100:.2f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Donut chart
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=labels,
        autopct="%1.2f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11, fontweight="bold")
    )
    ax2.set_title("Extreme Class Imbalance Ratio", fontsize=13, fontweight="bold")

    plt.suptitle("SanDisk Hackathon: Eligible Die Population Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = PLOTS_DIR / "1_target_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved to {out_path}")


def plot_spatial_feature_distributions(eligible_df):
    print("Generating Plot 2: Spatial Feature Distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    features = [
        ("radius", "Normalized Radius from Wafer Center", "Radius (0=Center, 1=Edge)"),
        ("distance_to_edge", "Proximity to Wafer Edge", "Norm Distance to Border"),
        ("old_fail_density_5x5", "Pre-Test Failure Density (5x5 Window)", "5x5 Fail Density"),
        ("distance_to_nearest_old_failure", "Distance to Nearest Pre-Test Defect", "Normalized Distance")
    ]

    pass_df = eligible_df[eligible_df["label"] == 0]
    fail_df = eligible_df[eligible_df["label"] == 1]

    for idx, (col, title, xlabel) in enumerate(features):
        ax = axes[idx]
        if col in eligible_df.columns:
            # KDE or histograms
            ax.hist(pass_df[col], bins=40, density=True, alpha=0.5, color="#2ecc71", label="Stayed Healthy (0)")
            ax.hist(fail_df[col], bins=40, density=True, alpha=0.6, color="#e67e22", label="Newly Failed (1)")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.legend(loc="upper right", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Spatial Feature Divergence: Healthy vs. Newly Failed Dies", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = PLOTS_DIR / "2_spatial_feature_distributions.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved to {out_path}")


def plot_block_feature_distributions(eligible_df):
    print("Generating Plot 3: Block Feature Distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    features = [
        ("max_rolling_mean_100", "Max Rolling Mean (Window=100)", "Max Rolling Mean (Normal~100)"),
        ("largest_contiguous_anomaly_run", "Largest Contiguous Anomaly Run", "Consecutive Outlier Blocks (|z|>2)"),
        ("block_mean_top50", "Mean of Top 50 Readings", "Top-50 Block Mean"),
        ("block_count_z_gt_3", "Extreme Outliers (|z| > 3)", "Block Count (|z| > 3)")
    ]

    pass_df = eligible_df[eligible_df["label"] == 0]
    fail_df = eligible_df[eligible_df["label"] == 1]

    for idx, (col, title, xlabel) in enumerate(features):
        ax = axes[idx]
        if col in eligible_df.columns:
            ax.hist(pass_df[col], bins=40, density=True, alpha=0.5, color="#2ecc71", label="Stayed Healthy (0)")
            ax.hist(fail_df[col], bins=40, density=True, alpha=0.6, color="#e74c3c", label="Newly Failed (1)")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.legend(loc="upper right", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Block-Level Anomaly Feature Divergence: Healthy vs. Newly Failed Dies", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = PLOTS_DIR / "3_block_feature_distributions.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved to {out_path}")


def plot_raw_block_traces(csv_path):
    print("Generating Plots 4 & 5: Raw 2000 Block Signal Traces...")
    # Sample 4 healthy and 4 new failure dies from CSV
    healthy_traces = []
    fail_traces = []
    healthy_meta = []
    fail_meta = []

    for chunk in pd.read_csv(csv_path, chunksize=10000, usecols=["wafer_id", "die_row", "die_col", "old_label", "label", "block_readings"]):
        el_pass = chunk[(chunk["old_label"] == 0) & (chunk["label"] == 0)]
        el_fail = chunk[(chunk["old_label"] == 0) & (chunk["label"] == 1)]

        for _, row in el_pass.iterrows():
            if len(healthy_traces) < 4:
                arr = np.fromstring(row["block_readings"], dtype=np.float32, sep=' ')
                if len(arr) == 2000:
                    healthy_traces.append(arr)
                    healthy_meta.append(f"{row['wafer_id']} (r={row['die_row']}, c={row['die_col']})")

        for _, row in el_fail.iterrows():
            if len(fail_traces) < 4:
                arr = np.fromstring(row["block_readings"], dtype=np.float32, sep=' ')
                if len(arr) == 2000:
                    fail_traces.append(arr)
                    fail_meta.append(f"{row['wafer_id']} (r={row['die_row']}, c={row['die_col']})")

        if len(healthy_traces) >= 4 and len(fail_traces) >= 4:
            break

    # Plot 4: New Failures
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i in range(4):
        ax = axes[i]
        ax.plot(fail_traces[i], color="#d35400", linewidth=0.8, alpha=0.85, label="2000 Block Readings")
        ax.axhline(100.0, color="black", linestyle="--", alpha=0.6, label="Base Mean (100.0)")
        ax.set_title(f"Newly Failed Die: {fail_meta[i]} - Notice Localized Anomaly Cluster", fontsize=11, fontweight="bold")
        ax.set_ylabel("Reading", fontsize=10)
        ax.set_ylim(40, 165)
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9)
    axes[-1].set_xlabel("Block Index (0 to 1999)", fontsize=11, fontweight="bold")
    plt.suptitle("Defect Signature: 2000 Block Readings in Newly Failed Dies", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_fail = PLOTS_DIR / "4_new_failure_block_traces.png"
    plt.savefig(out_fail, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved to {out_fail}")

    # Plot 5: Healthy Dies
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i in range(4):
        ax = axes[i]
        ax.plot(healthy_traces[i], color="#27ae60", linewidth=0.8, alpha=0.85, label="2000 Block Readings")
        ax.axhline(100.0, color="black", linestyle="--", alpha=0.6, label="Base Mean (100.0)")
        ax.set_title(f"Healthy Passing Die: {healthy_meta[i]} - Stationary Background Noise", fontsize=11, fontweight="bold")
        ax.set_ylabel("Reading", fontsize=10)
        ax.set_ylim(40, 165)
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9)
    axes[-1].set_xlabel("Block Index (0 to 1999)", fontsize=11, fontweight="bold")
    plt.suptitle("Baseline Signal: 2000 Block Readings in Healthy Passing Dies", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_pass = PLOTS_DIR / "5_healthy_block_traces.png"
    plt.savefig(out_pass, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved to {out_pass}")


def plot_sample_wafer_map(df):
    print("Generating Plot 6: Wafer Map Visualization...")
    # Find a wafer that has both pre-test failures and newly failed dies
    candidates = df.groupby("wafer_id").apply(
        lambda w: (w["old_label"] == 1).sum() > 20 and ((w["old_label"] == 0) & (w["label"] == 1)).sum() > 15
    )
    valid_wafers = candidates[candidates].index.tolist()
    chosen_wafer = valid_wafers[0] if valid_wafers else df["wafer_id"].iloc[0]

    w_df = df[df["wafer_id"] == chosen_wafer].copy()
    max_r = int(w_df["die_row"].max()) + 1
    max_c = int(w_df["die_col"].max()) + 1

    # 0 = background, 1 = pass, 2 = old fail, 3 = new fail
    grid_pre = np.zeros((max_r, max_c), dtype=int)
    grid_post = np.zeros((max_r, max_c), dtype=int)
    grid_diff = np.zeros((max_r, max_c), dtype=int)

    for _, r in w_df.iterrows():
        dr = int(r["die_row"])
        dc = int(r["die_col"])
        ol = int(r["old_label"])
        fl = int(r["label"])

        # Pre-test map: 1=pass, 2=old fail
        grid_pre[dr, dc] = 2 if ol == 1 else 1

        # Post-test map: 1=pass, 2=fail (old + new)
        grid_post[dr, dc] = 2 if fl == 1 else 1

        # Difference map: 1=pass, 2=old fail, 3=new fail
        if ol == 1:
            grid_diff[dr, dc] = 2
        elif fl == 1:
            grid_diff[dr, dc] = 3
        else:
            grid_diff[dr, dc] = 1

    from matplotlib.colors import ListedColormap
    # Colors: 0=white (bg), 1=green (pass), 2=red (old fail), 3=orange (new fail)
    cmap_pre = ListedColormap(["#ffffff", "#2ecc71", "#e74c3c"])
    cmap_post = ListedColormap(["#ffffff", "#2ecc71", "#e74c3c"])
    cmap_diff = ListedColormap(["#ffffff", "#2ecc71", "#e74c3c", "#e67e22"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.imshow(grid_pre, cmap=cmap_pre, vmin=0, vmax=2)
    ax1.set_title("Pre-Test State (old_label)\nGreen=Pass, Red=Pre-Existing Defect", fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(grid_post, cmap=cmap_post, vmin=0, vmax=2)
    ax2.set_title("Post-Test State (label)\nGreen=Pass, Red=All Defects", fontsize=12, fontweight="bold")
    ax2.axis("off")

    ax3.imshow(grid_diff, cmap=cmap_diff, vmin=0, vmax=3)
    ax3.set_title(f"Target Identification: {chosen_wafer}\nGreen=Pass, Red=Old Fail, Orange=NEW FAIL", fontsize=12, fontweight="bold")
    ax3.axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#2ecc71", edgecolor="black", label="Passing Die (0)"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="black", label="Old Failure (old_label=1)"),
        mpatches.Patch(facecolor="#e67e22", edgecolor="black", label="New Failure (TARGET: old_label=0 & label=1)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.05))

    plt.suptitle(f"Multi-Resolution Wafer Map Analysis: Wafer {chosen_wafer}", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_map = PLOTS_DIR / "6_sample_wafer_map.png"
    plt.savefig(out_map, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved to {out_map}")


def main():
    print(f"\n{'=' * 75}")
    print(f"GENERATING DIAGNOSTIC VISUALIZATIONS")
    print(f"{'=' * 75}")

    if not TRAIN_PARQUET.exists():
        print(f"ERROR: {TRAIN_PARQUET} not found. Run preprocessing first.")
        return

    print("Loading train dataset...")
    df = pd.read_parquet(TRAIN_PARQUET)
    eligible_df = df[df["old_label"] == 0].copy()

    # Generate all plots
    plot_target_distribution(eligible_df)
    plot_spatial_feature_distributions(eligible_df)
    plot_block_feature_distributions(eligible_df)
    plot_raw_block_traces(TRAIN_CSV)
    plot_sample_wafer_map(df)

    print(f"\nAll 6 diagnostic figures successfully generated in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
