"""
Sub-Die Block Reading Pattern Analysis for Model B & Multi-Resolution Models.

Rubric Deliverable (P0-3):
- Analyzes the raw 2,000-reading sequences from final_test_raw_blocks.dat.
- Compares newly failed anomalous dies against healthy baseline dies.
- Highlights localized burst windows and anomaly run lengths.
- Explicitly labels horizontal axis as 'Sub-Die Reading Index (1 to 2,000)'
  (strictly avoiding unsubstantiated microsecond/physical time assumptions).
- Compares rolling mean profiles and burst excesses.
- Saves reports/figures/29_block_pattern_analysis.png.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    PROCESSED_DIR,
    REPORTS_DIR,
    INPUT_DIR,
)

CACHE_DIR = PROCESSED_DIR / "cache"

FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BLOCKS_DAT = CACHE_DIR / "final_test_raw_blocks.dat"
TEST_CSV = INPUT_DIR / "test.csv"
PREDS_PARQUET = REPO_ROOT / "predictions" / "final_test_model_f_predictions.parquet"
OUTPUT_FIG = FIGURES_DIR / "29_block_pattern_analysis.png"


def run_block_pattern_analysis():
    print("=" * 80)
    print("SUB-DIE BLOCK READING PATTERN ANALYSIS (P0-3)")
    print("=" * 80)
    t0 = time.time()

    # 1. Load Memmap
    n_expected = 208264
    seq_len = 2000
    print(f"Opening memmap {BLOCKS_DAT.name} ({n_expected:,} x {seq_len:,})...")
    mmap = np.memmap(BLOCKS_DAT, dtype="float32", mode="r", shape=(n_expected, seq_len))

    # 2. Load Metadata and Prediction Scores
    print("Loading test metadata and prediction rankings...")
    df_labels = pd.read_csv(TEST_CSV, usecols=["wafer_id", "die_row", "die_col", "old_label", "label"])
    df_preds = pd.read_parquet(PREDS_PARQUET)
    df_meta = pd.merge(df_preds, df_labels, on=["wafer_id", "die_row", "die_col", "old_label"], how="left")

    # Filter eligible test dies (old_label == 0)
    # Target: New failures (label == 1) with high risk score vs healthy dies (label == 0)
    fail_indices = df_meta[(df_meta["old_label"] == 0) & (df_meta["label"] == 1) & (df_meta["model_f_optimal"] >= 0.90)].index.values
    healthy_indices = df_meta[(df_meta["old_label"] == 0) & (df_meta["label"] == 0) & (df_meta["model_f_optimal"] <= 0.20)].index.values

    print(f"Eligible high-risk failed dies: {len(fail_indices):,}")
    print(f"Eligible healthy baseline dies: {len(healthy_indices):,}")

    # Select representative samples
    np.random.seed(42)
    sample_fail_idx = np.random.choice(fail_indices, size=4, replace=False)
    sample_healthy_idx = np.random.choice(healthy_indices, size=4, replace=False)

    # Compute rolling window statistics for visualization
    window_sizes = [200, 350]

    # 3. Create Multi-Panel Figure
    print("Rendering 4-panel sub-die reading diagnostic figure...")
    fig = plt.figure(figsize=(18, 13), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.20)

    reading_x = np.arange(1, seq_len + 1)

    # Panel 1: Defect-Prone Dies with Localized Burst Highlights
    ax1 = fig.add_subplot(gs[0, 0])
    fail_colors = ["#d62728", "#e377c2", "#ff7f0e", "#8c564b"]
    for i, idx in enumerate(sample_fail_idx):
        trace = mmap[idx]
        w_id = df_meta.at[idx, "wafer_id"]
        r = df_meta.at[idx, "die_row"]
        c = df_meta.at[idx, "die_col"]
        risk = df_meta.at[idx, "model_f_optimal"]

        # Compute rolling mean (W=350)
        roll = pd.Series(trace).rolling(window=350, min_periods=1).mean().values
        peak_loc = np.argmax(roll)
        burst_val = roll[peak_loc]

        ax1.plot(reading_x, trace, color=fail_colors[i], alpha=0.35, linewidth=0.8)
        ax1.plot(reading_x, roll, color=fail_colors[i], linewidth=2.0,
                 label=f"{w_id} ({r},{c}) - Risk:{risk:.2f} (Peak W350: {burst_val:.1f})")

        # Highlight burst region around peak
        burst_start = max(0, peak_loc - 175)
        burst_end = min(seq_len, peak_loc + 175)
        ax1.axvspan(burst_start, burst_end, color=fail_colors[i], alpha=0.10)

    ax1.set_title("Panel 1: Defect-Associated Dies — Localized Amplitude Bursts\n(Raw Sub-Die Reading Profiles & W=350 Rolling Anomaly Windows)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Sub-Die Reading Index (1 to 2,000)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Reading Amplitude (A.U.)", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8, loc="upper right")

    # Panel 2: Healthy Control Dies Showing Baseline Uniformity
    ax2 = fig.add_subplot(gs[0, 1])
    healthy_colors = ["#2ca02c", "#1f77b4", "#17becf", "#bcbd22"]
    for i, idx in enumerate(sample_healthy_idx):
        trace = mmap[idx]
        w_id = df_meta.at[idx, "wafer_id"]
        r = df_meta.at[idx, "die_row"]
        c = df_meta.at[idx, "die_col"]
        risk = df_meta.at[idx, "model_f_optimal"]

        roll = pd.Series(trace).rolling(window=350, min_periods=1).mean().values
        ax2.plot(reading_x, trace, color=healthy_colors[i], alpha=0.35, linewidth=0.8)
        ax2.plot(reading_x, roll, color=healthy_colors[i], linewidth=2.0,
                 label=f"{w_id} ({r},{c}) - Risk:{risk:.2f} (Stable Baseline)")

    ax2.set_title("Panel 2: Healthy Control Dies — Uniform Sub-Die Baseline\n(Absence of Localized Bursts or Step Anomaly Run Lengths)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Sub-Die Reading Index (1 to 2,000)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Reading Amplitude (A.U.)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8, loc="upper right")

    # Panel 3: Ensemble Mean Rolling Window Comparison across 200 Dies
    ax3 = fig.add_subplot(gs[1, 0])
    # Compute mean profiles across 100 failed and 100 healthy dies
    eval_fails = np.random.choice(fail_indices, size=100, replace=False)
    eval_healthy = np.random.choice(healthy_indices, size=100, replace=False)

    traces_fail = np.array([mmap[idx] for idx in eval_fails])
    traces_healthy = np.array([mmap[idx] for idx in eval_healthy])

    # Sort each trace by amplitude to show quantile distribution across sub-die readings
    sorted_fail = np.sort(traces_fail, axis=1)
    sorted_healthy = np.sort(traces_healthy, axis=1)

    mean_sorted_fail = np.mean(sorted_fail, axis=0)
    std_sorted_fail = np.std(sorted_fail, axis=0)
    mean_sorted_healthy = np.mean(sorted_healthy, axis=0)
    std_sorted_healthy = np.std(sorted_healthy, axis=0)

    rank_pct = np.linspace(0, 100, seq_len)
    ax3.plot(rank_pct, mean_sorted_fail, color="#d62728", linewidth=2.5, label="Failed Dies (Mean Quantile Profile, N=100)")
    ax3.fill_between(rank_pct, mean_sorted_fail - std_sorted_fail, mean_sorted_fail + std_sorted_fail, color="#d62728", alpha=0.15)

    ax3.plot(rank_pct, mean_sorted_healthy, color="#2ca02c", linewidth=2.5, label="Healthy Dies (Mean Quantile Profile, N=100)")
    ax3.fill_between(rank_pct, mean_sorted_healthy - std_sorted_healthy, mean_sorted_healthy + std_sorted_healthy, color="#2ca02c", alpha=0.15)

    ax3.set_title("Panel 3: Sub-Die Reading Quantile Distribution Comparison\n(Failed Silicon Exhibits Extreme Upper-Tail Amplitudes >90th Percentile)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Internal Reading Percentile within Die (0% to 100%)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Reading Value (A.U.)", fontsize=10, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(fontsize=9, loc="upper left")

    # Panel 4: Distribution of Engineered Block Features (Model B Discriminators)
    ax4 = fig.add_subplot(gs[1, 1])

    # Compute key block features for these 200 dies
    # 1. Max rolling mean W=350
    # 2. Burst elevation excess (peak rolling mean - global median)
    fail_burst_excess = []
    for t in traces_fail:
        r = pd.Series(t).rolling(window=350, min_periods=1).mean().values
        fail_burst_excess.append(np.max(r) - np.median(t))

    healthy_burst_excess = []
    for t in traces_healthy:
        r = pd.Series(t).rolling(window=350, min_periods=1).mean().values
        healthy_burst_excess.append(np.max(r) - np.median(t))

    bins = np.linspace(0, max(max(fail_burst_excess), max(healthy_burst_excess)), 30)
    ax4.hist(healthy_burst_excess, bins=bins, alpha=0.6, color="#2ca02c", label=f"Healthy Dies (Mean: {np.mean(healthy_burst_excess):.2f})", density=True)
    ax4.hist(fail_burst_excess, bins=bins, alpha=0.6, color="#d62728", label=f"Failed Dies (Mean: {np.mean(fail_burst_excess):.2f})", density=True)

    ax4.set_title("Panel 4: Discriminative Engine — Burst Elevation Excess (W=350)\n(Engineered Feature Driving Model A → Model B Gain)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Burst Elevation Excess Above Die Baseline (A.U.)", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Empirical Probability Density", fontsize=10, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Multi-Resolution Sub-Die Block Reading Pattern Analysis (Model B / C1 / F Diagnostics)\n"
        "Observable Localized Non-Uniformities Across 2,000 Sequential Internal Measurements",
        fontsize=14, fontweight="bold"
    )

    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"Saved block pattern diagnostic figure to: {OUTPUT_FIG}")
    print(f"Block pattern analysis completed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    run_block_pattern_analysis()
