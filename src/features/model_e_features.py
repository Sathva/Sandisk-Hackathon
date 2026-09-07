"""
Feature Engineering Pipeline for Model E: AdversarialResNet Committee.

Constructs the full leak-free feature table on the canonical 1000-wafer split:
1. 555 Canonical Model B Features (500 parametric + 19 spatial + 36 block stats)
2. 10-Component Parametric PCA (fitted strictly on dev_train)
3. Cluster Topology & EDT Proximity (computed strictly from old_label == 1)
4. W=350 & W=400 Rolling Statistics and 1D Morphological Top-Hat Filters
5. Stepper Reticle & Zernike Orthogonal Polynomial Features
6. Cross-Resolution Bilinear Interaction Terms
7. 10 Block Signal Gradient & Curvature Features (burst shape dynamics)
8. 36 Wafer-Local Distributional & Rank Features (wrank, wzscore, wdev)

Strict Leak-Free Guarantees:
- Zero test data access (final 200 test wafers remain untouched).
- Evaluated strictly on eligible dies (old_label == 0).
- Excludes all leaky features (adversarial_score, te_wafer_fail_rate, etc.).
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# Ensure repo root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    DEV_TRAIN_PARQUET,
    DEV_VAL_PARQUET,
    PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
)
from src.models.common import (
    PARAMETRIC_FEATURES,
    SPATIAL_FEATURES,
    BLOCK_FEATURES,
    MODEL_B_FEATURES,
)

CACHE_DIR = PROCESSED_DIR / "cache"
TRAIN_RAW_BLOCKS_PATH = CACHE_DIR / "dev_train_raw_blocks.dat"
VAL_RAW_BLOCKS_PATH = CACHE_DIR / "dev_val_raw_blocks.dat"

DEV_TRAIN_ADDITIONAL_PARQUET = PROCESSED_DIR / "dev_train_model_d_additional.parquet"
DEV_VAL_ADDITIONAL_PARQUET = PROCESSED_DIR / "dev_val_model_d_additional.parquet"

MODEL_E_TRAIN_PARQUET = PROCESSED_DIR / "dev_train_model_e_features.parquet"
MODEL_E_VAL_PARQUET = PROCESSED_DIR / "dev_val_model_e_features.parquet"
MODEL_E_FEATURE_LIST_PATH = MODELS_DIR / "model_e_feature_list.json"


def compute_block_gradient_features(mmap_path, n_dies, seq_len=2000, chunk_size=50000):
    """
    Computes 10 first and second derivative block signal features:
    - 1st numerical derivative: grad_max, grad_min, grad_energy, grad_std, grad_peak_idx
    - 2nd numerical derivative: curv_max, curv_energy, curv_n_inflections, grad_ratio_max_mean, grad_burst_width
    """
    print(f"  Extracting gradient & curvature features from {mmap_path.name} ({n_dies:,} dies)...", flush=True)
    t0 = time.time()
    raw_mmap = np.memmap(mmap_path, dtype="float32", mode="r", shape=(n_dies, seq_len))

    grad_max = np.zeros(n_dies, dtype=np.float32)
    grad_min = np.zeros(n_dies, dtype=np.float32)
    grad_energy = np.zeros(n_dies, dtype=np.float32)
    grad_std = np.zeros(n_dies, dtype=np.float32)
    grad_peak_idx = np.zeros(n_dies, dtype=np.float32)
    curv_max = np.zeros(n_dies, dtype=np.float32)
    curv_energy = np.zeros(n_dies, dtype=np.float32)
    curv_n_inflections = np.zeros(n_dies, dtype=np.float32)
    grad_ratio_max_mean = np.zeros(n_dies, dtype=np.float32)
    grad_burst_width = np.zeros(n_dies, dtype=np.float32)

    for start in range(0, n_dies, chunk_size):
        end = min(start + chunk_size, n_dies)
        chunk = np.array(raw_mmap[start:end], dtype=np.float32)

        # 5-point moving average smoothing along sequence dimension
        smoothed = uniform_filter1d(chunk, size=5, axis=1, mode="reflect")

        # 1st and 2nd derivatives
        grad = np.diff(smoothed, axis=1)   # (chunk_n, 1999)
        curv = np.diff(grad, axis=1)       # (chunk_n, 1998)

        abs_grad = np.abs(grad)
        abs_curv = np.abs(curv)

        grad_max[start:end] = np.max(abs_grad, axis=1)
        grad_min[start:end] = np.min(grad, axis=1)
        grad_energy[start:end] = np.sum(grad**2, axis=1) / 1999.0
        g_std = np.std(grad, axis=1)
        grad_std[start:end] = g_std
        grad_peak_idx[start:end] = np.argmax(abs_grad, axis=1).astype(np.float32) / 2000.0

        curv_max[start:end] = np.max(abs_curv, axis=1)
        curv_energy[start:end] = np.sum(curv**2, axis=1) / 1998.0

        sign_changes = np.diff(np.sign(curv), axis=1)
        curv_n_inflections[start:end] = np.sum(np.abs(sign_changes) > 0, axis=1).astype(np.float32)

        mean_abs_grad = np.mean(abs_grad, axis=1)
        grad_ratio_max_mean[start:end] = np.where(
            mean_abs_grad > 1e-8,
            grad_max[start:end] / mean_abs_grad,
            0.0,
        )

        grad_threshold = 2.0 * g_std[:, np.newaxis]
        grad_burst_width[start:end] = np.sum(abs_grad > grad_threshold, axis=1).astype(np.float32)

    df_grad = pd.DataFrame({
        "grad_max": grad_max,
        "grad_min": grad_min,
        "grad_energy": grad_energy,
        "grad_std": grad_std,
        "grad_peak_idx": grad_peak_idx,
        "curv_max": curv_max,
        "curv_energy": curv_energy,
        "curv_n_inflections": curv_n_inflections,
        "grad_ratio_max_mean": grad_ratio_max_mean,
        "grad_burst_width": grad_burst_width,
    })

    print(f"    Done {n_dies:,} dies in {time.time() - t0:.2f}s.", flush=True)
    return df_grad


def compute_wafer_rank_and_distributional_features(df_train, df_val, target_cols):
    """
    Computes wafer-local rank percentiles, inter-wafer z-scores, and within-wafer deviations.
    Fitted strictly on dev_train for global means/stds to guarantee zero data leakage.
    """
    print(f"  Computing rank & distributional features for {len(target_cols)} features...", flush=True)
    t0 = time.time()

    # Pre-compute global stats on dev_train
    global_stats = {}
    for col in target_cols:
        global_stats[col] = {
            "mean": float(df_train[col].mean()),
            "std": float(max(df_train[col].std(), 1e-6)),
        }

    def process_split(df):
        rank_dict = {}
        wzscore_dict = {}
        wdev_dict = {}

        for col in target_cols:
            short = col.replace("max_rolling_mean_", "roll").replace("inter_pca01_x_", "ix_")
            
            # 1. Wafer-local percentile rank
            rank_col = f"wrank_{short}"
            rank_dict[rank_col] = df.groupby("wafer_id")[col].rank(pct=True).astype(np.float32).values

            # 2. Inter-wafer z-score
            wafer_mean = df.groupby("wafer_id")[col].transform("mean").values
            wzscore_col = f"wzscore_{short}"
            wzscore_dict[wzscore_col] = (
                (wafer_mean - global_stats[col]["mean"]) / global_stats[col]["std"]
            ).astype(np.float32)

            # 3. Within-wafer standardized deviation
            wafer_std = df.groupby("wafer_id")[col].transform("std").fillna(0.0).values
            wdev_col = f"wdev_{short}"
            wdev_dict[wdev_col] = (
                (df[col].values - wafer_mean) / (wafer_std + 1e-6)
            ).astype(np.float32)

        return pd.DataFrame({**rank_dict, **wzscore_dict, **wdev_dict}, index=df.index)

    df_rank_train = process_split(df_train)
    df_rank_val = process_split(df_val)

    print(f"    Generated {df_rank_train.shape[1]} rank/distributional features in {time.time() - t0:.2f}s.", flush=True)
    return df_rank_train, df_rank_val, list(df_rank_train.columns)


def build_model_e_features():
    print("=" * 80)
    print("MODEL E FEATURE ENGINEERING PIPELINE (CANONICAL 1000 WAFERS)")
    print("=" * 80)
    t_start = time.time()

    # 1. Load base features
    print("\n[Step 1/5] Loading base feature tables...")
    train_base = pd.read_parquet(DEV_TRAIN_PARQUET)
    val_base = pd.read_parquet(DEV_VAL_PARQUET)

    # Filter strictly to eligible dies: old_label == 0
    train_eligible_idx = train_base["old_label"] == 0
    val_eligible_idx = val_base["old_label"] == 0

    train_base = train_base.loc[train_eligible_idx].reset_index(drop=True)
    val_base = val_base.loc[val_eligible_idx].reset_index(drop=True)

    print(f"  dev_train eligible: {len(train_base):,} dies across {train_base['wafer_id'].nunique()} wafers")
    print(f"  dev_val eligible:   {len(val_base):,} dies across {val_base['wafer_id'].nunique()} wafers")

    # 2. Load Model D additional features (PCA, EDT, W=350, Top-Hat, Zernike)
    print("\n[Step 2/5] Loading Model D additional features...")
    train_add = pd.read_parquet(DEV_TRAIN_ADDITIONAL_PARQUET).reset_index(drop=True)
    val_add = pd.read_parquet(DEV_VAL_ADDITIONAL_PARQUET).reset_index(drop=True)
    assert len(train_add) == len(train_base)
    assert len(val_add) == len(val_base)

    # 3. Compute 10 Block Gradient & Curvature Features
    print("\n[Step 3/5] Computing Block Gradient & Curvature Features...")
    grad_train = compute_block_gradient_features(TRAIN_RAW_BLOCKS_PATH, len(train_base))
    grad_val = compute_block_gradient_features(VAL_RAW_BLOCKS_PATH, len(val_base))

    # 4. Compute Additional Bilinear Interaction Terms
    print("\n[Step 4/5] Computing Additional Bilinear Interactions...")
    def compute_bilinear(base_df, add_df):
        pc1 = add_df["pca_01"].values
        roll350_burst = np.maximum(0.0, add_df["max_rolling_mean_350"].values - 100.0)
        roll400_burst = np.maximum(0.0, base_df["max_rolling_mean_400"].values - 100.0)
        edge_dist = base_df["distance_to_edge"].values
        density5 = base_df["old_fail_density_5x5"].values
        burst_excess_350 = add_df["burst_excess_350"].values

        return pd.DataFrame({
            "inter_pc1_x_roll400": (pc1 * roll400_burst).astype(np.float32),
            "inter_pc1_x_edge_density": (pc1 * roll350_burst * (1.0 - edge_dist) * (density5 + 0.05)).astype(np.float32),
            "inter_pc1_x_burst_excess_350": (pc1 * burst_excess_350).astype(np.float32),
        })

    inter_train = compute_bilinear(train_base, train_add)
    inter_val = compute_bilinear(val_base, val_add)

    # Combine intermediate tables for rank features
    temp_train = pd.concat([
        train_base[["wafer_id", "max_rolling_mean_400", "block_std", "block_q99", "block_max_z", "distance_to_nearest_old_failure"]],
        train_add[["pca_01", "pca_02", "pca_03", "max_rolling_mean_350", "inter_pca01_x_roll350"]],
        grad_train[["grad_energy", "curv_energy"]],
    ], axis=1)

    temp_val = pd.concat([
        val_base[["wafer_id", "max_rolling_mean_400", "block_std", "block_q99", "block_max_z", "distance_to_nearest_old_failure"]],
        val_add[["pca_01", "pca_02", "pca_03", "max_rolling_mean_350", "inter_pca01_x_roll350"]],
        grad_val[["grad_energy", "curv_energy"]],
    ], axis=1)

    rank_target_cols = [
        "pca_01", "pca_02", "pca_03",
        "max_rolling_mean_350", "max_rolling_mean_400",
        "block_std", "block_q99", "block_max_z",
        "inter_pca01_x_roll350", "distance_to_nearest_old_failure",
        "grad_energy", "curv_energy",
    ]

    # 5. Compute Wafer-Local Distributional & Rank Features (36 features)
    print("\n[Step 5/5] Computing Wafer Distributional & Rank Features...")
    rank_train, rank_val, rank_cols = compute_wafer_rank_and_distributional_features(temp_train, temp_val, rank_target_cols)

    # Assemble Final Datasets
    print("\n" + "=" * 80)
    print("ASSEMBLING FINAL MODEL E FEATURE MATRICES")
    print("=" * 80)

    # Select base feature columns (Model B 555)
    model_b_cols = [c for c in MODEL_B_FEATURES if c in train_base.columns]
    print(f"  Model B Base Features:    {len(model_b_cols)}")
    print(f"  Model D Additional Feats: {train_add.shape[1]}")
    print(f"  Block Gradient Feats:     {grad_train.shape[1]}")
    print(f"  Additional Bilinear Feats:{inter_train.shape[1]}")
    print(f"  Wafer Rank/Dev Feats:     {rank_train.shape[1]}")

    train_full_e = pd.concat([
        train_base[["wafer_id", "die_row", "die_col", "old_label", "label"] + model_b_cols],
        train_add,
        grad_train,
        inter_train,
        rank_train,
    ], axis=1)

    val_full_e = pd.concat([
        val_base[["wafer_id", "die_row", "die_col", "old_label", "label"] + model_b_cols],
        val_add,
        grad_val,
        inter_val,
        rank_val,
    ], axis=1)

    # Drop duplicate column names if any
    train_full_e = train_full_e.loc[:, ~train_full_e.columns.duplicated()]
    val_full_e = val_full_e.loc[:, ~val_full_e.columns.duplicated()]

    # Verify zero NaNs
    n_nans_tr = train_full_e.isna().sum().sum()
    n_nans_val = val_full_e.isna().sum().sum()
    if n_nans_tr > 0 or n_nans_val > 0:
        print(f"  WARNING: Found {n_nans_tr} NaNs in train, {n_nans_val} in val. Filling with 0.0.")
        train_full_e = train_full_e.fillna(0.0)
        val_full_e = val_full_e.fillna(0.0)

    # Determine feature column list
    meta_cols = {"wafer_id", "die_row", "die_col", "old_label", "label"}
    leaky_cols = {"adversarial_score", "te_wafer_fail_rate", "te_wafer_residual", "te_wafer_fail_rate_smooth"}
    all_feature_cols = [c for c in train_full_e.columns if c not in meta_cols and c not in leaky_cols]

    print(f"\nFinal Feature Count: {len(all_feature_cols)}")
    print(f"Train Dataset Shape: {train_full_e.shape}")
    print(f"Val Dataset Shape:   {val_full_e.shape}")

    # Save to parquet
    print(f"\nSaving to:")
    print(f"  {MODEL_E_TRAIN_PARQUET}")
    print(f"  {MODEL_E_VAL_PARQUET}")
    train_full_e.to_parquet(MODEL_E_TRAIN_PARQUET, index=False)
    val_full_e.to_parquet(MODEL_E_VAL_PARQUET, index=False)

    with open(MODEL_E_FEATURE_LIST_PATH, "w") as f:
        json.dump(all_feature_cols, f, indent=2)
    print(f"Saved feature list ({len(all_feature_cols)} cols) to {MODEL_E_FEATURE_LIST_PATH}")

    print(f"\nModel E Feature Engineering Completed in {time.time() - t_start:.2f}s.")
    return train_full_e, val_full_e, all_feature_cols


if __name__ == "__main__":
    build_model_e_features()
