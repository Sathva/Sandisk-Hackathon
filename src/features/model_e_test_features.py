"""
Feature Engineering for Final Unseen Test Set (200 Wafers, 185,126 Eligible Dies).
Constructs the full 644-feature table for Model E strictly from unlabeled test files:
- validation_features.parquet (unlabeled)
- validation.csv / final_test_raw_blocks.dat (unlabeled)

Zero Label Access:
- Target 'label' is never read or referenced during feature creation.
- Normalization and PCA projections utilize frozen parameters from dev_train.
"""

import sys
import os
import time
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.ndimage import uniform_filter1d

# Ensure repo root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    SEED,
)
from src.models.common import (
    PARAMETRIC_FEATURES,
    SPATIAL_FEATURES,
    BLOCK_FEATURES,
    MODEL_B_FEATURES,
)

VALIDATION_PARQUET = PROCESSED_DIR / "validation_features.parquet"
CACHE_DIR = PROCESSED_DIR / "cache"
FINAL_TEST_BLOCKS_DAT = CACHE_DIR / "final_test_raw_blocks.dat"
DEV_TRAIN_MODEL_E_PARQUET = PROCESSED_DIR / "dev_train_model_e_features.parquet"
FINAL_TEST_MODEL_E_PARQUET = PROCESSED_DIR / "final_test_model_e_features.parquet"
MODEL_E_FEATURE_LIST_PATH = MODELS_DIR / "model_e_feature_list.json"
PCA_SCALER_PATH = MODELS_DIR / "model_d_pca_scaler.pkl"


def extract_test_cluster_topology(df_full, eligible_index):
    """Computes EDT distance and cluster sizes using strictly old_label == 1."""
    print(f"  Extracting wafer cluster topology & EDT ({len(df_full):,} dies)...", flush=True)
    t0 = time.time()
    results = []

    grouped = df_full.groupby("wafer_id", sort=False)
    for wafer_id, w_group in grouped:
        rows = w_group["die_row"].values.astype(int)
        cols = w_group["die_col"].values.astype(int)
        old_labels = w_group["old_label"].values

        max_r = int(rows.max()) + 1
        max_c = int(cols.max()) + 1
        diagonal = float(np.sqrt(max_r**2 + max_c**2))

        old_fail_mask = np.zeros((max_r, max_c), dtype=bool)
        old_fail_mask[rows, cols] = (old_labels == 1)

        n_old_fails = int(old_fail_mask.sum())

        if n_old_fails > 0:
            labeled_clusters, _ = ndi.label(old_fail_mask, structure=np.ones((3, 3), dtype=int))
            cluster_sizes = np.bincount(labeled_clusters.ravel())
            dist_grid, (nearest_r, nearest_c) = ndi.distance_transform_edt(~old_fail_mask, return_indices=True)

            die_dists = dist_grid[rows, cols]
            nearest_cluster_sizes = cluster_sizes[labeled_clusters[nearest_r[rows, cols], nearest_c[rows, cols]]]
            delta_r = (nearest_r[rows, cols] - rows) / max(diagonal, 1.0)
            delta_c = (nearest_c[rows, cols] - cols) / max(diagonal, 1.0)
            is_1hop = (die_dists <= 1.5).astype(np.int8)
            is_2hop = (die_dists <= 2.5).astype(np.int8)
        else:
            die_dists = np.full(len(rows), diagonal, dtype=np.float32)
            nearest_cluster_sizes = np.zeros(len(rows), dtype=np.float32)
            delta_r = np.zeros(len(rows), dtype=np.float32)
            delta_c = np.zeros(len(rows), dtype=np.float32)
            is_1hop = np.zeros(len(rows), dtype=np.int8)
            is_2hop = np.zeros(len(rows), dtype=np.int8)

        w_feat = pd.DataFrame({
            "exact_edt_distance": die_dists.astype(np.float32),
            "nearest_defect_cluster_size": nearest_cluster_sizes.astype(np.float32),
            "nearest_defect_log_cluster_size": np.log1p(nearest_cluster_sizes).astype(np.float32),
            "defect_vector_dr": delta_r.astype(np.float32),
            "defect_vector_dc": delta_c.astype(np.float32),
            "is_defect_neighbor_1hop": is_1hop,
            "is_defect_neighbor_2hop": is_2hop,
        }, index=w_group.index)
        results.append(w_feat)

    all_res = pd.concat(results).loc[eligible_index]
    print(f"    Done in {time.time() - t0:.2f}s.", flush=True)
    return all_res


def extract_test_block_signals(raw_blocks_path, eligible_indices, total_n=208264, chunk_size=25000):
    """
    Computes W=350 rolling stats, Top-Hat filters, and 10 block gradient & curvature
    features for the eligible dies directly from final_test_raw_blocks.dat.
    """
    print(f"  Extracting block rolling, top-hat, and gradient features ({len(eligible_indices):,} eligible dies)...", flush=True)
    t0 = time.time()
    blocks_mmap = np.memmap(raw_blocks_path, dtype="float32", mode="r", shape=(total_n, 2000))
    w = 350
    seq_len = 2000

    n_eligible = len(eligible_indices)
    
    # Pre-allocate arrays
    mean_rm_all = np.zeros(n_eligible, dtype=np.float32)
    max_rm_all = np.zeros(n_eligible, dtype=np.float32)
    min_rm_all = np.zeros(n_eligible, dtype=np.float32)
    max_rstd_all = np.zeros(n_eligible, dtype=np.float32)
    max_idx_all = np.zeros(n_eligible, dtype=np.float32)
    excess350_all = np.zeros(n_eligible, dtype=np.float32)

    th_peak100_all = np.zeros(n_eligible, dtype=np.float32)
    th_energy100_all = np.zeros(n_eligible, dtype=np.float32)
    th_peak200_all = np.zeros(n_eligible, dtype=np.float32)
    th_energy200_all = np.zeros(n_eligible, dtype=np.float32)

    grad_max_all = np.zeros(n_eligible, dtype=np.float32)
    grad_min_all = np.zeros(n_eligible, dtype=np.float32)
    grad_energy_all = np.zeros(n_eligible, dtype=np.float32)
    grad_std_all = np.zeros(n_eligible, dtype=np.float32)
    grad_peak_idx_all = np.zeros(n_eligible, dtype=np.float32)
    curv_max_all = np.zeros(n_eligible, dtype=np.float32)
    curv_energy_all = np.zeros(n_eligible, dtype=np.float32)
    curv_n_inflections_all = np.zeros(n_eligible, dtype=np.float32)
    grad_ratio_all = np.zeros(n_eligible, dtype=np.float32)
    grad_burst_w_all = np.zeros(n_eligible, dtype=np.float32)

    for start in range(0, n_eligible, chunk_size):
        end = min(start + chunk_size, n_eligible)
        chunk_raw_idx = eligible_indices[start:end]
        chunk = np.array(blocks_mmap[chunk_raw_idx], dtype=np.float32)
        n_c = len(chunk)

        # 1. W=350 rolling stats
        cumsum = np.empty((n_c, seq_len + 1), dtype=np.float32)
        cumsum[:, 0] = 0.0
        np.cumsum(chunk, axis=1, out=cumsum[:, 1:])

        cumsum2 = np.empty((n_c, seq_len + 1), dtype=np.float32)
        cumsum2[:, 0] = 0.0
        np.cumsum(chunk**2, axis=1, out=cumsum2[:, 1:])

        rm = (cumsum[:, w:] - cumsum[:, :-w]) / w
        rv = np.maximum(0.0, (cumsum2[:, w:] - cumsum2[:, :-w]) / w - rm**2)

        max_rm = np.max(rm, axis=1)
        mean_rm_all[start:end] = np.mean(rm, axis=1)
        max_rm_all[start:end] = max_rm
        min_rm_all[start:end] = np.min(rm, axis=1)
        max_rstd_all[start:end] = np.sqrt(np.max(rv, axis=1))
        max_idx_all[start:end] = np.argmax(rm, axis=1).astype(np.float32) / float(seq_len - w)
        block_mean = np.mean(chunk, axis=1)
        excess350_all[start:end] = np.maximum(0.0, max_rm - block_mean)

        # 2. 1D Top-hat filters
        th100 = ndi.white_tophat(chunk, size=(1, 100))
        th200 = ndi.white_tophat(chunk, size=(1, 200))
        th_peak100_all[start:end] = np.max(th100, axis=1)
        th_energy100_all[start:end] = np.mean(th100**2, axis=1)
        th_peak200_all[start:end] = np.max(th200, axis=1)
        th_energy200_all[start:end] = np.mean(th200**2, axis=1)

        # 3. Gradient and curvature
        smoothed = uniform_filter1d(chunk, size=5, axis=1, mode="reflect")
        grad = np.diff(smoothed, axis=1)
        curv = np.diff(grad, axis=1)
        abs_grad = np.abs(grad)
        abs_curv = np.abs(curv)

        g_max = np.max(abs_grad, axis=1)
        grad_max_all[start:end] = g_max
        grad_min_all[start:end] = np.min(grad, axis=1)
        grad_energy_all[start:end] = np.sum(grad**2, axis=1) / 1999.0
        g_std = np.std(grad, axis=1)
        grad_std_all[start:end] = g_std
        grad_peak_idx_all[start:end] = np.argmax(abs_grad, axis=1).astype(np.float32) / 2000.0

        curv_max_all[start:end] = np.max(abs_curv, axis=1)
        curv_energy_all[start:end] = np.sum(curv**2, axis=1) / 1998.0
        sign_changes = np.diff(np.sign(curv), axis=1)
        curv_n_inflections_all[start:end] = np.sum(np.abs(sign_changes) > 0, axis=1).astype(np.float32)

        mean_abs_g = np.mean(abs_grad, axis=1)
        grad_ratio_all[start:end] = np.where(mean_abs_g > 1e-8, g_max / mean_abs_g, 0.0)
        grad_burst_w_all[start:end] = np.sum(abs_grad > (2.0 * g_std[:, np.newaxis]), axis=1).astype(np.float32)

    df_w350 = pd.DataFrame({
        "block_rolling_mean_350": mean_rm_all,
        "max_rolling_mean_350": max_rm_all,
        "min_rolling_mean_350": min_rm_all,
        "max_rolling_std_350": max_rstd_all,
        "max_rolling_mean_350_start_idx": max_idx_all,
        "burst_excess_350": excess350_all,
    })

    df_th = pd.DataFrame({
        "tophat_peak_100": th_peak100_all,
        "tophat_energy_100": th_energy100_all,
        "tophat_peak_200": th_peak200_all,
        "tophat_energy_200": th_energy200_all,
    })

    df_grad = pd.DataFrame({
        "grad_max": grad_max_all,
        "grad_min": grad_min_all,
        "grad_energy": grad_energy_all,
        "grad_std": grad_std_all,
        "grad_peak_idx": grad_peak_idx_all,
        "curv_max": curv_max_all,
        "curv_energy": curv_energy_all,
        "curv_n_inflections": curv_n_inflections_all,
        "grad_ratio_max_mean": grad_ratio_all,
        "grad_burst_width": grad_burst_w_all,
    })

    print(f"    Done in {time.time() - t0:.2f}s.", flush=True)
    return df_w350, df_th, df_grad


def extract_test_geometric_features(df_eligible):
    """Computes orthogonal circular Zernike polynomials and stepper reticle features."""
    print(f"  Extracting geometric shape & Zernike features ({len(df_eligible):,} dies)...", flush=True)
    t0 = time.time()
    results = []

    for wafer_id, w_group in df_eligible.groupby("wafer_id", sort=False):
        r = w_group["die_row"].values.astype(np.float32)
        c = w_group["die_col"].values.astype(np.float32)
        r0 = (r.max() + r.min()) / 2.0
        c0 = (c.max() + c.min()) / 2.0
        dist = np.sqrt((r - r0)**2 + (c - c0)**2)
        R = max(float(dist.max()), 1.0)
        rho = dist / R
        phi = np.arctan2(r - r0, c - c0)

        z1_neg1 = (2.0 * rho * np.sin(phi)).astype(np.float32)
        z1_1 = (2.0 * rho * np.cos(phi)).astype(np.float32)
        z2_0 = (np.sqrt(3) * (2.0 * rho**2 - 1.0)).astype(np.float32)
        z2_neg2 = (np.sqrt(6) * rho**2 * np.sin(2.0 * phi)).astype(np.float32)
        z2_2 = (np.sqrt(6) * rho**2 * np.cos(2.0 * phi)).astype(np.float32)
        z4_0 = (np.sqrt(5) * (6.0 * rho**4 - 6.0 * rho**2 + 1.0)).astype(np.float32)

        reticle_r = (r.astype(int) % 4).astype(np.int8)
        reticle_c = (c.astype(int) % 4).astype(np.int8)
        reticle_pos = (reticle_r * 4 + reticle_c).astype(np.int8)
        is_reticle_corner = (((reticle_r == 0) | (reticle_r == 3)) & ((reticle_c == 0) | (reticle_c == 3))).astype(np.int8)

        w_feat = pd.DataFrame({
            "zernike_Z1_neg1": z1_neg1,
            "zernike_Z1_1": z1_1,
            "zernike_Z2_0": z2_0,
            "zernike_Z2_neg2": z2_neg2,
            "zernike_Z2_2": z2_2,
            "zernike_Z4_0": z4_0,
            "reticle_pos": reticle_pos,
            "is_reticle_corner": is_reticle_corner,
        }, index=w_group.index)
        results.append(w_feat)

    all_res = pd.concat(results).loc[df_eligible.index]
    print(f"    Done in {time.time() - t0:.2f}s.", flush=True)
    return all_res


def build_final_test_features():
    print("=" * 80)
    print("FINAL UNSEEN TEST MODEL E FEATURE ENGINEERING (ZERO-LABEL PIPELINE)")
    print("=" * 80)
    t_start = time.time()

    # 1. Load unlabeled validation features
    print("\n[Step 1/6] Loading validation_features.parquet (unlabeled test data)...")
    val_feat = pd.read_parquet(VALIDATION_PARQUET)
    assert "label" not in val_feat.columns, "SECURITY CHECK FAILED: validation_features contains labels!"

    eligible_mask = (val_feat["old_label"] == 0).values
    eligible_raw_indices = np.where(eligible_mask)[0]
    df_eligible = val_feat.iloc[eligible_raw_indices].reset_index(drop=True)
    print(f"  Total test dies: {len(val_feat):,} across {val_feat['wafer_id'].nunique()} wafers")
    print(f"  Eligible dies:   {len(df_eligible):,}")

    # 2. Extract Cluster Topology & EDT
    print("\n[Step 2/6] Computing test cluster topology & EDT...")
    df_cluster = extract_test_cluster_topology(val_feat, eligible_raw_indices).reset_index(drop=True)

    # 3. PCA Projection using frozen scaler
    print("\n[Step 3/6] Applying frozen PCA scaler to test parametric features...")
    with open(PCA_SCALER_PATH, "rb") as f:
        pca_artifacts = pickle.load(f)
    scaler = pca_artifacts["scaler"]
    pca = pca_artifacts["pca"]

    X_param_scaled = scaler.transform(df_eligible[PARAMETRIC_FEATURES].values)
    X_param_pca = pca.transform(X_param_scaled).astype(np.float32)
    pca_cols = [f"pca_{i+1:02d}" for i in range(10)]
    df_pca = pd.DataFrame(X_param_pca, columns=pca_cols)

    # 4. Extract Block Signals (W=350, Top-Hat, Gradient/Curvature)
    print("\n[Step 4/6] Streaming test block sequences from cache...")
    df_w350, df_th, df_grad = extract_test_block_signals(FINAL_TEST_BLOCKS_DAT, eligible_raw_indices, total_n=len(val_feat))

    # 5. Geometric & Bilinear Interaction Features
    print("\n[Step 5/6] Extracting geometric and bilinear interaction terms...")
    df_geom = extract_test_geometric_features(df_eligible).reset_index(drop=True)

    pc1 = df_pca["pca_01"].values
    roll350_burst = np.maximum(0.0, df_w350["max_rolling_mean_350"].values - 100.0)
    roll400_burst = np.maximum(0.0, df_eligible["max_rolling_mean_400"].values - 100.0)
    cluster_log_sz = df_cluster["nearest_defect_log_cluster_size"].values
    edt_dist = df_cluster["exact_edt_distance"].values
    edge_risk = (1.0 - df_eligible["distance_to_edge"].values).astype(np.float32)
    edge_dist = df_eligible["distance_to_edge"].values
    density5 = df_eligible["old_fail_density_5x5"].values
    burst_excess_350 = df_w350["burst_excess_350"].values

    df_bilinear = pd.DataFrame({
        "inter_pca01_x_roll350": (pc1 * roll350_burst).astype(np.float32),
        "inter_pca01_x_cluster_size": (pc1 * cluster_log_sz).astype(np.float32),
        "inter_pca01_x_edt": (pc1 / (edt_dist + 0.02)).astype(np.float32),
        "inter_pca01_x_edge": (pc1 * edge_risk).astype(np.float32),
        "inter_roll350_x_cluster": (roll350_burst * cluster_log_sz).astype(np.float32),
        "inter_pc1_x_roll400": (pc1 * roll400_burst).astype(np.float32),
        "inter_pc1_x_edge_density": (pc1 * roll350_burst * (1.0 - edge_dist) * (density5 + 0.05)).astype(np.float32),
        "inter_pc1_x_burst_excess_350": (pc1 * burst_excess_350).astype(np.float32),
    })

    # 6. Wafer-Local Distributional & Rank Features (using dev_train global statistics)
    print("\n[Step 6/6] Computing wafer-local rank and distributional features...")
    print("  Loading frozen global mean/std statistics from dev_train...")
    df_train_m_e = pd.read_parquet(DEV_TRAIN_MODEL_E_PARQUET, columns=[
        "pca_01", "pca_02", "pca_03", "max_rolling_mean_350", "max_rolling_mean_400",
        "block_std", "block_q99", "block_max_z", "inter_pca01_x_roll350",
        "distance_to_nearest_old_failure", "grad_energy", "curv_energy",
    ])

    rank_target_cols = [
        "pca_01", "pca_02", "pca_03",
        "max_rolling_mean_350", "max_rolling_mean_400",
        "block_std", "block_q99", "block_max_z",
        "inter_pca01_x_roll350", "distance_to_nearest_old_failure",
        "grad_energy", "curv_energy",
    ]

    global_stats = {}
    for col in rank_target_cols:
        global_stats[col] = {
            "mean": float(df_train_m_e[col].mean()),
            "std": float(max(df_train_m_e[col].std(), 1e-6)),
        }
    del df_train_m_e

    # Temporary dataframe for rank feature inputs
    temp_test = pd.concat([
        df_eligible[["wafer_id", "max_rolling_mean_400", "block_std", "block_q99", "block_max_z", "distance_to_nearest_old_failure"]],
        df_pca[["pca_01", "pca_02", "pca_03"]],
        df_w350[["max_rolling_mean_350"]],
        df_bilinear[["inter_pca01_x_roll350"]],
        df_grad[["grad_energy", "curv_energy"]],
    ], axis=1)

    rank_dict = {}
    wzscore_dict = {}
    wdev_dict = {}

    for col in rank_target_cols:
        short = col.replace("max_rolling_mean_", "roll").replace("inter_pca01_x_", "ix_")
        
        # 1. Wafer-local percentile rank
        rank_col = f"wrank_{short}"
        rank_dict[rank_col] = temp_test.groupby("wafer_id")[col].rank(pct=True).astype(np.float32).values

        # 2. Inter-wafer z-score using dev_train global stats
        wafer_mean = temp_test.groupby("wafer_id")[col].transform("mean").values
        wzscore_col = f"wzscore_{short}"
        wzscore_dict[wzscore_col] = (
            (wafer_mean - global_stats[col]["mean"]) / global_stats[col]["std"]
        ).astype(np.float32)

        # 3. Within-wafer standardized deviation
        wafer_std = temp_test.groupby("wafer_id")[col].transform("std").fillna(0.0).values
        wdev_col = f"wdev_{short}"
        wdev_dict[wdev_col] = (
            (temp_test[col].values - wafer_mean) / (wafer_std + 1e-6)
        ).astype(np.float32)

    df_rank = pd.DataFrame({**rank_dict, **wzscore_dict, **wdev_dict})

    # Combine all feature tables
    print("\nAssembling final test feature matrix...")
    model_b_cols = [c for c in MODEL_B_FEATURES if c in df_eligible.columns]

    df_final_test = pd.concat([
        df_eligible[["wafer_id", "die_row", "die_col", "old_label"] + model_b_cols],
        df_pca,
        df_cluster,
        df_w350,
        df_th,
        df_geom,
        df_bilinear,
        df_grad,
        df_rank,
    ], axis=1)

    # Drop duplicate column names if any
    df_final_test = df_final_test.loc[:, ~df_final_test.columns.duplicated()]

    # Verify column alignment with model_e_feature_list.json
    with open(MODEL_E_FEATURE_LIST_PATH, "r") as f:
        expected_cols = json.load(f)

    missing = [c for c in expected_cols if c not in df_final_test.columns]
    assert len(missing) == 0, f"Missing {len(missing)} features: {missing[:5]}"

    # Verify zero NaNs
    n_nans = df_final_test[expected_cols].isna().sum().sum()
    if n_nans > 0:
        print(f"  WARNING: Found {n_nans} NaNs in test features, filling with 0.0.")
        df_final_test[expected_cols] = df_final_test[expected_cols].fillna(0.0)

    print(f"\nFinal Test Feature Matrix: {df_final_test.shape} ({len(expected_cols)} features)")
    print(f"Saving to: {FINAL_TEST_MODEL_E_PARQUET}")
    df_final_test.to_parquet(FINAL_TEST_MODEL_E_PARQUET, index=False)
    print(f"Test feature engineering completed in {time.time() - t_start:.2f}s.")
    return df_final_test


if __name__ == "__main__":
    build_final_test_features()
