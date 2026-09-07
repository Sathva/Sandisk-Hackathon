"""
Targeted Feature Engineering Pipeline for Model D.
Extracts high-value, leak-free feature families inspired by teammate Adit's branch:
1. 10-Component Parametric PCA (fitted strictly on dev_train, zero leakage)
2. Wafer-Isolated Cluster Topology & Exact Distance Transform (EDT) using old_label == 1
3. Per-Die Independent W=350 Block Rolling Statistics from raw 2000-element readings
4. Per-Die Morphological White Top-Hat Filters (W=100, W=200)
5. Geometric / Shape-Derived Features (Zernike orthogonal polynomials & reticle grid)
6. Targeted Cross-Resolution Bilinear Interaction Terms

Saves additional feature sets to processed/ without modifying or touching final test data.
"""

import sys
import os
import time
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

MODEL_D_PCA_SCALER_PATH = MODELS_DIR / "model_d_pca_scaler.pkl"
MODEL_D_FEATURE_GROUPS_PATH = MODELS_DIR / "model_d_feature_groups.json"
DEV_TRAIN_ADDITIONAL_PARQUET = PROCESSED_DIR / "dev_train_model_d_additional.parquet"
DEV_VAL_ADDITIONAL_PARQUET = PROCESSED_DIR / "dev_val_model_d_additional.parquet"


# -----------------------------------------------------------------------------
# 1. PCA Latent Manifold (10 Components) Strictly on dev_train
# -----------------------------------------------------------------------------
def extract_parametric_pca(df_train, df_val):
    """
    Fits StandardScaler and 10-component PCA strictly on dev_train parametric features.
    Transforms dev_train and dev_val independently.
    Zero target leakage, zero validation data leakage.
    """
    print("\n--- 1. FITTING PARAMETRIC PCA (DEV_TRAIN ONLY) ---")
    t0 = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[PARAMETRIC_FEATURES].values)

    pca = PCA(n_components=10, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_scaled = scaler.transform(df_val[PARAMETRIC_FEATURES].values)
    X_val_pca = pca.transform(X_val_scaled)

    # Save pipeline object
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_D_PCA_SCALER_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca}, f)
    print(f"Saved PCA/Scaler pipeline to: {MODEL_D_PCA_SCALER_PATH.name}")

    var_ratio = pca.explained_variance_ratio_
    print(f"Explained variance top 3: {var_ratio[:3]} | Total top 10: {var_ratio.sum():.4f}")

    pca_cols = [f"pca_{i+1:02d}" for i in range(10)]
    df_pca_train = pd.DataFrame(X_train_pca.astype(np.float32), columns=pca_cols, index=df_train.index)
    df_pca_val = pd.DataFrame(X_val_pca.astype(np.float32), columns=pca_cols, index=df_val.index)

    print(f"Parametric PCA extraction complete in {time.time() - t0:.2f}s.")
    return df_pca_train, df_pca_val, pca_cols


# -----------------------------------------------------------------------------
# 2. Wafer-Isolated Cluster Topology & Exact Distance Transform (EDT)
# -----------------------------------------------------------------------------
def extract_wafer_cluster_topology(df_full, eligible_index):
    """
    Computes exact Euclidean Distance Transform, connected defect cluster sizes,
    and directional hazard vectors pointing toward defect centers.
    Strictly isolated per wafer using only pre-test information (old_label == 1).
    Never accesses post-test target 'label'.
    """
    print(f"\n--- 2. COMPUTING WAFER CLUSTER TOPOLOGY & EDT ({len(df_full):,} total dies) ---")
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
            # Connected components on pre-test defects (8-connectivity)
            labeled_clusters, _ = ndi.label(old_fail_mask, structure=np.ones((3, 3), dtype=int))
            cluster_sizes = np.bincount(labeled_clusters.ravel())

            # Distance transform with nearest failure index coordinates
            dist_grid, (nearest_r, nearest_c) = ndi.distance_transform_edt(~old_fail_mask, return_indices=True)

            die_dists = dist_grid[rows, cols]
            nearest_cluster_sizes = cluster_sizes[labeled_clusters[nearest_r[rows, cols], nearest_c[rows, cols]]]

            # Directional hazard vector from target die to nearest defect
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
    cluster_cols = [
        "exact_edt_distance",
        "nearest_defect_cluster_size",
        "nearest_defect_log_cluster_size",
        "defect_vector_dr",
        "defect_vector_dc",
        "is_defect_neighbor_1hop",
        "is_defect_neighbor_2hop",
    ]
    print(f"Cluster topology & EDT complete for {len(all_res):,} eligible dies in {time.time() - t0:.2f}s.")
    return all_res[cluster_cols], cluster_cols


# -----------------------------------------------------------------------------
# 3. Per-Die Independent W=350 Block Rolling Features & Top-Hat Filters
# -----------------------------------------------------------------------------
def extract_block_and_tophat_features(raw_blocks_path, n_dies, chunk_size=25000):
    """
    Computes rolling statistics at window W=350 and 1D morphological white top-hat filters
    strictly along axis 1 of each individual die's 2,000-reading sequence.
    Zero inter-die sequence leakage.
    """
    print(f"\n--- 3. EXTRACTING W=350 & TOP-HAT FEATURES ({n_dies:,} dies from {raw_blocks_path.name}) ---")
    t0 = time.time()
    blocks_mmap = np.memmap(raw_blocks_path, dtype="float32", mode="r", shape=(n_dies, 2000))
    w = 350
    seq_len = 2000

    records_w350 = []
    records_tophat = []

    for start_idx in range(0, n_dies, chunk_size):
        end_idx = min(start_idx + chunk_size, n_dies)
        chunk = np.array(blocks_mmap[start_idx:end_idx], dtype=np.float32)
        n_chunk = len(chunk)

        # Vectorized cumsum strictly along axis 1 (per die)
        cumsum = np.empty((n_chunk, seq_len + 1), dtype=np.float32)
        cumsum[:, 0] = 0.0
        np.cumsum(chunk, axis=1, out=cumsum[:, 1:])

        cumsum2 = np.empty((n_chunk, seq_len + 1), dtype=np.float32)
        cumsum2[:, 0] = 0.0
        np.cumsum(chunk**2, axis=1, out=cumsum2[:, 1:])

        rm = (cumsum[:, w:] - cumsum[:, :-w]) / w
        rv = np.maximum(0.0, (cumsum2[:, w:] - cumsum2[:, :-w]) / w - rm**2)

        max_rm = np.max(rm, axis=1)
        min_rm = np.min(rm, axis=1)
        mean_rm = np.mean(rm, axis=1)
        max_rstd = np.sqrt(np.max(rv, axis=1))
        max_idx = np.argmax(rm, axis=1).astype(np.float32) / float(seq_len - w)
        block_mean = np.mean(chunk, axis=1)
        excess350 = np.maximum(0.0, max_rm - block_mean)

        w350_df = pd.DataFrame({
            "block_rolling_mean_350": mean_rm.astype(np.float32),
            "max_rolling_mean_350": max_rm.astype(np.float32),
            "min_rolling_mean_350": min_rm.astype(np.float32),
            "max_rolling_std_350": max_rstd.astype(np.float32),
            "max_rolling_mean_350_start_idx": max_idx.astype(np.float32),
            "burst_excess_350": excess350.astype(np.float32),
        })
        records_w350.append(w350_df)

        # 1D Morphological White Top-Hat strictly along axis 1 (per die)
        th100 = ndi.white_tophat(chunk, size=(1, 100))
        th200 = ndi.white_tophat(chunk, size=(1, 200))
        th_df = pd.DataFrame({
            "tophat_peak_100": np.max(th100, axis=1).astype(np.float32),
            "tophat_energy_100": np.mean(th100**2, axis=1).astype(np.float32),
            "tophat_peak_200": np.max(th200, axis=1).astype(np.float32),
            "tophat_energy_200": np.mean(th200**2, axis=1).astype(np.float32),
        })
        records_tophat.append(th_df)

        elapsed = time.time() - t0
        rate = end_idx / max(elapsed, 0.001)
        print(f"   Processed {end_idx:,} / {n_dies:,} dies ({rate:.0f} dies/s)...", end="\r", flush=True)

    df_w350 = pd.concat(records_w350, ignore_index=True)
    df_tophat = pd.concat(records_tophat, ignore_index=True)
    print(f"\nCompleted W=350 and Top-Hat extraction in {time.time() - t0:.2f}s.")

    w350_cols = [
        "block_rolling_mean_350",
        "max_rolling_mean_350",
        "min_rolling_mean_350",
        "max_rolling_std_350",
        "max_rolling_mean_350_start_idx",
        "burst_excess_350",
    ]
    tophat_cols = [
        "tophat_peak_100",
        "tophat_energy_100",
        "tophat_peak_200",
        "tophat_energy_200",
    ]
    return df_w350, df_tophat, w350_cols, tophat_cols


# -----------------------------------------------------------------------------
# 4. Geometric / Shape-Derived Features
# -----------------------------------------------------------------------------
def extract_geometric_shape_features(df_eligible):
    """
    Computes orthogonal circular Zernike polynomial features and stepper reticle coordinates.
    Described strictly as geometric / shape-derived features.
    """
    print(f"\n--- 4. EXTRACTING GEOMETRIC / SHAPE-DERIVED FEATURES ({len(df_eligible):,} dies) ---")
    t0 = time.time()
    results = []

    for wafer_id, w_group in df_eligible.groupby("wafer_id", sort=False):
        r = w_group["die_row"].values.astype(np.float32)
        c = w_group["die_col"].values.astype(np.float32)

        r0 = (r.max() + r.min()) / 2.0
        c0 = (c.max() + c.min()) / 2.0
        dist = np.sqrt((r - r0)**2 + (c - c0)**2)
        R = max(float(dist.max()), 1.0)

        rho = dist / R  # Normalized radius [0, 1]
        phi = np.arctan2(r - r0, c - c0)  # Polar angle [-pi, pi]

        # Orthogonal Zernike polynomial terms
        z1_neg1 = (2.0 * rho * np.sin(phi)).astype(np.float32)
        z1_1 = (2.0 * rho * np.cos(phi)).astype(np.float32)
        z2_0 = (np.sqrt(3) * (2.0 * rho**2 - 1.0)).astype(np.float32)
        z2_neg2 = (np.sqrt(6) * rho**2 * np.sin(2.0 * phi)).astype(np.float32)
        z2_2 = (np.sqrt(6) * rho**2 * np.cos(2.0 * phi)).astype(np.float32)
        z4_0 = (np.sqrt(5) * (6.0 * rho**4 - 6.0 * rho**2 + 1.0)).astype(np.float32)

        # Stepper Reticle Field features (4x4 exposure grid)
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
    geom_cols = [
        "zernike_Z1_neg1",
        "zernike_Z1_1",
        "zernike_Z2_0",
        "zernike_Z2_neg2",
        "zernike_Z2_2",
        "zernike_Z4_0",
        "reticle_pos",
        "is_reticle_corner",
    ]
    print(f"Geometric shape features complete in {time.time() - t0:.2f}s.")
    return all_res[geom_cols], geom_cols


# -----------------------------------------------------------------------------
# 5. Targeted Cross-Resolution Bilinear Interactions
# -----------------------------------------------------------------------------
def compute_bilinear_interactions(pca_df, cluster_df, w350_df, base_df):
    """
    Computes targeted cross-resolution multiplicative interaction terms:
    - pca_01 x W=350 burst
    - pca_01 x nearest cluster size
    - pca_01 x exact EDT proximity
    - pca_01 x wafer edge risk
    - W=350 burst x nearest cluster size
    """
    print("\n--- 5. COMPUTING TARGETED CROSS-RESOLUTION BILINEAR INTERACTIONS ---")
    t0 = time.time()
    pc1 = pca_df["pca_01"].values
    roll350_burst = np.maximum(0.0, w350_df["max_rolling_mean_350"].values - 100.0)
    cluster_log_sz = cluster_df["nearest_defect_log_cluster_size"].values
    edt_dist = cluster_df["exact_edt_distance"].values
    edge_risk = (1.0 - base_df["distance_to_edge"].values).astype(np.float32)

    inter_df = pd.DataFrame({
        "inter_pca01_x_roll350": (pc1 * roll350_burst).astype(np.float32),
        "inter_pca01_x_cluster_size": (pc1 * cluster_log_sz).astype(np.float32),
        "inter_pca01_x_edt": (pc1 / (edt_dist + 0.02)).astype(np.float32),
        "inter_pca01_x_edge": (pc1 * edge_risk).astype(np.float32),
        "inter_roll350_x_cluster": (roll350_burst * cluster_log_sz).astype(np.float32),
    }, index=base_df.index)

    inter_cols = list(inter_df.columns)
    print(f"Bilinear interaction extraction complete in {time.time() - t0:.2f}s.")
    return inter_df, inter_cols


# -----------------------------------------------------------------------------
# 6. Main Orchestrator & Dataset Caching
# -----------------------------------------------------------------------------
def build_model_d_features():
    print("=" * 85)
    print("STARTING MODEL D FEATURE EXTRACTION PIPELINE (ZERO LEAKAGE)")
    print("=" * 85)
    print(f"Source Train Parquet: {DEV_TRAIN_PARQUET.name}")
    print(f"Source Val Parquet:   {DEV_VAL_PARQUET.name}")
    print(f"Output Directory:     {PROCESSED_DIR}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Parquet tables
    t0_load = time.time()
    print("\nLoading dev_train and dev_val base feature tables...")
    df_train_full = pd.read_parquet(DEV_TRAIN_PARQUET)
    df_val_full = pd.read_parquet(DEV_VAL_PARQUET)
    print(f"Loaded full tables in {time.time() - t0_load:.2f}s.")

    # Filter to eligible dies: old_label == 0
    train_eligible_mask = (df_train_full["old_label"] == 0).values
    val_eligible_mask = (df_val_full["old_label"] == 0).values

    df_train_el = df_train_full[train_eligible_mask].copy().reset_index(drop=True)
    df_val_el = df_val_full[val_eligible_mask].copy().reset_index(drop=True)

    N_TRAIN = len(df_train_el)
    N_VAL = len(df_val_el)
    print(f"Eligible dies: Train = {N_TRAIN:,} | Val = {N_VAL:,}")

    # 2. Extract PCA features (fitted on train only)
    pca_tr, pca_va, pca_cols = extract_parametric_pca(df_train_el, df_val_el)

    # 3. Extract Wafer Cluster Topology & EDT
    cluster_tr, cluster_cols = extract_wafer_cluster_topology(df_train_full, df_train_full.index[train_eligible_mask])
    cluster_va, _ = extract_wafer_cluster_topology(df_val_full, df_val_full.index[val_eligible_mask])
    cluster_tr.reset_index(drop=True, inplace=True)
    cluster_va.reset_index(drop=True, inplace=True)

    # 4. Extract W=350 and Top-Hat block features
    w350_tr, tophat_tr, w350_cols, tophat_cols = extract_block_and_tophat_features(
        TRAIN_RAW_BLOCKS_PATH, N_TRAIN
    )
    w350_va, tophat_va, _, _ = extract_block_and_tophat_features(
        VAL_RAW_BLOCKS_PATH, N_VAL
    )

    # 5. Extract Geometric Shape features
    geom_tr, geom_cols = extract_geometric_shape_features(df_train_el)
    geom_va, _ = extract_geometric_shape_features(df_val_el)
    geom_tr.reset_index(drop=True, inplace=True)
    geom_va.reset_index(drop=True, inplace=True)

    # 6. Extract Targeted Bilinear Interactions
    inter_tr, inter_cols = compute_bilinear_interactions(pca_tr, cluster_tr, w350_tr, df_train_el)
    inter_va, _ = compute_bilinear_interactions(pca_va, cluster_va, w350_va, df_val_el)

    # 7. Assemble and Cache Additional Features
    print("\n--- 7. ASSEMBLING AND SAVING MODEL D ADDITIONAL FEATURE SETS ---")
    add_train_df = pd.concat([pca_tr, cluster_tr, w350_tr, tophat_tr, geom_tr, inter_tr], axis=1)
    add_val_df = pd.concat([pca_va, cluster_va, w350_va, tophat_va, geom_va, inter_va], axis=1)

    print(f"Additional Features Shape: Train {add_train_df.shape} | Val {add_val_df.shape}")
    assert len(add_train_df) == N_TRAIN, f"Train row count mismatch: {len(add_train_df)} != {N_TRAIN}"
    assert len(add_val_df) == N_VAL, f"Val row count mismatch: {len(add_val_df)} != {N_VAL}"
    assert not add_train_df.isna().any().any(), "NaN values detected in training features!"
    assert not add_val_df.isna().any().any(), "NaN values detected in validation features!"

    t0_save = time.time()
    add_train_df.to_parquet(DEV_TRAIN_ADDITIONAL_PARQUET, index=False)
    add_val_df.to_parquet(DEV_VAL_ADDITIONAL_PARQUET, index=False)
    print(f"Saved {DEV_TRAIN_ADDITIONAL_PARQUET.name} ({DEV_TRAIN_ADDITIONAL_PARQUET.stat().st_size / (1024*1024):.1f} MB) in {time.time() - t0_save:.2f}s.")
    print(f"Saved {DEV_VAL_ADDITIONAL_PARQUET.name} ({DEV_VAL_ADDITIONAL_PARQUET.stat().st_size / (1024*1024):.1f} MB).")

    # Save feature group definitions
    feature_groups = {
        "PCA_FEATURES": pca_cols,
        "CLUSTER_EDT_FEATURES": cluster_cols,
        "W350_FEATURES": w350_cols,
        "TOPHAT_FEATURES": tophat_cols,
        "GEOMETRIC_FEATURES": geom_cols,
        "INTERACTION_FEATURES": inter_cols,
        "TOTAL_ADDITIONAL_COUNT": len(add_train_df.columns),
    }
    with open(MODEL_D_FEATURE_GROUPS_PATH, "w") as f:
        json.dump(feature_groups, f, indent=2)
    print(f"Saved feature group metadata: {MODEL_D_FEATURE_GROUPS_PATH.name}")

    print("\n" + "=" * 85)
    print("MODEL D FEATURE EXTRACTION PIPELINE SUCCESSFULLY COMPLETED")
    print("=" * 85 + "\n")
    return feature_groups


if __name__ == "__main__":
    build_model_d_features()
