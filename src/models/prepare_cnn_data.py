"""
High-Performance Memory-Mapped Cache Builder for Model C (1D CNN).
Streams train.csv in chunks, extracts raw 2,000-element block readings for eligible dies (old_label == 0),
verifies data integrity, computes global normalization statistics strictly from dev_train,
and writes memory-mapped binary files for zero-copy GPU training.
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure repo root is on path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    TRAIN_CSV,
    DEV_TRAIN_PARQUET,
    DEV_VAL_PARQUET,
    DEV_SPLIT_JSON,
    MODELS_DIR,
    REPORTS_DIR,
    PROCESSED_DIR,
)
from src.models.common import MODEL_A_FEATURES

CACHE_DIR = PROCESSED_DIR / "cache"


def prepare_cnn_cache(chunksize=25000):
    print(f"\n{'=' * 85}")
    print("STARTING MODEL C DATA PREPARATION & MEMORY-MAPPED CACHE BUILDER")
    print(f"{'=' * 85}")
    print(f"Source CSV:    {TRAIN_CSV}")
    print(f"Output Cache:  {CACHE_DIR}")
    print(f"{'=' * 85}\n")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Canonical Split Wafer IDs
    with open(DEV_SPLIT_JSON, "r") as f:
        split_data = json.load(f)
    train_wafers = set(split_data["train_wafer_ids"])
    val_wafers = set(split_data["dev_val_wafer_ids"])

    print(f"Loaded split metadata: {len(train_wafers)} train wafers, {len(val_wafers)} dev-val wafers.")

    # Target eligible die counts
    N_TRAIN = 651337
    N_VAL = 137576
    SEQ_LEN = 2000

    train_blocks_path = CACHE_DIR / "dev_train_raw_blocks.dat"
    val_blocks_path = CACHE_DIR / "dev_val_raw_blocks.dat"

    # Pre-allocate binary memmaps
    print(f"\nAllocating memory-mapped files on NVMe:")
    print(f"  Train blocks: {train_blocks_path} ({N_TRAIN} x {SEQ_LEN}, float32 ~{N_TRAIN*SEQ_LEN*4/(1024**3):.2f} GB)")
    print(f"  Val blocks:   {val_blocks_path} ({N_VAL} x {SEQ_LEN}, float32 ~{N_VAL*SEQ_LEN*4/(1024**3):.2f} GB)")

    train_blocks_mm = np.memmap(train_blocks_path, dtype="float32", mode="w+", shape=(N_TRAIN, SEQ_LEN))
    val_blocks_mm = np.memmap(val_blocks_path, dtype="float32", mode="w+", shape=(N_VAL, SEQ_LEN))

    train_labels = np.zeros(N_TRAIN, dtype="float32")
    val_labels = np.zeros(N_VAL, dtype="float32")

    train_offset = 0
    val_offset = 0
    total_rows_scanned = 0
    malformed_count = 0
    t0_stream = time.time()

    print(f"\nStreaming {TRAIN_CSV.name} in chunks of {chunksize:,} rows...")
    use_cols = ["wafer_id", "die_row", "die_col", "old_label", "label", "block_readings"]

    for chunk in pd.read_csv(TRAIN_CSV, usecols=use_cols, chunksize=chunksize):
        total_rows_scanned += len(chunk)

        # Filter strictly eligible dies: old_label == 0
        eligible_chunk = chunk[chunk["old_label"] == 0].reset_index(drop=True)
        if len(eligible_chunk) == 0:
            continue

        # Parse block_readings string into 2D float32 array
        parsed_arrays = []
        valid_indices = []
        for idx, row_str in enumerate(eligible_chunk["block_readings"]):
            try:
                arr = np.fromstring(row_str, dtype=np.float32, sep=" ")
                if len(arr) == SEQ_LEN:
                    parsed_arrays.append(arr)
                    valid_indices.append(idx)
                else:
                    malformed_count += 1
            except Exception:
                malformed_count += 1

        if not valid_indices:
            continue

        valid_eligible = eligible_chunk.iloc[valid_indices].reset_index(drop=True)
        parsed_matrix = np.vstack(parsed_arrays)

        # Partition into train vs val based on wafer_id
        is_train = valid_eligible["wafer_id"].isin(train_wafers).values
        is_val = valid_eligible["wafer_id"].isin(val_wafers).values

        n_chunk_train = int(is_train.sum())
        n_chunk_val = int(is_val.sum())

        if n_chunk_train > 0:
            train_blocks_mm[train_offset : train_offset + n_chunk_train] = parsed_matrix[is_train]
            train_labels[train_offset : train_offset + n_chunk_train] = valid_eligible.loc[is_train, "label"].values.astype(np.float32)
            train_offset += n_chunk_train

        if n_chunk_val > 0:
            val_blocks_mm[val_offset : val_offset + n_chunk_val] = parsed_matrix[is_val]
            val_labels[val_offset : val_offset + n_chunk_val] = valid_eligible.loc[is_val, "label"].values.astype(np.float32)
            val_offset += n_chunk_val

        elapsed = time.time() - t0_stream
        print(f"  Scanned {total_rows_scanned:,} dies | Train: {train_offset:,}/{N_TRAIN:,} | Val: {val_offset:,}/{N_VAL:,} ({elapsed:.1f}s)", end="\r", flush=True)

    # Flush memmaps to disk
    train_blocks_mm.flush()
    val_blocks_mm.flush()
    del train_blocks_mm
    del val_blocks_mm

    stream_time = time.time() - t0_stream
    print(f"\n\nStream & parsing completed in {stream_time:.1f}s ({stream_time / 60:.2f} min).")
    print(f"  Total eligible train dies parsed: {train_offset:,} (Expected: {N_TRAIN:,})")
    print(f"  Total eligible val dies parsed:   {val_offset:,} (Expected: {N_VAL:,})")
    print(f"  Malformed sequences encountered:  {malformed_count}")

    assert train_offset == N_TRAIN, f"Train count mismatch: {train_offset} != {N_TRAIN}"
    assert val_offset == N_VAL, f"Val count mismatch: {val_offset} != {N_VAL}"
    assert malformed_count == 0, f"Malformed rows detected: {malformed_count}"

    # Save labels
    np.save(CACHE_DIR / "dev_train_labels.npy", train_labels)
    np.save(CACHE_DIR / "dev_val_labels.npy", val_labels)
    print(f"  Saved train labels: {CACHE_DIR / 'dev_train_labels.npy'} (Positives: {int(train_labels.sum()):,})")
    print(f"  Saved val labels:   {CACHE_DIR / 'dev_val_labels.npy'} (Positives: {int(val_labels.sum()):,})")

    # -------------------------------------------------------------------------
    # Step 2: Compute Normalization Statistics Strictly from dev_train
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 2: COMPUTING NORMALIZATION STATISTICS (TRAIN ONLY) ---")
    t0_norm = time.time()
    train_blocks_ro = np.memmap(train_blocks_path, dtype="float32", mode="r", shape=(N_TRAIN, SEQ_LEN))

    # Fast sampling-free exact global mean and std across all 651,337 x 2000 = 1.3 billion values
    # Sample in chunks to avoid single massive float64 allocation
    total_elements = N_TRAIN * SEQ_LEN
    sum_vals = 0.0
    sum_sq_vals = 0.0
    calc_chunk = 50000

    for i in range(0, N_TRAIN, calc_chunk):
        c_data = train_blocks_ro[i : i + calc_chunk].astype(np.float64)
        sum_vals += np.sum(c_data)
        sum_sq_vals += np.sum(c_data ** 2)

    training_mean = float(sum_vals / total_elements)
    training_var = float(sum_sq_vals / total_elements - (training_mean ** 2))
    training_std = float(np.sqrt(max(training_var, 1e-8)))

    del train_blocks_ro
    print(f"Computed global block normalization statistics in {time.time() - t0_norm:.2f}s:")
    print(f"  training_mean: {training_mean:.6f}")
    print(f"  training_std:  {training_std:.6f}")

    # -------------------------------------------------------------------------
    # Step 3: Tabular Features Extraction & Normalization
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 3: TABULAR FEATURES EXTRACTION (519 FEATURES) ---")
    t0_tab = time.time()

    # Load Model A feature columns from parquet
    print(f"Loading {len(MODEL_A_FEATURES)} features from {DEV_TRAIN_PARQUET.name}...")
    df_train_tab = pd.read_parquet(DEV_TRAIN_PARQUET, columns=["old_label"] + MODEL_A_FEATURES)
    df_train_tab = df_train_tab[df_train_tab["old_label"] == 0].reset_index(drop=True)

    print(f"Loading {len(MODEL_A_FEATURES)} features from {DEV_VAL_PARQUET.name}...")
    df_val_tab = pd.read_parquet(DEV_VAL_PARQUET, columns=["old_label"] + MODEL_A_FEATURES)
    df_val_tab = df_val_tab[df_val_tab["old_label"] == 0].reset_index(drop=True)

    X_train_tab = df_train_tab[MODEL_A_FEATURES].values.astype(np.float32)
    X_val_tab = df_val_tab[MODEL_A_FEATURES].values.astype(np.float32)

    # Compute tabular mean and std strictly on dev_train
    tab_mean = np.mean(X_train_tab, axis=0)
    tab_std = np.std(X_train_tab, axis=0)
    tab_std[tab_std < 1e-6] = 1.0  # prevent div-by-zero on constant columns

    # Normalize tabular features
    X_train_tab_norm = (X_train_tab - tab_mean) / tab_std
    X_val_tab_norm = (X_val_tab - tab_mean) / tab_std

    np.save(CACHE_DIR / "dev_train_tabular_norm.npy", X_train_tab_norm)
    np.save(CACHE_DIR / "dev_val_tabular_norm.npy", X_val_tab_norm)
    print(f"  Saved normalized train tabular: {CACHE_DIR / 'dev_train_tabular_norm.npy'} {X_train_tab_norm.shape}")
    print(f"  Saved normalized val tabular:   {CACHE_DIR / 'dev_val_tabular_norm.npy'} {X_val_tab_norm.shape}")

    # -------------------------------------------------------------------------
    # Step 4: Save Normalization Metadata
    # -------------------------------------------------------------------------
    norm_meta = {
        "block_normalization": {
            "training_mean": training_mean,
            "training_std": training_std,
            "sequence_length": SEQ_LEN,
            "dtype": "float32",
        },
        "tabular_normalization": {
            "num_features": len(MODEL_A_FEATURES),
            "features": MODEL_A_FEATURES,
            "tab_mean": tab_mean.tolist(),
            "tab_std": tab_std.tolist(),
        },
        "dataset_statistics": {
            "num_train_eligible_dies": N_TRAIN,
            "num_val_eligible_dies": N_VAL,
            "train_positives": int(train_labels.sum()),
            "val_positives": int(val_labels.sum()),
            "pos_weight": float((N_TRAIN - train_labels.sum()) / train_labels.sum()),
        }
    }

    norm_json_path = MODELS_DIR / "model_c_normalization.json"
    with open(norm_json_path, "w") as f:
        json.dump(norm_meta, f, indent=2)
    print(f"\nSaved normalization metadata to: {norm_json_path}")

    # Also save validation metadata parquet for evaluation mapping
    val_meta_pq = pd.read_parquet(DEV_VAL_PARQUET, columns=["wafer_id", "die_row", "die_col", "old_label", "label"])
    val_meta_pq = val_meta_pq[val_meta_pq["old_label"] == 0].reset_index(drop=True)
    val_meta_pq.to_parquet(CACHE_DIR / "dev_val_meta.parquet", index=False)
    print(f"Saved validation metadata parquet: {CACHE_DIR / 'dev_val_meta.parquet'} ({len(val_meta_pq):,} dies)")

    print(f"\n{'=' * 85}")
    print("MODEL C CACHE PREPARATION COMPLETE AND VERIFIED!")
    print(f"{'=' * 85}\n")


if __name__ == "__main__":
    prepare_cnn_cache()
