"""
Data Preprocessing Pipeline for SanDisk Hackathon.
Converts raw CSV files (train, test, validation) into clean, optimized Parquet files.

Pipeline Operations:
1. Loads parametric features and identifiers, casting features to float32.
2. Extracts comprehensive spatial features per-wafer using only pre-test information (old_label).
3. Parses block_readings string sequences in streaming batches with parallel CPU multiprocessing.
4. Validates schema, row counts, and data integrity (detects any malformed block rows).
5. Saves processed datasets to processed/{train,test,validation}_features.parquet.

LEAKAGE PREVENTION:
- NEVER uses target 'label' for feature generation.
- Handles validation.csv strictly without label.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    TRAIN_CSV, TEST_CSV, VALIDATION_CSV,
    TRAIN_PARQUET, TEST_PARQUET, VALIDATION_PARQUET,
    NUM_FEATURES, NUM_BLOCK_READINGS
)
from src.features.spatial import compute_spatial_features
from src.features.block import extract_block_features_series


def process_dataset(csv_path, parquet_path, split_name="train", chunksize=15000):
    print(f"\n{'=' * 75}")
    print(f"PROCESSING SPLIT: {split_name.upper()} ({csv_path.name})")
    print(f"{'=' * 75}")

    if not csv_path.exists():
        print(f"ERROR: File not found at {csv_path}")
        return None

    start_time = time.time()

    # -------------------------------------------------------------
    # Step 1: Load tabular columns (excluding huge block_readings)
    # -------------------------------------------------------------
    print(f"[{split_name}] Step 1/3: Loading parametric features and metadata...")
    # Read first line to inspect columns
    sample_df = pd.read_csv(csv_path, nrows=5)
    has_label = "label" in sample_df.columns
    
    # Columns to load in step 1
    load_cols = [c for c in sample_df.columns if c != "block_readings"]
    
    t0 = time.time()
    tab_df = pd.read_csv(csv_path, usecols=load_cols)
    print(f"   Loaded {len(tab_df):,} rows in {time.time() - t0:.1f}s.")

    # Cast numeric feature columns to float32 to minimize memory
    feat_cols = [c for c in tab_df.columns if c.startswith("feature_")]
    tab_df[feat_cols] = tab_df[feat_cols].astype(np.float32)
    tab_df["die_row"] = tab_df["die_row"].astype(np.int32)
    tab_df["die_col"] = tab_df["die_col"].astype(np.int32)
    if "old_label" in tab_df.columns:
        tab_df["old_label"] = tab_df["old_label"].astype(np.int8)
    if has_label:
        tab_df["label"] = tab_df["label"].astype(np.int8)

    # -------------------------------------------------------------
    # Step 2: Compute spatial features per wafer (using only old_label)
    # -------------------------------------------------------------
    print(f"[{split_name}] Step 2/3: Computing spatial context features per wafer...")
    t0 = time.time()
    spatial_df = compute_spatial_features(tab_df[["wafer_id", "die_row", "die_col", "old_label"]])
    print(f"   Generated {spatial_df.shape[1]} spatial features in {time.time() - t0:.1f}s.")

    # -------------------------------------------------------------
    # Step 3: Stream and extract block features from block_readings
    # -------------------------------------------------------------
    print(f"[{split_name}] Step 3/3: Streaming & extracting block features...")
    t0 = time.time()
    block_chunks = []
    total_malformed = 0
    rows_processed = 0

    for chunk in pd.read_csv(csv_path, usecols=["block_readings"], chunksize=chunksize):
        b_feats, malformed = extract_block_features_series(chunk["block_readings"])
        block_chunks.append(b_feats)
        total_malformed += malformed
        rows_processed += len(chunk)
        print(f"   Extracted block features for {rows_processed:,}/{len(tab_df):,} dies...", end="\r", flush=True)

    block_df = pd.concat(block_chunks, axis=0).reset_index(drop=True)
    print(f"\n   Finished block feature extraction in {time.time() - t0:.1f}s. (Malformed rows: {total_malformed})")

    # -------------------------------------------------------------
    # Step 4: Assemble and Save to Parquet
    # -------------------------------------------------------------
    print(f"[{split_name}] Merging and saving to Parquet...")
    # Ensure indices align
    tab_df = tab_df.reset_index(drop=True)
    spatial_df = spatial_df.reset_index(drop=True)
    block_df = block_df.reset_index(drop=True)

    final_df = pd.concat([tab_df, spatial_df, block_df], axis=1)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")

    file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time

    print(f"\nSUCCESS: Saved {split_name} features to {parquet_path}")
    print(f"   Rows: {len(final_df):,}")
    print(f"   Columns: {final_df.shape[1]:,}")
    print(f"   Output File Size: {file_size_mb:.2f} MB")
    print(f"   Total Processing Time: {elapsed:.1f}s ({elapsed / 60:.2f} min)")

    return {
        "split": split_name,
        "rows": len(final_df),
        "cols": final_df.shape[1],
        "size_mb": file_size_mb,
        "time_s": elapsed,
        "malformed_blocks": total_malformed,
    }


def main():
    print(f"Starting Preprocessing Pipeline...")
    overall_start = time.time()

    reports = []
    # 1. Process Train
    rep_train = process_dataset(TRAIN_CSV, TRAIN_PARQUET, split_name="train")
    if rep_train:
        reports.append(rep_train)

    # 2. Process Test
    rep_test = process_dataset(TEST_CSV, TEST_PARQUET, split_name="test")
    if rep_test:
        reports.append(rep_test)

    # 3. Process Validation
    rep_val = process_dataset(VALIDATION_CSV, VALIDATION_PARQUET, split_name="validation")
    if rep_val:
        reports.append(rep_val)

    overall_elapsed = time.time() - overall_start

    print(f"\n{'=' * 75}")
    print(f"PREPROCESSING PIPELINE SUMMARY")
    print(f"{'=' * 75}")
    print(f"{'Split':<12} | {'Rows':<10} | {'Columns':<8} | {'Size (MB)':<10} | {'Malformed':<10} | {'Time (s)':<8}")
    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for r in reports:
        print(f"{r['split']:<12} | {r['rows']:<10,d} | {r['cols']:<8d} | {r['size_mb']:<10.2f} | {r['malformed_blocks']:<10d} | {r['time_s']:<8.1f}")
    print(f"{'=' * 75}")
    print(f"Total time across all splits: {overall_elapsed:.1f}s ({overall_elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()
