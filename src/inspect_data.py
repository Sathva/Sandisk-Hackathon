"""
Data Inspection Script for SanDisk Die Yield Prediction Hackathon.
Inspects train.csv and test.csv in a memory-conscious chunked fashion.
Validates schemas, missingness, duplicates, coordinates, distributions, and block readings.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import TRAIN_CSV, TEST_CSV, NUM_FEATURES


def inspect_file(file_path, is_train=True, chunksize=25000):
    print(f"\n{'=' * 75}")
    print(f"INSPECTING: {file_path.name} ({file_path})")
    print(f"{'=' * 75}")

    if not file_path.exists():
        print(f"ERROR: File not found at {file_path}")
        return

    start_time = time.time()
    total_rows = 0
    wafer_counts = {}
    old_label_counts = {}
    label_counts = {}
    eligible_pass_count = 0
    eligible_fail_count = 0

    missing_counts = {}
    col_dtypes = {}
    seen_coords = set()
    duplicate_coord_count = 0

    malformed_block_count = 0
    non_2000_block_count = 0

    coord_stats = {"min_row": 999999, "max_row": -1, "min_col": 999999, "max_col": -1}

    # Online summary for feature_1..feature_500
    feat_cols = [f"feature_{i}" for i in range(1, NUM_FEATURES + 1)]
    feat_sum = np.zeros(NUM_FEATURES, dtype=np.float64)
    feat_sq_sum = np.zeros(NUM_FEATURES, dtype=np.float64)
    feat_min = np.full(NUM_FEATURES, np.inf, dtype=np.float64)
    feat_max = np.full(NUM_FEATURES, -np.inf, dtype=np.float64)
    feat_missing = np.zeros(NUM_FEATURES, dtype=np.int64)

    print(f"Reading file in streaming chunks of {chunksize} rows...")
    chunk_idx = 0

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        chunk_idx += 1
        n_chunk = len(chunk)
        total_rows += n_chunk

        if chunk_idx == 1:
            col_dtypes = chunk.dtypes.to_dict()
            total_cols = len(chunk.columns)

        # 1. Missing values
        chunk_missing = chunk.isnull().sum()
        for col, m_cnt in chunk_missing.items():
            missing_counts[col] = missing_counts.get(col, 0) + m_cnt

        # 2. Wafers
        for wid, cnt in chunk["wafer_id"].value_counts().items():
            wafer_counts[wid] = wafer_counts.get(wid, 0) + cnt

        # 3. Coordinate sanity & duplicate check
        rows_arr = chunk["die_row"].values
        cols_arr = chunk["die_col"].values
        wids_arr = chunk["wafer_id"].values

        coord_stats["min_row"] = min(coord_stats["min_row"], int(rows_arr.min()))
        coord_stats["max_row"] = max(coord_stats["max_row"], int(rows_arr.max()))
        coord_stats["min_col"] = min(coord_stats["min_col"], int(cols_arr.min()))
        coord_stats["max_col"] = max(coord_stats["max_col"], int(cols_arr.max()))

        for wid, r, c in zip(wids_arr, rows_arr, cols_arr):
            key = (wid, int(r), int(c))
            if key in seen_coords:
                duplicate_coord_count += 1
            else:
                seen_coords.add(key)

        # 4. Labels
        if "old_label" in chunk.columns:
            for ol, cnt in chunk["old_label"].value_counts().items():
                old_label_counts[ol] = old_label_counts.get(ol, 0) + cnt

        if "label" in chunk.columns:
            for l, cnt in chunk["label"].value_counts().items():
                label_counts[l] = label_counts.get(l, 0) + cnt

            # Eligible dies: old_label == 0
            if "old_label" in chunk.columns:
                mask_el_pass = (chunk["old_label"] == 0) & (chunk["label"] == 0)
                mask_el_fail = (chunk["old_label"] == 0) & (chunk["label"] == 1)
                eligible_pass_count += int(mask_el_pass.sum())
                eligible_fail_count += int(mask_el_fail.sum())

        # 5. Feature 1..500 summary statistics
        present_feats = [c for c in feat_cols if c in chunk.columns]
        if len(present_feats) == NUM_FEATURES:
            feat_data = chunk[present_feats].values.astype(np.float64)
            # Nan mask
            nan_mask = np.isnan(feat_data)
            feat_missing += nan_mask.sum(axis=0)

            # Replace nan with 0 temporarily for sum calculation
            clean_feat = np.where(nan_mask, 0.0, feat_data)
            feat_sum += clean_feat.sum(axis=0)
            feat_sq_sum += (clean_feat ** 2).sum(axis=0)
            
            clean_min = np.where(nan_mask, np.inf, feat_data)
            clean_max = np.where(nan_mask, -np.inf, feat_data)
            feat_min = np.minimum(feat_min, clean_min.min(axis=0))
            feat_max = np.maximum(feat_max, clean_max.max(axis=0))

        # 6. Validate block readings
        if "block_readings" in chunk.columns:
            for s in chunk["block_readings"].values:
                if not isinstance(s, str) or not s.strip():
                    malformed_block_count += 1
                else:
                    vals = s.split()
                    if len(vals) != 2000:
                        non_2000_block_count += 1

        print(f"  Processed {total_rows:,} rows...", end="\r", flush=True)

    elapsed = time.time() - start_time
    print(f"\nCompleted inspection in {elapsed:.1f}s.")

    # -------------------------------------------------------------
    # Summary Output
    # -------------------------------------------------------------
    print(f"\n1. SHAPE & ROWS:")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Total columns: {total_cols}")

    print(f"\n2. WAFER METRICS:")
    n_wafers = len(wafer_counts)
    counts_arr = np.array(list(wafer_counts.values()))
    print(f"   Unique wafers: {n_wafers}")
    print(f"   Dies per wafer: min={counts_arr.min()}, max={counts_arr.max()}, "
          f"mean={counts_arr.mean():.1f}, median={np.median(counts_arr):.1f}")

    print(f"\n3. PRE-TEST STATUS (old_label):")
    for k, v in sorted(old_label_counts.items()):
        status = "Healthy / Eligible" if k == 0 else "Pre-Test Failed"
        print(f"   old_label={k} ({status}): {v:,} ({v / total_rows * 100:.2f}%)")

    if label_counts:
        print(f"\n4. POST-TEST STATUS (label):")
        for k, v in sorted(label_counts.items()):
            status = "Pass" if k == 0 else "Total Fail (Old + New)"
            print(f"   label={k} ({status}): {v:,} ({v / total_rows * 100:.2f}%)")

    if is_train and "old_label" in col_dtypes and "label" in col_dtypes:
        el_total = eligible_pass_count + eligible_fail_count
        pos_rate = eligible_fail_count / el_total * 100 if el_total > 0 else 0
        print(f"\n5. ELIGIBLE TARGET POPULATION (Train only, old_label == 0):")
        print(f"   Eligible healthy dies: {el_total:,} ({el_total / total_rows * 100:.2f}% of all dies)")
        print(f"   - Stayed Pass (old_label=0, label=0): {eligible_pass_count:,} ({eligible_pass_count / el_total * 100:.2f}%)")
        print(f"   - Newly Failed (old_label=0, label=1): {eligible_fail_count:,} ({pos_rate:.2f}% POSITIVE TARGET)")
        print(f"   - Target Positive Rate (Minority Class): {pos_rate:.3f}%")

    print(f"\n6. INTEGRITY & ANOMALY CHECKS:")
    total_missing = sum(missing_counts.values())
    print(f"   Total missing values across all columns: {total_missing}")
    print(f"   Duplicate (wafer_id, die_row, die_col) combinations: {duplicate_coord_count}")
    print(f"   Coordinate bounds: die_row in [{coord_stats['min_row']}, {coord_stats['max_row']}], "
          f"die_col in [{coord_stats['min_col']}, {coord_stats['max_col']}]")
    print(f"   Malformed block_readings strings: {malformed_block_count}")
    print(f"   Block readings with length != 2000: {non_2000_block_count}")

    # Feature 1..500 summary
    print(f"\n7. DIE-LEVEL PARAMETRIC FEATURES (feature_1..feature_500):")
    all_numeric = all(np.issubdtype(col_dtypes[c], np.number) for c in feat_cols if c in col_dtypes)
    print(f"   All 500 features present and strictly numeric: {all_numeric}")
    print(f"   Total missing values in 500 features: {int(feat_missing.sum())}")

    means = feat_sum / np.maximum(total_rows - feat_missing, 1)
    stds = np.sqrt(np.maximum(0.0, (feat_sq_sum / np.maximum(total_rows - feat_missing, 1)) - (means ** 2)))

    print(f"\n   Sample Feature Distribution (first 5 features):")
    print(f"   {'Feature':<14} | {'Mean':<12} | {'Std':<12} | {'Min':<12} | {'Max':<12}")
    print(f"   {'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for i in range(5):
        print(f"   {feat_cols[i]:<14} | {means[i]:<12.3f} | {stds[i]:<12.3f} | {feat_min[i]:<12.3f} | {feat_max[i]:<12.3f}")

    print(f"\n   Feature Scale Across all 500 Features:")
    print(f"   Overall min mean: {means.min():.3f}, max mean: {means.max():.3f}")
    print(f"   Overall min std:  {stds.min():.3f}, max std:  {stds.max():.3f}")


def main():
    print(f"SanDisk Hackathon Data Inspection Pipeline")
    print(f"Working Directory: {Path.cwd()}")
    print(f"Train File: {TRAIN_CSV}")
    print(f"Test File:  {TEST_CSV}")

    inspect_file(TRAIN_CSV, is_train=True)
    inspect_file(TEST_CSV, is_train=False)


if __name__ == "__main__":
    main()
