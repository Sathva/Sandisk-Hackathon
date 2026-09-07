"""
Development Train / Development Validation Split Module.
Partitions processed/train_features.parquet strictly at the WAFER level (80% / 20%).

Rules:
1. Complete Wafer Isolation: set(train_wafers) & set(val_wafers) == empty.
2. Full Population Retained: old_label == 1 rows are retained in both splits.
3. Zero Feature Modification: Exact row subsets of train_features.parquet.
4. Stratified Wafer-Level Selection: Wafers are partitioned across positive rate strata
   to ensure representative class distribution without violating wafer boundaries.
5. Canonical Artifact: Generates reports/development_split.json.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    TRAIN_PARQUET,
    DEV_TRAIN_PARQUET,
    DEV_VAL_PARQUET,
    DEV_SPLIT_JSON,
    SEED,
)


def compute_split_stats(df_split, split_name):
    """Computes comprehensive population statistics for a split."""
    n_wafers = int(df_split["wafer_id"].nunique())
    total_dies = int(len(df_split))
    pre_test_failed = int((df_split["old_label"] == 1).sum())
    
    # Eligible population: old_label == 0
    eligible_mask = df_split["old_label"] == 0
    eligible_dies = int(eligible_mask.sum())
    
    stayed_healthy = int(((df_split["old_label"] == 0) & (df_split["label"] == 0)).sum())
    new_failures = int(((df_split["old_label"] == 0) & (df_split["label"] == 1)).sum())
    
    pos_rate = (new_failures / eligible_dies * 100.0) if eligible_dies > 0 else 0.0

    return {
        "split_name": split_name,
        "wafers": n_wafers,
        "total_dies": total_dies,
        "eligible_dies": eligible_dies,
        "stayed_healthy": stayed_healthy,
        "new_failures": new_failures,
        "eligible_positive_rate": float(pos_rate),
        "pre_test_failed": pre_test_failed,
    }


def perform_wafer_split(train_parquet_path=TRAIN_PARQUET, seed=SEED):
    if not train_parquet_path.exists():
        raise FileNotFoundError(f"Source file not found: {train_parquet_path}")

    t0 = time.time()
    print(f"Loading {train_parquet_path.name}...")
    df = pd.read_parquet(train_parquet_path)
    print(f"Loaded {len(df):,} dies across {df['wafer_id'].nunique()} wafers in {time.time() - t0:.1f}s.")

    # -------------------------------------------------------------
    # Step 1: Compute Per-Wafer Metrics for Auditable Stratification
    # -------------------------------------------------------------
    wafer_summary = (
        df.groupby("wafer_id", sort=True)
        .apply(
            lambda g: pd.Series({
                "total_dies": len(g),
                "pre_test_failed": (g["old_label"] == 1).sum(),
                "eligible_dies": (g["old_label"] == 0).sum(),
                "new_failures": ((g["old_label"] == 0) & (g["label"] == 1)).sum(),
            })
        )
        .reset_index()
    )

    wafer_summary["eligible_fail_rate"] = (
        wafer_summary["new_failures"] / np.maximum(wafer_summary["eligible_dies"], 1)
    )

    # Bin wafers into strata based on eligible fail rate (4 quantiles)
    # This guarantees representative defect rates in both splits without mixing dies!
    num_bins = 4
    wafer_summary["strata"] = pd.qcut(
        wafer_summary["eligible_fail_rate"],
        q=num_bins,
        labels=False,
        duplicates="drop",
    )

    # -------------------------------------------------------------
    # Step 2: Stratified Wafer-Level Split (80% Train / 20% Dev-Val)
    # -------------------------------------------------------------
    total_wafers = len(wafer_summary)
    num_val_wafers = int(np.round(total_wafers * 0.20))  # 32
    num_train_wafers = total_wafers - num_val_wafers     # 128

    sss = StratifiedShuffleSplit(n_splits=1, test_size=num_val_wafers, random_state=seed)
    train_idx, val_idx = next(sss.split(wafer_summary, wafer_summary["strata"]))

    train_wafer_ids = sorted(wafer_summary.iloc[train_idx]["wafer_id"].tolist())
    val_wafer_ids = sorted(wafer_summary.iloc[val_idx]["wafer_id"].tolist())

    # -------------------------------------------------------------
    # Step 3: Strict Isolation & Integrity Checks
    # -------------------------------------------------------------
    intersection = set(train_wafer_ids) & set(val_wafer_ids)
    assert len(intersection) == 0, f"VIOLATION: Wafer overlap detected! {intersection}"
    assert len(train_wafer_ids) == num_train_wafers, f"Expected {num_train_wafers} train wafers, got {len(train_wafer_ids)}"
    assert len(val_wafer_ids) == num_val_wafers, f"Expected {num_val_wafers} val wafers, got {len(val_wafer_ids)}"
    assert len(train_wafer_ids) + len(val_wafer_ids) == total_wafers, "Wafer count mismatch!"

    # -------------------------------------------------------------
    # Step 4: Partition DataFrame by Wafer IDs
    # -------------------------------------------------------------
    dev_train_df = df[df["wafer_id"].isin(train_wafer_ids)].copy().reset_index(drop=True)
    dev_val_df = df[df["wafer_id"].isin(val_wafer_ids)].copy().reset_index(drop=True)

    # Check 1: Row counts add up exactly
    assert len(dev_train_df) + len(dev_val_df) == len(df), "Row count mismatch between partitions and source!"
    # Check 2: All columns match exactly
    assert list(dev_train_df.columns) == list(df.columns), "Train schema mismatch!"
    assert list(dev_val_df.columns) == list(df.columns), "Validation schema mismatch!"
    # Check 3: Unique wafers match
    assert dev_train_df["wafer_id"].nunique() == num_train_wafers
    assert dev_val_df["wafer_id"].nunique() == num_val_wafers

    # -------------------------------------------------------------
    # Step 5: Compute Statistics
    # -------------------------------------------------------------
    train_stats = compute_split_stats(dev_train_df, "DEV_TRAIN")
    val_stats = compute_split_stats(dev_val_df, "DEV_VALIDATION")

    # -------------------------------------------------------------
    # Step 6: Save Artifacts
    # -------------------------------------------------------------
    DEV_SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    split_meta = {
        "random_seed": seed,
        "split_ratio": "80_20",
        "num_train_wafers": num_train_wafers,
        "num_dev_val_wafers": num_val_wafers,
        "train_wafer_ids": train_wafer_ids,
        "dev_val_wafer_ids": val_wafer_ids,
        "train_statistics": train_stats,
        "dev_validation_statistics": val_stats,
        "total_source_dies": len(df),
        "total_source_wafers": total_wafers,
    }

    with open(DEV_SPLIT_JSON, "w") as f:
        json.dump(split_meta, f, indent=2)
    print(f"Saved split metadata to: {DEV_SPLIT_JSON}")

    # Materialize Parquet datasets
    print("Saving materialized split files...")
    dev_train_df.to_parquet(DEV_TRAIN_PARQUET, index=False, engine="pyarrow", compression="snappy")
    dev_val_df.to_parquet(DEV_VAL_PARQUET, index=False, engine="pyarrow", compression="snappy")
    print(f"  Saved: {DEV_TRAIN_PARQUET} ({DEV_TRAIN_PARQUET.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Saved: {DEV_VAL_PARQUET} ({DEV_VAL_PARQUET.stat().st_size / (1024*1024):.2f} MB)")

    # -------------------------------------------------------------
    # Step 7: Print Canonical Development Split Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEVELOPMENT SPLIT SUMMARY")
    print("=" * 60)
    print("\nTRAIN:")
    print(f"Wafers: {train_stats['wafers']}")
    print(f"Total dies: {train_stats['total_dies']:,}")
    print(f"Eligible dies: {train_stats['eligible_dies']:,}")
    print(f"Stayed healthy: {train_stats['stayed_healthy']:,}")
    print(f"New failures: {train_stats['new_failures']:,}")
    print(f"Eligible positive rate: {train_stats['eligible_positive_rate']:.3f}%")
    print(f"Pre-test failed: {train_stats['pre_test_failed']:,}")

    print("\nDEV VALIDATION:")
    print(f"Wafers: {val_stats['wafers']}")
    print(f"Total dies: {val_stats['total_dies']:,}")
    print(f"Eligible dies: {val_stats['eligible_dies']:,}")
    print(f"Stayed healthy: {val_stats['stayed_healthy']:,}")
    print(f"New failures: {val_stats['new_failures']:,}")
    print(f"Eligible positive rate: {val_stats['eligible_positive_rate']:.3f}%")
    print(f"Pre-test failed: {val_stats['pre_test_failed']:,}")

    print("\nWAFER OVERLAP:")
    print(f"Train/validation wafer intersection: {len(intersection)}")

    print("\nTOTAL:")
    print(f"Wafers: {train_stats['wafers'] + val_stats['wafers']}")
    print(f"Dies: {train_stats['total_dies'] + val_stats['total_dies']:,}")
    print("=" * 60 + "\n")


def main():
    perform_wafer_split()


if __name__ == "__main__":
    main()
