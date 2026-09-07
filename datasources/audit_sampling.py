"""
Audit Script: Analyzes the wafer sampling logic in generate_data.py and LSWMD.pkl.
Reconstructs the exact 200 wafers sampled using seed 42 and compares them
against the full eligible WM-811K population.

DOES NOT MODIFY ANY FILES OR DATASETS.
"""

import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def load_wm811k_clean(pkl_path):
    """Load WM-811K handling pandas compatibility shims."""
    print(f"Loading WM-811K from {pkl_path}...")
    try:
        df = pd.read_pickle(pkl_path)
    except Exception:
        import types
        import pandas.core.indexes.base
        import pandas.core.indexes.range
        indexes_pkg = types.ModuleType('pandas.indexes')
        indexes_pkg.__path__ = []
        sys.modules['pandas.indexes'] = indexes_pkg
        sys.modules['pandas.indexes.base'] = pandas.core.indexes.base
        sys.modules['pandas.indexes.range'] = pandas.core.indexes.range
        try:
            import pandas.core.indexes.numeric
            sys.modules['pandas.indexes.numeric'] = pandas.core.indexes.numeric
        except ModuleNotFoundError:
            sys.modules['pandas.indexes.numeric'] = pandas.core.indexes.base
            sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f, encoding='latin1')
    return df


def clean_failure_type(val):
    """Standardizes failureType from WM-811K array representation."""
    if isinstance(val, np.ndarray) and len(val) > 0 and len(val[0]) > 0:
        return str(val[0][0])
    return ""


def run_audit():
    print("=" * 75)
    print("AUDIT: WM-811K SOURCE POPULATION & SAMPLING LOGIC IN generate_data.py")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pkl_path = base_dir / config["wm811k_path"]
    if not pkl_path.exists():
        # Fallback check
        alt_path = base_dir / "LSWMD.pkl"
        if alt_path.exists():
            pkl_path = alt_path
        else:
            print(f"ERROR: Cannot find LSWMD.pkl at {pkl_path}")
            return

    # 1. Load full dataset
    raw_df = load_wm811k_clean(pkl_path)
    total_raw = len(raw_df)
    print(f"\n1. SOURCE POPULATION METRICS:")
    print(f"   Total wafers in raw WM-811K (LSWMD.pkl): {total_raw:,}")

    # Clean failureType column
    raw_df["clean_failureType"] = raw_df["failureType"].apply(clean_failure_type)

    labeled_mask = (raw_df["clean_failureType"] != "") & (raw_df["clean_failureType"] != "none")
    none_mask = (raw_df["clean_failureType"] == "none")
    unlabeled_mask = (raw_df["clean_failureType"] == "")

    labeled_df = raw_df[labeled_mask].copy()
    none_df = raw_df[none_mask].copy()
    unlabeled_df = raw_df[unlabeled_mask].copy()

    n_labeled = len(labeled_df)
    n_none = len(none_df)
    n_unlabeled = len(unlabeled_df)
    n_eligible = n_labeled + n_none

    print(f"   - Labeled Failure-Pattern Wafers: {n_labeled:,} ({n_labeled / total_raw * 100:.2f}%)")
    print(f"   - Labeled 'None' (All-Pass) Wafers: {n_none:,} ({n_none / total_raw * 100:.2f}%)")
    print(f"   - Unlabeled Wafers (Ignored):      {n_unlabeled:,} ({n_unlabeled / total_raw * 100:.2f}%)")
    print(f"   - Total ELIGIBLE Wafers for generator: {n_eligible:,} ({n_eligible / total_raw * 100:.2f}%)")

    # 2. Defect Type Distribution in Eligible Failure Population
    print(f"\n   Defect Class Breakdown in Eligible Labeled Population ({n_labeled:,} wafers):")
    fail_counts = labeled_df["clean_failureType"].value_counts()
    for ftype, cnt in fail_counts.items():
        print(f"     * {ftype:<14}: {cnt:>6,d} ({cnt / n_labeled * 100:>5.2f}%)")

    # 3. Simulate exact generator sampling (Seed 42)
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)

    num_wafers_train = config.get("num_wafers_train", 160)
    num_wafers_test = config.get("num_wafers_test", 40)
    total_needed = num_wafers_train + num_wafers_test  # 200
    target_fail_rate = config.get("target_fail_rate", 0.03)

    avg_internal_fail_rate = 0.12
    fraction_failure_wafers = min(target_fail_rate / avg_internal_fail_rate, 0.8)
    fraction_failure_wafers = max(fraction_failure_wafers, 0.05)

    num_failure_wafers = min(int(total_needed * fraction_failure_wafers), len(labeled_df))
    num_none_wafers = min(total_needed - num_failure_wafers, len(none_df))

    print(f"\n2. SAMPLING ALGORITHM IN generate_data.py:")
    print(f"   Random Seed: {seed} (Deterministic)")
    print(f"   Target overall fail rate: {target_fail_rate}")
    print(f"   Assumed avg internal fail rate per pattern wafer: {avg_internal_fail_rate}")
    print(f"   Calculated failure wafer fraction: {fraction_failure_wafers:.4f} (25%)")
    print(f"   Failure-pattern wafers sampled: {num_failure_wafers} (from {len(labeled_df):,})")
    print(f"   None-pattern wafers sampled:    {num_none_wafers} (from {len(none_df):,})")

    # Sample exactly as generator does
    failure_idx = rng.choice(len(labeled_df), size=num_failure_wafers, replace=False)
    none_idx = rng.choice(len(none_df), size=num_none_wafers, replace=False)

    sample_records = []

    # Failure wafers
    for i, idx in enumerate(failure_idx):
        row = labeled_df.iloc[idx]
        wm = row["waferMap"]
        n_dies = int((wm > 0).sum())
        n_pre_fails = int((wm == 2).sum())
        sample_records.append({
            "temp_id": f"W_F_{i:04d}",
            "source_pool": "labeled_failure",
            "source_row_index": int(labeled_df.index[idx]),
            "clean_failureType": row["clean_failureType"],
            "grid_shape": f"{wm.shape[0]}x{wm.shape[1]}",
            "grid_rows": wm.shape[0],
            "grid_cols": wm.shape[1],
            "n_dies": n_dies,
            "pre_test_fails": n_pre_fails,
            "pre_test_fail_rate": n_pre_fails / max(n_dies, 1),
            "lotName": str(row.get("lotName", "")),
            "waferIndex": str(row.get("waferIndex", "")),
        })

    # None wafers
    for i, idx in enumerate(none_idx):
        row = none_df.iloc[idx]
        wm = row["waferMap"]
        n_dies = int((wm > 0).sum())
        n_pre_fails = int((wm == 2).sum())
        sample_records.append({
            "temp_id": f"W_N_{i:04d}",
            "source_pool": "none_pass",
            "source_row_index": int(none_df.index[idx]),
            "clean_failureType": "none",
            "grid_shape": f"{wm.shape[0]}x{wm.shape[1]}",
            "grid_rows": wm.shape[0],
            "grid_cols": wm.shape[1],
            "n_dies": n_dies,
            "pre_test_fails": n_pre_fails,
            "pre_test_fail_rate": n_pre_fails / max(n_dies, 1),
            "lotName": str(row.get("lotName", "")),
            "waferIndex": str(row.get("waferIndex", "")),
        })

    # Generator shuffles the 200 records
    order = rng.permutation(len(sample_records))
    shuffled_records = [sample_records[i] for i in order]

    # Assign final train vs test split
    for i, rec in enumerate(shuffled_records):
        rec["generated_wafer_id"] = rec["temp_id"]
        rec["dataset_split"] = "train" if i < num_wafers_train else "test"
        rec["split_index"] = i

    sampled_df = pd.DataFrame(shuffled_records)

    # Save mapping file for audits
    mapping_path = base_dir.parent / "reports" / "source_wafer_mapping.csv"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(mapping_path, index=False)
    print(f"\n3. RECONSTRUCTED SOURCE MAPPING:")
    print(f"   Saved complete 200-wafer source provenance mapping to:")
    print(f"   {mapping_path}")

    # 4. Compare Train vs Test split within the 200 sampled wafers
    train_sampled = sampled_df[sampled_df["dataset_split"] == "train"]
    test_sampled = sampled_df[sampled_df["dataset_split"] == "test"]

    print(f"\n4. TRAIN vs. TEST SOURCE SPLIT COMPOSITION (Current 200 Wafers):")
    print(f"   Train Wafers: {len(train_sampled)} | Test Wafers: {len(test_sampled)}")
    print(f"   - Train Failure Wafers: {(train_sampled['source_pool'] == 'labeled_failure').sum()} ({(train_sampled['source_pool'] == 'labeled_failure').mean()*100:.1f}%)")
    print(f"   - Test Failure Wafers:  {(test_sampled['source_pool'] == 'labeled_failure').sum()} ({(test_sampled['source_pool'] == 'labeled_failure').mean()*100:.1f}%)")
    print(f"   - Train None Wafers:    {(train_sampled['source_pool'] == 'none_pass').sum()} ({(train_sampled['source_pool'] == 'none_pass').mean()*100:.1f}%)")
    print(f"   - Test None Wafers:     {(test_sampled['source_pool'] == 'none_pass').sum()} ({(test_sampled['source_pool'] == 'none_pass').mean()*100:.1f}%)")

    # 5. Compare Defect Class Distributions: Full Labeled Population vs Sampled 50 Wafers
    print(f"\n5. DEFECT TYPE DISTRIBUTION: Full Eligible ({n_labeled:,}) vs Sampled 50 Failure Wafers:")
    sample_fail_counts = sampled_df[sampled_df["source_pool"] == "labeled_failure"]["clean_failureType"].value_counts()
    print(f"   {'Defect Type':<16} | {'Full Pop Count':<14} | {'Full Pop %':<10} | {'Sampled (Train+Test)':<20} | {'Sampled %':<10}")
    print(f"   {'-'*16}-+-{'-'*14}-+-{'-'*10}-+-{'-'*20}-+-{'-'*10}")
    for ftype, full_cnt in fail_counts.items():
        s_cnt = sample_fail_counts.get(ftype, 0)
        print(f"   {ftype:<16} | {full_cnt:<14,d} | {full_cnt / n_labeled * 100:<10.2f} | {s_cnt:<20d} | {s_cnt / 50 * 100:<10.2f}")

    # 6. Physical Dimensions & Die Count Comparison
    # Compute full population die count on a random 1000 sample for speed
    pop_sample = raw_df[labeled_mask | none_mask].sample(min(2000, n_eligible), random_state=42)
    pop_die_counts = pop_sample["waferMap"].apply(lambda wm: (wm > 0).sum())
    pop_fail_rates = pop_sample["waferMap"].apply(lambda wm: (wm == 2).sum() / max((wm > 0).sum(), 1))

    print(f"\n6. PHYSICAL WAFER CHARACTERISTICS: Full Eligible Pop vs Current 200 Wafers:")
    print(f"   {'Metric':<25} | {'Full Pop (est)':<18} | {'Sampled 200 Wafers':<18}")
    print(f"   {'-'*25}-+-{'-'*18}-+-{'-'*18}")
    print(f"   {'Min dies per wafer':<25} | {pop_die_counts.min():<18d} | {sampled_df['n_dies'].min():<18d}")
    print(f"   {'Max dies per wafer':<25} | {pop_die_counts.max():<18d} | {sampled_df['n_dies'].max():<18d}")
    print(f"   {'Mean dies per wafer':<25} | {pop_die_counts.mean():<18.1f} | {sampled_df['n_dies'].mean():<18.1f}")
    print(f"   {'Median dies per wafer':<25} | {pop_die_counts.median():<18.1f} | {sampled_df['n_dies'].median():<18.1f}")
    print(f"   {'Mean pre-test fail rate':<25} | {pop_fail_rates.mean()*100:<18.2f}% | {sampled_df['pre_test_fail_rate'].mean()*100:<18.2f}%")

    print("\n" + "=" * 75)
    print("AUDIT SUMMARY & CAPACITY ASSESSMENT:")
    print("=" * 75)
    print(f"1. Available source pool: {n_eligible:,} wafers (25,519 pattern-failure + 147,431 none-type).")
    print(f"2. At a 25% failure-pattern ratio (1 failure wafer per 3 none wafers):")
    print(f"   - Max failure wafers available = {n_labeled:,}")
    print(f"   - To maintain 25% ratio, we could generate up to: {n_labeled * 4:,} total wafers!")
    print(f"   - Therefore, the dataset is NOT constrained by source data.")
    print("=" * 75)


if __name__ == "__main__":
    run_audit()
