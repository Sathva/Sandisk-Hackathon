"""
Feature Engineering for Final Unseen Test Set: Model F (Wafer-Conditional Manifold Detector).

Strict Test Isolation Protocol:
1. Uses ONLY unlabeled test features (final_test_model_e_features.parquet) and raw block traces.
2. Applies frozen LDA directions, scalers, and detrending metadata fitted strictly on dev_train.
3. Completely label-free: test.csv is NOT accessed during feature engineering.
"""

import sys
import os
import time
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    INPUT_DIR,
)
from src.features.model_f_features import (
    PARAM_COLS,
    WAFER_RELATIVE_BASE,
    detrend_parametric_within_wafer,
    apply_lda_features,
    compute_wafer_relative,
    compute_wafer_context_and_interactions,
)

INPUT_PARQUET = PROCESSED_DIR / "final_test_model_e_features.parquet"
OUTPUT_PARQUET = PROCESSED_DIR / "final_test_model_f_features.parquet"
VALIDATION_PARQUET = PROCESSED_DIR / "validation_features.parquet"
RAW_BLOCKS_DAT = PROCESSED_DIR / "cache" / "final_test_raw_blocks.dat"

LDA_MODEL_PATH = MODELS_DIR / "model_f_lda_direction.pkl"
DETREND_META_PATH = MODELS_DIR / "model_f_detrend_meta.pkl"


def compute_test_filter_bank_gpu(blocks_mmap, elig_mask, n_expected=185126, chunk_size=50000):
    print(f"  [C] GPU-Accelerated Filter Bank on {n_expected:,} eligible test dies...", flush=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows = [200, 300, 350, 400, 500, 600, 800]
    names = (
        [f"max_rolling_mean_{w}" for w in windows]
        + [f"max_rolling_std_{w}" for w in (300, 400, 600)]
        + [f"burst_excess_{w}" for w in (350, 400, 600)]
        + ["burst_peak_ratio", "burst_center_idx_600", "dwt_cD1_energy"]
    )
    out = np.zeros((n_expected, len(names)), dtype=np.float32)

    # Pre-slice eligible indices
    elig_indices = np.where(elig_mask)[0]
    assert len(elig_indices) == n_expected

    for s in range(0, n_expected, chunk_size):
        e = min(s + chunk_size, n_expected)
        idx_chunk = elig_indices[s:e]
        x_cpu = np.asarray(blocks_mmap[idx_chunk], dtype=np.float32)
        n_chunk, K = x_cpu.shape

        x_t = torch.from_numpy(x_cpu).to(device)
        cs = torch.zeros((n_chunk, K + 1), dtype=torch.float32, device=device)
        torch.cumsum(x_t, dim=1, out=cs[:, 1:])
        cs2 = torch.zeros((n_chunk, K + 1), dtype=torch.float32, device=device)
        torch.cumsum(x_t ** 2, dim=1, out=cs2[:, 1:])
        die_mean = x_t.mean(dim=1)

        col = 0
        roll_max = {}
        for w in windows:
            rm = (cs[:, w:] - cs[:, :-w]) / float(w)
            rm_max = rm.max(dim=1).values
            roll_max[w] = rm_max
            out[s:e, col] = rm_max.cpu().numpy()
            col += 1

        for w in (300, 400, 600):
            rm = (cs[:, w:] - cs[:, :-w]) / float(w)
            rv = torch.clamp((cs2[:, w:] - cs2[:, :-w]) / float(w) - rm ** 2, min=0.0)
            r_std = torch.sqrt(rv).max(dim=1).values
            out[s:e, col] = r_std.cpu().numpy()
            col += 1

        for w in (350, 400, 600):
            bex = torch.clamp(roll_max[w] - die_mean, min=0.0)
            out[s:e, col] = bex.cpu().numpy()
            col += 1

        denom = roll_max[600] - die_mean
        numer = roll_max[200] - die_mean
        ratio = torch.where(denom > 1e-6, numer / denom, torch.zeros_like(numer))
        out[s:e, col] = ratio.cpu().numpy()
        col += 1

        w600 = 600
        rm600 = (cs[:, w600:] - cs[:, :-w600]) / float(w600)
        idx600 = rm600.argmax(dim=1).to(torch.float32)
        out[s:e, col] = idx600.cpu().numpy()
        col += 1

        cd1 = (x_t[:, 1::2] - x_t[:, ::2]) / 1.41421356
        dwt_en = (cd1 ** 2).mean(dim=1)
        out[s:e, col] = dwt_en.cpu().numpy()
        col += 1

        del x_t, cs, cs2
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"      -> {len(names)} filter-bank columns computed on {device.type.upper()} in {time.time()-t0:.1f}s", flush=True)
    return pd.DataFrame(out, columns=names, index=pd.RangeIndex(n_expected))


def main():
    print("=" * 85)
    print("BUILDING FINAL UNSEEN TEST FEATURES FOR MODEL F (ZERO-LEAKAGE)")
    print("=" * 85)
    t_start = time.time()

    if OUTPUT_PARQUET.exists():
        print(f"Loaded existing Model F test features: {OUTPUT_PARQUET.name}")
        return OUTPUT_PARQUET

    # 1. Load unlabeled test input
    print("\n[Step 1] Loading unlabeled test features...")
    df = pd.read_parquet(INPUT_PARQUET).reset_index(drop=True)
    assert "label" not in df.columns, "Security check: test features must not contain labels!"
    print(f"  Base frame: {df.shape} ({len(df):,} dies across {df['wafer_id'].nunique()} test wafers)")

    # 2. Load frozen artifacts
    print("\n[Step 2] Loading frozen models and transforms...")
    with open(LDA_MODEL_PATH, "rb") as f:
        lda_model = pickle.load(f)
    with open(DETREND_META_PATH, "rb") as f:
        detrend_meta = pickle.load(f)
    global_stats = detrend_meta["global_stats"]

    # Alias pca_01 -> pca_comp_1 if needed
    for i in range(1, 4):
        if f"pca_0{i}" in df.columns and f"pca_comp_{i}" not in df.columns:
            df[f"pca_comp_{i}"] = df[f"pca_0{i}"]

    # 3. Wafer-conditional detrending
    print("\n[Step 3] Wafer-conditional detrending on test wafers...")
    resid_df, detrend_ctx = detrend_parametric_within_wafer(df)

    # 4. Apply frozen LDA & PCA
    print("\n[Step 4] Applying frozen LDA & detrended PCA...")
    lda_df = apply_lda_features(df, resid_df, lda_model)

    # 5. Compute test filter bank on GPU
    print("\n[Step 5] Computing filter bank on CUDA...")
    df_val = pd.read_parquet(VALIDATION_PARQUET, columns=["old_label"])
    elig_mask = (df_val["old_label"] == 0).values
    blocks_mmap = np.memmap(RAW_BLOCKS_DAT, dtype="float32", mode="r", shape=(len(df_val), 2000))
    fb = compute_test_filter_bank_gpu(blocks_mmap, elig_mask, n_expected=len(df))
    fb.index = df.index

    overlap = [c for c in fb.columns if c in df.columns]
    if overlap:
        df = df.drop(columns=overlap)

    stage = pd.concat([df, detrend_ctx, lda_df, fb], axis=1)
    stage = stage.loc[:, ~stage.columns.duplicated(keep="last")]

    # 6. Broadened within-wafer triplet using frozen global_stats
    print("\n[Step 6] Applying within-wafer triplet with frozen global stats...")
    rel_df, _ = compute_wafer_relative(stage, WAFER_RELATIVE_BASE, global_stats=global_stats)
    stage = pd.concat([stage, rel_df], axis=1)
    stage = stage.loc[:, ~stage.columns.duplicated(keep="last")]

    # 7. Wafer context & relative interactions
    print("\n[Step 7] Wafer context & relative interactions...")
    ctx_df = compute_wafer_context_and_interactions(stage)
    final = pd.concat([stage, ctx_df, resid_df], axis=1)
    final = final.loc[:, ~final.columns.duplicated(keep="last")]

    print(f"\nFinal test frame: {final.shape} (+{final.shape[1] - df.shape[1]} features)")
    final.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved {OUTPUT_PARQUET.name} ({OUTPUT_PARQUET.stat().st_size / (1024**2):.1f} MB) in {time.time() - t_start:.1f}s")
    return OUTPUT_PARQUET


if __name__ == "__main__":
    main()
