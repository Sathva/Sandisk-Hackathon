"""
Feature Engineering Engine for Model F: Wafer-Conditional Manifold Detector.

Adapted from Architecture 7 (branch adit, commit 0acf5a4) for our canonical dataset
with GPU-accelerated filter bank computation on the NVIDIA RTX 4500 Ada Generation GPU.

Key Novel Mechanisms:
1. Wafer-Conditional Detrending:
   Removes the injected per-wafer gradient field [1, r, r^2, xn, yn, xn*yn]
   from all 500 parametric features at source within each wafer.
2. Shrinkage LDA Discriminant & Detrended PCA:
   Directly estimates the Bayes-optimal class separation vector on dev_train,
   orienting the discriminant direction and extracting 8 detrended PCA components.
3. GPU-Accelerated Extended Filter Bank:
   Computes rolling-mean maxima across windows W in [200, 300, 350, 400, 500, 600, 800],
   max rolling std, burst excess, burst peak ratio, and Haar wavelet energy
   directly from the cached memmap raw blocks on CUDA.
4. Broadened Within-Wafer Relative Triplet:
   Extends wrank_*, wzscore_*, and wdev_* across ~35 key discriminants.
5. Wafer Context & Relative Interactions:
   Computes wafer-level aggregations and products of normalized deviations
   (e.g., wdev_ldadt * wdev_roll400 * edge).
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    DEV_TRAIN_PARQUET,
    DEV_VAL_PARQUET,
    SEED,
)

# File Paths
TRAIN_INPUT = PROCESSED_DIR / "dev_train_model_e_features.parquet"
VAL_INPUT = PROCESSED_DIR / "dev_val_model_e_features.parquet"
RAW_BLOCKS_CACHE = {
    "dev_train": PROCESSED_DIR / "cache" / "dev_train_raw_blocks.dat",
    "dev_val": PROCESSED_DIR / "cache" / "dev_val_raw_blocks.dat",
}

TRAIN_OUTPUT = PROCESSED_DIR / "dev_train_model_f_features.parquet"
VAL_OUTPUT = PROCESSED_DIR / "dev_val_model_f_features.parquet"

LDA_MODEL_PATH = MODELS_DIR / "model_f_lda_direction.pkl"
DETREND_META_PATH = MODELS_DIR / "model_f_detrend_meta.pkl"

PARAM_COLS = [f"feature_{i}" for i in range(1, 501)]

FORBIDDEN = {
    "wafer_id", "label", "old_label", "die_row", "die_col",
    "adversarial_score", "te_wafer_fail_rate", "te_wafer_residual",
    "te_wafer_fail_rate_smooth", "inter_te_residual_x_pc1", "inter_adv_x_pc1",
}

WAFER_RELATIVE_BASE = [
    "lda_score", "lda_score_detrended",
    "dpca_comp_1", "dpca_comp_2", "dpca_comp_3", "dpca_comp_4",
    "pca_01", "pca_02", "pca_03",
    "parametric_drift_l2", "detrend_resid_l2",
    "max_rolling_mean_200", "max_rolling_mean_300", "max_rolling_mean_350",
    "max_rolling_mean_400", "max_rolling_mean_500", "max_rolling_mean_600",
    "max_rolling_mean_800",
    "burst_excess_350", "burst_excess_400", "burst_excess_600",
    "tophat_peak_200", "tophat_energy_100", "tophat_energy_200",
    "block_mean", "block_std", "block_q99", "block_max_z",
    "block_mean_top100", "block_mean_top200",
    "grad_max", "grad_energy", "curv_energy", "grad_burst_width",
    "dwt_cD1_energy",
]


# ---------------------------------------------------------------------------
# A. Wafer-Conditional Detrending
# ---------------------------------------------------------------------------
def _position_basis(die_row, die_col):
    """
    Design matrix spanning the generator's injected spatial gradient field:
    [1, r, r^2, xn, yn, xn * yn].
    """
    r = die_row.astype(np.float64)
    c = die_col.astype(np.float64)
    r_mid, c_mid = (r.max() + r.min()) / 2.0, (c.max() + c.min()) / 2.0
    r_half = max((r.max() - r.min()) / 2.0, 1.0)
    c_half = max((c.max() - c.min()) / 2.0, 1.0)
    yn = (r - r_mid) / r_half
    xn = (c - c_mid) / c_half
    rad = np.sqrt(yn ** 2 + xn ** 2)
    return np.column_stack([np.ones_like(xn), rad, rad ** 2, xn, yn, xn * yn])


def detrend_parametric_within_wafer(df, param_cols=PARAM_COLS, ridge=1e-6):
    """
    Per wafer, regress every parametric feature on the position basis and keep residual.
    Label-free and wafer-local -> strictly leak-free for unseen wafers.
    """
    print(f"  [A] Wafer-conditional detrending of {len(param_cols)} parametric features...", flush=True)
    t0 = time.time()

    F = df[param_cols].to_numpy(np.float64, copy=False)
    resid = np.empty_like(F, dtype=np.float32)

    n_wafers = df["wafer_id"].nunique()
    coef_norm_rad = np.zeros(len(df), dtype=np.float32)
    coef_norm_lin = np.zeros(len(df), dtype=np.float32)

    for wid, g in df.groupby("wafer_id", sort=False):
        idx = df.index.get_indexer(g.index)
        P = _position_basis(g["die_row"].to_numpy(), g["die_col"].to_numpy())
        Fw = F[idx]
        PtP = P.T @ P + ridge * np.eye(P.shape[1])
        B = np.linalg.solve(PtP, P.T @ Fw)
        resid[idx] = (Fw - P @ B).astype(np.float32)
        coef_norm_rad[idx] = np.float32(np.linalg.norm(B[1:3]))
        coef_norm_lin[idx] = np.float32(np.linalg.norm(B[3:6]))

    resid_df = pd.DataFrame(
        resid, columns=[f"dt_{c}" for c in param_cols], index=df.index
    )
    ctx = pd.DataFrame(
        {
            "wafer_grad_radial_norm": coef_norm_rad,
            "wafer_grad_linear_norm": coef_norm_lin,
            "detrend_resid_l2": np.linalg.norm(resid, axis=1).astype(np.float32),
        },
        index=df.index,
    )
    print(f"      -> {n_wafers} wafers detrended in {time.time()-t0:.1f}s", flush=True)
    return resid_df, ctx


# ---------------------------------------------------------------------------
# B. Shrinkage LDA Discriminant & Detrended PCA
# ---------------------------------------------------------------------------
def fit_shrinkage_lda(X, y, shrinkage=0.20):
    """
    Estimate Bayes-optimal discriminant direction w proportional to (S_reg)^-1 (mu1 - mu0).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    mu1 = X[y == 1].mean(axis=0)
    mu0 = X[y == 0].mean(axis=0)
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / max(len(X) - 1, 1)
    trace_mean = np.trace(S) / S.shape[0]
    S_reg = (1.0 - shrinkage) * S + shrinkage * trace_mean * np.eye(S.shape[0])
    w = np.linalg.solve(S_reg, mu1 - mu0)
    nrm = np.linalg.norm(w)
    if nrm > 0:
        w = w / nrm
    return w


def build_lda_features(df_train, resid_train, shrinkage=0.20, n_pca=8):
    """
    Fits on dev_train ONLY:
    - Shrinkage LDA on raw parametric features
    - Shrinkage LDA on detrended residuals
    - PCA (8 comps) on detrended space
    """
    print(f"  [B] Fitting shrinkage LDA + detrended PCA on dev_train only...", flush=True)
    t0 = time.time()
    y = df_train["label"].to_numpy().astype(int)

    sc_raw = StandardScaler().fit(df_train[PARAM_COLS].to_numpy(np.float64))
    Z_raw = sc_raw.transform(df_train[PARAM_COLS].to_numpy(np.float64))
    w_raw = fit_shrinkage_lda(Z_raw, y, shrinkage)

    dt_cols = list(resid_train.columns)
    sc_dt = StandardScaler().fit(resid_train.to_numpy(np.float64))
    Z_dt = sc_dt.transform(resid_train.to_numpy(np.float64))
    w_dt = fit_shrinkage_lda(Z_dt, y, shrinkage)

    pca_dt = PCA(n_components=n_pca, random_state=SEED).fit(Z_dt)

    s_raw = Z_raw @ w_raw
    s_dt = Z_dt @ w_dt
    sign_raw = 1.0 if np.corrcoef(s_raw, y)[0, 1] >= 0 else -1.0
    sign_dt = 1.0 if np.corrcoef(s_dt, y)[0, 1] >= 0 else -1.0

    model = {
        "sc_raw": sc_raw, "w_raw": w_raw, "sign_raw": sign_raw,
        "sc_dt": sc_dt, "w_dt": w_dt, "sign_dt": sign_dt,
        "pca_dt": pca_dt, "dt_cols": dt_cols, "n_pca": n_pca,
        "shrinkage": shrinkage,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LDA_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"      -> saved {LDA_MODEL_PATH.name} in {time.time()-t0:.1f}s", flush=True)
    return model


def apply_lda_features(df, resid, model):
    Z_raw = model["sc_raw"].transform(df[PARAM_COLS].to_numpy(np.float64))
    Z_dt = model["sc_dt"].transform(resid[model["dt_cols"]].to_numpy(np.float64))
    out = {
        "lda_score": (model["sign_raw"] * (Z_raw @ model["w_raw"])).astype(np.float32),
        "lda_score_detrended": (model["sign_dt"] * (Z_dt @ model["w_dt"])).astype(np.float32),
    }
    P = model["pca_dt"].transform(Z_dt)
    for i in range(model["n_pca"]):
        out[f"dpca_comp_{i+1}"] = P[:, i].astype(np.float32)
    out["lda_gap"] = (out["lda_score_detrended"] - out["lda_score"]).astype(np.float32)
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# C. GPU-Accelerated Extended Filter Bank
# ---------------------------------------------------------------------------
def compute_extended_filter_bank_gpu(blocks_dat_path, n_expected, chunk_size=50_000):
    """
    Computes rolling-mean maxima across windows W in [200, 300, 350, 400, 500, 600, 800],
    rolling std, burst excess, peak ratio, and Haar wavelet energy on CUDA.
    """
    print(f"  [C] GPU-Accelerated Filter Bank from {blocks_dat_path.name} ({n_expected:,} dies)...", flush=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blocks = np.memmap(blocks_dat_path, dtype="float32", mode="r", shape=(n_expected, 2000))
    windows = [200, 300, 350, 400, 500, 600, 800]
    names = (
        [f"max_rolling_mean_{w}" for w in windows]
        + [f"max_rolling_std_{w}" for w in (300, 400, 600)]
        + [f"burst_excess_{w}" for w in (350, 400, 600)]
        + ["burst_peak_ratio", "burst_center_idx_600", "dwt_cD1_energy"]
    )
    out = np.zeros((n_expected, len(names)), dtype=np.float32)

    for s in range(0, n_expected, chunk_size):
        e = min(s + chunk_size, n_expected)
        x_cpu = np.asarray(blocks[s:e], dtype=np.float32)
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

        # Haar wavelet DWT cD1 energy
        cd1 = (x_t[:, 1::2] - x_t[:, ::2]) / 1.41421356
        dwt_en = (cd1 ** 2).mean(dim=1)
        out[s:e, col] = dwt_en.cpu().numpy()
        col += 1

        del x_t, cs, cs2
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"      -> {len(names)} filter-bank columns computed on {device.type.upper()} in {time.time()-t0:.1f}s", flush=True)
    return pd.DataFrame(out, columns=names, index=pd.RangeIndex(n_expected))


# ---------------------------------------------------------------------------
# D. Within-Wafer Relative Triplet
# ---------------------------------------------------------------------------
def compute_wafer_relative(df, base_cols, global_stats=None):
    """
    Computes wrank_*, wzscore_*, wdev_* across candidate columns.
    """
    cols = [c for c in base_cols if c in df.columns]
    print(f"  [D] Within-wafer relative triplet over {len(cols)} base columns...", flush=True)
    t0 = time.time()

    if global_stats is None:
        global_stats = {
            c: {"mean": float(df[c].mean()), "std": float(max(df[c].std(), 1e-8))}
            for c in cols
        }

    out = {}
    grouped = df.groupby("wafer_id")
    for c in cols:
        short = (
            c.replace("max_rolling_mean_", "roll")
            .replace("pca_comp_", "pc")
            .replace("pca_", "pc")
            .replace("dpca_comp_", "dpc")
            .replace("lda_score_detrended", "ldadt")
            .replace("lda_score", "lda")
            .replace("block_", "blk")
            .replace("tophat_", "th")
            .replace("burst_excess_", "bex")
        )
        s = df[c]
        out[f"wrank_{short}"] = grouped[c].rank(pct=True).astype(np.float32).values
        wmean = grouped[c].transform("mean")
        wstd = grouped[c].transform("std").clip(lower=1e-8)
        gs = global_stats.get(c, {"mean": float(s.mean()), "std": float(max(s.std(), 1e-8))})
        out[f"wzscore_{short}"] = ((wmean - gs["mean"]) / gs["std"]).astype(np.float32).values
        out[f"wdev_{short}"] = ((s.values - wmean.values) / wstd.values).astype(np.float32)

    print(f"      -> {len(out)} columns in {time.time()-t0:.1f}s", flush=True)
    return pd.DataFrame(out, index=df.index), global_stats


# ---------------------------------------------------------------------------
# E. Wafer Context & Relative Interactions
# ---------------------------------------------------------------------------
def compute_wafer_context_and_interactions(df):
    """
    Aggregates key signals by wafer and builds clean products of normalized deviations.
    """
    print("  [E] Wafer-context aggregates and wafer-relative interactions...", flush=True)
    t0 = time.time()
    out = {}
    g = df.groupby("wafer_id")

    for c in ("lda_score_detrended", "max_rolling_mean_400", "burst_excess_400"):
        if c in df.columns:
            short = c.replace("max_rolling_mean_", "roll").replace("lda_score_detrended", "ldadt").replace("burst_excess_", "bex")
            out[f"wctx_{short}_mean"] = g[c].transform("mean").astype(np.float32).values
            out[f"wctx_{short}_std"] = g[c].transform("std").fillna(0.0).astype(np.float32).values
            out[f"wctx_{short}_p90"] = g[c].transform(lambda s: s.quantile(0.90)).astype(np.float32).values

    def col(name):
        if name not in df.columns:
            return None
        v = df[name]
        if isinstance(v, pd.DataFrame):
            v = v.iloc[:, -1]
        return v.to_numpy(np.float32)

    lda = col("wdev_ldadt")
    for partner in ("wdev_roll350", "wdev_roll400", "wdev_bex400", "wdev_thenergy_200", "wdev_grad_energy"):
        p = col(partner)
        if lda is not None and p is not None:
            out[f"ix_ldadt_x_{partner.replace('wdev_','')}"] = (lda * p).astype(np.float32)

    edge = 1.0 - col("distance_to_edge") if col("distance_to_edge") is not None else None
    dens = col("old_fail_density_5x5")
    if lda is not None and edge is not None:
        out["ix_ldadt_x_edge"] = (lda * edge).astype(np.float32)
    if lda is not None and dens is not None:
        out["ix_ldadt_x_density"] = (lda * (dens + 0.05)).astype(np.float32)
    if lda is not None and edge is not None and col("wdev_roll400") is not None:
        out["ix_trimodal_wafer_rel"] = (lda * col("wdev_roll400") * edge).astype(np.float32)

    print(f"      -> {len(out)} columns in {time.time()-t0:.1f}s", flush=True)
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# Pipeline Builder
# ---------------------------------------------------------------------------
def build_dataset(split="dev_train", lda_model=None, global_stats=None, fit_mode=False):
    out_path = TRAIN_OUTPUT if split == "dev_train" else VAL_OUTPUT
    if out_path.exists():
        print(f"Reusing cached {out_path.name}")
        return out_path, lda_model, global_stats

    in_path = TRAIN_INPUT if split == "dev_train" else VAL_INPUT
    print(f"\n{'='*78}\nBUILDING MODEL F FEATURES: {split.upper()}  (base = {in_path.name})\n{'='*78}", flush=True)
    df = pd.read_parquet(in_path).reset_index(drop=True)
    print(f"  base frame: {df.shape}", flush=True)

    # Alias pca_01 -> pca_comp_1 if needed
    for i in range(1, 4):
        if f"pca_0{i}" in df.columns and f"pca_comp_{i}" not in df.columns:
            df[f"pca_comp_{i}"] = df[f"pca_0{i}"]

    # A. Wafer-Conditional Detrending
    resid_df, detrend_ctx = detrend_parametric_within_wafer(df)

    # B. Shrinkage LDA & Detrended PCA
    if fit_mode:
        lda_model = build_lda_features(df, resid_df)
    lda_df = apply_lda_features(df, resid_df, lda_model)

    # C. Extended Filter Bank on GPU
    blocks_dat = RAW_BLOCKS_CACHE[split]
    fb = compute_extended_filter_bank_gpu(blocks_dat, len(df))
    fb.index = df.index
    # Replace overlapping filter-bank columns with freshly computed ones
    overlap = [c for c in fb.columns if c in df.columns]
    if overlap:
        df = df.drop(columns=overlap)

    stage = pd.concat([df, detrend_ctx, lda_df, fb], axis=1)
    stage = stage.loc[:, ~stage.columns.duplicated(keep="last")]

    # D. Broadened Within-Wafer Relative Triplet
    rel_df, global_stats = compute_wafer_relative(stage, WAFER_RELATIVE_BASE, global_stats)
    stage = pd.concat([stage, rel_df], axis=1)
    stage = stage.loc[:, ~stage.columns.duplicated(keep="last")]

    # E. Wafer Context & Relative Interactions
    ctx_df = compute_wafer_context_and_interactions(stage)
    final = pd.concat([stage, ctx_df, resid_df], axis=1)

    final = final.loc[:, ~final.columns.duplicated(keep="last")]
    print(f"\n  final frame: {final.shape}  (+{final.shape[1]-df.shape[1]} new features)", flush=True)
    final.to_parquet(out_path, index=False)
    print(f"  saved {out_path.name} ({out_path.stat().st_size/(1024**2):.1f} MB)", flush=True)
    return out_path, lda_model, global_stats


def main():
    print("\n" + "=" * 78)
    print("MODEL F FEATURE ENGINEERING: WAFER-CONDITIONAL MANIFOLD DETECTOR")
    print("=" * 78)

    _, lda_model, global_stats = build_dataset("dev_train", fit_mode=True)
    build_dataset("dev_val", lda_model=lda_model, global_stats=global_stats, fit_mode=False)

    with open(DETREND_META_PATH, "wb") as f:
        pickle.dump({"global_stats": global_stats, "wafer_relative_base": WAFER_RELATIVE_BASE}, f)
    print(f"\nSaved {DETREND_META_PATH.name}")
    print("\nAll Model F features successfully engineered.\n")


if __name__ == "__main__":
    main()
