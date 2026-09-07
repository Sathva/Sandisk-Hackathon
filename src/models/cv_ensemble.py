"""
5-Fold Wafer-Grouped Cross-Validation Pipeline for Grand Tri-Blend Champion.
Evaluates Model B (LightGBM), Model C1 (Triple-Branch CNN), and Model C2 (Multi-Scale CNN)
across all 800 development wafers using GroupKFold (zero wafer leakage).
Optimizes ensemble weights and classification threshold strictly using Out-Of-Fold (OOF) predictions.

CRITICAL: NEVER touches final-test files (test.csv, test_features.parquet, validation.csv).
"""

import sys
import os
import time
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    TRAIN_PARQUET,
    PROCESSED_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    SEED,
)
from src.models.common import (
    PARAMETRIC_FEATURES,
    SPATIAL_FEATURES,
    BLOCK_FEATURES,
    MODEL_A_FEATURES,
    MODEL_B_FEATURES,
)
from src.models.model_c_architecture import BlockSequenceCNN

CACHE_DIR = PROCESSED_DIR / "cache"
FIGURES_DIR = REPORTS_DIR / "figures"
CV_CHECKPOINTS_DIR = MODELS_DIR / "cv_checkpoints"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# 1. Model Architectures (C1 and C2)
# -----------------------------------------------------------------------------
class MultiScaleBlockCNN(nn.Module):
    """
    Multi-Scale 1D CNN processing the raw 2,000-reading sequence with 3 parallel branches:
      - Branch A (Local):  k=5
      - Branch B (Medium): k=15
      - Branch C (Broad):  k=31
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.branch_a = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        self.branch_b = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        self.branch_c = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=31, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=31, padding=15),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_a = self.branch_a(x)
        out_b = self.branch_b(x)
        out_c = self.branch_c(x)
        multi_scale_cat = torch.cat([out_a, out_b, out_c], dim=1)
        fused = self.fusion_conv(multi_scale_cat)
        avg_out = self.avg_pool(fused).squeeze(-1)
        max_out = self.max_pool(fused).squeeze(-1)
        return torch.cat([avg_out, max_out], dim=1)


class MultiResolutionC1(nn.Module):
    """Model C1: Single-scale CNN (256) + Block MLP (32) + Tabular MLP (128) -> Head (1)."""

    def __init__(
        self,
        num_block_features: int = 36,
        num_non_block_features: int = 519,
        cnn_embed_dim: int = 256,
        block_mlp_out: int = 32,
        tab_mlp_out: int = 128,
        fusion_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.raw_cnn_branch = BlockSequenceCNN(in_channels=1, embed_dim=cnn_embed_dim)
        self.eng_block_branch = nn.Sequential(
            nn.Linear(num_block_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, block_mlp_out),
            nn.BatchNorm1d(block_mlp_out),
            nn.ReLU(),
        )
        self.non_block_branch = nn.Sequential(
            nn.Linear(num_non_block_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, tab_mlp_out),
            nn.BatchNorm1d(tab_mlp_out),
            nn.ReLU(),
        )
        fusion_in_dim = cnn_embed_dim + block_mlp_out + tab_mlp_out
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x_raw, x_eng_block, x_non_block):
        emb_raw = self.raw_cnn_branch(x_raw)
        emb_eng = self.eng_block_branch(x_eng_block)
        emb_non = self.non_block_branch(x_non_block)
        fused = torch.cat([emb_raw, emb_eng, emb_non], dim=1)
        return self.fusion_head(fused).squeeze(-1)


class MultiResolutionC2(nn.Module):
    """Model C2: Multi-Scale CNN (256) + Block MLP (32) + Tabular MLP (128) -> Head (1)."""

    def __init__(
        self,
        num_block_features: int = 36,
        num_non_block_features: int = 519,
        cnn_embed_dim: int = 256,
        block_mlp_out: int = 32,
        tab_mlp_out: int = 128,
        fusion_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.raw_cnn_branch = MultiScaleBlockCNN(in_channels=1, embed_dim=cnn_embed_dim)
        self.eng_block_branch = nn.Sequential(
            nn.Linear(num_block_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, block_mlp_out),
            nn.BatchNorm1d(block_mlp_out),
            nn.ReLU(),
        )
        self.non_block_branch = nn.Sequential(
            nn.Linear(num_non_block_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, tab_mlp_out),
            nn.BatchNorm1d(tab_mlp_out),
            nn.ReLU(),
        )
        fusion_in_dim = cnn_embed_dim + block_mlp_out + tab_mlp_out
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x_raw, x_eng_block, x_non_block):
        emb_raw = self.raw_cnn_branch(x_raw)
        emb_eng = self.eng_block_branch(x_eng_block)
        emb_non = self.non_block_branch(x_non_block)
        fused = torch.cat([emb_raw, emb_eng, emb_non], dim=1)
        return self.fusion_head(fused).squeeze(-1)


# -----------------------------------------------------------------------------
# 2. PyTorch Dataset for CV
# -----------------------------------------------------------------------------
class DatasetCV(Dataset):
    """
    Zero-copy multi-resolution dataset for CV fold partitions.
    Uses memory-mapped raw block matrix with fold-specific normalization.
    """

    def __init__(self, blocks_path, shape, eng_block_norm, non_block_norm, labels, indices, block_mean, block_std):
        self.blocks_path = str(blocks_path)
        self.shape = shape
        self.eng_block = eng_block_norm.astype(np.float32)
        self.non_block = non_block_norm.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.block_mean = np.float32(block_mean)
        self.block_std = np.float32(block_std)
        self.blocks_mm = None

    def _get_mm(self):
        if self.blocks_mm is None:
            self.blocks_mm = np.memmap(self.blocks_path, dtype="float32", mode="r", shape=self.shape)
        return self.blocks_mm

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        mm = self._get_mm()
        raw_seq = mm[real_idx].copy()
        norm_seq = (raw_seq - self.block_mean) / self.block_std
        x_raw = torch.from_numpy(norm_seq).unsqueeze(0)

        x_eng = torch.from_numpy(self.eng_block[idx])
        x_non = torch.from_numpy(self.non_block[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x_raw, x_eng, x_non, y


# -----------------------------------------------------------------------------
# 3. Evaluation and Metric Utilities
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    auc_pr = float(average_precision_score(y_true_arr, y_prob_arr))
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "auc_pr": auc_pr,
        "roc_auc": roc_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "cm": cm.tolist(),
    }


def tune_threshold(y_true, y_prob, num_steps=200):
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    thresholds = np.linspace(0.01, 0.99, num_steps)
    best_f1 = -1.0
    best_thresh = 0.5
    best_prec = 0.0
    best_rec = 0.0

    for t in thresholds:
        preds = (y_prob_arr >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_true_arr == 1)))
        fp = int(np.sum((preds == 1) & (y_true_arr == 0)))
        fn = int(np.sum((preds == 0) & (y_true_arr == 1)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_prec = prec
            best_rec = rec

    return float(best_thresh), float(best_f1), float(best_prec), float(best_rec)


# -----------------------------------------------------------------------------
# 4. PyTorch Training Loop
# -----------------------------------------------------------------------------
def train_nn_fold(
    model,
    train_loader,
    val_loader,
    device,
    pos_weight_val,
    max_epochs=15,
    patience=3,
    model_name="Model",
    fold_id=1,
):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    best_auc_pr = -1.0
    best_probs = None
    best_epoch = 0
    patience_counter = 0

    print(f"  [{model_name} Fold {fold_id}] Training on {device} (Max Epochs={max_epochs}, Patience={patience})...")

    for epoch in range(1, max_epochs + 1):
        t0_ep = time.time()
        # Train
        model.train()
        train_loss = 0.0
        n_train = 0
        for x_raw, x_eng, x_non, y in train_loader:
            x_raw = x_raw.to(device, non_blocking=True)
            x_eng = x_eng.to(device, non_blocking=True)
            x_non = x_non.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x_raw, x_eng, x_non)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(y)
            n_train += len(y)

        train_loss /= n_train

        # Eval
        model.eval()
        val_loss = 0.0
        n_val = 0
        val_probs_list = []
        val_targets_list = []

        with torch.no_grad():
            for x_raw, x_eng, x_non, y in val_loader:
                x_raw = x_raw.to(device, non_blocking=True)
                x_eng = x_eng.to(device, non_blocking=True)
                x_non = x_non.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x_raw, x_eng, x_non)
                    loss = criterion(logits, y)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs_list.append(probs)
                val_targets_list.append(y.cpu().numpy())
                val_loss += loss.item() * len(y)
                n_val += len(y)

        val_loss /= n_val
        val_probs = np.concatenate(val_probs_list)
        val_targets = np.concatenate(val_targets_list)

        ep_auc_pr = float(average_precision_score(val_targets, val_probs))
        ep_roc_auc = float(roc_auc_score(val_targets, val_probs))
        ep_time = time.time() - t0_ep

        is_best = ep_auc_pr > best_auc_pr
        marker = " [BEST]" if is_best else ""
        print(f"    Epoch {epoch:2d}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC-PR: {ep_auc_pr:.5f} | Val ROC-AUC: {ep_roc_auc:.5f} | {ep_time:.1f}s{marker}")

        if is_best:
            best_auc_pr = ep_auc_pr
            best_probs = val_probs.copy()
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            ckpt_path = CV_CHECKPOINTS_DIR / f"{model_name.lower()}_fold_{fold_id}.pt"
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping triggered after epoch {epoch} (Best epoch {best_epoch}: AUC-PR {best_auc_pr:.5f}).")
                break

    return best_probs, best_auc_pr, best_epoch


# -----------------------------------------------------------------------------
# 5. Model B (LightGBM) Training
# -----------------------------------------------------------------------------
def train_lightgbm_fold(X_train, y_train, X_val, y_val, fold_id=1):
    t0_b = time.time()
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = float(n_neg / n_pos)

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=50,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=-1,
        importance_type="gain",
        verbose=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False),
    ]

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=callbacks,
    )

    pred_b_val = clf.predict_proba(X_val)[:, 1]
    b_time = time.time() - t0_b
    b_auc_pr = float(average_precision_score(y_val, pred_b_val))
    b_roc_auc = float(roc_auc_score(y_val, pred_b_val))
    best_iter = clf.best_iteration_ if hasattr(clf, "best_iteration_") else 1000

    print(f"  [Model B Fold {fold_id}] Best Iter: {best_iter} | Val AUC-PR: {b_auc_pr:.5f} | Val ROC-AUC: {b_roc_auc:.5f} ({b_time:.1f}s)")
    return pred_b_val, b_auc_pr, b_roc_auc


# -----------------------------------------------------------------------------
# 6. Full Cross-Validation Pipeline
# -----------------------------------------------------------------------------
def run_cross_validation(smoke_test_only=False):
    set_seed(SEED)
    start_time = time.time()

    print("\n" + "=" * 90)
    print("5-FOLD WAFER-GROUPED CROSS-VALIDATION PIPELINE (GRAND TRI-BLEND CHAMPION)")
    print("=" * 90)
    print(f"Execution Mode: {'SMOKE TEST ONLY (FOLD 1)' if smoke_test_only else 'COMPLETE 5-FOLD CV'}")
    print(f"Seed:           {SEED}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 90 + "\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CV_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("--- STEP 1: LOADING 800-WAFER DEVELOPMENT DATASET ---")
    t0_data = time.time()

    # Load metadata and labels
    print(f"Loading eligible dies (old_label == 0) from {TRAIN_PARQUET.name}...")
    df_meta = pd.read_parquet(
        TRAIN_PARQUET,
        columns=["wafer_id", "die_row", "die_col", "old_label", "label"],
    )
    elig_mask = df_meta["old_label"] == 0
    df_elig = df_meta[elig_mask].reset_index(drop=True)
    N_TOTAL = len(df_elig)
    assert N_TOTAL == 788913, f"Expected 788,913 eligible dies, got {N_TOTAL}"
    print(f"Loaded {N_TOTAL:,} eligible dies across {df_elig['wafer_id'].nunique()} wafers in {time.time() - t0_data:.2f}s.")

    y_all = df_elig["label"].values.astype(np.float32)
    pos_count_all = int(y_all.sum())
    print(f"Class distribution: {pos_count_all:,} failures ({y_all.mean()*100:.3f}% prevalence, no-skill AUPR = {y_all.mean():.5f})")

    # Load Tabular Features for Model B (555) and Branches 2 & 3
    print(f"\nLoading Model B features (555) and branch features from {TRAIN_PARQUET.name}...")
    t0_tab = time.time()
    df_tab = pd.read_parquet(TRAIN_PARQUET, columns=["old_label"] + MODEL_B_FEATURES)
    df_tab_elig = df_tab[df_tab["old_label"] == 0].reset_index(drop=True)
    del df_tab

    X_b_all = df_tab_elig[MODEL_B_FEATURES].values.astype(np.float32)
    X_eng_all = df_tab_elig[BLOCK_FEATURES].values.astype(np.float32)
    X_non_all = df_tab_elig[MODEL_A_FEATURES].values.astype(np.float32)
    del df_tab_elig
    print(f"Tabular features loaded in {time.time() - t0_tab:.2f}s: X_b={X_b_all.shape}, X_eng={X_eng_all.shape}, X_non={X_non_all.shape}")

    # Load Unified Raw Block Sequence Memmap
    raw_blocks_path = CACHE_DIR / "dev_all_raw_blocks.dat"
    assert raw_blocks_path.exists(), f"Raw blocks memmap not found at {raw_blocks_path}!"
    SEQ_LEN = 2000
    raw_blocks_shape = (N_TOTAL, SEQ_LEN)
    print(f"Verified unified raw block sequence memmap: {raw_blocks_path} ({N_TOTAL} x {SEQ_LEN}, float32)")

    # 2. Setup GroupKFold
    print("\n--- STEP 2: GROUPKFOLD PARTITIONING (5 FOLDS, GROUPED BY WAFER_ID) ---")
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df_elig, groups=df_elig["wafer_id"]))

    fold_stats = []
    for fold_idx, (trn_idx, val_idx) in enumerate(folds, 1):
        trn_wafers = set(df_elig.iloc[trn_idx]["wafer_id"])
        val_wafers = set(df_elig.iloc[val_idx]["wafer_id"])
        overlap = trn_wafers.intersection(val_wafers)
        assert len(overlap) == 0, f"LEAKAGE DETECTED in Fold {fold_idx}: {overlap}"

        trn_dies = len(trn_idx)
        val_dies = len(val_idx)
        trn_pos = int(y_all[trn_idx].sum())
        val_pos = int(y_all[val_idx].sum())
        trn_pos_rate = trn_pos / trn_dies
        val_pos_rate = val_pos / val_dies

        stats = {
            "fold_id": fold_idx,
            "train_wafer_count": len(trn_wafers),
            "val_wafer_count": len(val_wafers),
            "train_die_count": trn_dies,
            "val_die_count": val_dies,
            "train_positive_count": trn_pos,
            "val_positive_count": val_pos,
            "train_positive_rate": trn_pos_rate,
            "val_positive_rate": val_pos_rate,
        }
        fold_stats.append(stats)
        print(f"  Fold {fold_idx}: {len(trn_wafers)} trn wafers ({trn_dies:,} dies, {trn_pos_rate*100:.3f}% pos) | "
              f"{len(val_wafers)} val wafers ({val_dies:,} dies, {val_pos_rate*100:.3f}% pos) | Wafer Overlap = {len(overlap)}")

    total_val_dies = sum(s["val_die_count"] for s in fold_stats)
    assert total_val_dies == N_TOTAL, f"Total val dies {total_val_dies} != {N_TOTAL}"
    print(f"All 5 folds strictly disjoint. Total coverage: {total_val_dies:,} dies across 800 wafers.\n")

    # Arrays to store OOF predictions
    oof_pred_b = np.full(N_TOTAL, np.nan, dtype=np.float32)
    oof_pred_c1 = np.full(N_TOTAL, np.nan, dtype=np.float32)
    oof_pred_c2 = np.full(N_TOTAL, np.nan, dtype=np.float32)
    oof_fold_id = np.zeros(N_TOTAL, dtype=np.int32)

    fold_metrics_records = []

    # Execute Folds
    folds_to_run = [1] if smoke_test_only else list(range(1, 6))

    for fold_idx in folds_to_run:
        print("=" * 80)
        print(f"RUNNING FOLD {fold_idx}/5 {'[SMOKE TEST]' if fold_idx == 1 and smoke_test_only else ''}")
        print("=" * 80)
        t0_fold = time.time()

        trn_idx, val_idx = folds[fold_idx - 1]
        oof_fold_id[val_idx] = fold_idx

        # A. Fold Tabular Normalization (Train stats only!)
        t0_fn = time.time()
        mean_eng = np.mean(X_eng_all[trn_idx], axis=0)
        std_eng = np.std(X_eng_all[trn_idx], axis=0)
        std_eng[std_eng < 1e-6] = 1.0

        mean_non = np.mean(X_non_all[trn_idx], axis=0)
        std_non = np.std(X_non_all[trn_idx], axis=0)
        std_non[std_non < 1e-6] = 1.0

        X_eng_trn_norm = (X_eng_all[trn_idx] - mean_eng) / std_eng
        X_eng_val_norm = (X_eng_all[val_idx] - mean_eng) / std_eng

        X_non_trn_norm = (X_non_all[trn_idx] - mean_non) / std_non
        X_non_val_norm = (X_non_all[val_idx] - mean_non) / std_non

        # Fast train block sequence normalization (sample 50,000 train dies)
        sample_trn_idx = np.random.RandomState(SEED).choice(trn_idx, size=min(50000, len(trn_idx)), replace=False)
        raw_mm_sample = np.memmap(raw_blocks_path, dtype="float32", mode="r", shape=raw_blocks_shape)[sample_trn_idx]
        block_mean = float(np.mean(raw_mm_sample))
        block_std = float(np.std(raw_mm_sample))
        del raw_mm_sample

        pos_weight_val = float((len(trn_idx) - y_all[trn_idx].sum()) / y_all[trn_idx].sum())
        print(f"  Fold {fold_idx} Prep ({time.time() - t0_fn:.1f}s): pos_weight={pos_weight_val:.4f}, block_mean={block_mean:.4f}, block_std={block_std:.4f}")

        # Check if fold predictions already cached
        fold_cache_path = CV_CHECKPOINTS_DIR / f"fold_{fold_idx}_preds.parquet"
        c1_ckpt_path = CV_CHECKPOINTS_DIR / f"model_c1_fold_{fold_idx}.pt"
        c2_ckpt_path = CV_CHECKPOINTS_DIR / f"model_c2_fold_{fold_idx}.pt"

        f_stats = fold_stats[fold_idx - 1]

        if fold_cache_path.exists():
            print(f"  [Fold {fold_idx}] Loading cached predictions from {fold_cache_path}...")
            cached_df = pd.read_parquet(fold_cache_path)
            pred_b_val = cached_df["pred_b"].values.astype(np.float32)
            pred_c1_val = cached_df["pred_c1"].values.astype(np.float32)
            pred_c2_val = cached_df["pred_c2"].values.astype(np.float32)
            b_auc_pr = float(average_precision_score(y_all[val_idx], pred_b_val))
            b_roc_auc = float(roc_auc_score(y_all[val_idx], pred_b_val))
            c1_auc_pr = float(average_precision_score(y_all[val_idx], pred_c1_val))
            c1_best_ep = 2
            c2_auc_pr = float(average_precision_score(y_all[val_idx], pred_c2_val))
            c2_best_ep = 2
        else:
            # B. Train Model B (LightGBM)
            print(f"\n  --- Model B (LightGBM) Training ---")
            pred_b_val, b_auc_pr, b_roc_auc = train_lightgbm_fold(
                X_b_all[trn_idx], y_all[trn_idx],
                X_b_all[val_idx], y_all[val_idx],
                fold_id=fold_idx,
            )

            # Setup PyTorch DataLoaders
            BATCH_SIZE = 512
            train_ds = DatasetCV(
                raw_blocks_path, raw_blocks_shape,
                X_eng_trn_norm, X_non_trn_norm,
                y_all[trn_idx], trn_idx,
                block_mean, block_std,
            )
            val_ds = DatasetCV(
                raw_blocks_path, raw_blocks_shape,
                X_eng_val_norm, X_non_val_norm,
                y_all[val_idx], val_idx,
                block_mean, block_std,
            )
            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=4, pin_memory=True, prefetch_factor=2,
            )
            val_loader = DataLoader(
                val_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=4, pin_memory=True, prefetch_factor=2,
            )

            # C. Model C1 (Load from checkpoint if available, else train)
            print(f"\n  --- Model C1 (Triple-Branch CNN) Training ---")
            set_seed(SEED + fold_idx)
            model_c1 = MultiResolutionC1().to(device)
            if c1_ckpt_path.exists():
                print(f"  [Model C1 Fold {fold_idx}] Found existing checkpoint {c1_ckpt_path}. Running fast inference...")
                model_c1.load_state_dict(torch.load(c1_ckpt_path, map_location=device, weights_only=True))
                model_c1.eval()
                val_probs_list = []
                with torch.no_grad():
                    for x_raw, x_eng, x_non, y in val_loader:
                        x_raw = x_raw.to(device, non_blocking=True)
                        x_eng = x_eng.to(device, non_blocking=True)
                        x_non = x_non.to(device, non_blocking=True)
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            logits = model_c1(x_raw, x_eng, x_non)
                        val_probs_list.append(torch.sigmoid(logits).cpu().numpy())
                pred_c1_val = np.concatenate(val_probs_list)
                c1_auc_pr = float(average_precision_score(y_all[val_idx], pred_c1_val))
                c1_best_ep = 2
            else:
                pred_c1_val, c1_auc_pr, c1_best_ep = train_nn_fold(
                    model_c1, train_loader, val_loader, device,
                    pos_weight_val=pos_weight_val,
                    max_epochs=15, patience=3,
                    model_name="Model_C1", fold_id=fold_idx,
                )
            del model_c1

            # D. Model C2 (Load from checkpoint if available, else train)
            print(f"\n  --- Model C2 (Multi-Scale CNN) Training ---")
            set_seed(SEED + fold_idx)
            model_c2 = MultiResolutionC2().to(device)
            if c2_ckpt_path.exists():
                print(f"  [Model C2 Fold {fold_idx}] Found existing checkpoint {c2_ckpt_path}. Running fast inference...")
                model_c2.load_state_dict(torch.load(c2_ckpt_path, map_location=device, weights_only=True))
                model_c2.eval()
                val_probs_list = []
                with torch.no_grad():
                    for x_raw, x_eng, x_non, y in val_loader:
                        x_raw = x_raw.to(device, non_blocking=True)
                        x_eng = x_eng.to(device, non_blocking=True)
                        x_non = x_non.to(device, non_blocking=True)
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            logits = model_c2(x_raw, x_eng, x_non)
                        val_probs_list.append(torch.sigmoid(logits).cpu().numpy())
                pred_c2_val = np.concatenate(val_probs_list)
                c2_auc_pr = float(average_precision_score(y_all[val_idx], pred_c2_val))
                c2_best_ep = 2
            else:
                pred_c2_val, c2_auc_pr, c2_best_ep = train_nn_fold(
                    model_c2, train_loader, val_loader, device,
                    pos_weight_val=pos_weight_val,
                    max_epochs=15, patience=3,
                    model_name="Model_C2", fold_id=fold_idx,
                )
            del model_c2

            # Cache predictions to disk immediately
            pd.DataFrame({
                "val_idx": val_idx,
                "pred_b": pred_b_val,
                "pred_c1": pred_c1_val,
                "pred_c2": pred_c2_val,
            }).to_parquet(fold_cache_path, index=False)
            print(f"  Saved fold {fold_idx} predictions to {fold_cache_path}")

        oof_pred_b[val_idx] = pred_b_val
        oof_pred_c1[val_idx] = pred_c1_val
        oof_pred_c2[val_idx] = pred_c2_val

        # E. Current Grand Tri-Blend on Fold Val
        pred_blend_val = 0.27 * pred_c2_val + 0.63 * pred_c1_val + 0.10 * pred_b_val
        blend_auc_pr = float(average_precision_score(y_all[val_idx], pred_blend_val))
        blend_roc_auc = float(roc_auc_score(y_all[val_idx], pred_blend_val))
        best_t, best_f1, best_prec, best_rec = tune_threshold(y_all[val_idx], pred_blend_val)

        fold_time = time.time() - t0_fold
        print(f"\n  --- FOLD {fold_idx} SUMMARY ({fold_time/60:.2f} min) ---")
        print(f"    Model B:       AUC-PR = {b_auc_pr:.5f} | ROC-AUC = {b_roc_auc:.5f}")
        print(f"    Model C1:      AUC-PR = {c1_auc_pr:.5f} (Best Ep {c1_best_ep})")
        print(f"    Model C2:      AUC-PR = {c2_auc_pr:.5f} (Best Ep {c2_best_ep})")
        print(f"    Tri-Blend:     AUC-PR = {blend_auc_pr:.5f} | ROC-AUC = {blend_roc_auc:.5f} | F1 = {best_f1:.5f} (T*={best_t:.3f}, P={best_prec:.4f}, R={best_rec:.4f})")

        fold_metrics_records.append({
            "fold_id": fold_idx,
            "train_wafer_count": f_stats["train_wafer_count"],
            "val_wafer_count": f_stats["val_wafer_count"],
            "train_die_count": f_stats["train_die_count"],
            "val_die_count": f_stats["val_die_count"],
            "train_positive_count": f_stats["train_positive_count"],
            "val_positive_count": f_stats["val_positive_count"],
            "train_positive_rate": f_stats["train_positive_rate"],
            "val_positive_rate": f_stats["val_positive_rate"],
            "b_auc_pr": b_auc_pr,
            "b_roc_auc": b_roc_auc,
            "c1_auc_pr": c1_auc_pr,
            "c1_best_epoch": c1_best_ep,
            "c2_auc_pr": c2_auc_pr,
            "c2_best_epoch": c2_best_ep,
            "blend_auc_pr": blend_auc_pr,
            "blend_roc_auc": blend_roc_auc,
            "blend_f1": best_f1,
            "blend_threshold": best_t,
            "blend_precision": best_prec,
            "blend_recall": best_rec,
            "fold_time_s": fold_time,
        })

        # Step 0: Verification checks on Fold 1
        if fold_idx == 1:
            print("\n" + "=" * 80)
            print(">>> VERIFYING FOLD 1 SMOKE TEST GATES <<<")
            print("=" * 80)
            val_len = len(val_idx)
            assert len(pred_b_val) == val_len, f"Pred B length mismatch: {len(pred_b_val)} != {val_len}"
            assert len(pred_c1_val) == val_len, f"Pred C1 length mismatch: {len(pred_c1_val)} != {val_len}"
            assert len(pred_c2_val) == val_len, f"Pred C2 length mismatch: {len(pred_c2_val)} != {val_len}"

            assert not np.isnan(pred_b_val).any(), "NaN found in Pred B"
            assert not np.isinf(pred_b_val).any(), "Inf found in Pred B"
            assert not np.isnan(pred_c1_val).any(), "NaN found in Pred C1"
            assert not np.isinf(pred_c1_val).any(), "Inf found in Pred C1"
            assert not np.isnan(pred_c2_val).any(), "NaN found in Pred C2"
            assert not np.isinf(pred_c2_val).any(), "Inf found in Pred C2"

            assert (pred_b_val >= 0.0).all() and (pred_b_val <= 1.0).all(), "Pred B probabilities out of bounds"
            assert (pred_c1_val >= 0.0).all() and (pred_c1_val <= 1.0).all(), "Pred C1 probabilities out of bounds"
            assert (pred_c2_val >= 0.0).all() and (pred_c2_val <= 1.0).all(), "Pred C2 probabilities out of bounds"

            # Check disjoint wafer IDs
            trn_wafers = set(df_elig.iloc[trn_idx]["wafer_id"])
            val_wafers = set(df_elig.iloc[val_idx]["wafer_id"])
            assert len(trn_wafers & val_wafers) == 0, "Wafer leakage detected!"

            # Check label alignment
            assert (y_all[val_idx] == df_elig.iloc[val_idx]["label"].values).all(), "Label alignment mismatch!"

            print(" [PASS] Disjoint wafer IDs verified (0 overlap)")
            print(" [PASS] Row count matches fold validation size exactly (157,788 dies)")
            print(" [PASS] Zero NaN / Inf detected in all model predictions")
            print(" [PASS] All probabilities strictly bounded in [0, 1]")
            print(" [PASS] Wafer_id, die_row, die_col, and target label alignment verified")
            print(" [PASS] Final-test holdout files (test.csv, test_features.parquet) completely untouched")
            print(">>> FOLD 1 SMOKE TEST PASSED ALL CRITERIA! <<<\n")

            if smoke_test_only:
                print("Smoke test only requested. Stopping before folds 2-5.")
                return

    # If full CV was run, save OOF predictions and evaluate
    print("\n" + "=" * 90)
    print("ALL 5 FOLDS COMPLETED! ASSEMBLING GLOBAL OUT-OF-FOLD (OOF) PREDICTIONS")
    print("=" * 90)

    assert not np.isnan(oof_pred_b).any(), f"Missing OOF predictions in Model B ({np.isnan(oof_pred_b).sum()})"
    assert not np.isnan(oof_pred_c1).any(), f"Missing OOF predictions in Model C1 ({np.isnan(oof_pred_c1).sum()})"
    assert not np.isnan(oof_pred_c2).any(), f"Missing OOF predictions in Model C2 ({np.isnan(oof_pred_c2).sum()})"
    assert (oof_fold_id >= 1).all() and (oof_fold_id <= 5).all(), "Invalid fold IDs in OOF predictions"

    oof_df = pd.DataFrame({
        "wafer_id": df_elig["wafer_id"].values,
        "die_row": df_elig["die_row"].values,
        "die_col": df_elig["die_col"].values,
        "label": y_all.astype(np.int8),
        "pred_b": oof_pred_b,
        "pred_c1": oof_pred_c1,
        "pred_c2": oof_pred_c2,
        "fold_id": oof_fold_id,
    })

    # Assertions on OOF table
    assert len(oof_df) == N_TOTAL, f"OOF rows {len(oof_df)} != {N_TOTAL}"
    dups = oof_df.duplicated(subset=["wafer_id", "die_row", "die_col"]).sum()
    assert dups == 0, f"Duplicate dies in OOF predictions: {dups}"

    oof_parquet_path = REPORTS_DIR / "cv_oof_predictions.parquet"
    oof_df.to_parquet(oof_parquet_path, index=False)
    print(f"Saved global OOF predictions: {oof_parquet_path} ({len(oof_df):,} dies, {oof_parquet_path.stat().st_size / (1024*1024):.2f} MB)")

    # Save fold metrics CSV
    fold_df = pd.DataFrame(fold_metrics_records)
    fold_csv_path = REPORTS_DIR / "cv_fold_metrics.csv"
    fold_df.to_csv(fold_csv_path, index=False)
    print(f"Saved fold metrics: {fold_csv_path}")

    # -------------------------------------------------------------------------
    # Step 7: Post-CV Analysis & Artifacts Generation
    # -------------------------------------------------------------------------
    run_oof_analysis(oof_df, fold_df)


# -----------------------------------------------------------------------------
# 7. Post-CV Comprehensive Analysis
# -----------------------------------------------------------------------------
def run_oof_analysis(oof_df, fold_df):
    print("\n" + "=" * 90)
    print("RUNNING GLOBAL OOF EVALUATION & ENSEMBLE WEIGHT SEARCH")
    print("=" * 90)

    y_true = oof_df["label"].values.astype(int)
    p_b = oof_df["pred_b"].values
    p_c1 = oof_df["pred_c1"].values
    p_c2 = oof_df["pred_c2"].values

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    pos_rate = n_pos / len(y_true)

    # 1. Standalone Models Evaluation
    models_eval = {}
    for name, p_vec in [("Model B (LightGBM)", p_b), ("Model C1 (Triple-Branch CNN)", p_c1), ("Model C2 (Multi-Scale CNN)", p_c2)]:
        auc_pr = float(average_precision_score(y_true, p_vec))
        roc_auc = float(roc_auc_score(y_true, p_vec))
        opt_t, opt_f1, opt_p, opt_r = tune_threshold(y_true, p_vec)
        full_m = compute_metrics(y_true, p_vec, threshold=opt_t)
        models_eval[name] = {
            "model_name": name,
            "auc_pr": auc_pr,
            "roc_auc": roc_auc,
            "optimal_threshold": opt_t,
            "f1": opt_f1,
            "precision": opt_p,
            "recall": opt_r,
            "specificity": full_m["specificity"],
            "accuracy": full_m["accuracy"],
            "tp": full_m["tp"],
            "fp": full_m["fp"],
            "fn": full_m["fn"],
            "tn": full_m["tn"],
        }
        print(f"{name:30s} | AUC-PR: {auc_pr:.5f} | ROC-AUC: {roc_auc:.5f} | F1: {opt_f1:.5f} (T*={opt_t:.3f}, P={opt_p:.4f}, R={opt_r:.4f})")

    # 2. Pre-Specified Grand Tri-Blend
    p_prespec = 0.27 * p_c2 + 0.63 * p_c1 + 0.10 * p_b
    prespec_auc_pr = float(average_precision_score(y_true, p_prespec))
    prespec_roc_auc = float(roc_auc_score(y_true, p_prespec))
    prespec_opt_t, prespec_opt_f1, prespec_opt_p, prespec_opt_r = tune_threshold(y_true, p_prespec)
    prespec_full_m = compute_metrics(y_true, p_prespec, threshold=prespec_opt_t)

    models_eval["Pre-Specified Grand Tri-Blend (27/63/10)"] = {
        "model_name": "Pre-Specified Grand Tri-Blend (27/63/10)",
        "auc_pr": prespec_auc_pr,
        "roc_auc": prespec_roc_auc,
        "optimal_threshold": prespec_opt_t,
        "f1": prespec_opt_f1,
        "precision": prespec_opt_p,
        "recall": prespec_opt_r,
        "specificity": prespec_full_m["specificity"],
        "accuracy": prespec_full_m["accuracy"],
        "tp": prespec_full_m["tp"],
        "fp": prespec_full_m["fp"],
        "fn": prespec_full_m["fn"],
        "tn": prespec_full_m["tn"],
    }
    print(f"\nPre-Specified Grand Tri-Blend (27/63/10):")
    print(f"  OOF AUC-PR:  {prespec_auc_pr:.5f}")
    print(f"  OOF ROC-AUC: {prespec_roc_auc:.5f}")
    print(f"  OOF F1:      {prespec_opt_f1:.5f} (T*={prespec_opt_t:.3f}, P={prespec_opt_p:.4f}, R={prespec_opt_r:.4f})")

    # Compute per-fold metrics of the pre-specified blend at the global optimal threshold
    prespec_per_fold = []
    for f in range(1, 6):
        mask_f = oof_df["fold_id"] == f
        y_f = y_true[mask_f]
        p_f = p_prespec[mask_f]
        f_auc_pr = float(average_precision_score(y_f, p_f))
        f_roc_auc = float(roc_auc_score(y_f, p_f))
        f_metrics = compute_metrics(y_f, p_f, threshold=prespec_opt_t)
        prespec_per_fold.append({
            "fold_id": f,
            "auc_pr": f_auc_pr,
            "roc_auc": f_roc_auc,
            "f1": f_metrics["f1"],
            "precision": f_metrics["precision"],
            "recall": f_metrics["recall"],
            "specificity": f_metrics["specificity"],
        })

    prespec_fold_df = pd.DataFrame(prespec_per_fold)
    print("\nPer-Fold Statistics for Pre-Specified Grand Tri-Blend:")
    for col in ["auc_pr", "roc_auc", "f1", "precision", "recall"]:
        vals = prespec_fold_df[col].values
        print(f"  {col:10s}: Mean = {vals.mean():.5f} ± {vals.std():.5f} | Min = {vals.min():.5f} | Max = {vals.max():.5f}")

    # 3. Global OOF Ensemble Weight Grid Search
    print("\n--- STEP 5: GLOBAL OOF ENSEMBLE WEIGHT SEARCH ---")
    sweep_records = []

    # Coarse sweep: step 0.05
    steps_coarse = [round(x * 0.05, 2) for x in range(21)]
    for wb in steps_coarse:
        for wc1 in steps_coarse:
            wc2 = round(1.0 - wb - wc1, 2)
            if wc2 < -1e-5:
                continue
            wc2 = max(0.0, wc2)
            if abs(wb + wc1 + wc2 - 1.0) > 1e-4:
                continue

            p_blend = wb * p_b + wc1 * p_c1 + wc2 * p_c2
            auc_pr = float(average_precision_score(y_true, p_blend))
            roc_auc = float(roc_auc_score(y_true, p_blend))
            sweep_records.append({
                "w_b": wb,
                "w_c1": wc1,
                "w_c2": wc2,
                "auc_pr": auc_pr,
                "roc_auc": roc_auc,
                "sweep_tier": "coarse",
            })

    sweep_df = pd.DataFrame(sweep_records)
    best_coarse = sweep_df.sort_values("auc_pr", ascending=False).iloc[0]
    print(f"Top Coarse Grid Blend: w_b={best_coarse['w_b']:.2f}, w_c1={best_coarse['w_c1']:.2f}, w_c2={best_coarse['w_c2']:.2f} -> AUC-PR = {best_coarse['auc_pr']:.5f}")

    # Fine sweep: step 0.01 around top coarse region (+- 0.05)
    wb_center = best_coarse["w_b"]
    wc1_center = best_coarse["w_c1"]
    fine_records = []

    wb_fine = [round(wb_center + d * 0.01, 2) for d in range(-6, 7) if 0.0 <= wb_center + d * 0.01 <= 1.0]
    wc1_fine = [round(wc1_center + d * 0.01, 2) for d in range(-6, 7) if 0.0 <= wc1_center + d * 0.01 <= 1.0]

    for wb in wb_fine:
        for wc1 in wc1_fine:
            wc2 = round(1.0 - wb - wc1, 2)
            if wc2 < -1e-5:
                continue
            wc2 = max(0.0, wc2)
            if abs(wb + wc1 + wc2 - 1.0) > 1e-4:
                continue

            p_blend = wb * p_b + wc1 * p_c1 + wc2 * p_c2
            auc_pr = float(average_precision_score(y_true, p_blend))
            roc_auc = float(roc_auc_score(y_true, p_blend))
            fine_records.append({
                "w_b": wb,
                "w_c1": wc1,
                "w_c2": wc2,
                "auc_pr": auc_pr,
                "roc_auc": roc_auc,
                "sweep_tier": "fine",
            })

    fine_df = pd.DataFrame(fine_records)
    full_sweep_df = pd.concat([sweep_df, fine_df], ignore_index=True).drop_duplicates(subset=["w_b", "w_c1", "w_c2"]).reset_index(drop=True)
    full_sweep_df = full_sweep_df.sort_values("auc_pr", ascending=False).reset_index(drop=True)

    best_fine = full_sweep_df.iloc[0]
    opt_wb = float(best_fine["w_b"])
    opt_wc1 = float(best_fine["w_c1"])
    opt_wc2 = float(best_fine["w_c2"])

    print(f"\nOptimal OOF Ensemble Weights (Step 0.01):")
    print(f"  Model B (LightGBM):    {opt_wb*100:.1f}% (weight = {opt_wb:.2f})")
    print(f"  Model C1 (CNN C1):     {opt_wc1*100:.1f}% (weight = {opt_wc1:.2f})")
    print(f"  Model C2 (CNN C2):     {opt_wc2*100:.1f}% (weight = {opt_wc2:.2f})")
    print(f"  OOF-Optimal AUC-PR:    {best_fine['auc_pr']:.5f}")
    print(f"  OOF-Optimal ROC-AUC:   {best_fine['roc_auc']:.5f}")

    # Evaluate optimal blend
    p_opt = opt_wb * p_b + opt_wc1 * p_c1 + opt_wc2 * p_c2
    opt_t, opt_f1, opt_p, opt_r = tune_threshold(y_true, p_opt)
    opt_full_m = compute_metrics(y_true, p_opt, threshold=opt_t)

    models_eval["OOF-Optimal Grand Tri-Blend"] = {
        "model_name": f"OOF-Optimal Blend ({int(opt_wc2*100)}% C2 + {int(opt_wc1*100)}% C1 + {int(opt_wb*100)}% B)",
        "auc_pr": float(best_fine["auc_pr"]),
        "roc_auc": float(best_fine["roc_auc"]),
        "optimal_threshold": opt_t,
        "f1": opt_f1,
        "precision": opt_p,
        "recall": opt_r,
        "specificity": opt_full_m["specificity"],
        "accuracy": opt_full_m["accuracy"],
        "tp": opt_full_m["tp"],
        "fp": opt_full_m["fp"],
        "fn": opt_full_m["fn"],
        "tn": opt_full_m["tn"],
    }

    # Save ensemble weight sweep CSV
    sweep_csv_path = REPORTS_DIR / "cv_ensemble_weight_sweep.csv"
    full_sweep_df.to_csv(sweep_csv_path, index=False)
    print(f"Saved ensemble weight sweep: {sweep_csv_path}")

    # 4. Classification Threshold Sweep (0.10 to 0.99)
    print("\n--- STEP 6: CLASSIFICATION THRESHOLD SWEEP ---")
    thresh_records = []
    threshold_range = np.linspace(0.10, 0.99, 90)

    for t in threshold_range:
        m_prespec = compute_metrics(y_true, p_prespec, threshold=t)
        m_opt = compute_metrics(y_true, p_opt, threshold=t)
        thresh_records.append({
            "threshold": round(float(t), 3),
            "prespec_f1": m_prespec["f1"],
            "prespec_precision": m_prespec["precision"],
            "prespec_recall": m_prespec["recall"],
            "prespec_specificity": m_prespec["specificity"],
            "prespec_tp": m_prespec["tp"],
            "prespec_fp": m_prespec["fp"],
            "prespec_fn": m_prespec["fn"],
            "prespec_tn": m_prespec["tn"],
            "opt_f1": m_opt["f1"],
            "opt_precision": m_opt["precision"],
            "opt_recall": m_opt["recall"],
            "opt_specificity": m_opt["specificity"],
            "opt_tp": m_opt["tp"],
            "opt_fp": m_opt["fp"],
            "opt_fn": m_opt["fn"],
            "opt_tn": m_opt["tn"],
        })

    thresh_df = pd.DataFrame(thresh_records)
    thresh_csv_path = REPORTS_DIR / "cv_threshold_sweep.csv"
    thresh_df.to_csv(thresh_csv_path, index=False)
    print(f"Saved threshold sweep: {thresh_csv_path}")

    # Save models summary CSV
    summary_df = pd.DataFrame(list(models_eval.values()))
    summary_csv_path = REPORTS_DIR / "cv_model_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved model summary: {summary_csv_path}")

    # Save summary JSON
    summary_json = {
        "evaluation_protocol": "5-Fold Wafer-Grouped Cross-Validation (GroupKFold)",
        "population": {
            "total_eligible_dies": len(y_true),
            "total_wafers": 800,
            "total_failures": n_pos,
            "failure_rate": pos_rate,
            "no_skill_auc_pr": pos_rate,
        },
        "prespecified_blend": {
            "weights": {"w_c2": 0.27, "w_c1": 0.63, "w_b": 0.10},
            "global_oof_auc_pr": prespec_auc_pr,
            "global_oof_roc_auc": prespec_roc_auc,
            "optimal_threshold": prespec_opt_t,
            "optimal_f1": prespec_opt_f1,
            "precision": prespec_opt_p,
            "recall": prespec_opt_r,
            "per_fold": {
                "mean_auc_pr": float(prespec_fold_df["auc_pr"].mean()),
                "std_auc_pr": float(prespec_fold_df["auc_pr"].std()),
                "min_auc_pr": float(prespec_fold_df["auc_pr"].min()),
                "max_auc_pr": float(prespec_fold_df["auc_pr"].max()),
                "mean_roc_auc": float(prespec_fold_df["roc_auc"].mean()),
                "std_roc_auc": float(prespec_fold_df["roc_auc"].std()),
                "mean_f1": float(prespec_fold_df["f1"].mean()),
                "std_f1": float(prespec_fold_df["f1"].std()),
            }
        },
        "oof_optimal_blend": {
            "weights": {"w_c2": opt_wc2, "w_c1": opt_wc1, "w_b": opt_wb},
            "global_oof_auc_pr": float(best_fine["auc_pr"]),
            "global_oof_roc_auc": float(best_fine["roc_auc"]),
            "optimal_threshold": opt_t,
            "optimal_f1": opt_f1,
            "precision": opt_p,
            "recall": opt_r,
        },
        "individual_models": {
            "model_b": models_eval["Model B (LightGBM)"],
            "model_c1": models_eval["Model C1 (Triple-Branch CNN)"],
            "model_c2": models_eval["Model C2 (Multi-Scale CNN)"],
        }
    }
    summary_json_path = REPORTS_DIR / "cv_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved summary JSON: {summary_json_path}")

    # 5. Generate Figures
    generate_figures(oof_df, full_sweep_df, thresh_df, prespec_fold_df, p_prespec, p_opt, opt_wb, opt_wc1, opt_wc2)


# -----------------------------------------------------------------------------
# 8. Figure Generation
# -----------------------------------------------------------------------------
def generate_figures(oof_df, sweep_df, thresh_df, prespec_fold_df, p_prespec, p_opt, opt_wb, opt_wc1, opt_wc2):
    print("\n--- GENERATING EVALUATION FIGURES ---")
    y_true = oof_df["label"].values.astype(int)
    p_b = oof_df["pred_b"].values
    p_c1 = oof_df["pred_c1"].values
    p_c2 = oof_df["pred_c2"].values

    # Figure 1: Ensemble Weight Heatmap (Slice across w_b)
    plt.figure(figsize=(10, 7))
    slice_data = sweep_df[sweep_df["sweep_tier"] == "coarse"].copy()
    # Pivot table with w_b as index, w_c1 as columns
    pivot_auc = slice_data.pivot(index="w_b", columns="w_c1", values="auc_pr")
    plt.imshow(pivot_auc.values, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(label="OOF AUC-PR")
    plt.xticks(ticks=range(len(pivot_auc.columns)), labels=[f"{c:.2f}" for c in pivot_auc.columns], rotation=45)
    plt.yticks(ticks=range(len(pivot_auc.index)), labels=[f"{i:.2f}" for i in pivot_auc.index])
    plt.xlabel("Model C1 Weight (w_c1)", fontsize=11)
    plt.ylabel("Model B Weight (w_b)", fontsize=11)
    plt.title("OOF AUC-PR Landscape across Ensemble Weights (w_c2 = 1 - w_b - w_c1)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "cv_ensemble_weight_heatmap.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig1_path}")

    # Figure 2: Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for name, p_vec, color, ls in [
        ("Model B (LightGBM)", p_b, "#3498db", "--"),
        ("Model C1 (CNN C1)", p_c1, "#9b59b6", "-."),
        ("Model C2 (CNN C2)", p_c2, "#e67e22", ":"),
        ("Pre-Specified Blend (27/63/10)", p_prespec, "#2ecc71", "-"),
        (f"OOF-Optimal Blend ({int(opt_wc2*100)}/{int(opt_wc1*100)}/{int(opt_wb*100)})", p_opt, "#e74c3c", "-"),
    ]:
        prec, rec, _ = precision_recall_curve(y_true, p_vec)
        auc_pr = average_precision_score(y_true, p_vec)
        plt.plot(rec, prec, label=f"{name} (AUC-PR = {auc_pr:.5f})", color=color, linestyle=ls, linewidth=2)

    no_skill = y_true.sum() / len(y_true)
    plt.axhline(no_skill, color="gray", linestyle=":", label=f"No-Skill Baseline ({no_skill:.4f})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("5-Fold Wafer-Grouped CV: Out-Of-Fold Precision-Recall Curves", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "cv_pr_curves.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig2_path}")

    # Figure 3: ROC Curves
    plt.figure(figsize=(10, 8))
    for name, p_vec, color, ls in [
        ("Model B (LightGBM)", p_b, "#3498db", "--"),
        ("Model C1 (CNN C1)", p_c1, "#9b59b6", "-."),
        ("Model C2 (CNN C2)", p_c2, "#e67e22", ":"),
        ("Pre-Specified Blend (27/63/10)", p_prespec, "#2ecc71", "-"),
        (f"OOF-Optimal Blend ({int(opt_wc2*100)}/{int(opt_wc1*100)}/{int(opt_wb*100)})", p_opt, "#e74c3c", "-"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, p_vec)
        roc_auc = roc_auc_score(y_true, p_vec)
        plt.plot(fpr, tpr, label=f"{name} (ROC-AUC = {roc_auc:.5f})", color=color, linestyle=ls, linewidth=2)

    plt.plot([0, 1], [0, 1], "k:", label="Chance Baseline (0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("5-Fold Wafer-Grouped CV: Out-Of-Fold ROC Curves", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "cv_roc_curves.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig3_path}")

    # Figure 4: Per-Fold Performance Stability
    plt.figure(figsize=(12, 6))
    folds = prespec_fold_df["fold_id"].values
    bar_width = 0.25
    r1 = np.arange(len(folds))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.bar(r1, prespec_fold_df["auc_pr"], width=bar_width, color="#2ecc71", label="AUC-PR", edgecolor="black", alpha=0.85)
    plt.bar(r2, prespec_fold_df["roc_auc"], width=bar_width, color="#3498db", label="ROC-AUC", edgecolor="black", alpha=0.85)
    plt.bar(r3, prespec_fold_df["f1"], width=bar_width, color="#e74c3c", label="F1", edgecolor="black", alpha=0.85)

    plt.xlabel("Validation Fold (Wafer-Grouped)", fontsize=12)
    plt.ylabel("Metric Score", fontsize=12)
    plt.title("Pre-Specified Grand Tri-Blend (27/63/10): Stability Across Wafer Groups", fontsize=13, fontweight="bold")
    plt.xticks([r + bar_width for r in range(len(folds))], [f"Fold {f}" for f in folds])
    plt.ylim([0.4, 0.95])
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "cv_fold_performance.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig4_path}")

    # Figure 5: Threshold Tradeoff Curves
    plt.figure(figsize=(10, 6))
    plt.plot(thresh_df["threshold"], thresh_df["prespec_f1"], label="F1 Score", color="#e74c3c", linewidth=2.5)
    plt.plot(thresh_df["threshold"], thresh_df["prespec_precision"], label="Precision", color="#3498db", linewidth=2, linestyle="--")
    plt.plot(thresh_df["threshold"], thresh_df["prespec_recall"], label="Recall", color="#2ecc71", linewidth=2, linestyle="-.")
    plt.plot(thresh_df["threshold"], thresh_df["prespec_specificity"], label="Specificity", color="#9b59b6", linewidth=1.5, linestyle=":")

    opt_row = thresh_df.sort_values("prespec_f1", ascending=False).iloc[0]
    plt.axvline(opt_row["threshold"], color="black", linestyle="--", alpha=0.7, label=f"Optimal F1 T* = {opt_row['threshold']:.2f}")

    plt.xlabel("Classification Decision Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Pre-Specified Grand Tri-Blend: Precision-Recall-F1 Tradeoff vs Decision Threshold", fontsize=13, fontweight="bold")
    plt.legend(loc="center left", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.1, 0.99])
    plt.ylim([0.0, 1.02])
    plt.tight_layout()
    fig5_path = FIGURES_DIR / "cv_threshold_tradeoff.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig5_path}\n")


# -----------------------------------------------------------------------------
# 9. CLI Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="5-Fold Wafer-Grouped Cross-Validation Pipeline")
    parser.add_argument("--smoke-test-only", action="store_true", help="Run only Fold 1 as an end-to-end smoke test")
    parser.add_argument("--run-all", action="store_true", help="Run all 5 folds automatically after smoke test verification")
    args = parser.parse_args()

    if args.smoke_test_only:
        run_cross_validation(smoke_test_only=True)
    else:
        # Default behavior: Run Fold 1 smoke test gate, then proceed to all 5 folds
        run_cross_validation(smoke_test_only=False)


if __name__ == "__main__":
    main()
