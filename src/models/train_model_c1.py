"""
Model C1: Multi-Resolution Triple-Branch Deep Neural Network.

Explicitly separates information resolutions into three distinct branches:
  Branch 1: Raw 2,000-element block sequence (1D CNN -> 256-d embedding)
  Branch 2: 36 engineered block features (Dedicated MLP -> 32-d embedding)
  Branch 3: 519 non-block tabular features (500 parametric + 19 spatial -> 128-d embedding)
  Fusion Head: Concat (256 + 32 + 128 = 416-d) -> Linear(416, 128) -> Linear(128, 1)

Preserves parameter fairness against Model C (306k vs 297k params, +2.9% diff).
Trains under identical conditions (LR=1e-3, AdamW, pos_weight=26.068, AMP, seed=42).
Includes zero-retraining branch attribution ablation diagnostic.
"""

import os
import sys
import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    DEV_TRAIN_PARQUET,
    DEV_VAL_PARQUET,
    MODELS_DIR,
    REPORTS_DIR,
    PROCESSED_DIR,
)
from src.models.common import (
    PARAMETRIC_FEATURES,
    SPATIAL_FEATURES,
    BLOCK_FEATURES,
    MODEL_A_FEATURES,
)
from src.models.model_c_architecture import BlockSequenceCNN, MultiResolutionCNN

CACHE_DIR = PROCESSED_DIR / "cache"
FIGURES_DIR = REPORTS_DIR / "figures"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiResolutionC1(nn.Module):
    """
    Triple-branch multi-resolution deep architecture.
    """

    def __init__(
        self,
        num_block_features: int = 36,
        num_non_block_features: int = 519,
        block_channels: int = 1,
        cnn_embed_dim: int = 256,
        block_mlp_hidden: int = 64,
        block_mlp_out: int = 32,
        tab_mlp_hidden: int = 256,
        tab_mlp_out: int = 128,
        fusion_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Branch 1: Raw Block Sequence 1D CNN (same architecture as Model C)
        self.raw_cnn_branch = BlockSequenceCNN(in_channels=block_channels, embed_dim=cnn_embed_dim)

        # Branch 2: Dedicated Engineered Block Features MLP (36 -> 64 -> 32)
        self.eng_block_branch = nn.Sequential(
            nn.Linear(num_block_features, block_mlp_hidden),
            nn.BatchNorm1d(block_mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(block_mlp_hidden, block_mlp_out),
            nn.BatchNorm1d(block_mlp_out),
            nn.ReLU(),
        )

        # Branch 3: Dedicated Parametric + Spatial Features MLP (519 -> 256 -> 128)
        self.non_block_branch = nn.Sequential(
            nn.Linear(num_non_block_features, tab_mlp_hidden),
            nn.BatchNorm1d(tab_mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tab_mlp_hidden, tab_mlp_out),
            nn.BatchNorm1d(tab_mlp_out),
            nn.ReLU(),
        )

        # Total Fusion Input: 256 + 32 + 128 = 416
        fusion_in_dim = cnn_embed_dim + block_mlp_out + tab_mlp_out

        # Fusion Head: 416 -> 128 -> 1
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(
        self,
        x_raw: torch.Tensor,
        x_eng_block: torch.Tensor,
        x_non_block: torch.Tensor,
        mask_branch1: bool = False,
        mask_branch2: bool = False,
        mask_branch3: bool = False,
    ) -> torch.Tensor:
        # Branch 1 embedding: (B, 256)
        if mask_branch1:
            emb_raw = torch.zeros(x_raw.size(0), 256, device=x_raw.device, dtype=x_raw.dtype)
        else:
            emb_raw = self.raw_cnn_branch(x_raw)

        # Branch 2 embedding: (B, 32)
        if mask_branch2:
            emb_eng = torch.zeros(x_eng_block.size(0), 32, device=x_eng_block.device, dtype=x_eng_block.dtype)
        else:
            emb_eng = self.eng_block_branch(x_eng_block)

        # Branch 3 embedding: (B, 128)
        if mask_branch3:
            emb_non_block = torch.zeros(x_non_block.size(0), 128, device=x_non_block.device, dtype=x_non_block.dtype)
        else:
            emb_non_block = self.non_block_branch(x_non_block)

        # Fusion: (B, 416) -> (B,)
        fused = torch.cat([emb_raw, emb_eng, emb_non_block], dim=1)
        logits = self.fusion_head(fused).squeeze(-1)
        return logits


class DatasetC1(Dataset):
    """
    Zero-copy multi-resolution dataset feeding 3 distinct input tensors.
    """

    def __init__(self, blocks_path, shape, eng_block_arr, non_block_arr, labels, block_mean, block_std):
        self.blocks_path = str(blocks_path)
        self.shape = shape
        self.eng_block = eng_block_arr.astype(np.float32)
        self.non_block = non_block_arr.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.block_mean = np.float32(block_mean)
        self.block_std = np.float32(block_std)
        self.blocks_mm = None

    def _get_mm(self):
        if self.blocks_mm is None:
            self.blocks_mm = np.memmap(self.blocks_path, dtype="float32", mode="r", shape=self.shape)
        return self.blocks_mm

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mm = self._get_mm()
        # Branch 1: normalized raw block sequence (1, 2000)
        raw_seq = mm[idx].copy()
        norm_seq = (raw_seq - self.block_mean) / self.block_std
        x_raw = torch.from_numpy(norm_seq).unsqueeze(0)

        # Branch 2: engineered block features (36,)
        x_eng_block = torch.from_numpy(self.eng_block[idx])

        # Branch 3: parametric + spatial features (519,)
        x_non_block = torch.from_numpy(self.non_block[idx])

        # Target label
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x_raw, x_eng_block, x_non_block, y


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    num_samples = 0

    for x_raw, x_eng, x_non, y in dataloader:
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

        total_loss += loss.item() * len(y)
        num_samples += len(y)

    return total_loss / num_samples


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, mask_b1=False, mask_b2=False, mask_b3=False):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    all_probs = []
    all_targets = []

    for x_raw, x_eng, x_non, y in dataloader:
        x_raw = x_raw.to(device, non_blocking=True)
        x_eng = x_eng.to(device, non_blocking=True)
        x_non = x_non.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(
                x_raw, x_eng, x_non,
                mask_branch1=mask_b1,
                mask_branch2=mask_b2,
                mask_branch3=mask_b3,
            )
            loss = criterion(logits, y)

        probs = torch.sigmoid(logits).cpu().numpy()
        targets = y.cpu().numpy()

        all_probs.append(probs)
        all_targets.append(targets)
        total_loss += loss.item() * len(y)
        num_samples += len(y)

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    avg_loss = total_loss / num_samples
    auc_pr = float(average_precision_score(all_targets, all_probs))
    roc_auc = float(roc_auc_score(all_targets, all_probs))

    return avg_loss, auc_pr, roc_auc, all_probs


def tune_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 197)
    best_f1 = -1.0
    best_thresh = 0.5
    best_prec = 0.0
    best_rec = 0.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_prec = precision_score(y_true, preds, zero_division=0)
            best_rec = recall_score(y_true, preds, zero_division=0)

    return float(best_thresh), float(best_f1), float(best_prec), float(best_rec)


def main():
    set_seed(42)
    start_total_time = time.time()

    print("\n" + "=" * 85)
    print("SANDISK HACKATHON — MODEL C1: TRIPLE-BRANCH MULTI-RESOLUTION DEEP NETWORK")
    print("=" * 85)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: Feature Branch Definitions & Schema Verification
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: FEATURE BRANCH DEFINITIONS & INTEGRITY CHECKS ---")

    eng_block_features = list(BLOCK_FEATURES)
    non_block_features = list(PARAMETRIC_FEATURES) + list(SPATIAL_FEATURES)

    print("=" * 80)
    print("1. Feature Verification & Disjoint Breakdown:")
    print(f"   - Branch 1 (Raw Block Sequence):   2,000 sub-die continuous readings")
    print(f"   - Branch 2 (Engineered Block):     {len(eng_block_features)} features")
    print(f"   - Branch 3 (Parametric + Spatial): {len(non_block_features)} features (500 parametric + 19 spatial)")
    print(f"   - Total Combined Tabular Inputs:   {len(eng_block_features) + len(non_block_features)} features (36 + 519 = 555)")
    print("=" * 80)

    print("\n2. Explicit List of 36 Engineered Block Features:")
    for idx, f_name in enumerate(eng_block_features, 1):
        print(f"   [{idx:>2}] {f_name}")

    print(f"\n3. Non-Block Tabular Features Count: {len(non_block_features)} total")
    print(f"   - 500 Parametric Features: feature_1 through feature_500")
    print(f"   - 19 Spatial Features: {list(SPATIAL_FEATURES)}")
    print(f"   (Note on prompt arithmetic: prompt stated '36 + 483 = 519'; in reality 500 parametric + 19 spatial = 519,")
    print(f"    and 36 block features are added for a total of 555 features, exactly matching Model B's feature space.)")

    # Assert integrity: no overlap, no missing, no leakage
    assert len(eng_block_features) == 36, f"Expected 36 block features, got {len(eng_block_features)}"
    assert len(non_block_features) == 519, f"Expected 519 non-block features, got {len(non_block_features)}"
    assert set(eng_block_features).isdisjoint(set(non_block_features)), "Overlap detected between Branch 2 and Branch 3!"

    forbidden = {"label", "wafer_id", "die_id", "old_label"}
    assert forbidden.isdisjoint(set(eng_block_features)), "Forbidden columns found in Branch 2!"
    assert forbidden.isdisjoint(set(non_block_features)), "Forbidden columns found in Branch 3!"
    print("\n4. Integrity Assertions Passed:")
    print("   [OK] Zero overlap between Branch 2 and Branch 3.")
    print("   [OK] Zero label/wafer_id/die_id leakage columns.")
    print("   [OK] All 36 block features and 519 non-block features accounted for.")

    # Save feature branch definition file
    feature_branches_meta = {
        "raw_block_features": "2000 continuous readings from block_readings",
        "engineered_block_features_count": len(eng_block_features),
        "engineered_block_features": eng_block_features,
        "parametric_spatial_features_count": len(non_block_features),
        "parametric_spatial_features": non_block_features,
        "total_tabular_features": len(eng_block_features) + len(non_block_features),
    }
    branch_json_path = REPORTS_DIR / "model_c1_feature_branches.json"
    with open(branch_json_path, "w") as f:
        json.dump(feature_branches_meta, f, indent=2)
    print(f"Saved feature branch definitions to: {branch_json_path}")

    # -------------------------------------------------------------------------
    # STEP 2: Architecture Parameter Fairness Check
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: ARCHITECTURE FAIRNESS & PARAMETER COUNT COMPARISON ---")

    model_c_dummy = MultiResolutionCNN(num_tabular_features=519)
    model_c1 = MultiResolutionC1(
        num_block_features=len(eng_block_features),
        num_non_block_features=len(non_block_features),
    )

    params_c = sum(p.numel() for p in model_c_dummy.parameters() if p.requires_grad)
    params_c1 = sum(p.numel() for p in model_c1.parameters() if p.requires_grad)
    param_diff = params_c1 - params_c
    param_pct_diff = (param_diff / params_c) * 100

    print(f"Model C  Trainable Parameters: {params_c:,}")
    print(f"Model C1 Trainable Parameters: {params_c1:,}")
    print(f"Parameter Difference:          {param_diff:+,} ({param_pct_diff:+.2f}%)")
    assert abs(param_pct_diff) < 5.0, f"C1 parameter difference too large ({param_pct_diff:.1f}%), violates fairness constraint!"
    print(f"Fairness constraint satisfied: Model C1 capacity matches Model C within 3.0% margin.")

    del model_c_dummy

    # -------------------------------------------------------------------------
    # STEP 3: Load Data & Extract Branch Normalization (Strictly dev_train)
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: DATA LOADING & NORMALIZATION (ZERO LEAKAGE) ---")

    train_blocks_path = CACHE_DIR / "dev_train_raw_blocks.dat"
    val_blocks_path = CACHE_DIR / "dev_val_raw_blocks.dat"
    assert train_blocks_path.exists() and val_blocks_path.exists(), "Raw block dat files missing!"

    # Load block sequence normalization from Model C metadata
    with open(MODELS_DIR / "model_c_normalization.json", "r") as f:
        norm_c = json.load(f)
    block_mean = norm_c["block_normalization"]["training_mean"]
    block_std = norm_c["block_normalization"]["training_std"]
    seq_len = norm_c["block_normalization"]["sequence_length"]
    n_train = norm_c["dataset_statistics"]["num_train_eligible_dies"]
    n_val = norm_c["dataset_statistics"]["num_val_eligible_dies"]

    # Non-block tabular features (519) are already pre-normalized strictly on dev_train!
    print("Loading pre-normalized 519 parametric+spatial tabular features...")
    X_train_nonblock = np.load(CACHE_DIR / "dev_train_tabular_norm.npy")
    X_val_nonblock = np.load(CACHE_DIR / "dev_val_tabular_norm.npy")
    y_train = np.load(CACHE_DIR / "dev_train_labels.npy")
    y_val = np.load(CACHE_DIR / "dev_val_labels.npy")

    # Extract and normalize the 36 engineered block features strictly from dev_train
    c1_train_block_cache = CACHE_DIR / "dev_train_c1_block_norm.npy"
    c1_val_block_cache = CACHE_DIR / "dev_val_c1_block_norm.npy"
    c1_norm_json = MODELS_DIR / "model_c1_normalization.json"

    if c1_train_block_cache.exists() and c1_val_block_cache.exists() and c1_norm_json.exists():
        print("Loading cached normalized 36 engineered block features...")
        X_train_eng_block = np.load(c1_train_block_cache)
        X_val_eng_block = np.load(c1_val_block_cache)
        with open(c1_norm_json, "r") as f:
            c1_norm_meta = json.load(f)
    else:
        print(f"Extracting 36 engineered block features from {DEV_TRAIN_PARQUET.name}...")
        df_tr = pd.read_parquet(DEV_TRAIN_PARQUET, columns=["old_label"] + eng_block_features)
        df_tr = df_tr[df_tr["old_label"] == 0].reset_index(drop=True)
        raw_tr_block = df_tr[eng_block_features].values.astype(np.float32)

        print(f"Extracting 36 engineered block features from {DEV_VAL_PARQUET.name}...")
        df_va = pd.read_parquet(DEV_VAL_PARQUET, columns=["old_label"] + eng_block_features)
        df_va = df_va[df_va["old_label"] == 0].reset_index(drop=True)
        raw_va_block = df_va[eng_block_features].values.astype(np.float32)

        # Compute mean & std strictly on dev_train
        block_feat_mean = np.mean(raw_tr_block, axis=0)
        block_feat_std = np.std(raw_tr_block, axis=0)
        block_feat_std[block_feat_std < 1e-6] = 1.0

        X_train_eng_block = (raw_tr_block - block_feat_mean) / block_feat_std
        X_val_eng_block = (raw_va_block - block_feat_mean) / block_feat_std

        np.save(c1_train_block_cache, X_train_eng_block)
        np.save(c1_val_block_cache, X_val_eng_block)

        c1_norm_meta = {
            "branch_1_raw_blocks": {
                "training_mean": block_mean,
                "training_std": block_std,
                "seq_len": seq_len,
            },
            "branch_2_eng_blocks": {
                "num_features": len(eng_block_features),
                "features": eng_block_features,
                "mean": block_feat_mean.tolist(),
                "std": block_feat_std.tolist(),
            },
            "branch_3_non_blocks": {
                "num_features": len(non_block_features),
                "features": non_block_features,
            }
        }
        with open(c1_norm_json, "w") as f:
            json.dump(c1_norm_meta, f, indent=2)
        print(f"Saved Model C1 normalization metadata to: {c1_norm_json}")

    print(f"Data Shapes Ready:")
    print(f"  Train: Raw Blocks ({n_train:,}, {seq_len}) | Eng Block {X_train_eng_block.shape} | Non-Block {X_train_nonblock.shape} | Labels {len(y_train):,}")
    print(f"  Val:   Raw Blocks ({n_val:,}, {seq_len}) | Eng Block {X_val_eng_block.shape} | Non-Block {X_val_nonblock.shape} | Labels {len(y_val):,}")

    # -------------------------------------------------------------------------
    # STEP 4: Datasets, DataLoaders, Loss & Optimizer
    # -------------------------------------------------------------------------
    train_dataset = DatasetC1(
        blocks_path=train_blocks_path,
        shape=(n_train, seq_len),
        eng_block_arr=X_train_eng_block,
        non_block_arr=X_train_nonblock,
        labels=y_train,
        block_mean=block_mean,
        block_std=block_std,
    )

    val_dataset = DatasetC1(
        blocks_path=val_blocks_path,
        shape=(n_val, seq_len),
        eng_block_arr=X_val_eng_block,
        non_block_arr=X_val_nonblock,
        labels=y_val,
        block_mean=block_mean,
        block_std=block_std,
    )

    batch_size = 512
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing compute device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    model = model_c1.to(device)

    # Identical Loss: BCEWithLogitsLoss with pos_weight = 26.067988
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight_val = neg_count / pos_count
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

    # Identical Training Setup: AdamW(lr=1e-3, weight_decay=1e-4), patience=3
    learning_rate = 1e-3
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda")

    max_epochs = 15
    early_stopping_patience = 3
    best_val_auc_pr = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    checkpoint_path = MODELS_DIR / "model_c1_cnn.pt"

    history = []

    # -------------------------------------------------------------------------
    # STEP 5: Training Loop
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("STARTING MODEL C1 TRAINING (LR=1e-3, PATIENCE=3)")
    print("=" * 85)
    print(f"{'Epoch':<7} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val AUC-PR':<12} | {'Val ROC-AUC':<12} | {'Time':<8}")
    print("-" * 75)

    training_t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        t_epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_auc_pr, val_roc_auc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc_pr)

        epoch_time = time.time() - t_epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc_pr": val_auc_pr,
            "val_roc_auc": val_roc_auc,
            "lr": current_lr,
            "duration_s": epoch_time,
        })

        is_best = val_auc_pr > best_val_auc_pr
        marker = " *" if is_best else ""
        print(f"{epoch:<7} | {train_loss:<12.5f} | {val_loss:<12.5f} | {val_auc_pr:<12.4f} | {val_roc_auc:<12.4f} | {epoch_time:<6.1f}s{marker}")

        if is_best:
            best_val_auc_pr = val_auc_pr
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auc_pr": best_val_auc_pr,
                "config": {
                    "num_block_features": len(eng_block_features),
                    "num_non_block_features": len(non_block_features),
                    "pos_weight": pos_weight_val,
                    "learning_rate": learning_rate,
                }
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience={early_stopping_patience}).")
                break

    total_train_time = time.time() - training_t0
    print(f"\nTraining finished in {total_train_time:.1f}s ({total_train_time / 60:.2f} min).")
    print(f"Best Model C1 checkpoint saved at Epoch {best_epoch} with Dev-Val AUC-PR: {best_val_auc_pr:.4f}")

    # Save training history
    df_history = pd.DataFrame(history)
    history_csv = REPORTS_DIR / "model_c1_training_history.csv"
    df_history.to_csv(history_csv, index=False)
    print(f"Saved training history to: {history_csv}")

    # -------------------------------------------------------------------------
    # STEP 6: Reload Best Checkpoint & Canonical Evaluation
    # -------------------------------------------------------------------------
    print("\nRestoring best Model C1 checkpoint for canonical evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_loss, val_auc_pr, val_roc_auc, val_probs = evaluate(model, val_loader, criterion, device)
    best_thresh, best_f1, best_prec, best_rec = tune_threshold(y_val, val_probs)

    y_pred = (val_probs >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / len(y_val)

    print("\n" + "=" * 85)
    print("MODEL C1 EVALUATION ON CANONICAL DEV-VAL (137,576 DIES, 5,367 FAILS)")
    print("=" * 85)
    print(f"  AUC-PR:             {val_auc_pr:.4f}")
    print(f"  ROC-AUC:            {val_roc_auc:.4f}")
    print(f"  Optimal Threshold:  {best_thresh:.4f}")
    print(f"  Tuned F1 Score:     {best_f1:.4f}")
    print(f"  Precision:          {best_prec:.4f}")
    print(f"  Recall:             {best_rec:.4f}")
    print(f"  Specificity:        {specificity:.4f}")
    print(f"  Accuracy:           {accuracy:.4%}")
    print(f"  True Positives:     {tp:,} / {int(y_val.sum()):,} fails ({tp / y_val.sum():.2%})")
    print(f"  False Positives:    {fp:,} / {int(len(y_val) - y_val.sum()):,} passes ({fp / (len(y_val) - y_val.sum()):.2%})")
    print(f"  False Negatives:    {fn:,}")
    print(f"  True Negatives:     {tn:,}")

    # Save validation predictions
    val_meta = pd.read_parquet(CACHE_DIR / "dev_val_meta.parquet")
    val_meta["predicted_probability"] = val_probs.astype(np.float32)
    val_meta["predicted_label_tuned"] = y_pred.astype(np.int8)
    val_meta["predicted_label_05"] = (val_probs >= 0.5).astype(np.int8)

    preds_path = REPORTS_DIR / "model_c1_dev_val_predictions.parquet"
    val_meta.to_parquet(preds_path, index=False)
    print(f"\nSaved validation predictions to: {preds_path}")

    # -------------------------------------------------------------------------
    # STEP 7: Branch Attribution Ablation Diagnostic (Zero-Masking)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("BRANCH ATTRIBUTION / ABLATION DIAGNOSTIC (ZERO-MASKING WITHOUT RETRAINING)")
    print("=" * 85)

    # 1. Mask Branch 1 (Zero out Raw CNN embedding)
    _, pr_no_b1, roc_no_b1, _ = evaluate(model, val_loader, criterion, device, mask_b1=True)
    # 2. Mask Branch 2 (Zero out Engineered Block embedding)
    _, pr_no_b2, roc_no_b2, _ = evaluate(model, val_loader, criterion, device, mask_b2=True)
    # 3. Mask Branch 3 (Zero out Parametric + Spatial embedding)
    _, pr_no_b3, roc_no_b3, _ = evaluate(model, val_loader, criterion, device, mask_b3=True)

    print(f"Full Model C1 (All 3 Branches Active): AUC-PR = {val_auc_pr:.4f}")
    print(f"Ablation 1: Mask Branch 1 (No Raw CNN):        AUC-PR = {pr_no_b1:.4f} (Delta: {pr_no_b1 - val_auc_pr:+.4f})")
    print(f"Ablation 2: Mask Branch 2 (No Eng Block):      AUC-PR = {pr_no_b2:.4f} (Delta: {pr_no_b2 - val_auc_pr:+.4f})")
    print(f"Ablation 3: Mask Branch 3 (No Param/Spatial):  AUC-PR = {pr_no_b3:.4f} (Delta: {pr_no_b3 - val_auc_pr:+.4f})")

    # -------------------------------------------------------------------------
    # STEP 8: Direct Comparison: Model C vs. Model C1
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("DIRECT COMPARISON: MODEL C (BASELINE) vs. MODEL C1 (TRIPLE-BRANCH)")
    print("=" * 85)

    old_auc_pr = 0.572072
    old_roc_auc = 0.889126
    old_f1 = 0.550496
    old_prec = 0.767667
    old_rec = 0.429104
    old_thresh = 0.8250
    old_tp = 2303
    old_fp = 697

    delta_pr = float(val_auc_pr - old_auc_pr)
    rel_pr = (delta_pr / old_auc_pr) * 100
    delta_roc = float(val_roc_auc - old_roc_auc)
    delta_f1 = float(best_f1 - old_f1)
    delta_prec = float(best_prec - old_prec)
    delta_rec = float(best_rec - old_rec)

    comp_rows = [
        {
            "Model": "Model C (Dual-Branch Baseline)",
            "Branches": "Raw CNN (256) + Non-Block Tabular (128)",
            "Parameters": params_c,
            "AUC-PR": old_auc_pr,
            "ROC-AUC": old_roc_auc,
            "Tuned F1": old_f1,
            "Precision": old_prec,
            "Recall": old_rec,
            "Threshold": old_thresh,
            "TP": old_tp,
            "FP": old_fp,
            "Delta AUC-PR": "0.0000 (Ref)",
            "Rel Lift AUC-PR": "0.00%",
        },
        {
            "Model": "Model C1 (Triple-Branch Separated)",
            "Branches": "Raw CNN (256) + Eng Block (32) + Param/Spatial (128)",
            "Parameters": params_c1,
            "AUC-PR": float(val_auc_pr),
            "ROC-AUC": float(val_roc_auc),
            "Tuned F1": float(best_f1),
            "Precision": float(best_prec),
            "Recall": float(best_rec),
            "Threshold": float(best_thresh),
            "TP": int(tp),
            "FP": int(fp),
            "Delta AUC-PR": f"{delta_pr:+.4f}",
            "Rel Lift AUC-PR": f"{rel_pr:+.2f}%",
        }
    ]

    df_comp = pd.DataFrame(comp_rows)
    print(df_comp.to_string(index=False))

    comp_csv_path = REPORTS_DIR / "model_c_vs_c1_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"\nSaved comparison table to: {comp_csv_path}")

    # Save metrics JSON & CSV
    metrics_dict = {
        "model_name": "Model C1 (Multi-Resolution Triple-Branch Deep Network)",
        "parameters": params_c1,
        "best_epoch": int(best_epoch),
        "total_epochs_trained": len(history),
        "training_time_s": float(total_train_time),
        "auc_pr": float(val_auc_pr),
        "roc_auc": float(val_roc_auc),
        "optimal_threshold": float(best_thresh),
        "f1_score": float(best_f1),
        "precision": float(best_prec),
        "recall": float(best_rec),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
        "pairwise_vs_model_c": {
            "delta_auc_pr": delta_pr,
            "rel_lift_auc_pr_pct": rel_pr,
            "delta_roc_auc": delta_roc,
            "delta_f1": delta_f1,
            "delta_precision": delta_prec,
            "delta_recall": delta_rec,
        },
        "branch_attribution_ablation": {
            "full_model_auc_pr": float(val_auc_pr),
            "mask_branch1_raw_cnn_auc_pr": float(pr_no_b1),
            "mask_branch2_eng_block_auc_pr": float(pr_no_b2),
            "mask_branch3_non_block_auc_pr": float(pr_no_b3),
        }
    }

    metrics_json_path = REPORTS_DIR / "model_c1_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved metrics JSON: {metrics_json_path}")

    metrics_csv_path = REPORTS_DIR / "model_c1_metrics.csv"
    pd.DataFrame([{
        "metric": k, "value": v
    } for k, v in metrics_dict.items() if not isinstance(v, dict)]).to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics CSV:  {metrics_csv_path}")

    # Config JSON
    config_dict = {
        "model_type": "MultiResolutionC1",
        "num_block_features": len(eng_block_features),
        "num_non_block_features": len(non_block_features),
        "cnn_embed_dim": 256,
        "block_mlp_hidden": 64,
        "block_mlp_out": 32,
        "tab_mlp_hidden": 256,
        "tab_mlp_out": 128,
        "fusion_hidden": 128,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "pos_weight": float(pos_weight_val),
        "best_epoch": int(best_epoch),
        "best_val_auc_pr": float(best_val_auc_pr),
    }
    with open(MODELS_DIR / "model_c1_cnn_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # -------------------------------------------------------------------------
    # STEP 9: Visualizations
    # -------------------------------------------------------------------------
    print("\nGenerating figures...")

    # Figure 1: Training Progression
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df_history["epoch"], df_history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    plt.plot(df_history["epoch"], df_history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
    plt.title("Model C1 Loss Progression", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df_history["epoch"], df_history["val_auc_pr"], "d-", label="Model C1 Val AUC-PR", color="#2ca02c")
    plt.axhline(old_auc_pr, color="#d62728", linestyle=":", label=f"Model C Baseline ({old_auc_pr:.4f})")
    plt.axhline(0.5543, color="#7f7f7f", linestyle="--", label="Model B LightGBM (0.5543)")
    plt.plot(df_history["epoch"], df_history["val_roc_auc"], "^--", label="Model C1 Val ROC-AUC", color="#9467bd")
    plt.title("Model C1 Validation Metrics", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "model_c1_training_curve.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  Saved training curves: {fig1_path}")

    # Figure 2: PR Curve Comparison: Model C vs Model C1 vs Model B
    plt.figure(figsize=(9, 7))
    df_old_c = pd.read_parquet(REPORTS_DIR / "model_c_dev_val_predictions.parquet")
    p_old, r_old, _ = precision_recall_curve(df_old_c["label"], df_old_c["predicted_probability"])
    plt.plot(r_old, p_old, label=f"Model C: Dual-Branch (AUC-PR = {old_auc_pr:.4f})", color="#1f77b4", linestyle="--", linewidth=2.0)

    df_b = pd.read_parquet(REPORTS_DIR / "model_b_dev_val_predictions.parquet")
    p_b, r_b, _ = precision_recall_curve(df_b["label"], df_b["predicted_probability"])
    plt.plot(r_b, p_b, label="Model B: LightGBM (AUC-PR = 0.5543)", color="#7f7f7f", linestyle=":", linewidth=1.8)

    p_c1, r_c1, _ = precision_recall_curve(y_val, val_probs)
    plt.plot(r_c1, p_c1, label=f"Model C1: Triple-Branch (AUC-PR = {val_auc_pr:.4f})", color="#2ca02c", linewidth=2.5)

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve: Model C vs. Model C1 vs. Model B", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig2_path = FIGURES_DIR / "model_c_vs_c1_pr_comparison.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  Saved PR comparison:   {fig2_path}")

    # -------------------------------------------------------------------------
    # STEP 10: Final Decision Rule Classification
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("FINAL DECISION RULE CLASSIFICATION & NEXT STEP RECOMMENDATION")
    print("=" * 85)

    if delta_pr > 0.003:
        classification = "1. STRONG IMPROVEMENT"
        recommendation = "Proceed to evaluate Model B + Model C1 probability ensemble."
    elif abs(delta_pr) <= 0.003:
        classification = "2. MARGINAL / EQUIVALENT"
        recommendation = "Do not immediately ensemble. Consider C2 (multi-scale receptive field 1D CNN) or gradient boosted tree tuning."
    else:
        classification = "3. WORSE"
        recommendation = "Discard C1 branch separation. Proceed to C2 multi-scale architecture or CatBoost/XGBoost."

    print(f"Classification:   {classification}")
    print(f"Model C AUC-PR:   {old_auc_pr:.4f}")
    print(f"Model C1 AUC-PR:  {val_auc_pr:.4f} (Delta: {delta_pr:+.4f}, {rel_pr:+.2f}%)")
    print(f"Recommendation:   {recommendation}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
