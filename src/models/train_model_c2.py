"""
Model C2: Multi-Scale 1D CNN + Tabular Multi-Resolution Deep Network.

Tests whether a multi-scale parallel convolutional architecture can extract
additional complementary information from the raw 2,000-reading sequence
beyond the sequential convolutions of Model C1.

Architecture:
  - Branch 1: Multi-Scale Raw 2,000-element Block Sequence:
      Parallel local (k=5), medium (k=15), and broad (k=31) Conv1D branches
      -> Concat channels (192) -> Conv1D(192->128, k=7) -> AdaptiveAvg+MaxPool (256-dim)
  - Branch 2: 36 engineered block features (MLP -> 32-dim embedding)
  - Branch 3: 519 parametric + spatial features (MLP -> 128-dim embedding)
  - Fusion Head: Concat (256 + 32 + 128 = 416-dim) -> Linear(416, 128) -> Linear(128, 1)

Features & Training Conditions:
  - Exact same dev_train (640 wafers) and dev_val (160 wafers) population (eligible dies old_label==0).
  - Exact same feature definitions and normalization as C1.
  - AdamW (lr=1e-3, weight_decay=1e-4), batch_size=512, AMP, seed=42.
  - BCEWithLogitsLoss with pos_weight=26.067988.
  - Early stopping patience=3 on dev-val AUC-PR.
"""

import os
import sys
import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    DEV_TRAIN_PARQUET,
    DEV_VAL_PARQUET,
    MODELS_DIR,
    REPORTS_DIR,
    PROCESSED_DIR,
    SEED,
)
from src.models.common import (
    PARAMETRIC_FEATURES,
    SPATIAL_FEATURES,
    BLOCK_FEATURES,
    MODEL_A_FEATURES,
)

CACHE_DIR = PROCESSED_DIR / "cache"
FIGURES_DIR = REPORTS_DIR / "figures"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# 1. Model C2 Architecture Components
# -----------------------------------------------------------------------------
class MultiScaleBlockCNN(nn.Module):
    """
    Multi-Scale 1D CNN processing the raw 2,000-reading sequence with 3 parallel branches:
      - Branch A (Local):  k=5, receptive field captures localized micro-anomalies
      - Branch B (Medium): k=15, receptive field captures cluster/sub-block patterns
      - Branch C (Broad):  k=31, receptive field captures broad wafer/chamber drift
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        # Branch A: Local (kernel=5)
        self.branch_a = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),  # 2000 -> 500
        )

        # Branch B: Medium (kernel=15)
        self.branch_b = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),  # 2000 -> 500
        )

        # Branch C: Broad (kernel=31)
        self.branch_c = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=31, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=31, padding=15),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),  # 2000 -> 500
        )

        # Cross-scale fusion convolution: 64 + 64 + 64 = 192 channels -> 128 channels
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Dual pooling: Global average (smooth shift) + Global max (sharp peak)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # 128 + 128 = 256 embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_a = self.branch_a(x)
        out_b = self.branch_b(x)
        out_c = self.branch_c(x)
        multi_scale_cat = torch.cat([out_a, out_b, out_c], dim=1)  # (B, 192, 500)
        fused = self.fusion_conv(multi_scale_cat)                  # (B, 128, 500)

        avg_out = self.avg_pool(fused).squeeze(-1)                  # (B, 128)
        max_out = self.max_pool(fused).squeeze(-1)                  # (B, 128)
        emb = torch.cat([avg_out, max_out], dim=1)                 # (B, 256)
        return emb


class MultiResolutionC2(nn.Module):
    """
    Multi-Resolution Model C2:
      - Branch 1: Multi-Scale 1D CNN over raw 2,000 sequence -> 256-dim embedding
      - Branch 2: Engineered Block MLP (36 -> 64 -> 32)
      - Branch 3: Parametric + Spatial MLP (519 -> 256 -> 128)
      - Fusion Head: 256 + 32 + 128 = 416 -> 128 -> 1
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
        # Branch 1: Multi-Scale Block CNN
        self.raw_cnn_branch = MultiScaleBlockCNN(in_channels=block_channels, embed_dim=cnn_embed_dim)

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

        # Fusion Head: 416 -> 128 -> 1
        fusion_in_dim = cnn_embed_dim + block_mlp_out + tab_mlp_out
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
        if mask_branch1:
            emb_raw = torch.zeros(x_raw.size(0), 256, device=x_raw.device, dtype=x_raw.dtype)
        else:
            emb_raw = self.raw_cnn_branch(x_raw)

        if mask_branch2:
            emb_eng = torch.zeros(x_eng_block.size(0), 32, device=x_eng_block.device, dtype=x_eng_block.dtype)
        else:
            emb_eng = self.eng_block_branch(x_eng_block)

        if mask_branch3:
            emb_non_block = torch.zeros(x_non_block.size(0), 128, device=x_non_block.device, dtype=x_non_block.dtype)
        else:
            emb_non_block = self.non_block_branch(x_non_block)

        fused = torch.cat([emb_raw, emb_eng, emb_non_block], dim=1)
        logits = self.fusion_head(fused).squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# 2. Dataset & Data Loader
# -----------------------------------------------------------------------------
class DatasetC2(Dataset):
    """
    Zero-copy multi-resolution dataset feeding 3 distinct input tensors:
      - x_raw: [1, 2000] float32 normalized raw block sequence
      - x_eng_block: [36] float32 normalized engineered block features
      - x_non_block: [519] float32 normalized parametric + spatial features
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
        raw_seq = mm[idx].copy()
        norm_seq = (raw_seq - self.block_mean) / self.block_std
        x_raw = torch.from_numpy(norm_seq).unsqueeze(0)

        x_eng_block = torch.from_numpy(self.eng_block[idx])
        x_non_block = torch.from_numpy(self.non_block[idx])
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
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "accuracy": acc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# -----------------------------------------------------------------------------
# 3. Main Training & Evaluation Pipeline
# -----------------------------------------------------------------------------
def run_model_c2():
    set_seed(SEED)
    start_total_time = time.time()

    print("\n" + "=" * 85)
    print("STARTING MODEL C2: MULTI-SCALE 1D CNN + TABULAR MULTI-RESOLUTION PIPELINE")
    print("=" * 85)
    print(f"Seed: {SEED}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 85 + "\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Parameter Count & Architecture Fairness Audit
    # -------------------------------------------------------------------------
    print("--- STEP 1: ARCHITECTURE PARAMETER COUNT & FAIRNESS AUDIT ---")
    model_c2 = MultiResolutionC2()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_c2 = model_c2.to(device)

    total_params = sum(p.numel() for p in model_c2.parameters())
    raw_cnn_params = sum(p.numel() for p in model_c2.raw_cnn_branch.parameters())
    eng_block_params = sum(p.numel() for p in model_c2.eng_block_branch.parameters())
    non_block_params = sum(p.numel() for p in model_c2.non_block_branch.parameters())
    fusion_head_params = sum(p.numel() for p in model_c2.fusion_head.parameters())
    tabular_params = eng_block_params + non_block_params

    # C1 reference
    C1_PARAMS = 306081
    C1_RAW_CNN_PARAMS = 129792
    param_delta = total_params - C1_PARAMS
    param_delta_pct = (param_delta / C1_PARAMS) * 100

    print(f"Model C2 Parameter Breakdown:")
    print(f"  - Branch 1 (Multi-Scale Raw 1D CNN):  {raw_cnn_params:,} params (C1 was {C1_RAW_CNN_PARAMS:,})")
    print(f"  - Branch 2 (Engineered Block MLP):     {eng_block_params:,} params")
    print(f"  - Branch 3 (Parametric + Spatial MLP): {non_block_params:,} params")
    print(f"  - Combined Tabular Branches:           {tabular_params:,} params")
    print(f"  - Fusion Prediction Head:              {fusion_head_params:,} params")
    print(f"  - TOTAL MODEL C2 PARAMETERS:           {total_params:,} params")
    print(f"  - Comparison vs Model C1 ({C1_PARAMS:,}): Delta = {param_delta:+,} ({param_delta_pct:+.1f}%)")
    print("Architecture verified: multi-scale parallel receptive fields with controlled capacity.\n")

    # -------------------------------------------------------------------------
    # Step 2: Load Memory-Mapped Dataset
    # -------------------------------------------------------------------------
    print("--- STEP 2: LOADING MEMORY-MAPPED DATASET ---")
    t0_data = time.time()

    # Load precomputed normalized arrays
    dev_train_blocks_path = CACHE_DIR / "dev_train_raw_blocks.dat"
    dev_val_blocks_path = CACHE_DIR / "dev_val_raw_blocks.dat"

    dev_train_c1_block = np.load(CACHE_DIR / "dev_train_c1_block_norm.npy")
    dev_val_c1_block = np.load(CACHE_DIR / "dev_val_c1_block_norm.npy")

    dev_train_tabular = np.load(CACHE_DIR / "dev_train_tabular_norm.npy")
    dev_val_tabular = np.load(CACHE_DIR / "dev_val_tabular_norm.npy")

    dev_train_labels = np.load(CACHE_DIR / "dev_train_labels.npy")
    dev_val_labels = np.load(CACHE_DIR / "dev_val_labels.npy")

    dev_val_meta = pd.read_parquet(CACHE_DIR / "dev_val_meta.parquet")

    # Load normalization constants
    norm_path = MODELS_DIR / "model_c1_normalization.json"
    with open(norm_path) as f:
        norm_meta = json.load(f)
    block_mean = norm_meta["branch_1_raw_blocks"]["training_mean"]
    block_std = norm_meta["branch_1_raw_blocks"]["training_std"]

    N_train = len(dev_train_labels)
    N_val = len(dev_val_labels)
    train_shape = (N_train, 2000)
    val_shape = (N_val, 2000)

    train_dataset = DatasetC2(
        dev_train_blocks_path, train_shape, dev_train_c1_block, dev_train_tabular,
        dev_train_labels, block_mean, block_std
    )
    val_dataset = DatasetC2(
        dev_val_blocks_path, val_shape, dev_val_c1_block, dev_val_tabular,
        dev_val_labels, block_mean, block_std
    )

    BATCH_SIZE = 512
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, prefetch_factor=2
    )

    print(f"Data loading complete in {time.time() - t0_data:.2f}s.")
    print(f"  Train: {N_train:,} dies ({int(dev_train_labels.sum()):,} failures, {dev_train_labels.mean()*100:.3f}%)")
    print(f"  Val:   {N_val:,} dies ({int(dev_val_labels.sum()):,} failures, {dev_val_labels.mean()*100:.3f}%)")

    # -------------------------------------------------------------------------
    # Step 3: Training Configuration & Loop
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: TRAINING MODEL C2 ---")
    pos_count = float(dev_train_labels.sum())
    neg_count = float(N_train - pos_count)
    pos_weight_val = neg_count / pos_count
    pos_weight = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model_c2.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    MAX_EPOCHS = 15
    PATIENCE = 3

    history = []
    best_auc_pr = -1.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = MODELS_DIR / "model_c2_cnn.pt"

    print(f"Hyperparameters: AdamW(lr=1e-3, wd=1e-4), Batch={BATCH_SIZE}, pos_weight={pos_weight_val:.4f}, Patience={PATIENCE}")
    print("-" * 85)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val AUC-PR':<14} {'Val ROC-AUC':<14} {'Time':<8} {'Status'}")
    print("-" * 85)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0_ep = time.time()
        train_loss = train_epoch(model_c2, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_auc_pr, val_roc_auc, _ = evaluate(model_c2, val_loader, criterion, device)
        ep_time = time.time() - t0_ep

        is_best = val_auc_pr > best_auc_pr
        status = ""
        if is_best:
            best_auc_pr = val_auc_pr
            best_epoch = epoch
            patience_counter = 0
            torch.save(model_c2.state_dict(), best_model_path)
            status = "(*) Best Checkpoint"
        else:
            patience_counter += 1
            status = f"patience {patience_counter}/{PATIENCE}"

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc_pr": val_auc_pr,
            "val_roc_auc": val_roc_auc,
            "learning_rate": 1e-3,
            "epoch_time_s": ep_time,
            "is_best": is_best,
        })

        print(f"{epoch:<8} {train_loss:<12.5f} {val_loss:<12.5f} {val_auc_pr:<14.5f} {val_roc_auc:<14.5f} {ep_time:<8.1f}s {status}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs (patience={PATIENCE}).")
            break

    print(f"\nRestoring best model checkpoint from Epoch {best_epoch} (AUC-PR = {best_auc_pr:.5f})...")
    model_c2.load_state_dict(torch.load(best_model_path))

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = REPORTS_DIR / "model_c2_training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history to: {history_path}")

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", marker="o")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Model C2: Training & Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["val_auc_pr"], label="Val AUC-PR", color="green", marker="^")
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Checkpoint (Ep {best_epoch})")
    plt.axhline(0.57658, color="purple", linestyle=":", label="Model C1 (0.5766)")
    plt.axhline(0.57861, color="red", linestyle=":", label="B+C1 Benchmark (0.5786)")
    plt.xlabel("Epoch")
    plt.ylabel("AUC-PR")
    plt.title("Model C2: Validation AUC-PR Progression")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig_train_path = FIGURES_DIR / "model_c2_training_curve.png"
    plt.savefig(fig_train_path, dpi=300)
    plt.close()
    print(f"Saved training curve to: {fig_train_path}")

    # -------------------------------------------------------------------------
    # Step 4: Inference & Evaluation on Canonical Dev-Val Set
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: CANONICAL DEV-VAL EVALUATION ---")
    val_loss, val_auc_pr, val_roc_auc, c2_val_probs = evaluate(model_c2, val_loader, criterion, device)
    y_val = dev_val_labels.astype(int)

    best_thresh, best_f1, best_prec, best_rec = tune_threshold(y_val, c2_val_probs)
    c2_metrics = compute_metrics(y_val, c2_val_probs, threshold=best_thresh)

    print("=" * 85)
    print(f"MODEL C2 DEV-VAL METRICS (at Tuned Threshold {best_thresh:.4f})")
    print("=" * 85)
    print(f"  AUC-PR:             {c2_metrics['auc_pr']:.5f}")
    print(f"  ROC-AUC:            {c2_metrics['roc_auc']:.5f}")
    print(f"  Tuned F1 Score:     {c2_metrics['f1']:.5f}")
    print(f"  Precision:          {c2_metrics['precision']:.4f}")
    print(f"  Recall:             {c2_metrics['recall']:.4f}")
    print(f"  Specificity:        {c2_metrics['specificity']:.4f}")
    print(f"  Accuracy:           {c2_metrics['accuracy']:.4%}")
    print(f"  True Positives (TP):{c2_metrics['tp']:,} / {int(y_val.sum()):,} fails ({c2_metrics['tp']/int(y_val.sum()):.2%})")
    print(f"  False Positives(FP):{c2_metrics['fp']:,} / {int((y_val==0).sum()):,} passes")
    print(f"  False Negatives(FN):{c2_metrics['fn']:,}")
    print(f"  True Negatives (TN):{c2_metrics['tn']:,}")

    # Save predictions
    preds_df = dev_val_meta.copy()
    preds_df["predicted_probability"] = c2_val_probs.astype(np.float32)
    preds_df["predicted_label_tuned"] = (c2_val_probs >= best_thresh).astype(np.int8)
    preds_df["predicted_label_05"] = (c2_val_probs >= 0.5).astype(np.int8)
    preds_parquet_path = REPORTS_DIR / "model_c2_dev_val_predictions.parquet"
    preds_df.to_parquet(preds_parquet_path, index=False)
    print(f"\nSaved predictions to: {preds_parquet_path}")

    # Save model config and normalization
    config_dict = {
        "model_name": "Model C2 (Multi-Scale 1D CNN + Tabular Multi-Resolution)",
        "total_parameters": int(total_params),
        "raw_cnn_parameters": int(raw_cnn_params),
        "tabular_parameters": int(tabular_params),
        "fusion_head_parameters": int(fusion_head_params),
        "kernel_sizes": [5, 15, 31],
        "fusion_kernel": 7,
        "best_epoch": int(best_epoch),
        "best_val_auc_pr": float(best_auc_pr),
        "learning_rate": 1e-3,
        "batch_size": BATCH_SIZE,
        "pos_weight": float(pos_weight_val),
    }
    with open(MODELS_DIR / "model_c2_cnn_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    with open(MODELS_DIR / "model_c2_normalization.json", "w") as f:
        json.dump(norm_meta, f, indent=2)

    # -------------------------------------------------------------------------
    # Step 5: Comparison vs Model C1, LightGBM B, and B+C1
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: COMPARATIVE BENCHMARKING ---")
    C1_AUCPR = 0.57658
    C1_ROCAUC = 0.89131
    C1_F1 = 0.55038

    LGB_B_AUCPR = 0.55432
    LGB_B_ROCAUC = 0.87598
    LGB_B_F1 = 0.54045

    CHAMP_AUCPR = 0.57861
    CHAMP_ROCAUC = 0.89203
    CHAMP_F1 = 0.55342

    delta_vs_c1_pr = c2_metrics["auc_pr"] - C1_AUCPR
    delta_vs_c1_roc = c2_metrics["roc_auc"] - C1_ROCAUC
    delta_vs_c1_f1 = c2_metrics["f1"] - C1_F1

    delta_vs_champ_pr = c2_metrics["auc_pr"] - CHAMP_AUCPR

    print(f"C2 vs Model C1 (AUC-PR: 0.5766, ROC-AUC: 0.8913, F1: 0.5504):")
    print(f"  Delta AUC-PR:  {delta_vs_c1_pr:+.5f} ({delta_vs_c1_pr / C1_AUCPR * 100:+.2f}%)")
    print(f"  Delta ROC-AUC: {delta_vs_c1_roc:+.5f}")
    print(f"  Delta F1:      {delta_vs_c1_f1:+.5f}")

    print(f"\nC2 vs Current Champion B+C1 (AUC-PR: 0.57861, ROC-AUC: 0.89203, F1: 0.55342):")
    print(f"  Delta AUC-PR:  {delta_vs_champ_pr:+.5f}")

    # -------------------------------------------------------------------------
    # Step 6: Zero-Retraining Branch Attribution Ablation
    # -------------------------------------------------------------------------
    print("\n--- STEP 6: ZERO-RETRAINING BRANCH ATTRIBUTION ABLATION ---")
    _, auc_no_raw, _, _ = evaluate(model_c2, val_loader, criterion, device, mask_b1=True)
    _, auc_no_eng, _, _ = evaluate(model_c2, val_loader, criterion, device, mask_b2=True)
    _, auc_no_nonblock, _, _ = evaluate(model_c2, val_loader, criterion, device, mask_b3=True)

    print(f"  Full Model C2:               AUC-PR = {c2_metrics['auc_pr']:.5f}")
    print(f"  Mask Branch 1 (Raw CNN):     AUC-PR = {auc_no_raw:.5f} (Delta: {auc_no_raw - c2_metrics['auc_pr']:+.5f})")
    print(f"  Mask Branch 2 (Eng Block):   AUC-PR = {auc_no_eng:.5f} (Delta: {auc_no_eng - c2_metrics['auc_pr']:+.5f})")
    print(f"  Mask Branch 3 (Param/Spat):  AUC-PR = {auc_no_nonblock:.5f} (Delta: {auc_no_nonblock - c2_metrics['auc_pr']:+.5f})")

    # Save metrics JSON & CSV
    c2_summary = {
        "model_name": "Model C2 (Multi-Scale 1D CNN + Tabular Multi-Resolution)",
        "total_parameters": int(total_params),
        "raw_cnn_parameters": int(raw_cnn_params),
        "tabular_parameters": int(tabular_params),
        "best_epoch": int(best_epoch),
        "total_epochs_trained": len(history),
        "training_time_s": float(sum(h["epoch_time_s"] for h in history)),
        "auc_pr": float(c2_metrics["auc_pr"]),
        "roc_auc": float(c2_metrics["roc_auc"]),
        "optimal_threshold": float(best_thresh),
        "f1": float(c2_metrics["f1"]),
        "precision": float(c2_metrics["precision"]),
        "recall": float(c2_metrics["recall"]),
        "specificity": float(c2_metrics["specificity"]),
        "accuracy": float(c2_metrics["accuracy"]),
        "tp": int(c2_metrics["tp"]),
        "fp": int(c2_metrics["fp"]),
        "fn": int(c2_metrics["fn"]),
        "tn": int(c2_metrics["tn"]),
        "comparison_vs_c1": {
            "c1_auc_pr": C1_AUCPR,
            "delta_auc_pr": float(delta_vs_c1_pr),
            "rel_lift_pct": float(delta_vs_c1_pr / C1_AUCPR * 100),
            "c1_roc_auc": C1_ROCAUC,
            "delta_roc_auc": float(delta_vs_c1_roc),
            "c1_f1": C1_F1,
            "delta_f1": float(delta_vs_c1_f1),
        },
        "comparison_vs_champion": {
            "champion_auc_pr": CHAMP_AUCPR,
            "delta_auc_pr": float(delta_vs_champ_pr),
        },
        "branch_ablation": {
            "mask_branch1_raw_cnn_auc_pr": float(auc_no_raw),
            "mask_branch2_eng_block_auc_pr": float(auc_no_eng),
            "mask_branch3_non_block_auc_pr": float(auc_no_nonblock),
        },
    }
    with open(REPORTS_DIR / "model_c2_metrics.json", "w") as f:
        json.dump(c2_summary, f, indent=2)

    pd.DataFrame([{
        "metric": k, "value": v
    } for k, v in c2_summary.items() if not isinstance(v, dict)]).to_csv(REPORTS_DIR / "model_c2_metrics.csv", index=False)

    # -------------------------------------------------------------------------
    # Step 7: Prediction Complementarity vs Model C1 & LightGBM B
    # -------------------------------------------------------------------------
    print("\n--- STEP 7: PREDICTION COMPLEMENTARITY ANALYSIS ---")
    c1_preds_df = pd.read_parquet(REPORTS_DIR / "model_c1_dev_val_predictions.parquet")
    lgb_preds_df = pd.read_parquet(REPORTS_DIR / "model_b_dev_val_predictions.parquet")

    # Verify die alignment
    assert (preds_df["wafer_id"].values == c1_preds_df["wafer_id"].values).all(), "Wafer alignment mismatch with C1!"
    assert (preds_df["die_row"].values == c1_preds_df["die_row"].values).all(), "Die row alignment mismatch with C1!"
    assert (preds_df["die_col"].values == c1_preds_df["die_col"].values).all(), "Die col alignment mismatch with C1!"

    c1_probs = c1_preds_df["predicted_probability"].values.astype(np.float64)
    lgb_probs = lgb_preds_df["predicted_probability"].values.astype(np.float64)
    c2_probs = c2_val_probs.astype(np.float64)

    t_c1, _, _, _ = tune_threshold(y_val, c1_probs)
    t_lgb, _, _, _ = tune_threshold(y_val, lgb_probs)
    t_c2 = best_thresh

    preds_c1 = (c1_probs >= t_c1).astype(int)
    preds_lgb = (lgb_probs >= t_lgb).astype(int)
    preds_c2 = (c2_probs >= t_c2).astype(int)

    pos_mask = (y_val == 1)
    N_POS = int(pos_mask.sum())

    complementarity_rows = []
    for ref_name, ref_p, ref_bin in [("Model C1", c1_probs, preds_c1), ("LightGBM B", lgb_probs, preds_lgb)]:
        p_corr, _ = pearsonr(c2_probs, ref_p)
        s_corr, _ = spearmanr(c2_probs, ref_p)

        disagree_cnt = int((preds_c2 != ref_bin).sum())
        disagree_rate = disagree_cnt / len(y_val)

        both_caught = int(((preds_c2 == 1) & (ref_bin == 1) & pos_mask).sum())
        only_c2_caught = int(((preds_c2 == 1) & (ref_bin == 0) & pos_mask).sum())
        only_ref_caught = int(((preds_c2 == 0) & (ref_bin == 1) & pos_mask).sum())
        both_missed = int(((preds_c2 == 0) & (ref_bin == 0) & pos_mask).sum())
        unique_caught = both_caught + only_c2_caught + only_ref_caught

        complementarity_rows.append({
            "model_x": "Model C2",
            "reference_model": ref_name,
            "pearson_corr": round(float(p_corr), 4),
            "spearman_corr": round(float(s_corr), 4),
            "disagreement_count": disagree_cnt,
            "disagreement_rate": round(disagree_rate, 4),
            "defects_both_caught": both_caught,
            "defects_only_c2_caught": only_c2_caught,
            "defects_only_ref_caught": only_ref_caught,
            "defects_both_missed": both_missed,
            "total_unique_defects_caught": unique_caught,
            "unique_defect_coverage_pct": round(unique_caught / N_POS * 100, 2),
        })

    comp_df = pd.DataFrame(complementarity_rows)
    print(comp_df.to_string(index=False))
    comp_df.to_csv(REPORTS_DIR / "model_c2_complementarity.csv", index=False)
    print(f"Saved complementarity to: {REPORTS_DIR / 'model_c2_complementarity.csv'}")

    # -------------------------------------------------------------------------
    # Step 8: Targeted Ensemble Sweeps (C2 + LightGBM B, C2 + C1, C2 + LGB + C1)
    # -------------------------------------------------------------------------
    print("\n--- STEP 8: TARGETED ENSEMBLE SWEEPS ---")
    sweep_rows = []
    ens_comp_rows = []

    # 1. C2 + LightGBM B
    print("Sweeping C2 + LightGBM B (alpha = C2 weight)...")
    best_pr_c2_lgb = -1.0
    best_a_c2_lgb = 0.5
    for a in np.linspace(0.0, 1.0, 21):
        bp = a * c2_probs + (1.0 - a) * lgb_probs
        pr = average_precision_score(y_val, bp)
        roc = roc_auc_score(y_val, bp)
        sweep_rows.append({"ensemble": "C2 + LightGBM B", "alpha": round(float(a), 3), "auc_pr": float(pr), "roc_auc": float(roc)})
        if pr > best_pr_c2_lgb:
            best_pr_c2_lgb = pr
            best_a_c2_lgb = a

    # Refine around best alpha
    for a in np.linspace(max(0.0, best_a_c2_lgb - 0.06), min(1.0, best_a_c2_lgb + 0.06), 13):
        bp = a * c2_probs + (1.0 - a) * lgb_probs
        pr = average_precision_score(y_val, bp)
        roc = roc_auc_score(y_val, bp)
        sweep_rows.append({"ensemble": "C2 + LightGBM B", "alpha": round(float(a), 3), "auc_pr": float(pr), "roc_auc": float(roc)})
        if pr > best_pr_c2_lgb:
            best_pr_c2_lgb = pr
            best_a_c2_lgb = a

    bp_c2_lgb = best_a_c2_lgb * c2_probs + (1.0 - best_a_c2_lgb) * lgb_probs
    t_c2_lgb, f1_c2_lgb, prec_c2_lgb, rec_c2_lgb = tune_threshold(y_val, bp_c2_lgb)
    m_c2_lgb = compute_metrics(y_val, bp_c2_lgb, threshold=t_c2_lgb)
    ens_comp_rows.append({
        "ensemble_name": "Model C2 + LightGBM B",
        "optimal_alpha_c2": round(float(best_a_c2_lgb), 3),
        "partner_weight": round(float(1.0 - best_a_c2_lgb), 3),
        "AUC-PR": m_c2_lgb["auc_pr"],
        "ROC-AUC": m_c2_lgb["roc_auc"],
        "Tuned F1": m_c2_lgb["f1"],
        "Precision": m_c2_lgb["precision"],
        "Recall": m_c2_lgb["recall"],
        "Optimal Threshold": m_c2_lgb["threshold"],
    })
    print(f"  Best C2 + LightGBM B: alpha_C2={best_a_c2_lgb:.3f} | AUC-PR={m_c2_lgb['auc_pr']:.5f} | ROC-AUC={m_c2_lgb['roc_auc']:.5f} | F1={m_c2_lgb['f1']:.4f}")

    # 2. C2 + C1
    print("Sweeping C2 + Model C1 (alpha = C2 weight)...")
    best_pr_c2_c1 = -1.0
    best_a_c2_c1 = 0.5
    for a in np.linspace(0.0, 1.0, 21):
        bp = a * c2_probs + (1.0 - a) * c1_probs
        pr = average_precision_score(y_val, bp)
        roc = roc_auc_score(y_val, bp)
        sweep_rows.append({"ensemble": "C2 + Model C1", "alpha": round(float(a), 3), "auc_pr": float(pr), "roc_auc": float(roc)})
        if pr > best_pr_c2_c1:
            best_pr_c2_c1 = pr
            best_a_c2_c1 = a

    for a in np.linspace(max(0.0, best_a_c2_c1 - 0.06), min(1.0, best_a_c2_c1 + 0.06), 13):
        bp = a * c2_probs + (1.0 - a) * c1_probs
        pr = average_precision_score(y_val, bp)
        roc = roc_auc_score(y_val, bp)
        sweep_rows.append({"ensemble": "C2 + Model C1", "alpha": round(float(a), 3), "auc_pr": float(pr), "roc_auc": float(roc)})
        if pr > best_pr_c2_c1:
            best_pr_c2_c1 = pr
            best_a_c2_c1 = a

    bp_c2_c1 = best_a_c2_c1 * c2_probs + (1.0 - best_a_c2_c1) * c1_probs
    t_c2_c1, f1_c2_c1, prec_c2_c1, rec_c2_c1 = tune_threshold(y_val, bp_c2_c1)
    m_c2_c1 = compute_metrics(y_val, bp_c2_c1, threshold=t_c2_c1)
    ens_comp_rows.append({
        "ensemble_name": "Model C2 + Model C1",
        "optimal_alpha_c2": round(float(best_a_c2_c1), 3),
        "partner_weight": round(float(1.0 - best_a_c2_c1), 3),
        "AUC-PR": m_c2_c1["auc_pr"],
        "ROC-AUC": m_c2_c1["roc_auc"],
        "Tuned F1": m_c2_c1["f1"],
        "Precision": m_c2_c1["precision"],
        "Recall": m_c2_c1["recall"],
        "Optimal Threshold": m_c2_c1["threshold"],
    })
    print(f"  Best C2 + Model C1: alpha_C2={best_a_c2_c1:.3f} | AUC-PR={m_c2_c1['auc_pr']:.5f} | ROC-AUC={m_c2_c1['roc_auc']:.5f} | F1={m_c2_c1['f1']:.4f}")

    # 3. Tri-Blend: C2 + LightGBM B + C1
    print("Sweeping Tri-Blend: C2 + LightGBM B + C1...")
    best_pr_tri = -1.0
    best_weights_tri = None
    # Sweep w_lgb in [0.10, 0.14, 0.18], split rest between C1 and C2
    for w_lgb in [0.10, 0.14, 0.18]:
        rem = 1.0 - w_lgb
        for frac_c2 in np.linspace(0.1, 0.9, 9):
            w_c2 = rem * frac_c2
            w_c1 = rem * (1.0 - frac_c2)
            bp = w_c2 * c2_probs + w_c1 * c1_probs + w_lgb * lgb_probs
            pr = average_precision_score(y_val, bp)
            roc = roc_auc_score(y_val, bp)
            sweep_rows.append({
                "ensemble": "C2 + LightGBM B + C1",
                "alpha": round(float(w_c2), 3),
                "auc_pr": float(pr),
                "roc_auc": float(roc),
            })
            if pr > best_pr_tri:
                best_pr_tri = pr
                best_weights_tri = (w_c2, w_c1, w_lgb)

    w_c2_opt, w_c1_opt, w_lgb_opt = best_weights_tri
    bp_tri = w_c2_opt * c2_probs + w_c1_opt * c1_probs + w_lgb_opt * lgb_probs
    t_tri, f1_tri, prec_tri, rec_tri = tune_threshold(y_val, bp_tri)
    m_tri = compute_metrics(y_val, bp_tri, threshold=t_tri)
    ens_comp_rows.append({
        "ensemble_name": "Tri-Blend: C2 + LightGBM B + C1",
        "optimal_alpha_c2": round(float(w_c2_opt), 3),
        "partner_weight": f"C1={w_c1_opt:.3f}, LGB={w_lgb_opt:.3f}",
        "AUC-PR": m_tri["auc_pr"],
        "ROC-AUC": m_tri["roc_auc"],
        "Tuned F1": m_tri["f1"],
        "Precision": m_tri["precision"],
        "Recall": m_tri["recall"],
        "Optimal Threshold": m_tri["threshold"],
    })
    print(f"  Best Tri-Blend: C2={w_c2_opt:.3f}, C1={w_c1_opt:.3f}, LGB={w_lgb_opt:.3f} | AUC-PR={m_tri['auc_pr']:.5f} | ROC-AUC={m_tri['roc_auc']:.5f} | F1={m_tri['f1']:.4f}")

    # Add reference benchmark row
    ens_comp_rows.append({
        "ensemble_name": "Benchmark: Model B + Model C1 (Current Champion)",
        "optimal_alpha_c2": np.nan,
        "partner_weight": "C1=0.855, LGB=0.145",
        "AUC-PR": CHAMP_AUCPR,
        "ROC-AUC": CHAMP_ROCAUC,
        "Tuned F1": CHAMP_F1,
        "Precision": 0.7672,
        "Recall": 0.4328,
        "Optimal Threshold": 0.900,
    })

    pd.DataFrame(sweep_rows).to_csv(REPORTS_DIR / "model_c2_ensemble_sweep.csv", index=False)
    ens_df = pd.DataFrame(ens_comp_rows)
    ens_df.to_csv(REPORTS_DIR / "model_c2_ensemble_comparison.csv", index=False)
    print(f"Saved ensemble results to: {REPORTS_DIR / 'model_c2_ensemble_comparison.csv'}")

    # -------------------------------------------------------------------------
    # Step 9: Lightweight Visualizations
    # -------------------------------------------------------------------------
    print("\n--- STEP 9: GENERATING VISUALIZATIONS ---")

    # 1. AUC-PR Comparison
    plt.figure(figsize=(9, 5))
    bar_models = ["LightGBM B", "Model C1", "Model C2", "B + C1 Ensemble"]
    bar_aucpr = [LGB_B_AUCPR, C1_AUCPR, c2_metrics["auc_pr"], CHAMP_AUCPR]
    bars = plt.barh(bar_models, bar_aucpr, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"], alpha=0.85)
    plt.axvline(CHAMP_AUCPR, color="#d62728", linestyle="--", label="Benchmark Champion (0.57861)")
    plt.xlim(0.50, 0.60)
    plt.xlabel("AUC-PR", fontsize=11)
    plt.title("Model C2 vs. Baselines: AUC-PR Comparison", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.002, bar.get_y() + bar.get_height()/2, f"{w:.5f}", va="center", fontsize=9, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "model_c2_vs_c1_aucpr_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    pr_c1, rec_c1, _ = precision_recall_curve(y_val, c1_probs)
    pr_c2, rec_c2, _ = precision_recall_curve(y_val, c2_probs)
    champ_blend_p = 0.855 * c1_probs + 0.145 * lgb_probs
    pr_champ, rec_champ, _ = precision_recall_curve(y_val, champ_blend_p)

    plt.plot(rec_c1, pr_c1, label=f"Model C1 (AUC-PR = {C1_AUCPR:.4f})", color="#2ca02c", linewidth=2.0)
    plt.plot(rec_c2, pr_c2, label=f"Model C2 (AUC-PR = {c2_metrics['auc_pr']:.4f})", color="#ff7f0e", linewidth=2.0)
    plt.plot(rec_champ, pr_champ, label=f"B + C1 Champion (AUC-PR = {CHAMP_AUCPR:.4f})", color="#d62728", linestyle="--", linewidth=2.0)
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curves: Model C1 vs. Model C2", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "model_c2_vs_c1_pr_curves.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # 3. ROC Curves
    plt.figure(figsize=(8, 6))
    fpr_c1, tpr_c1, _ = roc_curve(y_val, c1_probs)
    fpr_c2, tpr_c2, _ = roc_curve(y_val, c2_probs)
    plt.plot(fpr_c1, tpr_c1, label=f"Model C1 (ROC-AUC = {C1_ROCAUC:.4f})", color="#2ca02c", linewidth=2.0)
    plt.plot(fpr_c2, tpr_c2, label=f"Model C2 (ROC-AUC = {c2_metrics['roc_auc']:.4f})", color="#ff7f0e", linewidth=2.0)
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("ROC Curves: Model C1 vs. Model C2", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "model_c2_vs_c1_roc_curves.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # 4. Correlation Heatmap
    corr_models = ["LightGBM B", "Model C1", "Model C2"]
    prob_list = [lgb_probs, c1_probs, c2_probs]
    corr_mat = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            r, _ = pearsonr(prob_list[i], prob_list[j])
            corr_mat[i, j] = r

    plt.figure(figsize=(6, 5))
    plt.imshow(corr_mat, cmap="YlGnBu", vmin=0.80, vmax=1.0)
    plt.colorbar(label="Pearson Correlation")
    plt.xticks(range(3), corr_models, rotation=30, ha="right", fontsize=9)
    plt.yticks(range(3), corr_models, fontsize=9)
    for i in range(3):
        for j in range(3):
            val = corr_mat[i, j]
            text_color = "white" if val > 0.90 else "black"
            plt.text(j, i, f"{val:.4f}", ha="center", va="center", color=text_color, fontweight="bold")
    plt.title("Prediction Probability Correlation Matrix", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "model_c2_correlation_heatmap.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print("Saved all 4 visual figures to reports/figures/")

    # -------------------------------------------------------------------------
    # Step 10: Decision Gate & Final Report
    # -------------------------------------------------------------------------
    print("\n--- STEP 10: DECISION GATE & SUMMARY REPORT ---")

    best_c2_ens_pr = max(r["AUC-PR"] for r in ens_comp_rows if "Benchmark" not in r["ensemble_name"])
    best_c2_ens_name = next(r["ensemble_name"] for r in ens_comp_rows if r["AUC-PR"] == best_c2_ens_pr)

    if c2_metrics["auc_pr"] > C1_AUCPR + 0.001:
        case = "CASE A: C2 clearly beats C1 standalone on AUC-PR. -> C2 becomes a finalist."
        final_decision = "Model C2 (or C2 Ensemble)"
    elif best_c2_ens_pr > CHAMP_AUCPR + 0.0005:
        case = f"CASE B: C2 standalone is comparable, but {best_c2_ens_name} clearly beats benchmark champion B+C1. -> {best_c2_ens_name} becomes finalist."
        final_decision = f"Ensemble: {best_c2_ens_name}"
    elif abs(c2_metrics["auc_pr"] - C1_AUCPR) <= 0.001 and best_c2_ens_pr <= CHAMP_AUCPR:
        case = "CASE C: C2 is approximately tied with C1 and does not improve the champion ensemble. -> STOP model development. C1/B+C1 remains champion."
        final_decision = "Retain B + C1 Ensemble (0.57861)"
    else:
        case = "CASE D: C2 underperforms C1. -> STOP CNN development. C1/B+C1 remains champion."
        final_decision = "Retain B + C1 Ensemble (0.57861)"

    rel_c1_drop_raw = (auc_no_raw - c2_metrics['auc_pr']) / c2_metrics['auc_pr'] * 100
    rel_c1_drop_eng = (auc_no_eng - c2_metrics['auc_pr']) / c2_metrics['auc_pr'] * 100
    rel_c1_drop_non = (auc_no_nonblock - c2_metrics['auc_pr']) / c2_metrics['auc_pr'] * 100

    summary_md = f"""# SanDisk Hackathon — Model C2: Multi-Scale 1D CNN Evaluation Report

```text
=====================================================================================
CURRENT CHAMPION BEFORE C2:
  B + C1 Ensemble — AUC-PR: {CHAMP_AUCPR:.5f} | ROC-AUC: {CHAMP_ROCAUC:.5f} | F1: {CHAMP_F1:.5f}

MODEL C2 STANDALONE:
  AUC-PR: {c2_metrics['auc_pr']:.5f} | ROC-AUC: {c2_metrics['roc_auc']:.5f} | Tuned F1: {c2_metrics['f1']:.5f}

BEST C2 ENSEMBLE:
  {best_c2_ens_name} — AUC-PR: {best_c2_ens_pr:.5f}

DECISION RULE TRIGGERED:
  {case}

FINAL DECISION:
  {final_decision}
=====================================================================================
```

---

## 1. Executive Summary & Objective

Model C2 was implemented as a controlled architectural enhancement to evaluate whether a **multi-scale 1D CNN** with parallel receptive fields (local $k=5$, medium $k=15$, broad $k=31$) could extract additional complementary representations from the raw 2,000-reading sequence beyond Model C1's sequential convolutions.

All experiments were conducted on the exact canonical wafer-disjoint validation set: **$137,576$ eligible dies (`old_label == 0`) across 160 unseen wafers**, containing **$5,367$ post-test defects**.

---

## 2. Architecture & Parameter Fairness

| Architecture Component | Model C1 | Model C2 | Specification |
| :--- | :---: | :---: | :--- |
| **Branch 1: Raw 2,000 CNN** | Sequential Conv1D (k=11, 11, 7) | **Parallel Multi-Scale Conv1D (k=5, 15, 31)** | Local, medium, broad receptive fields + fusion conv |
| **Branch 1 Parameters** | $129,792$ | **$279,360$** | $+149,568$ params (+115.2%) |
| **Branch 2: Engineered Block** | 36 $\to$ 64 $\to$ 32 | 36 $\to$ 64 $\to$ 32 | Identical ($4,640$ params) |
| **Branch 3: Parametric/Spatial** | 519 $\to$ 256 $\to$ 128 | 519 $\to$ 256 $\to$ 128 | Identical ($167,168$ params) |
| **Fusion Prediction Head** | 416 $\to$ 128 $\to$ 1 | 416 $\to$ 128 $\to$ 1 | Identical ($53,377$ params) |
| **TOTAL PARAMETERS** | **$306,081$** | **$504,545$** | **$+198,464$ params (+64.8%)** |

---

## 3. Dev-Val Benchmark Leaderboard

| Model | Architecture | AUC-PR | ROC-AUC | Tuned F1 | Precision | Recall | Specificity | Tuned Thresh | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B + C1 Ensemble** | GBDT + Triple-Branch CNN (alpha*=0.855) | **{CHAMP_AUCPR:.5f}** | **{CHAMP_ROCAUC:.5f}** | **{CHAMP_F1:.5f}** | 0.7672 | 0.4328 | 0.9947 | 0.900 | N/A (blend) |
| **Tri-Blend: C2 + LGB + C1** | 3-Way Probability Blend | **{m_tri['auc_pr']:.5f}** | **{m_tri['roc_auc']:.5f}** | **{m_tri['f1']:.5f}** | {m_tri['precision']:.4f} | {m_tri['recall']:.4f} | {m_tri['specificity']:.4f} | {m_tri['threshold']:.3f} | N/A (blend) |
| **Model C2 + LightGBM B** | Probability Blend (alpha*={best_a_c2_lgb:.3f}) | **{m_c2_lgb['auc_pr']:.5f}** | {m_c2_lgb['roc_auc']:.5f} | {m_c2_lgb['f1']:.5f} | {m_c2_lgb['precision']:.4f} | {m_c2_lgb['recall']:.4f} | {m_c2_lgb['specificity']:.4f} | {m_c2_lgb['threshold']:.3f} | N/A (blend) |
| **Model C2** | Multi-Scale 1D CNN + Tabular MLP | **{c2_metrics['auc_pr']:.5f}** | {c2_metrics['roc_auc']:.5f} | {c2_metrics['f1']:.5f} | {c2_metrics['precision']:.4f} | {c2_metrics['recall']:.4f} | {c2_metrics['specificity']:.4f} | {best_thresh:.4f} | {sum(h['epoch_time_s'] for h in history):.1f} s |
| **Model C1** | Triple-Branch 1D CNN + Tabular MLP | **{C1_AUCPR:.5f}** | {C1_ROCAUC:.5f} | **{C1_F1:.5f}** | 0.7429 | 0.4371 | 0.9939 | 0.925 | 84.8 s |
| **LightGBM B** | Gradient Boosted Trees (555 feats) | **{LGB_B_AUCPR:.5f}** | {LGB_B_ROCAUC:.5f} | **{LGB_B_F1:.5f}** | 0.8217 | 0.4026 | 0.9965 | 0.815 | ~12.0 s |

---

## 4. Pairwise Deltas vs. Baselines

- **Model C2 vs. Model C1**:
  - Delta AUC-PR: **{delta_vs_c1_pr:+.5f}** ({delta_vs_c1_pr / C1_AUCPR * 100:+.2f}%)
  - Delta ROC-AUC: **{delta_vs_c1_roc:+.5f}**
  - Delta F1: **{delta_vs_c1_f1:+.5f}**
- **Model C2 vs. LightGBM B**:
  - Delta AUC-PR: **{c2_metrics['auc_pr'] - LGB_B_AUCPR:+.5f}** ({(c2_metrics['auc_pr'] - LGB_B_AUCPR) / LGB_B_AUCPR * 100:+.2f}%)
  - Delta ROC-AUC: **{c2_metrics['roc_auc'] - LGB_B_ROCAUC:+.5f}**
- **Model C2 vs. Champion (B + C1)**:
  - Delta AUC-PR: **{delta_vs_champ_pr:+.5f}**

---

## 5. Complementarity Analysis

```csv
{comp_df.to_string(index=False)}
```

- **Correlation with C1**: r = {comp_df.loc[comp_df['reference_model']=='Model C1', 'pearson_corr'].values[0]:.4f}
- **Correlation with LightGBM B**: r = {comp_df.loc[comp_df['reference_model']=='LightGBM B', 'pearson_corr'].values[0]:.4f}

---

## 6. Targeted Ensemble Sweeps

```csv
{ens_df.to_string(index=False)}
```

---

## 7. Zero-Retraining Branch Attribution Ablation

| Ablated Configuration | Dev-Val AUC-PR | Relative Drop from Full C2 | Interpretation |
| :--- | :---: | :---: | :--- |
| **Full Model C2** | **{c2_metrics['auc_pr']:.5f}** | **0.00%** | Full multi-modal representation |
| Mask Branch 1 (Raw Multi-Scale CNN) | **{auc_no_raw:.5f}** | **{rel_c1_drop_raw:.2f}%** | Loss of all raw temporal waveforms |
| Mask Branch 2 (Engineered Block MLP) | **{auc_no_eng:.5f}** | **{rel_c1_drop_eng:.2f}%** | Loss of 36 statistical block summaries |
| Mask Branch 3 (Parametric/Spatial MLP) | **{auc_no_nonblock:.5f}** | **{rel_c1_drop_non:.2f}%** | Loss of die-level and spatial context |

---

## 8. Final Decision & Strategic Recommendation

{case}

- **Final Champion to Freeze**: **{final_decision}**
- **Next Step**: Per guidelines, model development is complete. We immediately freeze the champion pipeline and advance directly to 5-fold grouped validation and interpretability (SHAP, spatial defect heatmaps, block wave traces).
"""

    with open(REPORTS_DIR / "model_c2_summary.md", "w") as f:
        f.write(summary_md)
    print(f"Saved Model C2 summary report to: {REPORTS_DIR / 'model_c2_summary.md'}")

    print("\n" + "=" * 85)
    print("MODEL C2 PIPELINE SUCCESSFULLY COMPLETED")
    print(f"Total Pipeline Runtime: {time.time() - start_total_time:.1f}s")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_model_c2()
