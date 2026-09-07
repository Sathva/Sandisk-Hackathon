"""
Model C Controlled Experiment: Learning Rate Ablation (LR = 3e-4, Patience = 5).

Tests whether lowering the learning rate from 1e-3 to 3e-4 prevents early overfitting
and improves validation generalization on the canonical dev-val split.
Preserves all original Model C artifacts without overwriting.
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
    MODELS_DIR,
    REPORTS_DIR,
    PROCESSED_DIR,
)
from src.models.common import MODEL_A_FEATURES
from src.models.model_c_architecture import MultiResolutionCNN, get_model_summary
from src.models.prepare_cnn_data import prepare_cnn_cache

CACHE_DIR = PROCESSED_DIR / "cache"
FIGURES_DIR = REPORTS_DIR / "figures"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiResolutionDataset(Dataset):
    """
    Zero-copy Dataset reading raw 2,000-element block sequences from memory-mapped disk,
    normalized with dev_train statistics, along with normalized tabular features.
    """

    def __init__(self, blocks_path, shape, tab_features, labels, block_mean, block_std):
        self.blocks_path = str(blocks_path)
        self.shape = shape
        self.tab = tab_features.astype(np.float32)
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
        x_block = torch.from_numpy(norm_seq).unsqueeze(0)  # (1, 2000)
        x_tab = torch.from_numpy(self.tab[idx])           # (519,)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x_block, x_tab, y


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    num_samples = 0

    for x_block, x_tab, y in dataloader:
        x_block = x_block.to(device, non_blocking=True)
        x_tab = x_tab.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x_block, x_tab)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(y)
        num_samples += len(y)

    return total_loss / num_samples


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    all_probs = []
    all_targets = []

    for x_block, x_tab, y in dataloader:
        x_block = x_block.to(device, non_blocking=True)
        x_tab = x_tab.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x_block, x_tab)
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
    print("SANDISK HACKATHON — MODEL C EXPERIMENT: LEARNING RATE 3e-4 (PATIENCE = 5)")
    print("=" * 85)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Ensure Cache Exists
    # -------------------------------------------------------------------------
    train_blocks_path = CACHE_DIR / "dev_train_raw_blocks.dat"
    val_blocks_path = CACHE_DIR / "dev_val_raw_blocks.dat"
    norm_meta_path = MODELS_DIR / "model_c_normalization.json"

    if not (train_blocks_path.exists() and val_blocks_path.exists() and norm_meta_path.exists()):
        print("\nCache files missing. Triggering data preparation & caching...")
        prepare_cnn_cache()
    else:
        print("\nVerified existing cache files in processed/cache/:")
        print(f"  Train blocks: {train_blocks_path} ({train_blocks_path.stat().st_size / (1024**3):.2f} GB)")
        print(f"  Val blocks:   {val_blocks_path} ({val_blocks_path.stat().st_size / (1024**3):.2f} GB)")

    with open(norm_meta_path, "r") as f:
        norm_meta = json.load(f)

    block_mean = norm_meta["block_normalization"]["training_mean"]
    block_std = norm_meta["block_normalization"]["training_std"]
    seq_len = norm_meta["block_normalization"]["sequence_length"]
    n_train = norm_meta["dataset_statistics"]["num_train_eligible_dies"]
    n_val = norm_meta["dataset_statistics"]["num_val_eligible_dies"]

    print(f"\nNormalization Parameters (Strictly dev_train, identical to baseline):")
    print(f"  Block Sequence Mean: {block_mean:.6f}")
    print(f"  Block Sequence Std:  {block_std:.6f}")
    print(f"  Sequence Length:     {seq_len}")
    print(f"  Train dies:          {n_train:,}")
    print(f"  Val dies:            {n_val:,}")

    # Load pre-normalized tabular data and labels
    print("\nLoading tabular arrays and labels...")
    X_train_tab = np.load(CACHE_DIR / "dev_train_tabular_norm.npy")
    X_val_tab = np.load(CACHE_DIR / "dev_val_tabular_norm.npy")
    y_train = np.load(CACHE_DIR / "dev_train_labels.npy")
    y_val = np.load(CACHE_DIR / "dev_val_labels.npy")

    print(f"  X_train_tab shape: {X_train_tab.shape} (Positives: {int(y_train.sum()):,})")
    print(f"  X_val_tab shape:   {X_val_tab.shape} (Positives: {int(y_val.sum()):,})")

    # -------------------------------------------------------------------------
    # Step 2: Datasets & DataLoaders
    # -------------------------------------------------------------------------
    train_dataset = MultiResolutionDataset(
        blocks_path=train_blocks_path,
        shape=(n_train, seq_len),
        tab_features=X_train_tab,
        labels=y_train,
        block_mean=block_mean,
        block_std=block_std,
    )

    val_dataset = MultiResolutionDataset(
        blocks_path=val_blocks_path,
        shape=(n_val, seq_len),
        tab_features=X_val_tab,
        labels=y_val,
        block_mean=block_mean,
        block_std=block_std,
    )

    batch_size = 512
    num_workers = 4
    pin_memory = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    # -------------------------------------------------------------------------
    # Step 3: Model, Loss, Optimizer, Scaler
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing compute device: {device}")
    if device.type == "cuda":
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        print(f"  Total VRAM:  {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    model = MultiResolutionCNN(num_tabular_features=len(MODEL_A_FEATURES)).to(device)

    # Loss function with positive weight strictly from dev_train
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight_val = neg_count / pos_count
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Configured BCEWithLogitsLoss with pos_weight: {pos_weight_val:.6f}")

    # EXPERIMENTAL CHANGE: LR = 3e-4, PATIENCE = 5
    learning_rate = 3e-4
    weight_decay = 1e-4
    early_stopping_patience = 5
    max_epochs = 15

    print(f"\nExperimental Hyperparameters:")
    print(f"  Learning Rate:           {learning_rate} (was 1e-3)")
    print(f"  Early Stopping Patience: {early_stopping_patience} epochs (was 3)")
    print(f"  Maximum Epochs:          {max_epochs}")
    print(f"  Optimizer:               AdamW (weight_decay={weight_decay})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda")

    # -------------------------------------------------------------------------
    # Step 4: Training Loop with Early Stopping
    # -------------------------------------------------------------------------
    best_val_auc_pr = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    checkpoint_path = MODELS_DIR / "model_c_lr3e4_cnn.pt"

    history = []

    print("\n" + "=" * 85)
    print("STARTING MODEL C (LR=3e-4) TRAINING")
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
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auc_pr": best_val_auc_pr,
                "config": {
                    "num_tabular_features": len(MODEL_A_FEATURES),
                    "block_sequence_length": seq_len,
                    "pos_weight": pos_weight_val,
                    "learning_rate": learning_rate,
                    "patience": early_stopping_patience,
                }
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience={early_stopping_patience}).")
                break

    total_train_time = time.time() - training_t0
    print(f"\nTraining finished in {total_train_time:.1f}s ({total_train_time / 60:.2f} min).")
    print(f"Best model achieved at Epoch {best_epoch} with Dev-Val AUC-PR: {best_val_auc_pr:.4f}")

    # Save training history
    df_history = pd.DataFrame(history)
    history_csv = REPORTS_DIR / "model_c_lr3e4_training_history.csv"
    df_history.to_csv(history_csv, index=False)
    print(f"Saved training history to {history_csv}")

    # -------------------------------------------------------------------------
    # Step 5: Post-Training Evaluation with Best Restored Checkpoint
    # -------------------------------------------------------------------------
    print("\nRestoring best model checkpoint for evaluation...")
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
    print("MODEL C (LR=3e-4) EVALUATION ON CANONICAL DEV-VAL (ELIGIBLE DIES, OLD_LABEL == 0)")
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

    preds_path = REPORTS_DIR / "model_c_lr3e4_dev_val_predictions.parquet"
    val_meta.to_parquet(preds_path, index=False)
    print(f"\nSaved validation predictions to: {preds_path}")

    # Save metrics JSON & CSV
    metrics_dict = {
        "model_name": "Model C (Multi-Resolution 1D CNN, LR=3e-4)",
        "learning_rate": learning_rate,
        "early_stopping_patience": early_stopping_patience,
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
    }

    metrics_json_path = REPORTS_DIR / "model_c_lr3e4_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved metrics JSON: {metrics_json_path}")

    metrics_csv_path = REPORTS_DIR / "model_c_lr3e4_metrics.csv"
    pd.DataFrame([{
        "metric": k, "value": v
    } for k, v in metrics_dict.items() if not isinstance(v, dict)]).to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics CSV:  {metrics_csv_path}")

    # Save model config JSON
    config_dict = {
        "model_type": "MultiResolutionCNN",
        "num_tabular_features": len(MODEL_A_FEATURES),
        "block_channels": 1,
        "block_sequence_length": seq_len,
        "block_embed_dim": 256,
        "tab_hidden_dim": 256,
        "tab_out_dim": 128,
        "fusion_hidden_dim": 128,
        "dropout": 0.2,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "early_stopping_patience": early_stopping_patience,
        "pos_weight": float(pos_weight_val),
        "best_epoch": int(best_epoch),
        "best_val_auc_pr": float(best_val_auc_pr),
    }
    with open(MODELS_DIR / "model_c_lr3e4_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # -------------------------------------------------------------------------
    # Step 6: Direct Comparison: Old Model C (1e-3) vs. New Model C (3e-4)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("DIRECT COMPARISON: MODEL C (LR=1e-3) vs. MODEL C (LR=3e-4)")
    print("=" * 85)

    old_auc_pr = 0.572072
    old_roc_auc = 0.889126
    old_f1 = 0.550496
    old_prec = 0.767667
    old_rec = 0.429104
    old_best_epoch = 2

    delta_auc_pr = float(val_auc_pr - old_auc_pr)
    rel_auc_pr = (delta_auc_pr / old_auc_pr) * 100

    delta_roc_auc = float(val_roc_auc - old_roc_auc)
    rel_roc_auc = (delta_roc_auc / old_roc_auc) * 100

    delta_f1 = float(best_f1 - old_f1)
    rel_f1 = (delta_f1 / old_f1) * 100

    delta_rec = float(best_rec - old_rec)
    delta_prec = float(best_prec - old_prec)

    comparison_data = [
        {
            "Experiment": "Model C (Baseline LR=1e-3)",
            "Learning Rate": "1e-3",
            "Patience": 3,
            "Best Epoch": old_best_epoch,
            "AUC-PR": old_auc_pr,
            "ROC-AUC": old_roc_auc,
            "Tuned F1": old_f1,
            "Precision": old_prec,
            "Recall": old_rec,
            "Threshold": 0.8250,
            "Delta AUC-PR": "0.0000 (Ref)",
            "Rel Lift AUC-PR": "0.00%",
        },
        {
            "Experiment": "Model C (New LR=3e-4)",
            "Learning Rate": "3e-4",
            "Patience": early_stopping_patience,
            "Best Epoch": int(best_epoch),
            "AUC-PR": float(val_auc_pr),
            "ROC-AUC": float(val_roc_auc),
            "Tuned F1": float(best_f1),
            "Precision": float(best_prec),
            "Recall": float(best_rec),
            "Threshold": float(best_thresh),
            "Delta AUC-PR": f"{delta_auc_pr:+.4f}",
            "Rel Lift AUC-PR": f"{rel_auc_pr:+.2f}%",
        }
    ]

    df_comp = pd.DataFrame(comparison_data)
    print(df_comp.to_string(index=False))

    comp_csv_path = REPORTS_DIR / "model_c_vs_lr3e4_comparison.csv"
    comp_json_path = REPORTS_DIR / "model_c_vs_lr3e4_comparison.json"
    df_comp.to_csv(comp_csv_path, index=False)
    with open(comp_json_path, "w") as f:
        json.dump(comparison_data, f, indent=2)

    print(f"\nSaved comparison to: {comp_csv_path}")

    # -------------------------------------------------------------------------
    # Step 7: Generate Visualizations
    # -------------------------------------------------------------------------
    print("\nGenerating evaluation figures...")

    # Figure 1: Training Curve (Loss & Metrics progression)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df_history["epoch"], df_history["train_loss"], "o-", label="Train Loss (BCE)", color="#1f77b4")
    plt.plot(df_history["epoch"], df_history["val_loss"], "s--", label="Val Loss (BCE)", color="#ff7f0e")
    plt.title("Model C (LR=3e-4) Loss Progression", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df_history["epoch"], df_history["val_auc_pr"], "d-", label="Val AUC-PR (LR=3e-4)", color="#2ca02c")
    plt.axhline(old_auc_pr, color="#d62728", linestyle=":", label=f"Model C Baseline (1e-3, {old_auc_pr:.4f})")
    plt.axhline(0.5543, color="#7f7f7f", linestyle="--", label="Model B LightGBM (0.5543)")
    plt.plot(df_history["epoch"], df_history["val_roc_auc"], "^--", label="Val ROC-AUC", color="#9467bd")
    plt.title("Model C (LR=3e-4) Validation Metrics", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    curve_fig_path = FIGURES_DIR / "model_c_lr3e4_training_curve.png"
    plt.savefig(curve_fig_path, dpi=300)
    plt.close()
    print(f"  Saved training curve: {curve_fig_path}")

    # Figure 2: PR Curve Comparison: Model C (1e-3) vs Model C (3e-4) vs Model B
    plt.figure(figsize=(9, 7))

    # Old Model C
    df_old_c = pd.read_parquet(REPORTS_DIR / "model_c_dev_val_predictions.parquet")
    p_old_c, r_old_c, _ = precision_recall_curve(df_old_c["label"], df_old_c["predicted_probability"])
    plt.plot(r_old_c, p_old_c, label=f"Model C (LR=1e-3): AUC-PR = {old_auc_pr:.4f}", color="#1f77b4", linestyle="--", linewidth=2.0)

    # Model B
    df_b = pd.read_parquet(REPORTS_DIR / "model_b_dev_val_predictions.parquet")
    p_b, r_b, _ = precision_recall_curve(df_b["label"], df_b["predicted_probability"])
    plt.plot(r_b, p_b, label="Model B (LightGBM): AUC-PR = 0.5543", color="#7f7f7f", linestyle=":", linewidth=1.8)

    # New Model C (3e-4)
    p_new_c, r_new_c, _ = precision_recall_curve(y_val, val_probs)
    plt.plot(r_new_c, p_new_c, label=f"Model C (LR=3e-4): AUC-PR = {val_auc_pr:.4f}", color="#2ca02c", linewidth=2.5)

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve: Model C (LR=1e-3) vs. Model C (LR=3e-4)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    pr_fig_path = FIGURES_DIR / "model_c_vs_lr3e4_pr_comparison.png"
    plt.savefig(pr_fig_path, dpi=300)
    plt.close()
    print(f"  Saved PR comparison:  {pr_fig_path}")

    print("\n" + "=" * 85)
    print("MODEL C (LR=3e-4) EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    main()
