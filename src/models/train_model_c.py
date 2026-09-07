"""
End-to-End Training and Evaluation Pipeline for Model C (1D CNN + Tabular MLP).

Fuses raw 2,000-element block sequences (Branch 1) with 519 tabular features (Branch 2).
Evaluates on the canonical wafer-disjoint dev-val set and benchmarks against Model A and Model B.
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
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    TRAIN_CSV,
    DEV_TRAIN_PARQUET,
    DEV_VAL_PARQUET,
    DEV_SPLIT_JSON,
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
        # Lazy initialization per DataLoader worker
        if self.blocks_mm is None:
            self.blocks_mm = np.memmap(self.blocks_path, dtype="float32", mode="r", shape=self.shape)
        return self.blocks_mm

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mm = self._get_mm()
        # Extract raw block reading sequence, normalize, and shape to (1, 2000)
        raw_seq = mm[idx].copy()
        norm_seq = (raw_seq - self.block_mean) / self.block_std
        x_block = torch.from_numpy(norm_seq).unsqueeze(0)  # (1, 2000)

        # Tabular features (519,)
        x_tab = torch.from_numpy(self.tab[idx])

        # Target label
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
    print("SANDISK HACKATHON — MODEL C: MULTI-RESOLUTION 1D CNN TRAINING PIPELINE")
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

    print(f"\nNormalization Parameters (Zero Leakage, dev_train strictly):")
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
    arch_summary = get_model_summary(model, str(device))
    print(f"\n{arch_summary}")

    # Save architecture summary
    with open(REPORTS_DIR / "model_c_architecture.txt", "w") as f:
        f.write(arch_summary)

    # Loss function with positive weight strictly from dev_train
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight_val = neg_count / pos_count
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nConfigured BCEWithLogitsLoss with pos_weight: {pos_weight_val:.6f}")

    learning_rate = 1e-3
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda")

    # -------------------------------------------------------------------------
    # Step 4: Training Loop with Early Stopping
    # -------------------------------------------------------------------------
    max_epochs = 15
    early_stopping_patience = 3
    best_val_auc_pr = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    checkpoint_path = MODELS_DIR / "model_c_cnn.pt"

    history = []

    print("\n" + "=" * 85)
    print("STARTING MODEL C TRAINING")
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

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc_pr": val_auc_pr,
            "val_roc_auc": val_roc_auc,
            "lr": optimizer.param_groups[0]["lr"],
            "duration_s": epoch_time,
        })

        is_best = val_auc_pr > best_val_auc_pr
        marker = " *" if is_best else ""
        print(f"{epoch:<7} | {train_loss:<12.5f} | {val_loss:<12.5f} | {val_auc_pr:<12.4f} | {val_roc_auc:<12.4f} | {epoch_time:<6.1f}s{marker}")

        if is_best:
            best_val_auc_pr = val_auc_pr
            best_epoch = epoch
            epochs_without_improvement = 0
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auc_pr": best_val_auc_pr,
                "config": {
                    "num_tabular_features": len(MODEL_A_FEATURES),
                    "block_sequence_length": seq_len,
                    "pos_weight": pos_weight_val,
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
    history_csv = REPORTS_DIR / "model_c_training_history.csv"
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
    print("MODEL C EVALUATION ON CANONICAL DEV-VAL (ELIGIBLE DIES, OLD_LABEL == 0)")
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

    preds_path = REPORTS_DIR / "model_c_dev_val_predictions.parquet"
    val_meta.to_parquet(preds_path, index=False)
    print(f"\nSaved validation predictions to: {preds_path}")

    # Save metrics JSON & CSV
    metrics_dict = {
        "model_name": "Model C (Multi-Resolution 1D CNN + Tabular MLP)",
        "branch_1": "Raw 2,000-element block sequence (3-layer Conv1D + multi-scale pool)",
        "branch_2": f"519 tabular features ({len(MODEL_A_FEATURES)}: 500 parametric + 19 spatial)",
        "num_parameters": sum(p.numel() for p in model.parameters()),
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

    metrics_json_path = REPORTS_DIR / "model_c_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved metrics JSON: {metrics_json_path}")

    metrics_csv_path = REPORTS_DIR / "model_c_metrics.csv"
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
        "pos_weight": float(pos_weight_val),
        "best_epoch": int(best_epoch),
        "best_val_auc_pr": float(best_val_auc_pr),
    }
    with open(MODELS_DIR / "model_c_cnn_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # -------------------------------------------------------------------------
    # Step 6: Side-by-Side Comparison against Model A and Model B
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("BENCHMARK COMPARISON: MODEL A vs MODEL B-NO-SPATIAL vs MODEL B vs MODEL C")
    print("=" * 85)

    comparison_data = [
        {
            "Model": "Model A (Baseline)",
            "Features": "500 Parametric + 19 Spatial (519)",
            "Block Representation": "None",
            "AUC-PR": 0.4937,
            "ROC-AUC": 0.8318,
            "Tuned F1": 0.5200,
            "Precision": 0.9441,
            "Recall": 0.3589,
            "Threshold": 0.8127,
            "Training Time": "109.1s",
        },
        {
            "Model": "Model B-Without-Spatial",
            "Features": "500 Parametric + 36 Block (536)",
            "Block Representation": "36 Engineered Features",
            "AUC-PR": 0.5517,
            "ROC-AUC": 0.8732,
            "Tuned F1": 0.5392,
            "Precision": 0.7933,
            "Recall": 0.4084,
            "Threshold": 0.8033,
            "Training Time": "136.2s",
        },
        {
            "Model": "Model B (LightGBM Full)",
            "Features": "500 Param + 19 Spat + 36 Block (555)",
            "Block Representation": "36 Engineered Features",
            "AUC-PR": 0.5543,
            "ROC-AUC": 0.8760,
            "Tuned F1": 0.5397,
            "Precision": 0.8266,
            "Recall": 0.4006,
            "Threshold": 0.8176,
            "Training Time": "142.7s",
        },
        {
            "Model": "Model C (Multi-Res 1D CNN)",
            "Features": "519 Tabular + Raw 2,000 Seq",
            "Block Representation": "Learned 1D CNN (256-dim embedding)",
            "AUC-PR": float(val_auc_pr),
            "ROC-AUC": float(val_roc_auc),
            "Tuned F1": float(best_f1),
            "Precision": float(best_prec),
            "Recall": float(best_rec),
            "Threshold": float(best_thresh),
            "Training Time": f"{total_train_time:.1f}s",
        }
    ]

    df_comp = pd.DataFrame(comparison_data)
    print(df_comp.to_string(index=False))

    comp_csv_path = REPORTS_DIR / "model_comparison_a_b_c.csv"
    comp_json_path = REPORTS_DIR / "model_comparison_a_b_c.json"
    df_comp.to_csv(comp_csv_path, index=False)
    with open(comp_json_path, "w") as f:
        json.dump(comparison_data, f, indent=2)

    # -------------------------------------------------------------------------
    # Step 7: Generate Publication-Grade Visualizations
    # -------------------------------------------------------------------------
    print("\nGenerating evaluation figures...")

    # Figure 1: Training Curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df_history["epoch"], df_history["train_loss"], "o-", label="Train Loss (BCE)", color="#1f77b4")
    plt.plot(df_history["epoch"], df_history["val_loss"], "s--", label="Val Loss (BCE)", color="#ff7f0e")
    plt.title("Model C Loss Progression", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df_history["epoch"], df_history["val_auc_pr"], "d-", label="Val AUC-PR", color="#2ca02c")
    plt.plot(df_history["epoch"], df_history["val_roc_auc"], "^--", label="Val ROC-AUC", color="#9467bd")
    plt.axhline(0.5543, color="#d62728", linestyle=":", label="Model B Baseline (0.5543)")
    plt.title("Model C Validation Metrics", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    curve_fig_path = FIGURES_DIR / "model_c_training_curve.png"
    plt.savefig(curve_fig_path, dpi=300)
    plt.close()
    print(f"  Saved training curves: {curve_fig_path}")

    # Figure 2: Precision-Recall Curve Comparison
    plt.figure(figsize=(9, 7))

    # Load Model A predictions
    df_a_preds = pd.read_parquet(REPORTS_DIR / "model_a_dev_val_predictions.parquet")
    p_a, r_a, _ = precision_recall_curve(df_a_preds["label"], df_a_preds["predicted_probability"])
    plt.plot(r_a, p_a, label=f"Model A: Parametric + Spatial (AUC-PR = 0.4937)", color="#7f7f7f", linestyle="--", linewidth=1.8)

    # Load Model B-no-spatial predictions
    df_b_no_preds = pd.read_parquet(REPORTS_DIR / "model_b_no_spatial_dev_val_predictions.parquet")
    p_b_no, r_b_no, _ = precision_recall_curve(df_b_no_preds["label"], df_b_no_preds["predicted_probability"])
    plt.plot(r_b_no, p_b_no, label=f"Model B (No Spatial): Parametric + 36 Block (AUC-PR = 0.5517)", color="#ff7f0e", linestyle="-.", linewidth=1.8)

    # Load Model B predictions
    df_b_preds = pd.read_parquet(REPORTS_DIR / "model_b_dev_val_predictions.parquet")
    p_b, r_b, _ = precision_recall_curve(df_b_preds["label"], df_b_preds["predicted_probability"])
    plt.plot(r_b, p_b, label=f"Model B: Parametric + Spatial + 36 Block (AUC-PR = 0.5543)", color="#1f77b4", linewidth=2.2)

    # Model C predictions
    p_c, r_c, _ = precision_recall_curve(y_val, val_probs)
    plt.plot(r_c, p_c, label=f"Model C: 1D CNN Raw Seq + Tabular (AUC-PR = {val_auc_pr:.4f})", color="#2ca02c", linewidth=2.5)

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves: Model A vs B-no-spatial vs B vs C", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    pr_fig_path = FIGURES_DIR / "model_c_pr_comparison.png"
    plt.savefig(pr_fig_path, dpi=300)
    plt.close()
    print(f"  Saved PR comparison:   {pr_fig_path}")

    print("\n" + "=" * 85)
    print("MODEL C PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    main()
