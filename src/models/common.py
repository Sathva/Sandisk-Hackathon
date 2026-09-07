"""
Common Utilities, Feature Definitions, and Evaluation Routines for Model A and Model B.
Modular and reusable across baseline and advanced modeling experiments.
"""

import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
)

# -----------------------------------------------------------------------------
# 1. Feature Definitions
# -----------------------------------------------------------------------------
PARAMETRIC_FEATURES = [f"feature_{i}" for i in range(1, 501)]

SPATIAL_FEATURES = [
    "normalized_row",
    "normalized_col",
    "radius",
    "radius_squared",
    "distance_to_edge",
    "distance_to_nearest_old_failure",
    "wafer_die_count",
    "wafer_old_fail_count",
    "wafer_old_fail_rate",
    "old_fail_count_3x3",
    "old_fail_density_3x3",
    "old_fail_count_5x5",
    "old_fail_density_5x5",
    "old_fail_count_7x7",
    "old_fail_density_7x7",
    "old_fail_count_9x9",
    "old_fail_density_9x9",
    "old_fail_count_11x11",
    "old_fail_density_11x11",
]

BLOCK_FEATURES = [
    "block_mean",
    "block_std",
    "block_min",
    "block_max",
    "block_range",
    "block_median",
    "block_q01",
    "block_q05",
    "block_q25",
    "block_q50",
    "block_q75",
    "block_q95",
    "block_q99",
    "block_mean_top10",
    "block_mean_top50",
    "block_mean_top100",
    "block_mean_top200",
    "block_mean_bottom10",
    "block_mean_bottom50",
    "block_mean_bottom100",
    "block_mean_bottom200",
    "block_count_z_gt_2",
    "block_count_z_gt_3",
    "block_count_z_gt_4",
    "block_max_z",
    "block_count_mad_gt_3",
    "block_max_mad_deviation",
    "max_rolling_mean_50",
    "max_rolling_mean_100",
    "max_rolling_mean_200",
    "max_rolling_mean_400",
    "max_rolling_mean_100_start_idx",
    "max_rolling_std_50",
    "max_rolling_std_100",
    "max_rolling_std_200",
    "largest_contiguous_anomaly_run",
]

# Model feature subsets
MODEL_A_FEATURES = PARAMETRIC_FEATURES + SPATIAL_FEATURES  # 519 features
MODEL_B_FEATURES = PARAMETRIC_FEATURES + SPATIAL_FEATURES + BLOCK_FEATURES  # 555 features

IDENTIFIER_COLS = ["wafer_id", "die_row", "die_col"]
LABEL_COLS = ["old_label", "label"]


# -----------------------------------------------------------------------------
# 2. Data Loading Helpers
# -----------------------------------------------------------------------------
def load_dataset(parquet_path, feature_cols, eligible_only=True):
    """
    Loads parquet dataset with memory-efficient column projection.
    Optionally filters strictly to eligible dies (old_label == 0).

    Returns:
        X (pd.DataFrame): feature matrix
        y (pd.Series): binary target label
        meta (pd.DataFrame): wafer_id, die_row, die_col, old_label, label
    """
    cols_to_load = IDENTIFIER_COLS + LABEL_COLS + feature_cols
    print(f"Loading {parquet_path.name} ({len(cols_to_load)} columns)...")
    t0 = time.time()
    df = pd.read_parquet(parquet_path, columns=cols_to_load)
    load_time = time.time() - t0
    print(f"   Loaded {len(df):,} total dies in {load_time:.2f}s.")

    total_rows = len(df)
    n_wafers = df["wafer_id"].nunique()

    if eligible_only:
        mask = (df["old_label"] == 0)
        df_eligible = df[mask].reset_index(drop=True)
    else:
        df_eligible = df.reset_index(drop=True)

    eligible_rows = len(df_eligible)
    n_pos = int((df_eligible["label"] == 1).sum())
    n_neg = eligible_rows - n_pos
    pos_rate = (n_pos / eligible_rows * 100) if eligible_rows > 0 else 0.0

    print(f"   Wafers: {n_wafers} | Total Dies: {total_rows:,} | Eligible Dies (old_label==0): {eligible_rows:,}")
    print(f"   Eligible Negatives (Pass): {n_neg:,} ({100 - pos_rate:.2f}%)")
    print(f"   Eligible Positives (Newly Failed): {n_pos:,} ({pos_rate:.3f}%)")

    X = df_eligible[feature_cols]
    y = df_eligible["label"].astype(np.int8)
    meta = df_eligible[IDENTIFIER_COLS + LABEL_COLS]

    return X, y, meta, {
        "total_rows": total_rows,
        "eligible_rows": eligible_rows,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": pos_rate,
        "n_wafers": n_wafers,
        "load_time_s": load_time,
    }


# -----------------------------------------------------------------------------
# 3. Evaluation Metrics & Confusion Matrix
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes comprehensive binary classification metrics on the eligible population.
    y_true: binary ground truth (1 = new fail, 0 = stay pass)
    y_prob: continuous predicted failure probability
    threshold: decision boundary for positive class
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    # Threshold predictions
    y_pred = (y_prob_arr >= threshold).astype(int)

    # Ranking metrics (independent of threshold)
    auc_pr = float(average_precision_score(y_true_arr, y_prob_arr))
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    # Confusion matrix:
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Classification metrics
    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    # Negative-class (Pass) metrics
    pass_prec = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    pass_rec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    pass_f1 = float(2 * pass_prec * pass_rec / (pass_prec + pass_rec)) if (pass_prec + pass_rec) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "auc_pr": auc_pr,
        "roc_auc": roc_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pass_precision": pass_prec,
        "pass_recall": pass_rec,
        "pass_f1": pass_f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "confusion_matrix": cm.tolist(),
    }


def compute_predict_all_pass_baseline(y_true):
    """
    Baseline predicting all eligible dies pass (pred = 0).
    Quantifies the class imbalance trap.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    n_total = len(y_true_arr)
    n_pos = int(np.sum(y_true_arr))
    n_neg = n_total - n_pos

    # Predictions are identically 0
    y_prob = np.zeros(n_total, dtype=float)
    y_pred = np.zeros(n_total, dtype=int)

    pos_rate = n_pos / n_total if n_total > 0 else 0.0
    acc = n_neg / n_total if n_total > 0 else 0.0

    return {
        "baseline_name": "Predict-All-Pass (Trivial Majority Class)",
        "threshold": 0.5,
        "auc_pr": float(pos_rate),  # No-skill baseline AUC-PR equals positive rate
        "roc_auc": 0.5,
        "accuracy": float(acc),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "pass_precision": float(acc),
        "pass_recall": 1.0,
        "pass_f1": float(2 * acc / (acc + 1.0)) if acc > 0 else 0.0,
        "tp": 0,
        "fp": 0,
        "fn": n_pos,
        "tn": n_neg,
        "confusion_matrix": [[n_neg, 0], [n_pos, 0]],
    }


# -----------------------------------------------------------------------------
# 4. Threshold Tuning
# -----------------------------------------------------------------------------
def tune_threshold(y_true, y_prob, num_steps=200):
    """
    Finds the optimal probability threshold T* on dev_val that maximizes positive-class F1.
    Evaluates candidate thresholds across [0.01, 0.99].
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    thresholds = np.linspace(0.01, 0.99, num_steps)
    best_thresh = 0.5
    best_f1 = -1.0
    best_metrics = None

    history = []

    for t in thresholds:
        y_pred = (y_prob_arr >= t).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true_arr == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true_arr == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true_arr == 1)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        history.append({"threshold": float(t), "precision": float(prec), "recall": float(rec), "f1": float(f1)})

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    # Compute full metrics at best threshold
    best_metrics = compute_metrics(y_true_arr, y_prob_arr, threshold=best_thresh)

    return {
        "best_threshold": best_thresh,
        "best_f1": best_f1,
        "metrics": best_metrics,
        "history": history,
    }


# -----------------------------------------------------------------------------
# 5. Formatted Reporting Table
# -----------------------------------------------------------------------------
def format_confusion_table(metrics_dict):
    """
    Generates the exact confusion table specified in the hackathon guidelines.
    """
    tp = metrics_dict["tp"]
    fp = metrics_dict["fp"]
    fn = metrics_dict["fn"]
    tn = metrics_dict["tn"]
    fail_rec = metrics_dict["recall"]
    pass_rec = metrics_dict["pass_recall"]

    header = f"{'':15s} {'Pred Fail':>10s}  {'Pred Pass':>10s}       {'Metric':<20s} {'Value':>10s}"
    row1 = f"{'Actual Fail':15s} {tp:10d}  {fn:10d}       {'Fail Accuracy (Rec)':<20s} {fail_rec:10.6f}"
    row2 = f"{'Actual Pass':15s} {fp:10d}  {tn:10d}       {'Pass Accuracy (Rec)':<20s} {pass_rec:10.6f}"

    return "\n".join([header, row1, row2])
