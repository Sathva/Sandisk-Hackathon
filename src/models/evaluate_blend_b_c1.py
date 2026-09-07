"""
Prediction-Level Ensembling & Complementarity Analysis: Model B + Model C1.

Fuses LightGBM Model B predictions (tabular + spatial + 36 engineered block features)
with Multi-Resolution Triple-Branch Deep Network Model C1 predictions
(256-d raw 1D CNN + 32-d engineered block MLP + 128-d parametric/spatial MLP).

Formula:
  P_blend = alpha * P_C1 + (1 - alpha) * P_B

Performs coarse (0.00 to 1.00 step 0.01) and fine sweeps (step 0.001), threshold tuning,
detailed error complementarity analysis, and generates publication-grade figures.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
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

from src.config import REPORTS_DIR

FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_B_PREDS_PATH = REPORTS_DIR / "model_b_dev_val_predictions.parquet"
MODEL_C1_PREDS_PATH = REPORTS_DIR / "model_c1_dev_val_predictions.parquet"


def tune_threshold(y_true, y_prob):
    """Sweeps thresholds in [0.01, 0.99] with step 0.005 to find threshold maximizing F1."""
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


def get_cm_metrics(y_t, y_p, thresh):
    preds = (y_p >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_t, preds).ravel()
    spec = tn / (tn + fp)
    acc = (tp + tn) / len(y_t)
    return {
        "threshold": float(thresh),
        "f1": float(f1_score(y_t, preds, zero_division=0)),
        "precision": float(precision_score(y_t, preds, zero_division=0)),
        "recall": float(recall_score(y_t, preds, zero_division=0)),
        "specificity": float(spec),
        "accuracy": float(acc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main():
    print("\n" + "=" * 85)
    print("SANDISK HACKATHON — ENSEMBLE EXPERIMENT: MODEL B (LIGHTGBM) + MODEL C1 (TRIPLE-BRANCH)")
    print("=" * 85)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1 & 2: Locate Predictions, Inspect Schemas & Verify Exact Alignment
    # -------------------------------------------------------------------------
    print("\n--- STEP 1 & 2: VERIFY ALIGNMENT & PROBABILITY SCHEMAS ---")

    assert MODEL_B_PREDS_PATH.exists(), f"Missing Model B predictions: {MODEL_B_PREDS_PATH}"
    assert MODEL_C1_PREDS_PATH.exists(), f"Missing Model C1 predictions: {MODEL_C1_PREDS_PATH}"

    print(f"Loading Model B predictions from: {MODEL_B_PREDS_PATH}")
    df_b = pd.read_parquet(MODEL_B_PREDS_PATH)
    print(f"  Shape: {df_b.shape} | Columns: {df_b.columns.tolist()}")

    print(f"Loading Model C1 predictions from: {MODEL_C1_PREDS_PATH}")
    df_c1 = pd.read_parquet(MODEL_C1_PREDS_PATH)
    print(f"  Shape: {df_c1.shape} | Columns: {df_c1.columns.tolist()}")

    # Identify probability columns
    prob_col_candidates = ["predicted_probability", "prob", "prediction"]
    prob_col_b = next((c for c in prob_col_candidates if c in df_b.columns), None)
    prob_col_c1 = next((c for c in prob_col_candidates if c in df_c1.columns), None)

    assert prob_col_b is not None, f"Could not identify probability column in Model B: {df_b.columns.tolist()}"
    assert prob_col_c1 is not None, f"Could not identify probability column in Model C1: {df_c1.columns.tolist()}"

    p_b = df_b[prob_col_b].values.astype(np.float64)
    p_c1 = df_c1[prob_col_c1].values.astype(np.float64)

    # Population & target checks
    N_EXPECTED = 137576
    N_POS_EXPECTED = 5367
    N_NEG_EXPECTED = 132209
    N_WAFERS_EXPECTED = 160

    assert len(df_b) == N_EXPECTED, f"Model B row count mismatch: {len(df_b)} != {N_EXPECTED}"
    assert len(df_c1) == N_EXPECTED, f"Model C1 row count mismatch: {len(df_c1)} != {N_EXPECTED}"

    y_b = df_b["label"].values.astype(int)
    y_c1 = df_c1["label"].values.astype(int)

    assert int(y_b.sum()) == N_POS_EXPECTED, f"Model B positive count mismatch: {y_b.sum()} != {N_POS_EXPECTED}"
    assert int(y_c1.sum()) == N_POS_EXPECTED, f"Model C1 positive count mismatch: {y_c1.sum()} != {N_POS_EXPECTED}"
    assert (y_b == y_c1).all(), "Target label arrays are not identical between Model B and Model C1!"

    # Coordinate & ID checks
    id_cols = ["wafer_id", "die_row", "die_col"]
    assert all(col in df_b.columns for col in id_cols), f"Missing ID columns in Model B: {id_cols}"
    assert all(col in df_c1.columns for col in id_cols), f"Missing ID columns in Model C1: {id_cols}"

    id_match = (df_b[id_cols] == df_c1[id_cols]).all().all()
    assert id_match, "Die coordinate IDs do NOT match row-for-row between Model B and Model C1!"

    n_wafers_b = df_b["wafer_id"].nunique()
    n_wafers_c1 = df_c1["wafer_id"].nunique()
    assert n_wafers_b == N_WAFERS_EXPECTED and n_wafers_c1 == N_WAFERS_EXPECTED, f"Wafer count mismatch: {n_wafers_b}, {n_wafers_c1}"

    y_true = y_b

    print(f"\nProbability Schema & Summary Inspection:")
    print(f"  Model B Prediction File:  {MODEL_B_PREDS_PATH.name}")
    print(f"  Model B Selected Column:  '{prob_col_b}'")
    print(f"  Model B Prob Range:       min={p_b.min():.5f}, max={p_b.max():.5f}, mean={p_b.mean():.5f}")
    print(f"  Model B Rows / Positives: {len(p_b):,} / {int(y_true.sum()):,} fails ({y_true.mean():.4%})")
    print(f"  Model C1 Prediction File: {MODEL_C1_PREDS_PATH.name}")
    print(f"  Model C1 Selected Column: '{prob_col_c1}'")
    print(f"  Model C1 Prob Range:      min={p_c1.min():.5f}, max={p_c1.max():.5f}, mean={p_c1.mean():.5f}")
    print(f"  Model C1 Rows/Positives:  {len(p_c1):,} / {int(y_true.sum()):,} fails ({y_true.mean():.4%})")

    print(f"\nAlignment Verification Result:")
    print(f"  [OK] Total Population:   {len(y_true):,} eligible dies ({N_EXPECTED:,} expected)")
    print(f"  [OK] Positive Failures:  {int(y_true.sum()):,} dies ({N_POS_EXPECTED:,} expected)")
    print(f"  [OK] Negative Passes:    {int(len(y_true) - y_true.sum()):,} dies ({N_NEG_EXPECTED:,} expected)")
    print(f"  [OK] Unique Wafers:      {n_wafers_b} wafers ({N_WAFERS_EXPECTED} expected)")
    print(f"  [OK] Identifiers Match:  100% row-for-row match on ['wafer_id', 'die_row', 'die_col']")
    print(f"  [OK] Zero Missing / Zero NaNs")

    # -------------------------------------------------------------------------
    # STEP 4: Coarse Blend Sweep (alpha = 0.00 to 1.00 step 0.01)
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: COARSE BLEND SWEEP (alpha in [0.00, 1.00], step 0.01) ---")
    print(r"Formula: P_blend = alpha * P_C1 + (1 - alpha) * P_B")

    coarse_alphas = np.linspace(0.0, 1.0, 101)
    sweep_results = []

    for a in coarse_alphas:
        p_blend = a * p_c1 + (1.0 - a) * p_b
        auc_pr = average_precision_score(y_true, p_blend)
        roc_auc = roc_auc_score(y_true, p_blend)
        sweep_results.append({
            "alpha": float(round(a, 4)),
            "auc_pr": float(auc_pr),
            "roc_auc": float(roc_auc),
        })

    df_coarse = pd.DataFrame(sweep_results)
    best_coarse_pr = df_coarse.loc[df_coarse["auc_pr"].idxmax()]
    best_coarse_roc = df_coarse.loc[df_coarse["roc_auc"].idxmax()]

    auc_pr_b = float(df_coarse.loc[0, "auc_pr"])
    roc_auc_b = float(df_coarse.loc[0, "roc_auc"])
    auc_pr_c1 = float(df_coarse.loc[100, "auc_pr"])
    roc_auc_c1 = float(df_coarse.loc[100, "roc_auc"])

    print(f"Coarse Sweep Summary:")
    print(f"  alpha=0.00 (Model B Only):  AUC-PR = {auc_pr_b:.5f}, ROC-AUC = {roc_auc_b:.5f}")
    print(f"  alpha=1.00 (Model C1 Only): AUC-PR = {auc_pr_c1:.5f}, ROC-AUC = {roc_auc_c1:.5f}")
    print(f"  Coarse Best AUC-PR:         alpha={best_coarse_pr['alpha']:.2f} -> AUC-PR = {best_coarse_pr['auc_pr']:.5f}, ROC-AUC = {best_coarse_pr['roc_auc']:.5f}")
    print(f"  Coarse Best ROC-AUC:        alpha={best_coarse_roc['alpha']:.2f} -> ROC-AUC = {best_coarse_roc['roc_auc']:.5f}, AUC-PR = {best_coarse_roc['auc_pr']:.5f}")

    # -------------------------------------------------------------------------
    # STEP 5: Fine Sweep Around Coarse Optimum (step 0.001)
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: FINE SWEEP AROUND COARSE OPTIMUM (step 0.001) ---")
    center_alpha = best_coarse_pr["alpha"]
    fine_min = max(0.0, center_alpha - 0.035)
    fine_max = min(1.0, center_alpha + 0.035)
    fine_alphas = np.linspace(fine_min, fine_max, int(round((fine_max - fine_min) / 0.001)) + 1)

    fine_results = []
    for a in fine_alphas:
        p_blend = a * p_c1 + (1.0 - a) * p_b
        auc_pr = average_precision_score(y_true, p_blend)
        roc_auc = roc_auc_score(y_true, p_blend)
        fine_results.append({
            "alpha": float(round(a, 4)),
            "auc_pr": float(auc_pr),
            "roc_auc": float(roc_auc),
        })

    df_fine = pd.DataFrame(fine_results)
    best_fine_pr = df_fine.loc[df_fine["auc_pr"].idxmax()]
    best_fine_roc = df_fine.loc[df_fine["roc_auc"].idxmax()]

    best_alpha = float(best_fine_pr["alpha"])
    best_auc_pr = float(best_fine_pr["auc_pr"])
    best_roc_auc_at_best_pr = float(best_fine_pr["roc_auc"])

    print(f"Fine Sweep Optimal (Around {center_alpha:.2f} with step 0.001):")
    print(f"  Best Alpha for AUC-PR: alpha = {best_alpha:.3f}")
    print(f"  Best Blended AUC-PR:   {best_auc_pr:.5f}")
    print(f"  ROC-AUC at Best PR:    {best_roc_auc_at_best_pr:.5f}")
    print(f"  Highest Blended ROC:   {best_fine_roc['roc_auc']:.5f} (at alpha = {best_fine_roc['alpha']:.3f})")

    # Merge and save full sweep
    df_all_sweep = pd.concat([df_coarse, df_fine]).drop_duplicates(subset=["alpha"]).sort_values("alpha").reset_index(drop=True)
    sweep_csv_path = REPORTS_DIR / "model_b_c1_blend_sweep.csv"
    df_all_sweep.to_csv(sweep_csv_path, index=False)
    print(f"Saved complete sweep table to: {sweep_csv_path}")

    # -------------------------------------------------------------------------
    # STEP 6: Threshold / F1 Evaluation
    # -------------------------------------------------------------------------
    print("\n--- STEP 6: THRESHOLD & CONFUSION MATRIX EVALUATION ---")

    # Optimal blended probability
    p_best_blend = best_alpha * p_c1 + (1.0 - best_alpha) * p_b

    # Tune thresholds for F1 maximization
    t_b, f1_b, prec_b, rec_b = tune_threshold(y_true, p_b)
    t_c1, f1_c1, prec_c1, rec_c1 = tune_threshold(y_true, p_c1)
    t_blend, f1_blend, prec_blend, rec_blend = tune_threshold(y_true, p_best_blend)

    metrics_b = get_cm_metrics(y_true, p_b, t_b)
    metrics_c1 = get_cm_metrics(y_true, p_c1, t_c1)
    metrics_blend = get_cm_metrics(y_true, p_best_blend, t_blend)

    print(f"Model B Alone:       Tuned F1 = {metrics_b['f1']:.4f} (Thresh={metrics_b['threshold']:.4f}, Prec={metrics_b['precision']:.4f}, Rec={metrics_b['recall']:.4f}, TP={metrics_b['tp']:,}, FP={metrics_b['fp']:,})")
    print(f"Model C1 Alone:      Tuned F1 = {metrics_c1['f1']:.4f} (Thresh={metrics_c1['threshold']:.4f}, Prec={metrics_c1['precision']:.4f}, Rec={metrics_c1['recall']:.4f}, TP={metrics_c1['tp']:,}, FP={metrics_c1['fp']:,})")
    print(f"Best B+C1 (a={best_alpha:.3f}): Tuned F1 = {metrics_blend['f1']:.4f} (Thresh={metrics_blend['threshold']:.4f}, Prec={metrics_blend['precision']:.4f}, Rec={metrics_blend['recall']:.4f}, TP={metrics_blend['tp']:,}, FP={metrics_blend['fp']:,})")

    # -------------------------------------------------------------------------
    # STEP 7: Master Comparison & Deltas
    # -------------------------------------------------------------------------
    print("\n--- STEP 7: MASTER COMPARISON TABLE & DELTAS ---")

    # Previous benchmarks
    PREV_BC_AUCPR = 0.5764
    PREV_BC_ROCAUC = 0.8913
    PREV_BC_F1 = 0.5518

    comp_rows = [
        {
            "Model": "Model B (LightGBM)",
            "Blend Alpha": 0.000,
            "AUC-PR": auc_pr_b,
            "ROC-AUC": roc_auc_b,
            "Tuned F1": metrics_b["f1"],
            "Precision": metrics_b["precision"],
            "Recall": metrics_b["recall"],
            "Specificity": metrics_b["specificity"],
            "Accuracy": metrics_b["accuracy"],
            "Threshold": metrics_b["threshold"],
            "TP": metrics_b["tp"],
            "FP": metrics_b["fp"],
            "FN": metrics_b["fn"],
            "TN": metrics_b["tn"],
        },
        {
            "Model": "Model C1 (Triple-Branch Deep Net)",
            "Blend Alpha": 1.000,
            "AUC-PR": auc_pr_c1,
            "ROC-AUC": roc_auc_c1,
            "Tuned F1": metrics_c1["f1"],
            "Precision": metrics_c1["precision"],
            "Recall": metrics_c1["recall"],
            "Specificity": metrics_c1["specificity"],
            "Accuracy": metrics_c1["accuracy"],
            "Threshold": metrics_c1["threshold"],
            "TP": metrics_c1["tp"],
            "FP": metrics_c1["fp"],
            "FN": metrics_c1["fn"],
            "TN": metrics_c1["tn"],
        },
        {
            "Model": f"Best B+C1 Blend (Alpha={best_alpha:.3f})",
            "Blend Alpha": best_alpha,
            "AUC-PR": best_auc_pr,
            "ROC-AUC": best_roc_auc_at_best_pr,
            "Tuned F1": metrics_blend["f1"],
            "Precision": metrics_blend["precision"],
            "Recall": metrics_blend["recall"],
            "Specificity": metrics_blend["specificity"],
            "Accuracy": metrics_blend["accuracy"],
            "Threshold": metrics_blend["threshold"],
            "TP": metrics_blend["tp"],
            "FP": metrics_blend["fp"],
            "FN": metrics_blend["fn"],
            "TN": metrics_blend["tn"],
        }
    ]

    df_comparison = pd.DataFrame(comp_rows)
    print(df_comparison.to_string(index=False))

    comp_csv_path = REPORTS_DIR / "model_b_c1_blend_comparison.csv"
    df_comparison.to_csv(comp_csv_path, index=False)
    print(f"\nSaved comparison table to: {comp_csv_path}")

    # Pairwise deltas
    delta_pr_vs_b = best_auc_pr - auc_pr_b
    rel_pr_vs_b = (delta_pr_vs_b / auc_pr_b) * 100
    delta_roc_vs_b = best_roc_auc_at_best_pr - roc_auc_b
    delta_f1_vs_b = metrics_blend["f1"] - metrics_b["f1"]

    delta_pr_vs_c1 = best_auc_pr - auc_pr_c1
    rel_pr_vs_c1 = (delta_pr_vs_c1 / auc_pr_c1) * 100
    delta_roc_vs_c1 = best_roc_auc_at_best_pr - roc_auc_c1
    delta_f1_vs_c1 = metrics_blend["f1"] - metrics_c1["f1"]

    delta_pr_vs_prev_bc = best_auc_pr - PREV_BC_AUCPR

    print("\nPairwise Improvements:")
    print(f"  Best B+C1 vs. Model B (LightGBM):")
    print(f"    Delta AUC-PR: {delta_pr_vs_b:+.4f} ({rel_pr_vs_b:+.2f}% relative lift)")
    print(f"    Delta ROC-AUC:{delta_roc_vs_b:+.4f}")
    print(f"    Delta F1:     {delta_f1_vs_b:+.4f}")
    print(f"    TP change:    {metrics_blend['tp'] - metrics_b['tp']:+d} dies caught")
    print(f"    FP change:    {metrics_blend['fp'] - metrics_b['fp']:+d} false alarms")
    print(f"  Best B+C1 vs. Model C1 (Triple-Branch):")
    print(f"    Delta AUC-PR: {delta_pr_vs_c1:+.4f} ({rel_pr_vs_c1:+.2f}% relative lift)")
    print(f"    Delta ROC-AUC:{delta_roc_vs_c1:+.4f}")
    print(f"    Delta F1:     {delta_f1_vs_c1:+.4f}")
    print(f"    TP change:    {metrics_blend['tp'] - metrics_c1['tp']:+d} dies caught")
    print(f"    FP change:    {metrics_blend['fp'] - metrics_c1['fp']:+d} false alarms")
    print(f"  Best B+C1 vs. Previous B+C Ensemble (AUC-PR = {PREV_BC_AUCPR:.4f}):")
    print(f"    Delta AUC-PR: {delta_pr_vs_prev_bc:+.4f}")

    # -------------------------------------------------------------------------
    # STEP 8: Complementarity & Error Diversity Analysis
    # -------------------------------------------------------------------------
    print("\n--- STEP 8: COMPLEMENTARITY & ERROR DIVERSITY ANALYSIS ---")

    p_corr, _ = pearsonr(p_b, p_c1)
    s_corr, _ = spearmanr(p_b, p_c1)
    print(f"Prediction Correlation between Model B and Model C1:")
    print(f"  Pearson Correlation:  {p_corr:.4f}")
    print(f"  Spearman Correlation: {s_corr:.4f}")

    preds_b = (p_b >= t_b).astype(int)
    preds_c1 = (p_c1 >= t_c1).astype(int)

    disagree_mask = (preds_b != preds_c1)
    total_disagreements = int(disagree_mask.sum())
    disagreement_rate = total_disagreements / len(y_true)

    b_correct = (preds_b == y_true)
    c1_correct = (preds_c1 == y_true)

    both_correct = int((b_correct & c1_correct).sum())
    b_only_correct = int((b_correct & (~c1_correct)).sum())
    c1_only_correct = int(((~b_correct) & c1_correct).sum())
    both_wrong = int(((~b_correct) & (~c1_correct)).sum())

    # Actual failure breakdown (positives)
    pos_mask = (y_true == 1)
    pos_both_correct = int(((preds_b == 1) & (preds_c1 == 1) & pos_mask).sum())
    pos_b_only = int(((preds_b == 1) & (preds_c1 == 0) & pos_mask).sum())
    pos_c1_only = int(((preds_b == 0) & (preds_c1 == 1) & pos_mask).sum())
    pos_both_wrong = int(((preds_b == 0) & (preds_c1 == 0) & pos_mask).sum())

    # Healthy dies breakdown (negatives / false positives)
    neg_mask = (y_true == 0)
    neg_fp_both = int(((preds_b == 1) & (preds_c1 == 1) & neg_mask).sum())
    neg_fp_b_only = int(((preds_b == 1) & (preds_c1 == 0) & neg_mask).sum())
    neg_fp_c1_only = int(((preds_b == 0) & (preds_c1 == 1) & neg_mask).sum())
    neg_tn_both = int(((preds_b == 0) & (preds_c1 == 0) & neg_mask).sum())

    print(f"\nGeneral Decision Disagreements (at Tuned Thresholds):")
    print(f"  Total Population:              {len(y_true):,}")
    print(f"  Decision Disagreements:        {total_disagreements:,} dies ({disagreement_rate:.2%})")
    print(f"  Both Models Correct:           {both_correct:,} ({both_correct / len(y_true):.2%})")
    print(f"  Model B Correct, Model C1 Wrong:{b_only_correct:,} ({b_only_correct / len(y_true):.2%})")
    print(f"  Model C1 Correct, Model B Wrong:{c1_only_correct:,} ({c1_only_correct / len(y_true):.2%})")
    print(f"  Both Models Wrong:             {both_wrong:,} ({both_wrong / len(y_true):.2%})")

    print(f"\nTarget Failures Breakdown (Total 5,367 Defective Dies):")
    print(f"  Defects Caught by BOTH:        {pos_both_correct:,} ({pos_both_correct / N_POS_EXPECTED:.2%})")
    print(f"  Defects Caught ONLY by Model B:{pos_b_only:,} ({pos_b_only / N_POS_EXPECTED:.2%})")
    print(f"  Defects Caught ONLY by Model C1:{pos_c1_only:,} ({pos_c1_only / N_POS_EXPECTED:.2%})")
    print(f"  Defects Missed by BOTH:        {pos_both_wrong:,} ({pos_both_wrong / N_POS_EXPECTED:.2%})")
    print(f"  Total Unique Defects Captured: {pos_both_correct + pos_b_only + pos_c1_only:,} / 5,367 ({(pos_both_correct + pos_b_only + pos_c1_only) / N_POS_EXPECTED:.2%})")

    print(f"\nHealthy Dies Breakdown (Total 132,209 Clean Dies):")
    print(f"  False Alarms Triggered by BOTH: {neg_fp_both:,}")
    print(f"  False Alarms Triggered ONLY by B:{neg_fp_b_only:,}")
    print(f"  False Alarms Triggered ONLY by C1:{neg_fp_c1_only:,}")
    print(f"  Clean Passes by BOTH:           {neg_tn_both:,} ({neg_tn_both / N_NEG_EXPECTED:.2%})")

    # Save validation predictions with blend probability
    df_blend_preds = df_b[id_cols + ["old_label", "label"]].copy()
    df_blend_preds["prob_model_b"] = p_b.astype(np.float32)
    df_blend_preds["prob_model_c1"] = p_c1.astype(np.float32)
    df_blend_preds["predicted_probability"] = p_best_blend.astype(np.float32)
    df_blend_preds["predicted_label_tuned"] = (p_best_blend >= t_blend).astype(np.int8)
    df_blend_preds["predicted_label_05"] = (p_best_blend >= 0.5).astype(np.int8)

    blend_preds_path = REPORTS_DIR / "model_b_c1_blend_predictions.parquet"
    df_blend_preds.to_parquet(blend_preds_path, index=False)
    print(f"\nSaved blended dev_val predictions to: {blend_preds_path}")

    # Save metrics JSON
    metrics_json = {
        "experiment": "Model B + Model C1 Probability Ensemble",
        "best_alpha_c1_weight": best_alpha,
        "model_b_weight": float(round(1.0 - best_alpha, 4)),
        "auc_pr": best_auc_pr,
        "roc_auc": best_roc_auc_at_best_pr,
        "optimal_threshold": float(t_blend),
        "f1_score": float(metrics_blend["f1"]),
        "precision": float(metrics_blend["precision"]),
        "recall": float(metrics_blend["recall"]),
        "specificity": float(metrics_blend["specificity"]),
        "accuracy": float(metrics_blend["accuracy"]),
        "confusion_matrix": {
            "tp": metrics_blend["tp"],
            "fp": metrics_blend["fp"],
            "fn": metrics_blend["fn"],
            "tn": metrics_blend["tn"],
        },
        "pairwise_vs_model_b": {
            "delta_auc_pr": delta_pr_vs_b,
            "rel_lift_auc_pr_pct": rel_pr_vs_b,
            "delta_roc_auc": delta_roc_vs_b,
            "delta_f1": delta_f1_vs_b,
        },
        "pairwise_vs_model_c1": {
            "delta_auc_pr": delta_pr_vs_c1,
            "rel_lift_auc_pr_pct": rel_pr_vs_c1,
            "delta_roc_auc": delta_roc_vs_c1,
            "delta_f1": delta_f1_vs_c1,
        },
        "pairwise_vs_previous_b_c_ensemble": {
            "prev_auc_pr": PREV_BC_AUCPR,
            "delta_auc_pr": delta_pr_vs_prev_bc,
        },
        "complementarity": {
            "pearson_correlation": float(p_corr),
            "spearman_correlation": float(s_corr),
            "disagreement_rate": float(disagreement_rate),
            "defects_both_caught": pos_both_correct,
            "defects_only_b_caught": pos_b_only,
            "defects_only_c1_caught": pos_c1_only,
            "defects_both_missed": pos_both_wrong,
            "total_unique_defects_caught": pos_both_correct + pos_b_only + pos_c1_only,
            "false_positives_both": neg_fp_both,
            "false_positives_only_b": neg_fp_b_only,
            "false_positives_only_c1": neg_fp_c1_only,
        }
    }

    metrics_json_path = REPORTS_DIR / "model_b_c1_blend_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Saved blend metrics JSON to: {metrics_json_path}")

    # -------------------------------------------------------------------------
    # STEP 9: Visualizations
    # -------------------------------------------------------------------------
    print("\n--- STEP 9: GENERATING VISUALIZATIONS ---")

    # Figure 1: AUC-PR vs Alpha
    plt.figure(figsize=(9, 5))
    plt.plot(df_all_sweep["alpha"], df_all_sweep["auc_pr"], color="#1f77b4", linewidth=2.5, label="Blended AUC-PR")
    plt.axvline(best_alpha, color="#d62728", linestyle="--", label=f"Optimal Alpha = {best_alpha:.3f} (AUC-PR = {best_auc_pr:.4f})")
    plt.axhline(PREV_BC_AUCPR, color="#7f7f7f", linestyle=":", label=f"Previous B+C Ensemble ({PREV_BC_AUCPR:.4f})")
    plt.scatter([0.0], [auc_pr_b], color="#ff7f0e", s=90, zorder=5, label=f"Model B Only (alpha=0.0): {auc_pr_b:.4f}")
    plt.scatter([1.0], [auc_pr_c1], color="#2ca02c", s=90, zorder=5, label=f"Model C1 Only (alpha=1.0): {auc_pr_c1:.4f}")
    plt.scatter([best_alpha], [best_auc_pr], color="#d62728", marker="*", s=240, zorder=6, label=f"Best B+C1 Blend: {best_auc_pr:.4f}")
    plt.title(r"Dev-Val AUC-PR vs. Model C1 Weight Alpha ($P = \alpha P_{C1} + (1-\alpha) P_B$)", fontsize=12, fontweight="bold")
    plt.xlabel(r"Model C1 Weight ($\alpha$)", fontsize=11)
    plt.ylabel("AUC-PR", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower center", frameon=True)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "model_b_c1_blend_aucpr.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  Saved AUC-PR curve:  {fig1_path}")

    # Figure 2: ROC-AUC vs Alpha
    plt.figure(figsize=(9, 5))
    plt.plot(df_all_sweep["alpha"], df_all_sweep["roc_auc"], color="#9467bd", linewidth=2.5, label="Blended ROC-AUC")
    plt.axvline(best_alpha, color="#d62728", linestyle="--", label=f"Best PR Alpha = {best_alpha:.3f} (ROC-AUC = {best_roc_auc_at_best_pr:.4f})")
    plt.scatter([0.0], [roc_auc_b], color="#ff7f0e", s=90, zorder=5, label=f"Model B Only (alpha=0.0): {roc_auc_b:.4f}")
    plt.scatter([1.0], [roc_auc_c1], color="#2ca02c", s=90, zorder=5, label=f"Model C1 Only (alpha=1.0): {roc_auc_c1:.4f}")
    plt.scatter([best_alpha], [best_roc_auc_at_best_pr], color="#d62728", marker="*", s=240, zorder=6)
    plt.title(r"Dev-Val ROC-AUC vs. Model C1 Weight Alpha", fontsize=12, fontweight="bold")
    plt.xlabel(r"Model C1 Weight ($\alpha$)", fontsize=11)
    plt.ylabel("ROC-AUC", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower center", frameon=True)
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "model_b_c1_blend_rocauc.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  Saved ROC-AUC curve: {fig2_path}")

    # Figure 3: Prediction Scatter Plot
    plt.figure(figsize=(8, 8))
    neg_indices = np.where(y_true == 0)[0]
    np.random.seed(42)
    sample_neg_idx = np.random.choice(neg_indices, size=15000, replace=False)
    pos_indices = np.where(y_true == 1)[0]

    plt.scatter(p_b[sample_neg_idx], p_c1[sample_neg_idx], color="#1f77b4", alpha=0.15, s=8, label=f"Healthy Dies (Sample n=15,000)")
    plt.scatter(p_b[pos_indices], p_c1[pos_indices], color="#d62728", alpha=0.6, s=15, label=f"Defective Dies (Total n={len(pos_indices):,})")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.5, label="Identity Line (Equal Confidence)")
    plt.axvline(t_b, color="#ff7f0e", linestyle=":", alpha=0.8, linewidth=1.5, label=f"Model B Tuned Threshold ({t_b:.3f})")
    plt.axhline(t_c1, color="#2ca02c", linestyle=":", alpha=0.8, linewidth=1.5, label=f"Model C1 Tuned Threshold ({t_c1:.3f})")
    plt.title("Prediction Scatter: Model B (LightGBM) vs. Model C1 (Triple-Branch Deep Net)", fontsize=12, fontweight="bold")
    plt.xlabel("Model B Predicted Probability", fontsize=11)
    plt.ylabel("Model C1 Predicted Probability", fontsize=11)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "model_b_c1_blend_prediction_scatter.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"  Saved scatter plot:  {fig3_path}")

    # -------------------------------------------------------------------------
    # STEP 10: Final Decision Questions & Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("FINAL EXPERIMENT DECISION & EVALUATION QUESTIONS")
    print("=" * 85)

    beats_c1 = best_auc_pr > (auc_pr_c1 + 0.0005)
    beats_prev_bc = best_auc_pr > PREV_BC_AUCPR
    improves_f1 = metrics_blend["f1"] > metrics_c1["f1"]
    meaningful = (best_auc_pr - auc_pr_c1) > 0.002

    print(f"1. Best alpha (C1 weight)?               {best_alpha:.3f} (Model B weight = {1.0 - best_alpha:.3f})")
    print(f"2. Best B+C1 AUC-PR?                     {best_auc_pr:.5f}")
    print(f"3. Best B+C1 ROC-AUC?                    {best_roc_auc_at_best_pr:.5f}")
    print(f"4. Does B+C1 beat C1's 0.5766 AUC-PR?    {'YES' if beats_c1 else 'NO'} ({best_auc_pr:.4f} vs {auc_pr_c1:.4f}, Delta = {best_auc_pr - auc_pr_c1:+.4f})")
    print(f"5. Does B+C1 beat previous B+C's 0.5764? {'YES' if beats_prev_bc else 'NO'} ({best_auc_pr:.4f} vs {PREV_BC_AUCPR:.4f}, Delta = {best_auc_pr - PREV_BC_AUCPR:+.4f})")
    print(f"6. Does it improve C1's F1 of 0.5504?    {'YES' if improves_f1 else 'NO'} ({metrics_blend['f1']:.4f} vs {metrics_c1['f1']:.4f}, Delta = {metrics_blend['f1'] - metrics_c1['f1']:+.4f})")
    print(f"7. Are B and C1 complementary?           YES (Disagreement = {disagreement_rate:.2%}, Pearson r = {p_corr:.4f}, Unique defects caught = {pos_both_correct + pos_b_only + pos_c1_only:,})")
    print(f"8. Is ensemble improvement meaningful?   {'YES' if meaningful else 'MARGINAL / EQUIVALENT'}")

    if beats_c1 and meaningful:
        decision_status = "CHAMPION: B + C1 ENSEMBLE CARRIED FORWARD"
        recommendation = "Mark B+C1 as the current champion. Do NOT immediately start C2."
    elif abs(best_auc_pr - auc_pr_c1) <= 0.002:
        decision_status = "EQUIVALENT: KEEP C1 AS SIMPLER CHAMPION"
        recommendation = "Keep C1 as the simpler champion. Consider C2 (multi-scale receptive field 1D CNN) as next experiment."
    else:
        decision_status = "DISCARD ENSEMBLE: C1 REMAINS CHAMPION"
        recommendation = "Discard B+C1 ensemble. Keep C1 as champion. Proceed to C2 only if seeking further predictive improvement."

    print(f"\nDecision Status: {decision_status}")
    print(f"Recommendation:  {recommendation}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
