"""
Training and Evaluation Pipeline for XGBoost Model A.
Predicts die yield failure probability for eligible dies (old_label == 0)
using 500 parametric features + 19 spatial context features = 519 features.
Excludes the 36 block features (reserved for Model B).
Uses histogram-based GPU tree method with early stopping on dev_val AUC-PR.
"""

import sys
import os
import time
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# Ensure repo root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    DEV_TRAIN_PARQUET,
    DEV_VAL_PARQUET,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
)
from src.models.common import (
    MODEL_A_FEATURES,
    load_dataset,
    compute_metrics,
    tune_threshold,
)


def run_xgb_a():
    print(f"\n{'=' * 85}")
    print("STARTING XGBOOST MODEL A TRAINING & EVALUATION PIPELINE")
    print(f"{'=' * 85}")
    print(f"Seed: {SEED}")
    print(f"Features: {len(MODEL_A_FEATURES)} (500 Parametric + 19 Spatial Context)")
    print(f"Train Source: {DEV_TRAIN_PARQUET.name}")
    print(f"Validation Source: {DEV_VAL_PARQUET.name}")
    print(f"Predictive Task: ONLY Eligible Dies (old_label == 0)")
    print(f"Target: label (0 = stayed healthy, 1 = newly failed)")
    print(f"{'=' * 85}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load Datasets
    # -------------------------------------------------------------------------
    print("--- STEP 1: LOADING DATASETS ---")
    t0_load = time.time()
    X_train, y_train, meta_train, train_stats = load_dataset(
        DEV_TRAIN_PARQUET, MODEL_A_FEATURES, eligible_only=True
    )
    X_val, y_val, meta_val, val_stats = load_dataset(
        DEV_VAL_PARQUET, MODEL_A_FEATURES, eligible_only=True
    )
    print(f"Data loading complete in {time.time() - t0_load:.2f}s.\n")

    # Verify no overlap in wafers
    train_wafers = set(meta_train["wafer_id"].unique())
    val_wafers = set(meta_val["wafer_id"].unique())
    wafer_leak = train_wafers.intersection(val_wafers)
    assert not wafer_leak, f"CRITICAL ERROR: Wafer leakage detected! {len(wafer_leak)} wafers overlap."
    print(f"Wafer-disjoint verification: 0 overlap between {len(train_wafers)} train wafers and {len(val_wafers)} dev-val wafers.")

    # -------------------------------------------------------------------------
    # Step 2: Handle Class Imbalance with scale_pos_weight
    # -------------------------------------------------------------------------
    neg_count = train_stats["n_neg"]
    pos_count = train_stats["n_pos"]
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"\n--- STEP 2: CLASS IMBALANCE CONFIGURATION ---")
    print(f"Train Negatives: {neg_count:,} | Positives: {pos_count:,}")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.6f} (exactly {neg_count}/{pos_count})")

    # -------------------------------------------------------------------------
    # Step 3: Model Configuration & Training
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 3: TRAINING XGBOOST MODEL A ---")
    clf = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device="cuda",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=-1,
    )

    t0_train = time.time()
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    train_time = time.time() - t0_train
    best_iteration = clf.best_iteration
    print(f"\nXGBoost Model A training complete in {train_time:.2f}s ({train_time/60:.2f} min).")
    print(f"Best Iteration: {best_iteration} (of {clf.n_estimators})")

    # -------------------------------------------------------------------------
    # Step 4: Inference on Dev-Val Set
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 4: PREDICTING PROBABILITIES ON CANONICAL DEV-VAL SET ---")
    t0_pred = time.time()
    val_probs = clf.predict_proba(X_val)[:, 1]
    pred_time = time.time() - t0_pred
    print(f"Inference complete on {len(val_probs):,} dies in {pred_time:.2f}s ({len(val_probs)/pred_time:,.0f} dies/sec).")

    # -------------------------------------------------------------------------
    # Step 5: Threshold Tuning & Metrics Computation
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 5: THRESHOLD TUNING & COMPREHENSIVE EVALUATION ---")
    tuning_res = tune_threshold(y_val, val_probs, num_steps=200)
    best_thresh = tuning_res["best_threshold"]
    metrics = tuning_res["metrics"]
    best_f1 = metrics["f1"]
    best_prec = metrics["precision"]
    best_rec = metrics["recall"]
    print(f"Optimal F1 Threshold: {best_thresh:.4f} (Max F1: {best_f1:.4f}, Prec: {best_prec:.4f}, Rec: {best_rec:.4f})")

    metrics_05 = compute_metrics(y_val, val_probs, threshold=0.5)

    print("\n" + "=" * 85)
    print(f"XGBOOST MODEL A CANONICAL DEV-VAL METRICS (at Tuned Threshold {best_thresh:.4f})")
    print("=" * 85)
    print(f"  AUC-PR:             {metrics['auc_pr']:.4f}")
    print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"  Tuned F1 Score:     {metrics['f1']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  Specificity:        {metrics['pass_recall']:.4f}")
    print(f"  Accuracy:           {metrics['accuracy']:.4%}")
    print(f"  True Positives (TP):{metrics['tp']:,} / {val_stats['n_pos']:,} fails ({metrics['tp']/val_stats['n_pos']:.2%})")
    print(f"  False Positives(FP):{metrics['fp']:,} / {val_stats['n_neg']:,} passes ({metrics['fp']/val_stats['n_neg']:.2%})")
    print(f"  False Negatives(FN):{metrics['fn']:,}")
    print(f"  True Negatives (TN):{metrics['tn']:,}")
    print(f"  Training Time:      {train_time:.1f} s")

    # Comparison against LightGBM Model A Baseline
    LGBM_A_AUCPR = 0.4937
    LGBM_A_ROCAUC = 0.8318
    LGBM_A_F1 = 0.5200

    print("\n" + "=" * 85)
    print("COMPARISON: XGBOOST MODEL A vs. LIGHTGBM MODEL A (BASELINE)")
    print("=" * 85)
    delta_pr = metrics["auc_pr"] - LGBM_A_AUCPR
    delta_roc = metrics["roc_auc"] - LGBM_A_ROCAUC
    delta_f1 = metrics["f1"] - LGBM_A_F1
    print(f"  AUC-PR:  XGBoost={metrics['auc_pr']:.4f} vs LightGBM={LGBM_A_AUCPR:.4f} (Delta: {delta_pr:+.4f}, {delta_pr/LGBM_A_AUCPR*100:+.2f}%)")
    print(f"  ROC-AUC: XGBoost={metrics['roc_auc']:.4f} vs LightGBM={LGBM_A_ROCAUC:.4f} (Delta: {delta_roc:+.4f})")
    print(f"  F1:      XGBoost={metrics['f1']:.4f} vs LightGBM={LGBM_A_F1:.4f} (Delta: {delta_f1:+.4f})")

    # -------------------------------------------------------------------------
    # Step 6: Save Artifacts
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 6: SAVING ARTIFACTS ---")
    model_path = MODELS_DIR / "xgb_a.json"
    clf.save_model(model_path)
    print(f"Saved model to: {model_path}")

    # Save predictions
    meta_val = meta_val.copy()
    meta_val["predicted_probability"] = val_probs.astype(np.float32)
    meta_val["predicted_label_tuned"] = (val_probs >= best_thresh).astype(np.int8)
    meta_val["predicted_label_05"] = (val_probs >= 0.5).astype(np.int8)

    preds_path = REPORTS_DIR / "xgb_a_dev_val_predictions.parquet"
    meta_val.to_parquet(preds_path, index=False)
    print(f"Saved predictions to: {preds_path}")

    # Metrics JSON & CSV
    metrics_summary = {
        "model_name": "XGBoost Model A",
        "feature_set": "Model A (500 Parametric + 19 Spatial)",
        "num_features": len(MODEL_A_FEATURES),
        "best_iteration": int(best_iteration),
        "training_time_seconds": float(train_time),
        "scale_pos_weight": float(scale_pos_weight),
        "auc_pr": float(metrics["auc_pr"]),
        "roc_auc": float(metrics["roc_auc"]),
        "optimal_threshold": float(best_thresh),
        "f1": float(metrics["f1"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "specificity": float(metrics["pass_recall"]),
        "accuracy": float(metrics["accuracy"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tn": int(metrics["tn"]),
        "metrics_at_05": {
            "f1": float(metrics_05["f1"]),
            "precision": float(metrics_05["precision"]),
            "recall": float(metrics_05["recall"]),
            "specificity": float(metrics_05["pass_recall"]),
            "accuracy": float(metrics_05["accuracy"]),
        },
        "comparison_vs_lgbm_a": {
            "lgbm_a_auc_pr": LGBM_A_AUCPR,
            "delta_auc_pr": float(delta_pr),
            "rel_lift_auc_pr_pct": float(delta_pr / LGBM_A_AUCPR * 100),
            "lgbm_a_roc_auc": LGBM_A_ROCAUC,
            "delta_roc_auc": float(delta_roc),
            "lgbm_a_f1": LGBM_A_F1,
            "delta_f1": float(delta_f1),
        }
    }

    metrics_json_path = REPORTS_DIR / "xgb_a_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics JSON: {metrics_json_path}")

    metrics_csv_path = REPORTS_DIR / "xgb_a_metrics.csv"
    pd.DataFrame([{
        "metric": k, "value": v
    } for k, v in metrics_summary.items() if not isinstance(v, dict)]).to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics CSV:  {metrics_csv_path}")

    print("\nXGBoost Model A pipeline successfully finished!\n")
    return metrics_summary


if __name__ == "__main__":
    run_xgb_a()
