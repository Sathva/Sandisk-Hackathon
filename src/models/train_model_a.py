"""
Training and Evaluation Pipeline for Model A (LightGBM Baseline).
Predicts die yield failure probability for eligible dies (old_label == 0)
using 500 parametric features + 19 spatial context features.
Excludes the 36 block features (reserved for Model B).
Wafer ID is NEVER used as a predictive feature.
"""

import sys
import os
import time
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

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
    compute_predict_all_pass_baseline,
    tune_threshold,
    format_confusion_table,
)


def run_model_a():
    print(f"\n{'=' * 85}")
    print("STARTING MODEL A LIGHTGBM TRAINING & EVALUATION PIPELINE")
    print(f"{'=' * 85}")
    print(f"Seed: {SEED}")
    print(f"Features: {len(MODEL_A_FEATURES)} (500 Parametric + 19 Spatial Context)")
    print(f"Train Source: {DEV_TRAIN_PARQUET}")
    print(f"Validation Source: {DEV_VAL_PARQUET}")
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
    if wafer_leak:
        raise ValueError(f"CRITICAL ERROR: Wafer leakage detected! {len(wafer_leak)} wafers overlap.")
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
    print(f"\n--- STEP 3: TRAINING LIGHTGBM MODEL A ---")
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
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    t0_train = time.time()
    clf.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="average_precision",
        callbacks=callbacks,
    )
    train_time_s = time.time() - t0_train
    best_iter = clf.best_iteration_ if hasattr(clf, "best_iteration_") else 1000
    print(f"\nTraining completed in {train_time_s:.2f}s ({train_time_s / 60:.2f} min). Best iteration: {best_iter}")

    # -------------------------------------------------------------------------
    # Step 4: Inference on dev_val
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 4: INFERENCE ON DEV_VAL ---")
    t0_infer = time.time()
    y_prob = clf.predict_proba(X_val)[:, 1]
    infer_time_s = time.time() - t0_infer
    throughput = len(y_val) / infer_time_s if infer_time_s > 0 else 0
    print(f"Inference completed in {infer_time_s:.4f}s ({throughput:,.1f} dies/sec, {infer_time_s / len(y_val) * 1000:.4f} ms/die)")

    # -------------------------------------------------------------------------
    # Step 5: Baselines & Evaluation
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 5: EVALUATION & BASELINES ---")

    # Baseline: Predict all pass
    all_pass_baseline = compute_predict_all_pass_baseline(y_val)

    # Model A: Default threshold (0.5)
    metrics_05 = compute_metrics(y_val, y_prob, threshold=0.5)

    # Model A: Threshold tuning on dev_val
    print("Tuning decision threshold to maximize positive-class F1...")
    tuning_result = tune_threshold(y_val, y_prob, num_steps=200)
    tuned_threshold = tuning_result["best_threshold"]
    metrics_tuned = tuning_result["metrics"]
    print(f"Optimal Threshold T*: {tuned_threshold:.4f} (Max Dev-Val F1: {metrics_tuned['f1']:.4f})")

    # -------------------------------------------------------------------------
    # Step 6: Feature Importances
    # -------------------------------------------------------------------------
    importances_gain = clf.feature_importances_
    booster = clf.booster_
    importances_split = booster.feature_importance(importance_type="split")

    fi_df = pd.DataFrame({
        "feature": MODEL_A_FEATURES,
        "importance_gain": importances_gain,
        "importance_split": importances_split,
        "feature_type": ["spatial" if f in SPATIAL_FEATURES else "parametric" for f in MODEL_A_FEATURES]
    }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Step 7: Save All Artifacts
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 6: SAVING ARTIFACTS ---")
    model_joblib_path = MODELS_DIR / "model_a.joblib"
    model_txt_path = MODELS_DIR / "model_a.txt"
    joblib.dump(clf, model_joblib_path)
    booster.save_model(str(model_txt_path))
    print(f"  Saved model (joblib): {model_joblib_path} ({model_joblib_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  Saved model (txt):    {model_txt_path} ({model_txt_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Predictions DataFrame
    y_pred_tuned = (y_prob >= tuned_threshold).astype(int)
    y_pred_05 = (y_prob >= 0.5).astype(int)

    preds_df = meta_val.copy()
    preds_df["predicted_probability"] = y_prob.astype(np.float32)
    preds_df["predicted_label_tuned"] = y_pred_tuned.astype(np.int8)
    preds_df["predicted_label_05"] = y_pred_05.astype(np.int8)

    preds_parquet_path = REPORTS_DIR / "model_a_dev_val_predictions.parquet"
    preds_df.to_parquet(preds_parquet_path, index=False, engine="pyarrow", compression="snappy")
    print(f"  Saved predictions:    {preds_parquet_path} ({preds_parquet_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Features list JSON
    features_json_path = REPORTS_DIR / "model_a_features.json"
    with open(features_json_path, "w") as f:
        json.dump({
            "model_name": "Model A (LightGBM Baseline)",
            "num_features": len(MODEL_A_FEATURES),
            "num_parametric": len(PARAMETRIC_FEATURES),
            "num_spatial": len(SPATIAL_FEATURES),
            "features": MODEL_A_FEATURES,
            "parametric_features": PARAMETRIC_FEATURES,
            "spatial_features": SPATIAL_FEATURES,
        }, f, indent=2)
    print(f"  Saved features list:  {features_json_path}")

    # Feature Importance CSV
    fi_csv_path = REPORTS_DIR / "model_a_feature_importance.csv"
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"  Saved feature importance: {fi_csv_path}")

    # Metrics JSON & CSV
    metrics_summary = {
        "model_name": "Model A (LightGBM Baseline)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": SEED,
        "timing": {
            "training_time_seconds": round(train_time_s, 2),
            "inference_time_seconds": round(infer_time_s, 4),
            "inference_throughput_dies_per_sec": round(throughput, 1),
            "best_iteration": int(best_iter),
        },
        "dataset_statistics": {
            "dev_train_wafers": train_stats["n_wafers"],
            "dev_train_total_dies": train_stats["total_rows"],
            "dev_train_eligible_dies": train_stats["eligible_rows"],
            "dev_train_positives": train_stats["n_pos"],
            "dev_train_negatives": train_stats["n_neg"],
            "dev_train_positive_rate_pct": round(train_stats["pos_rate"], 4),
            "scale_pos_weight": round(scale_pos_weight, 6),
            "dev_val_wafers": val_stats["n_wafers"],
            "dev_val_total_dies": val_stats["total_rows"],
            "dev_val_eligible_dies": val_stats["eligible_rows"],
            "dev_val_positives": val_stats["n_pos"],
            "dev_val_negatives": val_stats["n_neg"],
            "dev_val_positive_rate_pct": round(val_stats["pos_rate"], 4),
        },
        "baselines": {
            "predict_all_pass": all_pass_baseline,
        },
        "evaluation_default_threshold_05": metrics_05,
        "evaluation_tuned_threshold": {
            "optimal_threshold": round(tuned_threshold, 4),
            **metrics_tuned,
        },
        "artifacts_created": [
            str(model_joblib_path),
            str(model_txt_path),
            str(preds_parquet_path),
            str(features_json_path),
            str(fi_csv_path),
            str(REPORTS_DIR / "model_a_metrics.json"),
            str(REPORTS_DIR / "model_a_metrics.csv"),
        ]
    }

    metrics_json_path = REPORTS_DIR / "model_a_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  Saved metrics JSON:   {metrics_json_path}")

    # Metrics comparison table CSV
    comp_df = pd.DataFrame([
        {
            "Model / Setup": "Baseline: Predict-All-Pass",
            "Threshold": "N/A",
            "AUC-PR": f"{all_pass_baseline['auc_pr']:.4f}",
            "ROC-AUC": f"{all_pass_baseline['roc_auc']:.4f}",
            "Precision (Fail)": f"{all_pass_baseline['precision']:.4f}",
            "Recall (Fail)": f"{all_pass_baseline['recall']:.4f}",
            "F1 (Fail)": f"{all_pass_baseline['f1']:.4f}",
            "Accuracy": f"{all_pass_baseline['accuracy'] * 100:.2f}%",
        },
        {
            "Model / Setup": "Model A (Default Threshold 0.5)",
            "Threshold": "0.5000",
            "AUC-PR": f"{metrics_05['auc_pr']:.4f}",
            "ROC-AUC": f"{metrics_05['roc_auc']:.4f}",
            "Precision (Fail)": f"{metrics_05['precision']:.4f}",
            "Recall (Fail)": f"{metrics_05['recall']:.4f}",
            "F1 (Fail)": f"{metrics_05['f1']:.4f}",
            "Accuracy": f"{metrics_05['accuracy'] * 100:.2f}%",
        },
        {
            "Model / Setup": f"Model A (Tuned Threshold T* = {tuned_threshold:.4f})",
            "Threshold": f"{tuned_threshold:.4f}",
            "AUC-PR": f"{metrics_tuned['auc_pr']:.4f}",
            "ROC-AUC": f"{metrics_tuned['roc_auc']:.4f}",
            "Precision (Fail)": f"{metrics_tuned['precision']:.4f}",
            "Recall (Fail)": f"{metrics_tuned['recall']:.4f}",
            "F1 (Fail)": f"{metrics_tuned['f1']:.4f}",
            "Accuracy": f"{metrics_tuned['accuracy'] * 100:.2f}%",
        }
    ])
    metrics_csv_path = REPORTS_DIR / "model_a_metrics.csv"
    comp_df.to_csv(metrics_csv_path, index=False)
    print(f"  Saved metrics CSV:    {metrics_csv_path}")

    # -------------------------------------------------------------------------
    # Step 8: Comprehensive Terminal Output
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 85}")
    print("MODEL A EVALUATION REPORT & SUMMARY")
    print(f"{'=' * 85}")
    print(f"1. Training Time:             {train_time_s:.2f} s ({train_time_s / 60:.2f} min)")
    print(f"2. Inference Time:            {infer_time_s:.4f} s ({throughput:,.1f} dies/s)")
    print(f"3. Eligible Training Dies:    {train_stats['eligible_rows']:,} (out of {train_stats['total_rows']:,})")
    print(f"   Eligible Validation Dies:  {val_stats['eligible_rows']:,} (out of {val_stats['total_rows']:,})")
    print(f"4. Positive Rate:             Train: {train_stats['pos_rate']:.3f}% | Dev-Val: {val_stats['pos_rate']:.3f}%")
    print(f"5. AUC-PR (Average Prec):     {metrics_tuned['auc_pr']:.4f} (Baseline: {all_pass_baseline['auc_pr']:.4f})")
    print(f"6. Tuned F1 Score:            {metrics_tuned['f1']:.4f} (at T* = {tuned_threshold:.4f})")
    print(f"7. Precision (Fail Class):    {metrics_tuned['precision']:.4f} (at T*) | {metrics_05['precision']:.4f} (at 0.5)")
    print(f"8. Recall (Fail Class):       {metrics_tuned['recall']:.4f} (at T*) | {metrics_05['recall']:.4f} (at 0.5)")
    print(f"9. Tuned Threshold:           {tuned_threshold:.4f}")
    print(f"{'=' * 85}\n")

    print("--- COMPARISON SUMMARY TABLE ---")
    print(comp_df.to_string(index=False))

    print(f"\n--- CONFUSION MATRIX (TUNED THRESHOLD T* = {tuned_threshold:.4f}) ---")
    print(format_confusion_table(metrics_tuned))

    print(f"\n--- CONFUSION MATRIX (DEFAULT THRESHOLD = 0.5000) ---")
    print(format_confusion_table(metrics_05))

    print(f"\n--- TOP 10 IMPORTANT FEATURES (GAIN) ---")
    print(f"{'Rank':<5} {'Feature':<35} {'Type':<12} {'Gain':<15} {'Splits':<8}")
    print(f"{'-'*5}-+-{'-'*35}-+-{'-'*12}-+-{'-'*15}-+-{'-'*8}")
    for i, row in fi_df.head(10).iterrows():
        print(f"{i+1:<5} {row['feature']:<35} {row['feature_type']:<12} {row['importance_gain']:<15.2f} {int(row['importance_split']):<8d}")

    print(f"\n{'=' * 85}")
    print("ALL MODEL A ARTIFACTS SUCCESSFULLY GENERATED")
    print(f"{'=' * 85}")


if __name__ == "__main__":
    run_model_a()
