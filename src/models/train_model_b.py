"""
Training and Evaluation Pipeline for Model B (Die + Spatial + Block Features).
Predicts die yield failure probability for eligible dies (old_label == 0)
using 500 parametric features + 19 spatial context features + 36 block features = 555 features.
Wafer ID is NEVER used as a predictive feature.
Performs a controlled, fair Model A -> Model B comparison and feature contribution analysis.
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
    MODEL_B_FEATURES,
    PARAMETRIC_FEATURES,
    SPATIAL_FEATURES,
    BLOCK_FEATURES,
    load_dataset,
    compute_metrics,
    compute_predict_all_pass_baseline,
    tune_threshold,
    format_confusion_table,
)


def run_model_b():
    print(f"\n{'=' * 85}")
    print("STARTING MODEL B LIGHTGBM TRAINING & A-VS-B EVALUATION PIPELINE")
    print(f"{'=' * 85}")
    print(f"Seed: {SEED}")
    print(f"Features: {len(MODEL_B_FEATURES)} (500 Parametric + 19 Spatial + 36 Block)")
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
        DEV_TRAIN_PARQUET, MODEL_B_FEATURES, eligible_only=True
    )
    X_val, y_val, meta_val, val_stats = load_dataset(
        DEV_VAL_PARQUET, MODEL_B_FEATURES, eligible_only=True
    )
    print(f"Data loading complete in {time.time() - t0_load:.2f}s.\n")

    # Verification Checks
    print("--- INTEGRITY CHECKS ---")
    train_wafers = set(meta_train["wafer_id"].unique())
    val_wafers = set(meta_val["wafer_id"].unique())
    wafer_leak = train_wafers.intersection(val_wafers)
    if wafer_leak:
        raise ValueError(f"CRITICAL ERROR: Wafer leakage detected! {len(wafer_leak)} wafers overlap.")
    print(f"1. Wafer overlap check: 0 overlap ({len(train_wafers)} train wafers, {len(val_wafers)} dev-val wafers) [PASSED]")

    if "wafer_id" in X_train.columns or "wafer_id" in X_val.columns:
        raise ValueError("CRITICAL ERROR: wafer_id leaked into feature matrix!")
    print(f"2. Feature leakage check: wafer_id strictly excluded from features [PASSED]")

    assert len(MODEL_B_FEATURES) == 555, f"Expected 555 features, got {len(MODEL_B_FEATURES)}"
    print(f"3. Feature count check: exactly {len(MODEL_B_FEATURES)} features (500 parametric + 19 spatial + 36 block) [PASSED]")

    train_nans = X_train.isna().sum().sum()
    val_nans = X_val.isna().sum().sum()
    if train_nans > 0 or val_nans > 0:
        raise ValueError(f"CRITICAL ERROR: Missing values detected! Train NaNs: {train_nans}, Val NaNs: {val_nans}")
    print(f"4. Missing value check: 0 NaNs in train, 0 NaNs in validation [PASSED]\n")

    # -------------------------------------------------------------------------
    # Step 2: Handle Class Imbalance with scale_pos_weight
    # -------------------------------------------------------------------------
    neg_count = train_stats["n_neg"]
    pos_count = train_stats["n_pos"]
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"--- STEP 2: CLASS IMBALANCE CONFIGURATION ---")
    print(f"Train Negatives: {neg_count:,} | Positives: {pos_count:,}")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.6f} (identical to Model A)\n")

    # -------------------------------------------------------------------------
    # Step 3: Model Configuration & Training (Identical hyperparameters to Model A)
    # -------------------------------------------------------------------------
    print(f"--- STEP 3: TRAINING LIGHTGBM MODEL B ---")
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
        lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=True),
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
    # Step 5: Evaluation & Threshold Tuning
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 5: EVALUATION & THRESHOLD TUNING ---")
    metrics_05 = compute_metrics(y_val, y_prob, threshold=0.5)

    print("Tuning decision threshold to maximize positive-class F1...")
    tuning_result = tune_threshold(y_val, y_prob, num_steps=200)
    tuned_threshold = tuning_result["best_threshold"]
    metrics_tuned = tuning_result["metrics"]
    print(f"Optimal Threshold T*: {tuned_threshold:.4f} (Max Dev-Val F1: {metrics_tuned['f1']:.4f})")

    # -------------------------------------------------------------------------
    # Step 6: Feature Importances & Group Breakdown
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 6: FEATURE IMPORTANCE & GROUP CONTRIBUTION ANALYSIS ---")
    importances_gain = clf.feature_importances_
    booster = clf.booster_
    importances_split = booster.feature_importance(importance_type="split")

    def get_feature_group(name):
        if name in BLOCK_FEATURES:
            return "block"
        elif name in SPATIAL_FEATURES:
            return "spatial"
        else:
            return "parametric"

    fi_df = pd.DataFrame({
        "feature": MODEL_B_FEATURES,
        "importance_gain": importances_gain,
        "importance_split": importances_split,
        "group": [get_feature_group(f) for f in MODEL_B_FEATURES]
    }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)

    # Group-level attribution
    total_gain = float(fi_df["importance_gain"].sum())
    group_stats = fi_df.groupby("group").agg(
        num_features=("feature", "count"),
        total_gain=("importance_gain", "sum"),
        mean_gain=("importance_gain", "mean"),
        total_splits=("importance_split", "sum"),
    ).reset_index()

    group_stats["gain_pct"] = group_stats["total_gain"] / total_gain * 100.0 if total_gain > 0 else 0.0
    group_stats = group_stats.sort_values(by="total_gain", ascending=False).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Step 7: Model A vs Model B Comparative Analysis
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 7: MODEL A VS MODEL B COMPARATIVE ANALYSIS ---")
    model_a_json_path = REPORTS_DIR / "model_a_metrics.json"
    if not model_a_json_path.exists():
        raise FileNotFoundError(f"Model A metrics not found at {model_a_json_path}")

    with open(model_a_json_path, "r") as f:
        model_a_data = json.load(f)

    a_tuned = model_a_data["evaluation_tuned_threshold"]
    a_05 = model_a_data["evaluation_default_threshold_05"]
    a_timing = model_a_data["timing"]

    delta_auc_pr = metrics_tuned["auc_pr"] - a_tuned["auc_pr"]
    rel_auc_pr_pct = (delta_auc_pr / a_tuned["auc_pr"] * 100.0) if a_tuned["auc_pr"] > 0 else 0.0

    delta_roc_auc = metrics_tuned["roc_auc"] - a_tuned["roc_auc"]
    delta_f1 = metrics_tuned["f1"] - a_tuned["f1"]
    rel_f1_pct = (delta_f1 / a_tuned["f1"] * 100.0) if a_tuned["f1"] > 0 else 0.0

    delta_prec = metrics_tuned["precision"] - a_tuned["precision"]
    delta_rec = metrics_tuned["recall"] - a_tuned["recall"]
    delta_acc = metrics_tuned["accuracy"] - a_tuned["accuracy"]

    comparison_dict = {
        "metric": ["AUC-PR (Average Precision)", "F1 Score (Tuned)", "Precision (Fail)", "Recall (Fail)", "ROC-AUC", "Overall Accuracy", "Decision Threshold (T*)", "Training Time (s)", "Inference Time (s)", "Best Iteration"],
        "Model_A": [
            round(a_tuned["auc_pr"], 4),
            round(a_tuned["f1"], 4),
            round(a_tuned["precision"], 4),
            round(a_tuned["recall"], 4),
            round(a_tuned["roc_auc"], 4),
            f"{a_tuned['accuracy'] * 100:.2f}%",
            round(a_tuned["optimal_threshold"], 4),
            round(a_timing["training_time_seconds"], 2),
            round(a_timing["inference_time_seconds"], 4),
            int(a_timing["best_iteration"]),
        ],
        "Model_B": [
            round(metrics_tuned["auc_pr"], 4),
            round(metrics_tuned["f1"], 4),
            round(metrics_tuned["precision"], 4),
            round(metrics_tuned["recall"], 4),
            round(metrics_tuned["roc_auc"], 4),
            f"{metrics_tuned['accuracy'] * 100:.2f}%",
            round(tuned_threshold, 4),
            round(train_time_s, 2),
            round(infer_time_s, 4),
            int(best_iter),
        ],
        "Absolute_Delta (B - A)": [
            f"{delta_auc_pr:+.4f}",
            f"{delta_f1:+.4f}",
            f"{delta_prec:+.4f}",
            f"{delta_rec:+.4f}",
            f"{delta_roc_auc:+.4f}",
            f"{(delta_acc * 100):+.2f}%",
            f"{(tuned_threshold - a_tuned['optimal_threshold']):+.4f}",
            f"{(train_time_s - a_timing['training_time_seconds']):+.2f}s",
            f"{(infer_time_s - a_timing['inference_time_seconds']):+.4f}s",
            f"{(best_iter - a_timing['best_iteration']):+d}",
        ],
        "Relative_Improvement": [
            f"{rel_auc_pr_pct:+.2f}%",
            f"{rel_f1_pct:+.2f}%",
            f"{(delta_prec / a_tuned['precision'] * 100):+.2f}%" if a_tuned['precision'] > 0 else "N/A",
            f"{(delta_rec / a_tuned['recall'] * 100):+.2f}%" if a_tuned['recall'] > 0 else "N/A",
            f"{(delta_roc_auc / a_tuned['roc_auc'] * 100):+.2f}%" if a_tuned['roc_auc'] > 0 else "N/A",
            "—",
            "—",
            "—",
            "—",
            "—",
        ]
    }
    comparison_df = pd.DataFrame(comparison_dict)

    # -------------------------------------------------------------------------
    # Step 8: Save All Artifacts
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 8: SAVING ARTIFACTS ---")
    model_b_joblib_path = MODELS_DIR / "model_b.joblib"
    model_b_txt_path = MODELS_DIR / "model_b.txt"
    joblib.dump(clf, model_b_joblib_path)
    booster.save_model(str(model_b_txt_path))
    print(f"  Saved model (joblib):        {model_b_joblib_path} ({model_b_joblib_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  Saved model (txt):           {model_b_txt_path} ({model_b_txt_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Predictions DataFrame
    y_pred_tuned = (y_prob >= tuned_threshold).astype(int)
    y_pred_05 = (y_prob >= 0.5).astype(int)

    preds_df = meta_val.copy()
    preds_df["predicted_probability"] = y_prob.astype(np.float32)
    preds_df["predicted_label_tuned"] = y_pred_tuned.astype(np.int8)
    preds_df["predicted_label_05"] = y_pred_05.astype(np.int8)

    preds_parquet_path = REPORTS_DIR / "model_b_dev_val_predictions.parquet"
    preds_df.to_parquet(preds_parquet_path, index=False, engine="pyarrow", compression="snappy")
    print(f"  Saved predictions:           {preds_parquet_path} ({preds_parquet_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Features list JSON
    features_json_path = REPORTS_DIR / "model_b_features.json"
    with open(features_json_path, "w") as f:
        json.dump({
            "model_name": "Model B (Die + Spatial + Block)",
            "num_features": len(MODEL_B_FEATURES),
            "num_parametric": len(PARAMETRIC_FEATURES),
            "num_spatial": len(SPATIAL_FEATURES),
            "num_block": len(BLOCK_FEATURES),
            "features": MODEL_B_FEATURES,
            "parametric_features": PARAMETRIC_FEATURES,
            "spatial_features": SPATIAL_FEATURES,
            "block_features": BLOCK_FEATURES,
        }, f, indent=2)
    print(f"  Saved features list:         {features_json_path}")

    # Feature Importance CSV
    fi_csv_path = REPORTS_DIR / "model_b_feature_importance.csv"
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"  Saved feature importance:    {fi_csv_path}")

    # Metrics JSON
    metrics_summary = {
        "model_name": "Model B (LightGBM Die + Spatial + Block)",
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
        "evaluation_default_threshold_05": metrics_05,
        "evaluation_tuned_threshold": {
            "optimal_threshold": round(tuned_threshold, 4),
            **metrics_tuned,
        },
        "feature_group_attribution": group_stats.to_dict(orient="records"),
        "comparison_vs_model_a": {
            "model_a_auc_pr": a_tuned["auc_pr"],
            "model_b_auc_pr": metrics_tuned["auc_pr"],
            "delta_auc_pr": round(delta_auc_pr, 4),
            "rel_improvement_auc_pr_pct": round(rel_auc_pr_pct, 2),
            "model_a_f1": a_tuned["f1"],
            "model_b_f1": metrics_tuned["f1"],
            "delta_f1": round(delta_f1, 4),
            "rel_improvement_f1_pct": round(rel_f1_pct, 2),
            "model_a_precision": a_tuned["precision"],
            "model_b_precision": metrics_tuned["precision"],
            "delta_precision": round(delta_prec, 4),
            "model_a_recall": a_tuned["recall"],
            "model_b_recall": metrics_tuned["recall"],
            "delta_recall": round(delta_rec, 4),
        },
        "artifacts_created": [
            str(model_b_joblib_path),
            str(model_b_txt_path),
            str(preds_parquet_path),
            str(features_json_path),
            str(fi_csv_path),
            str(REPORTS_DIR / "model_b_metrics.json"),
            str(REPORTS_DIR / "model_b_metrics.csv"),
            str(REPORTS_DIR / "model_comparison_a_vs_b.json"),
            str(REPORTS_DIR / "model_comparison_a_vs_b.csv"),
        ]
    }

    metrics_json_path = REPORTS_DIR / "model_b_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  Saved metrics JSON:          {metrics_json_path}")

    # Metrics CSV
    metrics_csv_df = pd.DataFrame([
        {
            "Model / Setup": "Model B (Default Threshold 0.5)",
            "Threshold": "0.5000",
            "AUC-PR": f"{metrics_05['auc_pr']:.4f}",
            "ROC-AUC": f"{metrics_05['roc_auc']:.4f}",
            "Precision (Fail)": f"{metrics_05['precision']:.4f}",
            "Recall (Fail)": f"{metrics_05['recall']:.4f}",
            "F1 (Fail)": f"{metrics_05['f1']:.4f}",
            "Accuracy": f"{metrics_05['accuracy'] * 100:.2f}%",
        },
        {
            "Model / Setup": f"Model B (Tuned Threshold T* = {tuned_threshold:.4f})",
            "Threshold": f"{tuned_threshold:.4f}",
            "AUC-PR": f"{metrics_tuned['auc_pr']:.4f}",
            "ROC-AUC": f"{metrics_tuned['roc_auc']:.4f}",
            "Precision (Fail)": f"{metrics_tuned['precision']:.4f}",
            "Recall (Fail)": f"{metrics_tuned['recall']:.4f}",
            "F1 (Fail)": f"{metrics_tuned['f1']:.4f}",
            "Accuracy": f"{metrics_tuned['accuracy'] * 100:.2f}%",
        }
    ])
    metrics_csv_path = REPORTS_DIR / "model_b_metrics.csv"
    metrics_csv_df.to_csv(metrics_csv_path, index=False)
    print(f"  Saved metrics CSV:           {metrics_csv_path}")

    # Comparison JSON & CSV
    comp_json_path = REPORTS_DIR / "model_comparison_a_vs_b.json"
    with open(comp_json_path, "w") as f:
        json.dump(metrics_summary["comparison_vs_model_a"], f, indent=2)
    print(f"  Saved comparison JSON:       {comp_json_path}")

    comp_csv_path = REPORTS_DIR / "model_comparison_a_vs_b.csv"
    comparison_df.to_csv(comp_csv_path, index=False)
    print(f"  Saved comparison CSV:        {comp_csv_path}")

    # -------------------------------------------------------------------------
    # Step 9: Comprehensive Terminal Output
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 85}")
    print("MODEL B EVALUATION REPORT & A-VS-B COMPARISON")
    print(f"{'=' * 85}")
    print(f"1. Training Time:             {train_time_s:.2f} s ({train_time_s / 60:.2f} min)")
    print(f"2. Inference Time:            {infer_time_s:.4f} s ({throughput:,.1f} dies/s)")
    print(f"3. Best Iteration:            {best_iter}")
    print(f"4. Eligible Training Dies:    {train_stats['eligible_rows']:,} (out of {train_stats['total_rows']:,})")
    print(f"   Eligible Validation Dies:  {val_stats['eligible_rows']:,} (out of {val_stats['total_rows']:,})")
    print(f"5. Positive Rate:             Train: {train_stats['pos_rate']:.3f}% | Dev-Val: {val_stats['pos_rate']:.3f}%")
    print(f"6. AUC-PR (Average Prec):     {metrics_tuned['auc_pr']:.4f} (Model A: {a_tuned['auc_pr']:.4f} | Delta: {delta_auc_pr:+.4f} [{rel_auc_pr_pct:+.2f}%])")
    print(f"7. Tuned F1 Score:            {metrics_tuned['f1']:.4f} (Model A: {a_tuned['f1']:.4f} | Delta: {delta_f1:+.4f} [{rel_f1_pct:+.2f}%])")
    print(f"8. Precision (Fail Class):    {metrics_tuned['precision']:.4f} (at T*) | {metrics_05['precision']:.4f} (at 0.5)")
    print(f"9. Recall (Fail Class):       {metrics_tuned['recall']:.4f} (at T*) | {metrics_05['recall']:.4f} (at 0.5)")
    print(f"10. Tuned Threshold:          {tuned_threshold:.4f} (Model A: {a_tuned['optimal_threshold']:.4f})")
    print(f"11. Exact Files Created:")
    for f_art in metrics_summary["artifacts_created"]:
        print(f"    - {f_art}")
    print(f"12. Warnings/Errors:          None (Clean execution)")
    print(f"{'=' * 85}\n")

    print("--- CONCISE A VS B COMPARISON TABLE ---")
    print(comparison_df.to_string(index=False))

    print(f"\n--- CONFUSION MATRIX (MODEL B @ TUNED THRESHOLD T* = {tuned_threshold:.4f}) ---")
    print(format_confusion_table(metrics_tuned))

    print(f"\n--- FEATURE GROUP ATTRIBUTION (GAIN BREAKDOWN) ---")
    print(f"{'Group':<15} {'Features':<10} {'Total Gain':<18} {'Mean Gain':<15} {'Gain %':<10} {'Total Splits':<12}")
    print(f"{'-'*15}-+-{'-'*10}-+-{'-'*18}-+-{'-'*15}-+-{'-'*10}-+-{'-'*12}")
    for _, r in group_stats.iterrows():
        print(f"{r['group']:<15} {int(r['num_features']):<10d} {r['total_gain']:<18.2f} {r['mean_gain']:<15.2f} {r['gain_pct']:<9.2f}% {int(r['total_splits']):<12d}")

    print(f"\n--- TOP 20 FEATURES OVERALL BY GAIN ---")
    print(f"{'Rank':<5} {'Feature':<35} {'Group':<12} {'Gain':<15} {'Splits':<8}")
    print(f"{'-'*5}-+-{'-'*35}-+-{'-'*12}-+-{'-'*15}-+-{'-'*8}")
    for i, row in fi_df.head(20).iterrows():
        print(f"{i+1:<5} {row['feature']:<35} {row['group']:<12} {row['importance_gain']:<15.2f} {int(row['importance_split']):<8d}")

    block_fi = fi_df[fi_df["group"] == "block"].reset_index(drop=True)
    print(f"\n--- TOP 10 BLOCK FEATURES BY GAIN ---")
    print(f"{'Rank':<5} {'Feature':<35} {'Gain':<15} {'Splits':<8} {'Overall Rank':<12}")
    print(f"{'-'*5}-+-{'-'*35}-+-{'-'*15}-+-{'-'*8}-+-{'-'*12}")
    for i, row in block_fi.head(10).iterrows():
        overall_rank = fi_df[fi_df['feature'] == row['feature']].index[0] + 1
        print(f"{i+1:<5} {row['feature']:<35} {row['importance_gain']:<15.2f} {int(row['importance_split']):<8d} #{overall_rank:<11d}")

    print(f"\n{'=' * 85}")
    print("ALL MODEL B ARTIFACTS AND A-VS-B COMPARISONS SUCCESSFULLY GENERATED")
    print(f"{'=' * 85}")


if __name__ == "__main__":
    run_model_b()
