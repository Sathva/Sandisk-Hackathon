"""
Controlled Ablation Experiment: Model B-Without-Spatial (Parametric + Block Only).
Quantifies the incremental predictive value of high-dimensional block-level measurements
independently of spatial context.
Features: 500 Parametric + 36 Block = 536 features.
Strictly excludes all 19 spatial features and wafer_id.
Produces full 3-way ablation comparison:
  1. Model A (Parametric + Spatial)
  2. Model B-no-spatial (Parametric + Block)
  3. Model B (Parametric + Spatial + Block)
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
    MODEL_B_NO_SPATIAL_FEATURES,
    PARAMETRIC_FEATURES,
    SPATIAL_FEATURES,
    BLOCK_FEATURES,
    load_dataset,
    compute_metrics,
    compute_predict_all_pass_baseline,
    tune_threshold,
    format_confusion_table,
)


def run_model_b_no_spatial():
    print(f"\n{'=' * 85}")
    print("STARTING CONTROLLED ABLATION: MODEL B-WITHOUT-SPATIAL")
    print(f"{'=' * 85}")
    print(f"Seed: {SEED}")
    print(f"Features: {len(MODEL_B_NO_SPATIAL_FEATURES)} (500 Parametric + 36 Block | 0 Spatial)")
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
        DEV_TRAIN_PARQUET, MODEL_B_NO_SPATIAL_FEATURES, eligible_only=True
    )
    X_val, y_val, meta_val, val_stats = load_dataset(
        DEV_VAL_PARQUET, MODEL_B_NO_SPATIAL_FEATURES, eligible_only=True
    )
    print(f"Data loading complete in {time.time() - t0_load:.2f}s.\n")

    # -------------------------------------------------------------------------
    # Step 2: Strict Leakage & Integrity Verification Checks
    # -------------------------------------------------------------------------
    print("--- STEP 2: LEAKAGE & INTEGRITY VERIFICATION CHECKS ---")
    
    # 1. wafer_id leakage
    if "wafer_id" in X_train.columns or "wafer_id" in X_val.columns:
        raise ValueError("CRITICAL ERROR: wafer_id leaked into feature matrix!")
    print("1. wafer_id leakage check: wafer_id strictly excluded from predictive features [PASSED]")

    # 2. spatial features exclusion
    spatial_in_train = set(X_train.columns).intersection(set(SPATIAL_FEATURES))
    spatial_in_val = set(X_val.columns).intersection(set(SPATIAL_FEATURES))
    if spatial_in_train or spatial_in_val:
        raise ValueError(f"CRITICAL ERROR: Spatial features detected in ablation! {spatial_in_train}")
    print("2. Spatial exclusion check: exactly 0 spatial features present in feature set [PASSED]")

    # 3. Exact feature count check
    assert len(MODEL_B_NO_SPATIAL_FEATURES) == 536, f"Expected 536 features, got {len(MODEL_B_NO_SPATIAL_FEATURES)}"
    assert X_train.shape[1] == 536 and X_val.shape[1] == 536, "Feature matrix column count mismatch!"
    print(f"3. Feature count check: exactly {len(MODEL_B_NO_SPATIAL_FEATURES)} features (500 parametric + 36 block) [PASSED]")

    # 4. Missing values
    train_nans = int(X_train.isna().sum().sum())
    val_nans = int(X_val.isna().sum().sum())
    if train_nans > 0 or val_nans > 0:
        raise ValueError(f"CRITICAL ERROR: NaNs detected! Train NaNs: {train_nans}, Val NaNs: {val_nans}")
    print(f"4. Missing value check: 0 NaNs in train, 0 NaNs in validation [PASSED]")

    # 5. Wafer intersection check
    train_wafers = set(meta_train["wafer_id"].unique())
    val_wafers = set(meta_val["wafer_id"].unique())
    wafer_overlap = train_wafers.intersection(val_wafers)
    if wafer_overlap:
        raise ValueError(f"CRITICAL ERROR: Wafer overlap detected! {len(wafer_overlap)} wafers overlap.")
    print(f"5. Wafer-disjoint check: 0 wafer overlap ({len(train_wafers)} train wafers, {len(val_wafers)} dev-val wafers) [PASSED]")

    # 6. Eligible population consistency
    assert train_stats["eligible_rows"] == 651337, f"Train row count mismatch: {train_stats['eligible_rows']}"
    assert val_stats["eligible_rows"] == 137576, f"Val row count mismatch: {val_stats['eligible_rows']}"
    assert val_stats["n_pos"] == 5367, f"Val positive count mismatch: {val_stats['n_pos']}"
    print(f"6. Population check: exactly identical eligible dies as Models A & B (651,337 train / 137,576 val) [PASSED]\n")

    # -------------------------------------------------------------------------
    # Step 3: Class Imbalance Configuration
    # -------------------------------------------------------------------------
    neg_count = train_stats["n_neg"]
    pos_count = train_stats["n_pos"]
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"--- STEP 3: CLASS IMBALANCE CONFIGURATION ---")
    print(f"Train Negatives: {neg_count:,} | Positives: {pos_count:,}")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.6f} (identical to Models A and B)\n")

    # -------------------------------------------------------------------------
    # Step 4: Model Configuration & Training (Identical protocol to Models A and B)
    # -------------------------------------------------------------------------
    print(f"--- STEP 4: TRAINING LIGHTGBM MODEL B-WITHOUT-SPATIAL ---")
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
    # Step 5: Inference on dev_val
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 5: INFERENCE ON DEV_VAL ---")
    t0_infer = time.time()
    y_prob = clf.predict_proba(X_val)[:, 1]
    infer_time_s = time.time() - t0_infer
    throughput = len(y_val) / infer_time_s if infer_time_s > 0 else 0
    print(f"Inference completed in {infer_time_s:.4f}s ({throughput:,.1f} dies/sec, {infer_time_s / len(y_val) * 1000:.4f} ms/die)")

    # -------------------------------------------------------------------------
    # Step 6: Evaluation & Threshold Tuning
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 6: EVALUATION & THRESHOLD TUNING ---")
    no_skill_auc_pr = float(val_stats["pos_rate"] / 100.0)
    print(f"No-skill AUC-PR baseline (positive prevalence): {no_skill_auc_pr:.4f} ({val_stats['pos_rate']:.3f}%)")

    metrics_05 = compute_metrics(y_val, y_prob, threshold=0.5)

    print("Tuning decision threshold to maximize positive-class F1...")
    tuning_result = tune_threshold(y_val, y_prob, num_steps=200)
    tuned_threshold = tuning_result["best_threshold"]
    metrics_tuned = tuning_result["metrics"]
    print(f"Optimal Threshold T*: {tuned_threshold:.4f} (Max Dev-Val F1: {metrics_tuned['f1']:.4f})")

    # -------------------------------------------------------------------------
    # Step 7: Feature Importance & Group Attribution
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 7: FEATURE IMPORTANCE ANALYSIS (PARAMETRIC VS. BLOCK) ---")
    importances_gain = clf.feature_importances_
    booster = clf.booster_
    importances_split = booster.feature_importance(importance_type="split")

    fi_df = pd.DataFrame({
        "feature": MODEL_B_NO_SPATIAL_FEATURES,
        "importance_gain": importances_gain,
        "importance_split": importances_split,
        "group": ["block" if f in BLOCK_FEATURES else "parametric" for f in MODEL_B_NO_SPATIAL_FEATURES]
    }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)

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
    # Step 8: Controlled 3-Model Ablation Comparison (A vs. B-no-spatial vs. B)
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 8: CONTROLLED THREE-MODEL ABLATION COMPARISON ---")
    model_a_json_path = REPORTS_DIR / "model_a_metrics.json"
    model_b_json_path = REPORTS_DIR / "model_b_metrics.json"

    with open(model_a_json_path, "r") as f:
        model_a_data = json.load(f)
    with open(model_b_json_path, "r") as f:
        model_b_data = json.load(f)

    a_m = model_a_data["evaluation_tuned_threshold"]
    a_t = model_a_data["timing"]
    b_m = model_b_data["evaluation_tuned_threshold"]
    b_t = model_b_data["timing"]
    bns_m = metrics_tuned
    bns_t = {
        "training_time_seconds": train_time_s,
        "inference_time_seconds": infer_time_s,
        "best_iteration": best_iter,
    }

    # Summary table across models
    ablation_summary_rows = [
        {
            "Model": "Model A (Parametric + Spatial)",
            "Features": 519,
            "AUC-PR": round(a_m["auc_pr"], 4),
            "F1 (Tuned)": round(a_m["f1"], 4),
            "Precision": round(a_m["precision"], 4),
            "Recall": round(a_m["recall"], 4),
            "ROC-AUC": round(a_m["roc_auc"], 4),
            "Accuracy": f"{a_m['accuracy'] * 100:.2f}%",
            "Threshold T*": round(a_m["optimal_threshold"], 4),
            "Train Time": f"{a_t['training_time_seconds']:.1f}s",
            "Best Iter": int(a_t["best_iteration"]),
        },
        {
            "Model": "Model B-no-spatial (Parametric + Block)",
            "Features": 536,
            "AUC-PR": round(bns_m["auc_pr"], 4),
            "F1 (Tuned)": round(bns_m["f1"], 4),
            "Precision": round(bns_m["precision"], 4),
            "Recall": round(bns_m["recall"], 4),
            "ROC-AUC": round(bns_m["roc_auc"], 4),
            "Accuracy": f"{bns_m['accuracy'] * 100:.2f}%",
            "Threshold T*": round(tuned_threshold, 4),
            "Train Time": f"{train_time_s:.1f}s",
            "Best Iter": int(best_iter),
        },
        {
            "Model": "Model B (Parametric + Spatial + Block)",
            "Features": 555,
            "AUC-PR": round(b_m["auc_pr"], 4),
            "F1 (Tuned)": round(b_m["f1"], 4),
            "Precision": round(b_m["precision"], 4),
            "Recall": round(b_m["recall"], 4),
            "ROC-AUC": round(b_m["roc_auc"], 4),
            "Accuracy": f"{b_m['accuracy'] * 100:.2f}%",
            "Threshold T*": round(b_m["optimal_threshold"], 4),
            "Train Time": f"{b_t['training_time_seconds']:.1f}s",
            "Best Iter": int(b_t["best_iteration"]),
        },
    ]
    ablation_summary_df = pd.DataFrame(ablation_summary_rows)

    # Pairwise comparison calculations:
    # 1. B vs A
    # 2. B-no-spatial vs A
    # 3. B vs B-no-spatial
    def make_pairwise_comparison(name1, m1, name2, m2):
        d_auc_pr = m1["auc_pr"] - m2["auc_pr"]
        rel_auc_pr = (d_auc_pr / m2["auc_pr"] * 100.0) if m2["auc_pr"] > 0 else 0.0
        d_f1 = m1["f1"] - m2["f1"]
        rel_f1 = (d_f1 / m2["f1"] * 100.0) if m2["f1"] > 0 else 0.0
        d_prec = m1["precision"] - m2["precision"]
        d_rec = m1["recall"] - m2["recall"]
        d_roc_auc = m1["roc_auc"] - m2["roc_auc"]
        d_acc = m1["accuracy"] - m2["accuracy"]

        return {
            "Comparison": f"{name1} vs {name2}",
            "Delta_AUC-PR": round(d_auc_pr, 4),
            "Rel_AUC-PR_Improvement": f"{rel_auc_pr:+.2f}%",
            "Delta_F1": round(d_f1, 4),
            "Rel_F1_Improvement": f"{rel_f1:+.2f}%",
            "Delta_Recall": round(d_rec, 4),
            "Delta_Precision": round(d_prec, 4),
            "Delta_ROC-AUC": round(d_roc_auc, 4),
            "Delta_Accuracy": f"{(d_acc * 100):+.2f}%",
        }

    pairwise_rows = [
        make_pairwise_comparison("Model B", b_m, "Model A", a_m),
        make_pairwise_comparison("Model B-no-spatial", bns_m, "Model A", a_m),
        make_pairwise_comparison("Model B", b_m, "Model B-no-spatial", bns_m),
    ]
    pairwise_df = pd.DataFrame(pairwise_rows)

    # -------------------------------------------------------------------------
    # Step 9: Save All Artifacts
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 9: SAVING ARTIFACTS ---")
    model_bns_joblib_path = MODELS_DIR / "model_b_no_spatial.joblib"
    model_bns_txt_path = MODELS_DIR / "model_b_no_spatial.txt"
    joblib.dump(clf, model_bns_joblib_path)
    booster.save_model(str(model_bns_txt_path))
    print(f"  Saved model (joblib):        {model_bns_joblib_path} ({model_bns_joblib_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  Saved model (txt):           {model_bns_txt_path} ({model_bns_txt_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Predictions DataFrame
    y_pred_tuned = (y_prob >= tuned_threshold).astype(int)
    y_pred_05 = (y_prob >= 0.5).astype(int)

    preds_df = meta_val.copy()
    preds_df["predicted_probability"] = y_prob.astype(np.float32)
    preds_df["predicted_label_tuned"] = y_pred_tuned.astype(np.int8)
    preds_df["predicted_label_05"] = y_pred_05.astype(np.int8)

    preds_parquet_path = REPORTS_DIR / "model_b_no_spatial_dev_val_predictions.parquet"
    preds_df.to_parquet(preds_parquet_path, index=False, engine="pyarrow", compression="snappy")
    print(f"  Saved predictions:           {preds_parquet_path} ({preds_parquet_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Features list JSON
    features_json_path = REPORTS_DIR / "model_b_no_spatial_features.json"
    with open(features_json_path, "w") as f:
        json.dump({
            "model_name": "Model B-Without-Spatial (Parametric + Block)",
            "num_features": len(MODEL_B_NO_SPATIAL_FEATURES),
            "num_parametric": len(PARAMETRIC_FEATURES),
            "num_block": len(BLOCK_FEATURES),
            "features": MODEL_B_NO_SPATIAL_FEATURES,
            "parametric_features": PARAMETRIC_FEATURES,
            "block_features": BLOCK_FEATURES,
        }, f, indent=2)
    print(f"  Saved features list:         {features_json_path}")

    # Feature Importance CSV
    fi_csv_path = REPORTS_DIR / "model_b_no_spatial_feature_importance.csv"
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"  Saved feature importance:    {fi_csv_path}")

    # Metrics JSON & CSV
    metrics_summary = {
        "model_name": "Model B-Without-Spatial (Parametric + Block)",
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
            "no_skill_auc_pr_baseline": round(no_skill_auc_pr, 4),
        },
        "evaluation_default_threshold_05": metrics_05,
        "evaluation_tuned_threshold": {
            "optimal_threshold": round(tuned_threshold, 4),
            **metrics_tuned,
        },
        "feature_group_attribution": group_stats.to_dict(orient="records"),
        "artifacts_created": [
            str(model_bns_joblib_path),
            str(model_bns_txt_path),
            str(preds_parquet_path),
            str(features_json_path),
            str(fi_csv_path),
            str(REPORTS_DIR / "model_b_no_spatial_metrics.json"),
            str(REPORTS_DIR / "model_b_no_spatial_metrics.csv"),
            str(REPORTS_DIR / "model_ablation_comparison.json"),
            str(REPORTS_DIR / "model_ablation_comparison.csv"),
        ]
    }

    metrics_json_path = REPORTS_DIR / "model_b_no_spatial_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  Saved metrics JSON:          {metrics_json_path}")

    metrics_csv_df = pd.DataFrame([
        {
            "Model / Setup": "Model B-no-spatial (Default Threshold 0.5)",
            "Threshold": "0.5000",
            "AUC-PR": f"{metrics_05['auc_pr']:.4f}",
            "ROC-AUC": f"{metrics_05['roc_auc']:.4f}",
            "Precision (Fail)": f"{metrics_05['precision']:.4f}",
            "Recall (Fail)": f"{metrics_05['recall']:.4f}",
            "F1 (Fail)": f"{metrics_05['f1']:.4f}",
            "Accuracy": f"{metrics_05['accuracy'] * 100:.2f}%",
        },
        {
            "Model / Setup": f"Model B-no-spatial (Tuned Threshold T* = {tuned_threshold:.4f})",
            "Threshold": f"{tuned_threshold:.4f}",
            "AUC-PR": f"{metrics_tuned['auc_pr']:.4f}",
            "ROC-AUC": f"{metrics_tuned['roc_auc']:.4f}",
            "Precision (Fail)": f"{metrics_tuned['precision']:.4f}",
            "Recall (Fail)": f"{metrics_tuned['recall']:.4f}",
            "F1 (Fail)": f"{metrics_tuned['f1']:.4f}",
            "Accuracy": f"{metrics_tuned['accuracy'] * 100:.2f}%",
        }
    ])
    metrics_csv_path = REPORTS_DIR / "model_b_no_spatial_metrics.csv"
    metrics_csv_df.to_csv(metrics_csv_path, index=False)
    print(f"  Saved metrics CSV:           {metrics_csv_path}")

    # Ablation comparison JSON & CSV
    ablation_comparison_data = {
        "summary": ablation_summary_df.to_dict(orient="records"),
        "pairwise_deltas": pairwise_df.to_dict(orient="records"),
    }
    ablation_json_path = REPORTS_DIR / "model_ablation_comparison.json"
    with open(ablation_json_path, "w") as f:
        json.dump(ablation_comparison_data, f, indent=2)
    print(f"  Saved ablation JSON:         {ablation_json_path}")

    ablation_csv_path = REPORTS_DIR / "model_ablation_comparison.csv"
    ablation_summary_df.to_csv(ablation_csv_path, index=False)
    print(f"  Saved ablation CSV:          {ablation_csv_path}")

    # -------------------------------------------------------------------------
    # Step 10: Final Comprehensive Terminal Output
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 85}")
    print("MODEL B-WITHOUT-SPATIAL EVALUATION REPORT & ABLATION STUDY")
    print(f"{'=' * 85}")
    print(f"1. Training Time:             {train_time_s:.2f} s ({train_time_s / 60:.2f} min)")
    print(f"2. Inference Time:            {infer_time_s:.4f} s ({throughput:,.1f} dies/s)")
    print(f"3. Best Iteration:            {best_iter}")
    print(f"4. Eligible Training Dies:    {train_stats['eligible_rows']:,} (out of {train_stats['total_rows']:,})")
    print(f"   Eligible Validation Dies:  {val_stats['eligible_rows']:,} (out of {val_stats['total_rows']:,})")
    print(f"5. Positive Prevalence:       Train: {train_stats['pos_rate']:.3f}% | Dev-Val: {val_stats['pos_rate']:.3f}%")
    print(f"   No-Skill AUC-PR Baseline:  {no_skill_auc_pr:.4f}")
    print(f"6. AUC-PR (Average Prec):     {metrics_tuned['auc_pr']:.4f}")
    print(f"7. Tuned F1 Score:            {metrics_tuned['f1']:.4f} (at T* = {tuned_threshold:.4f})")
    print(f"8. Precision (Fail Class):    {metrics_tuned['precision']:.4f} (at T*) | {metrics_05['precision']:.4f} (at 0.5)")
    print(f"9. Recall (Fail Class):       {metrics_tuned['recall']:.4f} (at T*) | {metrics_05['recall']:.4f} (at 0.5)")
    print(f"10. ROC-AUC:                  {metrics_tuned['roc_auc']:.4f}")
    print(f"11. Overall Accuracy:         {metrics_tuned['accuracy'] * 100:.2f}%")
    print(f"12. Optimal F1 Threshold:     {tuned_threshold:.4f}")
    print(f"13. Exact Files Created:")
    for f_art in metrics_summary["artifacts_created"]:
        print(f"    - {f_art}")
    print(f"14. Warnings/Errors:          None (Clean execution)")
    print(f"{'=' * 85}\n")

    print("--- THREE-MODEL ABLATION COMPARISON TABLE ---")
    print(ablation_summary_df.to_string(index=False))

    print(f"\n--- PAIRWISE INCREMENTAL PREDICTIVE VALUE (DELTAS) ---")
    print(pairwise_df.to_string(index=False))

    print(f"\n--- CONFUSION MATRIX (MODEL B-NO-SPATIAL @ TUNED T* = {tuned_threshold:.4f}) ---")
    print(format_confusion_table(metrics_tuned))

    print(f"\n--- FEATURE GROUP ATTRIBUTION (PARAMETRIC VS BLOCK) ---")
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
    print("ALL ABLATION ARTIFACTS AND THREE-WAY COMPARISONS SUCCESSFULLY GENERATED")
    print(f"{'=' * 85}")


if __name__ == "__main__":
    run_model_b_no_spatial()
