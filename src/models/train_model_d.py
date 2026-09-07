"""
Model D Training and Controlled Ablation Pipeline.
Runs the first experimental evaluation of Model D on the canonical development split:
- dev_train: 640 wafers (~651,337 eligible dies)
- dev_val: 160 wafers (~137,576 eligible dies, 5,367 validation positives)
- Final 200-wafer test set remains COMPLETELY UNTOUCHED.

Controlled Ablations:
- D0: Canonical Model B (555 features) + CatBoost (baseline)
- D1: Model B + 10 PCA components (565 features)
- D2: D1 + Cluster / EDT topology features (572 features)
- D3: D2 + W=350 block rolling features (578 features)
- D4: Full Model D: D3 + Top-Hat + Geometric Shape + Bilinear Interactions (595 features)

Evaluates on identical dev_val rows and generates comprehensive diagnostic artifacts.
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Ensure repo root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
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
    MODEL_B_FEATURES,
    load_dataset,
    compute_metrics,
    tune_threshold,
)

MODEL_D_FEATURE_GROUPS_PATH = MODELS_DIR / "model_d_feature_groups.json"
DEV_TRAIN_ADDITIONAL_PARQUET = PROCESSED_DIR / "dev_train_model_d_additional.parquet"
DEV_VAL_ADDITIONAL_PARQUET = PROCESSED_DIR / "dev_val_model_d_additional.parquet"


def run_model_d_experiment():
    print(f"\n{'=' * 85}")
    print("STARTING MODEL D CONTROLLED ABLATION EXPERIMENT (D0 -> D4)")
    print(f"{'=' * 85}")
    print(f"Seed: {SEED}")
    print(f"Train Source: {DEV_TRAIN_PARQUET.name}")
    print(f"Validation Source: {DEV_VAL_PARQUET.name}")
    print(f"Predictive Task: Eligible Dies Only (old_label == 0)")
    print(f"Target: label (0 = healthy, 1 = newly failed)")
    print(f"{'=' * 85}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load Base Datasets (Canonical Model B Features)
    # -------------------------------------------------------------------------
    print("--- STEP 1: LOADING CANONICAL BASE FEATURES ---")
    t0_load = time.time()
    X_train_base, y_train, meta_train, train_stats = load_dataset(
        DEV_TRAIN_PARQUET, MODEL_B_FEATURES, eligible_only=True
    )
    X_val_base, y_val, meta_val, val_stats = load_dataset(
        DEV_VAL_PARQUET, MODEL_B_FEATURES, eligible_only=True
    )
    print(f"Base data loaded in {time.time() - t0_load:.2f}s.")
    print(f"Eligible counts: Train = {len(X_train_base):,} | Val = {len(X_val_base):,}")

    # Wafer isolation verification
    train_wafers = set(meta_train["wafer_id"].unique())
    val_wafers = set(meta_val["wafer_id"].unique())
    assert len(train_wafers & val_wafers) == 0, "CRITICAL ERROR: Wafer leakage detected!"
    print(f"Wafer isolation verified: 0 overlap between {len(train_wafers)} train wafers and {len(val_wafers)} val wafers.")

    # -------------------------------------------------------------------------
    # Step 2: Load Additional Feature Sets
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: LOADING ADDITIONAL MODEL D FEATURES ---")
    t0_add = time.time()
    assert DEV_TRAIN_ADDITIONAL_PARQUET.exists() and DEV_VAL_ADDITIONAL_PARQUET.exists(), (
        f"Missing additional feature parquets! Run src/features/model_d_features.py first."
    )
    add_train_df = pd.read_parquet(DEV_TRAIN_ADDITIONAL_PARQUET)
    add_val_df = pd.read_parquet(DEV_VAL_ADDITIONAL_PARQUET)
    print(f"Loaded additional features ({add_train_df.shape[1]} columns) in {time.time() - t0_add:.2f}s.")

    with open(MODEL_D_FEATURE_GROUPS_PATH, "r") as f:
        feature_groups = json.load(f)

    pca_cols = feature_groups["PCA_FEATURES"]
    cluster_cols = feature_groups["CLUSTER_EDT_FEATURES"]
    w350_cols = feature_groups["W350_FEATURES"]
    tophat_cols = feature_groups["TOPHAT_FEATURES"]
    geom_cols = feature_groups["GEOMETRIC_FEATURES"]
    inter_cols = feature_groups["INTERACTION_FEATURES"]

    print(f"Feature Groups:")
    print(f"  PCA Features:        {len(pca_cols)} ({pca_cols[0]} ... {pca_cols[-1]})")
    print(f"  Cluster/EDT:         {len(cluster_cols)} ({cluster_cols})")
    print(f"  W=350 Block:         {len(w350_cols)} ({w350_cols})")
    print(f"  Top-Hat Filters:     {len(tophat_cols)} ({tophat_cols})")
    print(f"  Geometric Shape:     {len(geom_cols)} ({geom_cols})")
    print(f"  Bilinear Multiplier: {len(inter_cols)} ({inter_cols})")

    # -------------------------------------------------------------------------
    # Step 3: Define Ablation Variants D0 -> D4
    # -------------------------------------------------------------------------
    ablation_definitions = {
        "D0": {
            "name": "D0 (Model B Baseline)",
            "description": "Canonical 555 Model B features + CatBoost baseline",
            "feature_cols": MODEL_B_FEATURES,
            "add_cols": [],
        },
        "D1": {
            "name": "D1 (+PCA)",
            "description": "Model B + 10-Component Parametric PCA",
            "feature_cols": MODEL_B_FEATURES + pca_cols,
            "add_cols": pca_cols,
        },
        "D2": {
            "name": "D2 (+Cluster/EDT)",
            "description": "D1 + Wafer Cluster Topology & Exact Distance Transform",
            "feature_cols": MODEL_B_FEATURES + pca_cols + cluster_cols,
            "add_cols": pca_cols + cluster_cols,
        },
        "D3": {
            "name": "D3 (+W350 Block)",
            "description": "D2 + W=350 Per-Die Block Rolling Statistics",
            "feature_cols": MODEL_B_FEATURES + pca_cols + cluster_cols + w350_cols,
            "add_cols": pca_cols + cluster_cols + w350_cols,
        },
        "D4": {
            "name": "D4 (Full Model D)",
            "description": "D3 + Top-Hat + Geometric Shape + Bilinear Interactions",
            "feature_cols": (
                MODEL_B_FEATURES
                + pca_cols
                + cluster_cols
                + w350_cols
                + tophat_cols
                + geom_cols
                + inter_cols
            ),
            "add_cols": pca_cols + cluster_cols + w350_cols + tophat_cols + geom_cols + inter_cols,
        },
    }

    # Class imbalance scale_pos_weight
    neg_count = train_stats["n_neg"]
    pos_count = train_stats["n_pos"]
    scale_pos_weight = float(neg_count / pos_count)
    print(f"\nClass imbalance scale_pos_weight: {scale_pos_weight:.6f} ({neg_count:,} neg / {pos_count:,} pos)")

    # -------------------------------------------------------------------------
    # Step 4: Run Ablation Experiments
    # -------------------------------------------------------------------------
    results = {}
    fitted_models = {}
    val_predictions = {}
    feature_importances = {}

    for var_key in ["D0", "D1", "D2", "D3", "D4"]:
        var_info = ablation_definitions[var_key]
        var_name = var_info["name"]
        add_cols = var_info["add_cols"]

        print(f"\n{'=' * 75}")
        print(f"TRAINING VARIANT: {var_name}")
        print(f"Features: {len(var_info['feature_cols'])} ({var_info['description']})")
        print(f"{'=' * 75}")

        # Construct feature matrices
        if not add_cols:
            X_tr = X_train_base
            X_va = X_val_base
        else:
            X_tr = pd.concat([X_train_base, add_train_df[add_cols]], axis=1)
            X_va = pd.concat([X_val_base, add_val_df[add_cols]], axis=1)

        # Standardized CatBoost GPU configuration
        clf = cb.CatBoostClassifier(
            iterations=1200,
            learning_rate=0.04,
            depth=7,
            l2_leaf_reg=6.0,
            loss_function="Logloss",
            eval_metric="Logloss",
            scale_pos_weight=scale_pos_weight,
            task_type="GPU",
            early_stopping_rounds=50,
            random_seed=SEED,
            verbose=200,
        )

        t0_fit = time.time()
        clf.fit(X_tr, y_train, eval_set=(X_va, y_val), use_best_model=True)
        fit_time = time.time() - t0_fit
        best_iter = clf.get_best_iteration()

        # Inference
        t0_inf = time.time()
        probs = clf.predict_proba(X_va)[:, 1]
        inf_time = time.time() - t0_inf

        # Threshold tuning
        tuning_res = tune_threshold(y_val, probs, num_steps=200)
        best_thresh = tuning_res["best_threshold"]
        m = tuning_res["metrics"]

        # Feature importance
        fi = clf.get_feature_importance()
        fi_df = pd.DataFrame({
            "feature": var_info["feature_cols"],
            "importance": fi,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        res_dict = {
            "variant": var_key,
            "name": var_name,
            "description": var_info["description"],
            "num_features": len(var_info["feature_cols"]),
            "best_iteration": int(best_iter),
            "train_time_seconds": float(fit_time),
            "inference_time_seconds": float(inf_time),
            "auc_pr": float(m["auc_pr"]),
            "roc_auc": float(m["roc_auc"]),
            "tuned_threshold": float(best_thresh),
            "f1": float(m["f1"]),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "specificity": float(m["pass_recall"]),
            "accuracy": float(m["accuracy"]),
            "tp": int(m["tp"]),
            "fp": int(m["fp"]),
            "fn": int(m["fn"]),
            "tn": int(m["tn"]),
        }

        results[var_key] = res_dict
        fitted_models[var_key] = clf
        val_predictions[var_key] = probs
        feature_importances[var_key] = fi_df

        print(f"\n--> {var_key} RESULTS (at Tuned Threshold {best_thresh:.4f}):")
        print(f"    AUC-PR:      {m['auc_pr']:.4f}")
        print(f"    ROC-AUC:     {m['roc_auc']:.4f}")
        print(f"    F1-Score:    {m['f1']:.4f} (Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f})")
        print(f"    Defects TP:  {m['tp']:,} / {val_stats['n_pos']:,} ({m['tp']/val_stats['n_pos']:.2%}) | FP: {m['fp']:,}")
        print(f"    Train Time:  {fit_time:.1f}s | Best Iteration: {best_iter}")

    # -------------------------------------------------------------------------
    # Step 5: Summary Ablation Table & Canonical Benchmarking
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("MODEL D CONTROLLED ABLATION SUMMARY TABLE (D0 -> D4)")
    print("=" * 95)
    ablation_df = pd.DataFrame(list(results.values()))
    print(ablation_df[["variant", "num_features", "auc_pr", "roc_auc", "f1", "precision", "recall", "tp", "fp", "train_time_seconds"]].to_string(index=False))

    # Identify Best Model D Variant
    best_variant_key = max(results.keys(), key=lambda k: results[k]["auc_pr"])
    best_res = results[best_variant_key]
    best_model = fitted_models[best_variant_key]
    best_probs = val_predictions[best_variant_key]
    best_fi = feature_importances[best_variant_key]

    print("\n" + "=" * 95)
    print(f"WINNING VARIANT: {best_res['name']} with AUC-PR = {best_res['auc_pr']:.4f}")
    print("=" * 95)

    # Compare against Canonical Benchmarks
    CANONICAL_BENCHMARKS = {
        "Model A LightGBM": {"auc_pr": 0.4937, "roc_auc": 0.8228, "f1": 0.4984},
        "Model B LightGBM": {"auc_pr": 0.5543, "roc_auc": 0.8760, "f1": 0.5397},
        "Model C1 CNN": {"auc_pr": 0.5766, "roc_auc": 0.8920, "f1": 0.5524},
        "Model C2 CNN": {"auc_pr": 0.5717, "roc_auc": 0.8872, "f1": 0.5480},
        "Grand Tri-Blend Champion": {"auc_pr": 0.57945, "roc_auc": 0.89335, "f1": 0.55530},
    }

    print("\nBENCHMARK COMPARISON AGAINST EXISTING MODELS (CANONICAL DEV-VAL):")
    for b_name, b_vals in CANONICAL_BENCHMARKS.items():
        delta_pr = best_res["auc_pr"] - b_vals["auc_pr"]
        delta_f1 = best_res["f1"] - b_vals["f1"]
        print(f"  vs. {b_name:<26}: AUC-PR {b_vals['auc_pr']:.4f} -> {best_res['auc_pr']:.4f} (Delta: {delta_pr:+.4f}) | F1 (Delta: {delta_f1:+.4f})")

    # -------------------------------------------------------------------------
    # Step 6: Save Model D Artifacts
    # -------------------------------------------------------------------------
    print("\n--- STEP 6: SAVING MODEL D ARTIFACTS ---")
    cbm_path = MODELS_DIR / "model_d_catboost.cbm"
    best_model.save_model(str(cbm_path))
    print(f"Saved best model checkpoint: {cbm_path.name}")

    config_path = MODELS_DIR / "model_d_catboost_config.json"
    config_dict = {
        "best_variant": best_variant_key,
        "variant_name": best_res["name"],
        "num_features": best_res["num_features"],
        "hyperparameters": {
            "iterations": 1200,
            "learning_rate": 0.04,
            "depth": 7,
            "l2_leaf_reg": 6.0,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "scale_pos_weight": scale_pos_weight,
            "task_type": "GPU",
            "early_stopping_rounds": 50,
            "random_seed": SEED,
        },
        "ablation_results": results,
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved configuration: {config_path.name}")

    feature_list_path = MODELS_DIR / "model_d_feature_list.json"
    with open(feature_list_path, "w") as f:
        json.dump({
            "variant": best_variant_key,
            "num_features": len(ablation_definitions[best_variant_key]["feature_cols"]),
            "features": ablation_definitions[best_variant_key]["feature_cols"],
        }, f, indent=2)
    print(f"Saved feature list: {feature_list_path.name}")

    fi_path = MODELS_DIR / "model_d_feature_importance.csv"
    best_fi.to_csv(fi_path, index=False)
    print(f"Saved feature importances: {fi_path.name}")

    metrics_path = MODELS_DIR / "model_d_metrics.json"
    metrics_export = {
        "model_name": f"Model D ({best_res['name']})",
        "best_variant": best_variant_key,
        "metrics": best_res,
        "canonical_benchmarks": CANONICAL_BENCHMARKS,
        "ablation_summary": results,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"Saved metrics JSON: {metrics_path.name}")

    # Save predictions parquet
    meta_val_out = meta_val.copy()
    meta_val_out["predicted_probability"] = best_probs.astype(np.float32)
    meta_val_out["predicted_label"] = (best_probs >= best_res["tuned_threshold"]).astype(np.int8)
    preds_path = MODELS_DIR / "model_d_predictions.parquet"
    meta_val_out.to_parquet(preds_path, index=False)
    print(f"Saved validation predictions: {preds_path.name}")

    # -------------------------------------------------------------------------
    # Step 7: Generate Comprehensive Evaluation Report
    # -------------------------------------------------------------------------
    generate_markdown_report(results, best_res, best_fi, CANONICAL_BENCHMARKS)

    print("\n" + "=" * 85)
    print("MODEL D PIPELINE COMPLETE. STOPPING BEFORE CV AS INSTRUCTED.")
    print("=" * 85 + "\n")
    return results, best_res


def generate_markdown_report(results, best_res, best_fi, canonical_benchmarks):
    """
    Generates reports/MODEL_D_EVALUATION.md incorporating all user specifications.
    """
    report_path = REPORTS_DIR / "MODEL_D_EVALUATION.md"

    # Feature Provenance Table
    provenance_table = r"""
| Feature Name / Group | Information Used | Inference Available? | Uses `old_label`? | Fitted on `dev_train`? |
| :--- | :--- | :---: | :---: | :---: |
| **Model B Base (555)** | 500 Parametric + 19 Spatial + 36 Block Stats | ✅ Yes | ✅ Yes (pre-test only) | ❌ No (deterministic) |
| **`pca_01` ... `pca_10`** | 500 Parametric Tests | ✅ Yes | ❌ No | ✅ **Yes** (`StandardScaler` + `PCA(10)` on `dev_train`) |
| **`exact_edt_distance`** | Pre-test wafer defect coordinates | ✅ Yes | ✅ Yes (`old_label == 1`) | ❌ No (deterministic per wafer) |
| **`nearest_defect_cluster_size`** | Pre-test wafer defect 2D connected components | ✅ Yes | ✅ Yes (`old_label == 1`) | ❌ No (deterministic per wafer) |
| **`nearest_defect_log_cluster_size`** | Log of nearest pre-test cluster size | ✅ Yes | ✅ Yes (`old_label == 1`) | ❌ No (deterministic per wafer) |
| **`defect_vector_dr`, `defect_vector_dc`** | Relative vector from die to nearest pre-test defect | ✅ Yes | ✅ Yes (`old_label == 1`) | ❌ No (deterministic per wafer) |
| **`is_defect_neighbor_1hop`, `2hop`** | Discrete spatial adjacency flags (distance $\le 1.5, 2.5$) | ✅ Yes | ✅ Yes (`old_label == 1`) | ❌ No (deterministic per wafer) |
| **`block_rolling_mean_350`** | Raw 2,000 block sequence (window 350) | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`max_rolling_mean_350`** | Maximum rolling mean over window 350 | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`min_rolling_mean_350`** | Minimum rolling mean over window 350 | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`max_rolling_std_350`** | Maximum rolling std over window 350 | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`max_rolling_mean_350_start_idx`** | Location index $[0, 1]$ of maximum rolling mean | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`burst_excess_350`** | Excess burst over die global block mean | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`tophat_peak_100`, `200`** | 1D Morphological white top-hat filter peak | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`tophat_energy_100`, `200`** | 1D Morphological white top-hat squared energy | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`zernike_Z1_neg1` ... `Z4_0`** | Normalized circular die coordinates $(\rho, \phi)$ | ✅ Yes | ❌ No | ❌ No (closed-form formula) |
| **`reticle_pos`, `is_reticle_corner`** | $4\times 4$ stepper exposure grid coordinates | ✅ Yes | ❌ No | ❌ No (closed-form formula) |
| **`inter_pca01_x_roll350`** | Product: `pca_01` $\times \max(0, \text{Roll350} - 100)$ | ✅ Yes | ❌ No | ✅ Indirect (uses `pca_01`) |
| **`inter_pca01_x_cluster_size`** | Product: `pca_01` $\times$ `log_cluster_size` | ✅ Yes | ✅ Yes (`old_label == 1`) | ✅ Indirect (uses `pca_01`) |
| **`inter_pca01_x_edt`** | Ratio: `pca_01` / (`exact_edt_distance` + 0.02) | ✅ Yes | ✅ Yes (`old_label == 1`) | ✅ Indirect (uses `pca_01`) |
| **`inter_pca01_x_edge`** | Product: `pca_01` $\times (1 - \text{edge\_dist})$ | ✅ Yes | ❌ No | ✅ Indirect (uses `pca_01`) |
| **`inter_roll350_x_cluster`** | Product: `burst_350` $\times$ `log_cluster_size` | ✅ Yes | ✅ Yes (`old_label == 1`) | ❌ No (deterministic) |
"""

    # Ablation markdown table
    ablation_rows = []
    for k, v in results.items():
        ablation_rows.append(
            f"| **{v['variant']}** | {v['num_features']} | **{v['auc_pr']:.4f}** | {v['roc_auc']:.4f} | {v['f1']:.4f} | "
            f"{v['precision']:.4f} | {v['recall']:.4f} | {v['specificity']:.4f} | {v['tp']:,} | {v['fp']:,} | "
            f"{v['tuned_threshold']:.4f} | {v['train_time_seconds']:.1f}s |"
        )
    ablation_table_str = "\n".join(ablation_rows)

    # Top 25 feature importances
    top_fi_rows = []
    for idx, row in best_fi.head(25).iterrows():
        top_fi_rows.append(f"| {idx+1} | `{row['feature']}` | {row['importance']:.4f} |")
    top_fi_table_str = "\n".join(top_fi_rows)

    # Benchmark comparison table
    benchmark_rows = []
    for b_name, b_val in canonical_benchmarks.items():
        d_pr = best_res["auc_pr"] - b_val["auc_pr"]
        d_f1 = best_res["f1"] - b_val["f1"]
        benchmark_rows.append(
            f"| **{b_name}** | {b_val['auc_pr']:.4f} | {b_val['roc_auc']:.4f} | {b_val['f1']:.4f} | **{d_pr:+.4f}** | {d_f1:+.4f} |"
        )
    benchmark_table_str = "\n".join(benchmark_rows)

    # Determine recommendation
    lift_over_b = best_res["auc_pr"] - canonical_benchmarks["Model B LightGBM"]["auc_pr"]
    lift_over_c1 = best_res["auc_pr"] - canonical_benchmarks["Model C1 CNN"]["auc_pr"]
    lift_over_champ = best_res["auc_pr"] - canonical_benchmarks["Grand Tri-Blend Champion"]["auc_pr"]

    if lift_over_b > 0.010 and best_res["auc_pr"] > 0.570:
        rec_verdict = "PROMISING — CONSIDER FOR 5-FOLD CV EVALUATION"
        rec_details = (
            f"Model D ({best_res['name']}) achieved AUC-PR = {best_res['auc_pr']:.4f}, demonstrating a decisive lift of "
            f"{lift_over_b:+.4f} over Model B LightGBM (0.5543). It captures significant complementary signal from the "
            f"Adit-inspired feature engineering."
        )
    elif lift_over_b > 0.002:
        rec_verdict = "MARGINAL IMPROVEMENT OVER MODEL B — CAUTION ON 5-FOLD PROMOTION"
        rec_details = (
            f"Model D achieved AUC-PR = {best_res['auc_pr']:.4f}, yielding a modest lift of {lift_over_b:+.4f} over Model B, "
            f"but remains below Model C1 CNN (0.5766). While feature engineering improved tabular performance, the compute cost "
            f"of running full 5-fold CV should be weighed against direct interpretability and final test preparation."
        )
    else:
        rec_verdict = "DISCARD OR REFINE — INSUFFICIENT LIFT OVER MODEL B"
        rec_details = (
            f"Model D achieved AUC-PR = {best_res['auc_pr']:.4f}, failing to meaningfully improve over baseline Model B LightGBM (0.5543). "
            f"Do NOT proceed to 5-fold CV. The existing Grand Tri-Blend champion remains our strongest model."
        )

    content = f"""# Model D Evaluation & Controlled Feature Ablation Report

**Date**: September 8, 2026  
**Subject**: Evaluation of Model D (Targeted Hybrid CatBoost) on Canonical Development Split  
**Evaluation Scope**: Strictly Canonical Development Split (640 train wafers, 160 dev-val wafers, 137,576 eligible dies, 5,367 validation positives).  
**Holdout Protection**: **Final 200 test wafers remain COMPLETELY UNTOUCHED.** Zero 5-fold CV executed in this step.

---

## 1. Executive Summary & Core Results

Model D investigates whether targeted, leak-free feature engineering inspired by teammate Adit's branch can improve tabular performance over our canonical **Model B (LightGBM baseline, AUC-PR = 0.5543)** and complement our deep sequence models (**Model C1 CNN, AUC-PR = 0.5766**).

### Winning Model D Configuration:
- **Best Variant**: **{best_res['name']}**
- **Total Features**: {best_res['num_features']} (555 base + targeted additions)
- **Validation AUC-PR**: **{best_res['auc_pr']:.4f}**
- **Validation ROC-AUC**: **{best_res['roc_auc']:.4f}**
- **Tuned F1-Score**: **{best_res['f1']:.4f}** (Precision: {best_res['precision']:.4f}, Recall: {best_res['recall']:.4f})
- **Defects Caught (TP)**: **{best_res['tp']:,} / 5,367** ({best_res['tp']/5367:.2%}) at {best_res['fp']:,} False Positives
- **Optimal Threshold**: $T^* = {best_res['tuned_threshold']:.4f}$

---

## 2. Controlled Feature Ablation Table (D0 through D4)

Every variant was evaluated on the **exact same 137,576 eligible validation dies** using identical CatBoost GPU hyperparameters (`iterations=1200, lr=0.04, depth=7, l2_leaf_reg=6.0, scale_pos_weight=26.07, seed=42`):

| Variant | Features | AUC-PR | ROC-AUC | F1 | Precision | Recall | Specificity | TP | FP | Thresh | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{ablation_table_str}

---

## 3. Comparison Against Canonical Benchmarks

| Model Architecture | Canonical AUC-PR | Canonical ROC-AUC | Canonical F1 | Model D Delta (AUC-PR) | Model D Delta (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
{benchmark_table_str}

---

## 4. Feature Provenance & Leakage Audit

All additional features obey strict causal and inference constraints:

{provenance_table}

---

## 5. Feature Importance Analysis (Top 25 Features)

| Rank | Feature Name | CatBoost Gain Importance |
| :---: | :--- | :---: |
{top_fi_table_str}

---

## 6. Analysis: Which Feature Groups Actually Helped?

1. **Baseline CatBoost (D0 vs. LightGBM B)**:
   - D0 established the CatBoost baseline on canonical 555 features at AUC-PR = **{results['D0']['auc_pr']:.4f}** (compared to LightGBM B at 0.5543).
2. **Impact of Parametric PCA (D1 vs. D0)**:
   - Adding 10 PCA components shifted AUC-PR from **{results['D0']['auc_pr']:.4f} to {results['D1']['auc_pr']:.4f}** (Delta: **{results['D1']['auc_pr'] - results['D0']['auc_pr']:+.4f}**).
3. **Impact of Wafer Cluster Topology & EDT (D2 vs. D1)**:
   - Adding cluster connected component sizes and exact Euclidean distance transforms shifted AUC-PR from **{results['D1']['auc_pr']:.4f} to {results['D2']['auc_pr']:.4f}** (Delta: **{results['D2']['auc_pr'] - results['D1']['auc_pr']:+.4f}**).
4. **Impact of W=350 Block Features (D3 vs. D2)**:
   - Adding per-die W=350 rolling mean and std features shifted AUC-PR from **{results['D2']['auc_pr']:.4f} to {results['D3']['auc_pr']:.4f}** (Delta: **{results['D3']['auc_pr'] - results['D2']['auc_pr']:+.4f}**).
5. **Impact of Full Model D (D4 vs. D3)**:
   - Adding top-hat filters, geometric Zernike coordinates, and cross-resolution bilinear interactions shifted AUC-PR from **{results['D3']['auc_pr']:.4f} to {results['D4']['auc_pr']:.4f}** (Delta: **{results['D4']['auc_pr'] - results['D3']['auc_pr']:+.4f}**).

---

## 7. Decisive Recommendation

### Status: **{rec_verdict}**

{rec_details}

> [!IMPORTANT]
> **Next Steps**:
> In accordance with instructions, execution has stopped. No 5-fold CV has been started, and our champion Grand Tri-Blend remains completely untouched. Please review these results to decide whether to proceed with 5-fold GroupKFold promotion or freeze the current champion.
"""

    with open(report_path, "w") as f:
        f.write(content)
    print(f"Saved comprehensive evaluation report to: {report_path.name}")


if __name__ == "__main__":
    run_model_d_experiment()
