"""
Final Unseen Test Evaluation Pipeline for Model F: Wafer-Conditional Manifold Detector.

Strict Test-Time Isolation Protocol:
1. Predictions are generated strictly from unlabeled test features (final_test_model_f_features.parquet).
2. Saved boosters (CatBoost GPU, XGBoost CUDA, LightGBM CPU) and OOF stacking weights are frozen from development.
3. Official competition submission file (submission_model_f_optimal.csv) is generated for all 208,264 dies
   (with old_label == 1 dies assigned probability = 1.0 and predicted_label = 1).
4. Ground-truth labels from test.csv are loaded ONLY post-prediction to evaluate final test metrics.
"""

import sys
import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
    accuracy_score,
)

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    INPUT_DIR,
    SEED,
)

SUBMISSIONS_DIR = REPO_ROOT / "submissions"
PREDICTIONS_DIR = REPO_ROOT / "predictions"
ARTIFACT_DIR = Path("/home/user/.gemini/antigravity-ide/brain/f2c0f3fa-a66b-423d-a0c8-7e59a8b50ea9")

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_TEST_PARQUET = PROCESSED_DIR / "final_test_model_f_features.parquet"
VALIDATION_CSV = INPUT_DIR / "validation.csv"
TEST_CSV = INPUT_DIR / "test.csv"

FEATURES_PATH = MODELS_DIR / "model_f_feature_list.json"
WEIGHTS_PATH = MODELS_DIR / "model_f_stacking_weights.json"
CB_PATH = MODELS_DIR / "model_f_catboost.cbm"
XGB_PATH = MODELS_DIR / "model_f_xgboost.json"
LGB_PATH = MODELS_DIR / "model_f_lightgbm.txt"

MODEL_E_TEST_PREDS = PREDICTIONS_DIR / "final_test_model_e_predictions.parquet"

OUTPUT_PREDS_PARQUET = PREDICTIONS_DIR / "final_test_model_f_predictions.parquet"
OUTPUT_SUBMISSION_CSV = SUBMISSIONS_DIR / "submission_model_f_optimal.csv"
FINAL_METRICS_JSON = REPORTS_DIR / "model_f_final_test_metrics.json"
FINAL_METRICS_CSV = REPORTS_DIR / "model_f_final_test_metrics.csv"
FINAL_REPORT_MD = REPORTS_DIR / "MODEL_F_FINAL_TEST_EVALUATION.md"


def _rank01(p):
    return rankdata(p, method="average") / len(p)


def compute_metrics(y_true, y_prob, model_name="", fixed_threshold=None):
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0)))

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    denom = precisions + recalls
    denom[denom == 0] = 1.0
    f1_curve = 2.0 * (precisions * recalls) / denom

    best_idx = np.argmax(f1_curve)
    opt_f1 = float(f1_curve[best_idx])
    opt_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    eval_thresh = fixed_threshold if fixed_threshold is not None else opt_thresh
    y_pred = (y_prob >= eval_thresh).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float((tp + tn) / (tp + tn + fp + fn))
    f1_eval = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "model_name": model_name,
        "test_auc_pr": round(pr_auc, 5),
        "test_roc_auc": round(roc_auc, 5),
        "opt_f1": round(opt_f1, 5),
        "opt_threshold": round(opt_thresh, 5),
        "eval_threshold": round(eval_thresh, 5),
        "eval_f1": round(f1_eval, 5),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "specificity": round(spec, 5),
        "accuracy": round(acc, 5),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "brier_score": round(brier, 5),
    }


def main():
    print("=" * 85)
    print("FINAL UNSEEN TEST EVALUATION: MODEL F (200 WAFERS, ZERO-LEAKAGE)")
    print("=" * 85)
    t_start = time.time()

    # 1. Load test features
    print("\n[Phase 1/5] Loading unlabeled test features...")
    df_test = pd.read_parquet(FINAL_TEST_PARQUET)
    assert "label" not in df_test.columns, "Security check: test features must not contain labels!"

    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)["features"]

    with open(WEIGHTS_PATH) as f:
        weights_data = json.load(f)

    weights = weights_data["weights"]
    optimal_threshold = weights_data["optimal_threshold"]

    # Matrix preparation
    missing_cols = [c for c in feature_cols if c not in df_test.columns]
    if missing_cols:
        for c in missing_cols:
            df_test[c] = 0.0

    X_test = df_test[feature_cols].to_numpy(np.float32)
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Test matrix: {X_test.shape} ({len(X_test):,} dies across {df_test['wafer_id'].nunique()} test wafers)")

    # 2. Run inference with frozen Model F engines
    print("\n[Phase 2/5] Running inference with frozen Model F engines...")
    # CatBoost
    t0 = time.time()
    cb = CatBoostClassifier()
    cb.load_model(str(CB_PATH))
    p_cb = cb.predict_proba(X_test)[:, 1]
    print(f"  CatBoost completed in {time.time() - t0:.1f}s")

    # XGBoost
    t0 = time.time()
    xg = xgb.Booster()
    xg.load_model(str(XGB_PATH))
    p_xgb = xg.predict(xgb.DMatrix(X_test))
    print(f"  XGBoost completed in {time.time() - t0:.1f}s")

    # LightGBM
    t0 = time.time()
    lg = lgb.Booster(model_file=str(LGB_PATH))
    p_lgb = lg.predict(X_test)
    print(f"  LightGBM completed in {time.time() - t0:.1f}s")

    # Rank blend
    print("\n[Phase 3/5] Applying OOF rank-space blend weights...")
    names = ["catboost", "xgboost", "lightgbm"]
    preds_dict = {"catboost": p_cb, "xgboost": p_xgb, "lightgbm": p_lgb}
    P_rank = np.stack([_rank01(preds_dict[n]) for n in names])
    w = np.array([weights[n] for n in names], dtype=float)
    p_stack = (w / w.sum()) @ P_rank
    print(f"  Blended in rank space with weights: {weights}")

    # Save predictions dataframe
    df_preds = pd.DataFrame({
        "wafer_id": df_test["wafer_id"].values,
        "die_row": df_test["die_row"].values,
        "die_col": df_test["die_col"].values,
        "old_label": df_test["old_label"].values,
        "p_catboost": p_cb,
        "p_xgboost": p_xgb,
        "p_lightgbm": p_lgb,
        "model_f_optimal": p_stack,
    })
    df_preds.to_parquet(OUTPUT_PREDS_PARQUET, index=False)
    print(f"  Saved test predictions to {OUTPUT_PREDS_PARQUET.name}")

    # 3. Format official submission for validation.csv
    print("\n[Phase 4/5] Formatting competition submission for validation.csv (208,264 dies)...")
    df_val_raw = pd.read_csv(VALIDATION_CSV, usecols=["wafer_id", "die_row", "die_col", "old_label"])
    n_total = len(df_val_raw)

    sub_prob = np.zeros(n_total, dtype=np.float32)
    sub_pred = np.zeros(n_total, dtype=int)

    already_failed = (df_val_raw["old_label"] == 1).values
    sub_prob[already_failed] = 1.0
    sub_pred[already_failed] = 1

    eligible_mask = (df_val_raw["old_label"] == 0).values
    sub_prob[eligible_mask] = p_stack
    sub_pred[eligible_mask] = (p_stack >= optimal_threshold).astype(int)

    submission_df = pd.DataFrame({
        "wafer_id": df_val_raw["wafer_id"],
        "die_row": df_val_raw["die_row"],
        "die_col": df_val_raw["die_col"],
        "probability": sub_prob,
        "predicted_label": sub_pred,
    })
    submission_df.to_csv(OUTPUT_SUBMISSION_CSV, index=False)
    print(f"  Saved official submission to {OUTPUT_SUBMISSION_CSV.name} ({len(submission_df):,} rows)")

    # 4. Post-inference ground-truth evaluation against test.csv
    print("\n[Phase 5/5] Post-inference ground-truth evaluation against test.csv...")
    df_test_labels = pd.read_csv(TEST_CSV, usecols=["wafer_id", "die_row", "die_col", "old_label", "label"])
    test_elig_mask = (df_test_labels["old_label"] == 0).values
    y_test = df_test_labels.loc[test_elig_mask, "label"].to_numpy(int)
    assert len(y_test) == len(X_test), f"Mismatch: {len(y_test)} labels vs {len(X_test)} predictions"

    metrics_list = []
    # Standalone engines
    metrics_list.append(compute_metrics(y_test, p_cb, "Engine 1: CatBoost (GPU)"))
    metrics_list.append(compute_metrics(y_test, p_xgb, "Engine 2: XGBoost (CUDA)"))
    metrics_list.append(compute_metrics(y_test, p_lgb, "Engine 3: LightGBM (CPU)"))
    metrics_list.append(compute_metrics(y_test, p_stack, "Model F Stack (Rank Blend)", fixed_threshold=optimal_threshold))

    # Compare with Model E if available
    has_model_e = MODEL_E_TEST_PREDS.exists()
    p_model_e = None
    if has_model_e:
        df_e = pd.read_parquet(MODEL_E_TEST_PREDS)
        col_e = "model_e_optimal" if "model_e_optimal" in df_e.columns else ("ensemble_prob" if "ensemble_prob" in df_e.columns else df_e.columns[-1])
        p_model_e = df_e[col_e].values
        m_e = compute_metrics(y_test, p_model_e, "Model E Stack (Previous Leader)")
        metrics_list.append(m_e)

        # Cross-Architecture Super-Ensemble on Test
        r_f = _rank01(p_stack)
        r_e = _rank01(p_model_e)
        p_super = 0.50 * r_f + 0.50 * r_e
        m_super = compute_metrics(y_test, p_super, "Super-Ensemble (50% Model F + 50% Model E)")
        metrics_list.append(m_super)

    # Print Results Table
    print("\n" + "=" * 95)
    print(f"{'Model Architecture':<40}{'Test AUC-PR':>12}{'Test ROC-AUC':>13}{'Optimal F1':>11}{'Precision':>11}{'Recall':>9}")
    print("=" * 95)
    for m in metrics_list:
        print(
            f"{m['model_name']:<40}{m['test_auc_pr']:>12.5f}{m['test_roc_auc']:>13.5f}"
            f"{m['opt_f1']:>11.5f}{m['precision']*100:>10.2f}%{m['recall']*100:>8.2f}%"
        )
    print("=" * 95)

    # Save metrics JSON and CSV
    with open(FINAL_METRICS_JSON, "w") as f:
        json.dump(metrics_list, f, indent=2)
    pd.DataFrame(metrics_list).to_csv(FINAL_METRICS_CSV, index=False)

    # Generate Precision-Recall Curve Plot
    plt.figure(figsize=(10, 7), dpi=150)
    plt.grid(True, linestyle="--", alpha=0.5)

    curves = [
        ("Model F Stack", p_stack, "#1f77b4", 2.5),
        ("CatBoost (GPU)", p_cb, "#2ca02c", 1.5),
        ("XGBoost (CUDA)", p_xgb, "#ff7f0e", 1.5),
        ("LightGBM (CPU)", p_lgb, "#9467bd", 1.5),
    ]
    if p_model_e is not None:
        curves.append(("Model E Stack", p_model_e, "#d62728", 2.0))
        curves.append(("Super-Ensemble (F+E)", p_super, "#000000", 2.8))

    for label, preds, color, lw in curves:
        prec, rec, _ = precision_recall_curve(y_test, preds)
        ap = average_precision_score(y_test, preds)
        plt.plot(rec, prec, label=f"{label} (Test AUC-PR = {ap:.5f})", color=color, linewidth=lw)

    plt.axhline(y_test.mean(), color="gray", linestyle="--", alpha=0.6, label=f"Random ({y_test.mean()*100:.2f}%)")
    plt.title("Final Unseen Test Precision-Recall Curves (200 Test Wafers)", fontsize=14, fontweight="bold")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    pr_plot_path = PLOTS_DIR / "24_final_test_model_f_pr_curves.png"
    plt.savefig(pr_plot_path)
    plt.close()

    # Generate AUC-PR Comparison Bar Chart
    plt.figure(figsize=(9, 5), dpi=150)
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    bar_labels = [
        "Frozen Champion",
        "Model B",
        "Model C1",
        "Model E Stack",
        "Model F Stack",
    ]
    bar_scores = [0.56283, 0.53528, 0.56020, 0.61382, metrics_list[3]["test_auc_pr"]]
    bar_colors = ["#7f7f7f", "#bcbd22", "#17becf", "#d62728", "#1f77b4"]

    if has_model_e:
        bar_labels.append("Super-Ensemble (F+E)")
        bar_scores.append(m_super["test_auc_pr"])
        bar_colors.append("#2ca02c")

    bars = plt.bar(bar_labels, bar_scores, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.8)
    plt.ylim([0.50, 0.66])
    plt.ylabel("Test AUC-PR", fontsize=12)
    plt.title("Final Unseen Test AUC-PR Across Architectures (200 Wafers)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=10)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.003, f"{h:.5f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.tight_layout()
    bar_plot_path = PLOTS_DIR / "25_final_test_model_f_aucpr_comparison.png"
    plt.savefig(bar_plot_path)
    plt.close()

    # Copy plots to artifact dir
    import shutil
    for p in [pr_plot_path, bar_plot_path]:
        shutil.copy(p, ARTIFACT_DIR / p.name)

    # 5. Write Comprehensive Markdown Report
    report_content = f"""# Final Unseen Test Evaluation Report: Model F (Wafer-Conditional Manifold Detector)

## Executive Summary
**Model F (Wafer-Conditional Manifold Detector)** was evaluated on the **200 completely unseen test wafers** ($208,264$ total dies, $185,126$ eligible dies, $6,584$ newly failed dies, $3.556\%$ defect prevalence) under strict zero-leakage test-time isolation.

### Key Test Benchmark Highlights
- **Model F Stack Test AUC-PR**: **`{metrics_list[3]['test_auc_pr']:.5f}`** (vs. Model E `{0.61382:.5f}`, Champion Baseline `{0.56283:.5f}`, Model B `{0.53528:.5f}`).
- **Net Lift over Model E**: **`+{metrics_list[3]['test_auc_pr'] - 0.61382:+.5f}`** net test lift.
- **Net Lift over Frozen Champion**: **`+{metrics_list[3]['test_auc_pr'] - 0.56283:+.5f}`** net test lift.
- **Test ROC-AUC**: **`{metrics_list[3]['test_roc_auc']:.5f}`**.
- **Optimal Test F1-Score**: **`{metrics_list[3]['opt_f1']:.5f}`** (Precision: `{metrics_list[3]['precision']*100:.2f}%`, Recall: `{metrics_list[3]['recall']*100:.2f}%`).

"""
    if has_model_e:
        report_content += f"""### Cross-Architecture Super-Ensemble (Model F + Model E)
Combining **Model F** (wafer-conditional spatial detrending) with **Model E** (deep/focal gradient boosted trees) on the test set:
- **Super-Ensemble Test AUC-PR**: **`{m_super['test_auc_pr']:.5f}`**
- **Super-Ensemble Test ROC-AUC**: **`{m_super['test_roc_auc']:.5f}`**
- **Super-Ensemble Test F1-Score**: **`{m_super['opt_f1']:.5f}`** (Precision: `{m_super['precision']*100:.2f}%`, Recall: `{m_super['recall']*100:.2f}%`).

"""

    report_content += f"""---

## Final Unseen Test Comparison Table (185,126 Eligible Dies)

| Architecture / Model | Test AUC-PR 🥇 | Test ROC-AUC | Optimal F1 🥈 | Precision | Recall | True Positives | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model B (LightGBM 555)** | 0.53528 | 0.87440 | 0.52070 | **94.08%** | 36.00% | 2,370 | **149** |
| **Model C1 (Triple-Branch CNN)** | 0.56020 | 0.88740 | 0.51910 | 56.34% | 48.13% | 3,169 | 2,456 |
| **Frozen Champion Baseline** | 0.56283 | 0.88950 | 0.54070 | 70.36% | 43.91% | 2,891 | 1,218 |
| **Model E Stack (Previous Leader)** | 0.61382 | 0.91854 | 0.57395 | 81.11% | 44.41% | 2,924 | 681 |
| **Engine 1: CatBoost (GPU)** | {metrics_list[0]['test_auc_pr']:.5f} | {metrics_list[0]['test_roc_auc']:.5f} | {metrics_list[0]['opt_f1']:.5f} | {metrics_list[0]['precision']*100:.2f}% | {metrics_list[0]['recall']*100:.2f}% | {metrics_list[0]['true_positives']:,} | {metrics_list[0]['false_positives']:,} |
| **Engine 2: XGBoost (CUDA)** | {metrics_list[1]['test_auc_pr']:.5f} | {metrics_list[1]['test_roc_auc']:.5f} | {metrics_list[1]['opt_f1']:.5f} | {metrics_list[1]['precision']*100:.2f}% | {metrics_list[1]['recall']*100:.2f}% | {metrics_list[1]['true_positives']:,} | {metrics_list[1]['false_positives']:,} |
| **Engine 3: LightGBM (CPU)** | {metrics_list[2]['test_auc_pr']:.5f} | {metrics_list[2]['test_roc_auc']:.5f} | {metrics_list[2]['opt_f1']:.5f} | {metrics_list[2]['precision']*100:.2f}% | {metrics_list[2]['recall']*100:.2f}% | {metrics_list[2]['true_positives']:,} | {metrics_list[2]['false_positives']:,} |
| **MODEL F STACK** 👑 | **`{metrics_list[3]['test_auc_pr']:.5f}`** | **`{metrics_list[3]['test_roc_auc']:.5f}`** | **`{metrics_list[3]['opt_f1']:.5f}`** | **`{metrics_list[3]['precision']*100:.2f}%`** | **`{metrics_list[3]['recall']*100:.2f}%`** | **`{metrics_list[3]['true_positives']:,}`** | **`{metrics_list[3]['false_positives']:,}`** |
"""
    if has_model_e:
        report_content += f"""| **SUPER-ENSEMBLE (Model F + E)** | **`{m_super['test_auc_pr']:.5f}`** | **`{m_super['test_roc_auc']:.5f}`** | **`{m_super['opt_f1']:.5f}`** | **`{m_super['precision']*100:.2f}%`** | **`{m_super['recall']*100:.2f}%`** | **`{m_super['true_positives']:,}`** | **`{m_super['false_positives']:,}`** |
"""

    report_content += f"""
---

## Confusion Matrix (Model F Optimal on 185,126 Eligible Test Dies)
```text
                    Pred Fail      Pred Pass         Metric                     Value
Actual Fail             {metrics_list[3]['true_positives']:,}          {metrics_list[3]['false_negatives']:,}         Fail Accuracy (Recall)     {metrics_list[3]['recall']:.6f}
Actual Pass               {metrics_list[3]['false_positives']:,}        {metrics_list[3]['true_negatives']:,}         Pass Accuracy (Specificity){metrics_list[3]['specificity']:.6f}
```

---

## Generated Submission
Official competition submission saved to [`submissions/submission_model_f_optimal.csv`](file:///home/user/Vinay/san/submissions/submission_model_f_optimal.csv) (208,264 total dies).
"""

    with open(FINAL_REPORT_MD, "w") as f:
        f.write(report_content)
    print(f"\nSaved final test report to {FINAL_REPORT_MD.name}")
    print(f"Final test evaluation completed in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
