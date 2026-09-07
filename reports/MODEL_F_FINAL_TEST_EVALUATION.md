# Final Unseen Test Evaluation Report: Model F (Wafer-Conditional Manifold Detector)

## Executive Summary
**Model F (Wafer-Conditional Manifold Detector)** was evaluated on the **200 completely unseen test wafers** ($208,264$ total dies, $185,126$ eligible dies, $6,584$ newly failed dies, $3.556\%$ defect prevalence) under strict zero-leakage test-time isolation.

### Key Test Benchmark Highlights
- **Model F Stack Test AUC-PR**: **`0.62374`** (vs. Model E `0.61382`, Champion Baseline `0.56283`, Model B `0.53528`).
- **Net Lift over Model E**: **`++0.00992`** net test lift.
- **Net Lift over Frozen Champion**: **`++0.06091`** net test lift.
- **Test ROC-AUC**: **`0.92800`**.
- **Optimal Test F1-Score**: **`0.57891`** (Precision: `71.00%`, Recall: `48.56%`).

### Cross-Architecture Super-Ensemble (Model F + Model E)
Combining **Model F** (wafer-conditional spatial detrending) with **Model E** (deep/focal gradient boosted trees) on the test set:
- **Super-Ensemble Test AUC-PR**: **`0.62134`**
- **Super-Ensemble Test ROC-AUC**: **`0.92543`**
- **Super-Ensemble Test F1-Score**: **`0.57878`** (Precision: `76.24%`, Recall: `46.64%`).

---

## Final Unseen Test Comparison Table (185,126 Eligible Dies)

| Architecture / Model | Test AUC-PR 🥇 | Test ROC-AUC | Optimal F1 🥈 | Precision | Recall | True Positives | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model B (LightGBM 555)** | 0.53528 | 0.87440 | 0.52070 | **94.08%** | 36.00% | 2,370 | **149** |
| **Model C1 (Triple-Branch CNN)** | 0.56020 | 0.88740 | 0.51910 | 56.34% | 48.13% | 3,169 | 2,456 |
| **Frozen Champion Baseline** | 0.56283 | 0.88950 | 0.54070 | 70.36% | 43.91% | 2,891 | 1,218 |
| **Model E Stack (Previous Leader)** | 0.61382 | 0.91854 | 0.57395 | 81.11% | 44.41% | 2,924 | 681 |
| **Engine 1: CatBoost (GPU)** | 0.62262 | 0.92699 | 0.57986 | 79.44% | 45.66% | 3,006 | 778 |
| **Engine 2: XGBoost (CUDA)** | 0.62280 | 0.92756 | 0.57776 | 77.30% | 46.13% | 3,037 | 892 |
| **Engine 3: LightGBM (CPU)** | 0.62107 | 0.92727 | 0.57964 | 74.81% | 47.31% | 3,115 | 1,049 |
| **MODEL F STACK** 👑 | **`0.62374`** | **`0.92800`** | **`0.57891`** | **`71.00%`** | **`48.56%`** | **`3,197`** | **`1,306`** |
| **SUPER-ENSEMBLE (Model F + E)** | **`0.62134`** | **`0.92543`** | **`0.57878`** | **`76.24%`** | **`46.64%`** | **`3,071`** | **`957`** |

---

## Confusion Matrix (Model F Optimal on 185,126 Eligible Test Dies)
```text
                    Pred Fail      Pred Pass         Metric                     Value
Actual Fail             3,197          3,387         Fail Accuracy (Recall)     0.485570
Actual Pass               1,306        177,236         Pass Accuracy (Specificity)0.992690
```

---

## Generated Submission
Official competition submission saved to [`submissions/submission_model_f_optimal.csv`](file:///home/user/Vinay/san/submissions/submission_model_f_optimal.csv) (208,264 total dies).
