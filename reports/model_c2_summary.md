# SanDisk Hackathon — Model C2: Multi-Scale 1D CNN Evaluation Report

```text
=====================================================================================
CURRENT CHAMPION BEFORE C2:
  B + C1 Ensemble — AUC-PR: 0.57861 | ROC-AUC: 0.89203 | F1: 0.55342

MODEL C2 STANDALONE:
  AUC-PR: 0.57173 | ROC-AUC: 0.88932 | Tuned F1: 0.55085

BEST C2 ENSEMBLE:
  Tri-Blend: C2 + LightGBM B + C1 — AUC-PR: 0.57945

DECISION RULE TRIGGERED:
  CASE B: C2 standalone is comparable, but Tri-Blend: C2 + LightGBM B + C1 clearly beats benchmark champion B+C1. -> Tri-Blend: C2 + LightGBM B + C1 becomes finalist.

FINAL DECISION:
  Ensemble: Tri-Blend: C2 + LightGBM B + C1
=====================================================================================
```

---

## 1. Executive Summary & Objective

Model C2 was implemented as a controlled architectural enhancement to evaluate whether a **multi-scale 1D CNN** with parallel receptive fields (local $k=5$, medium $k=15$, broad $k=31$) could extract additional complementary representations from the raw 2,000-reading sequence beyond Model C1's sequential convolutions.

All experiments were conducted on the exact canonical wafer-disjoint validation set: **$137,576$ eligible dies (`old_label == 0`) across 160 unseen wafers**, containing **$5,367$ post-test defects**.

---

## 2. Architecture & Parameter Fairness

| Architecture Component | Model C1 | Model C2 | Specification |
| :--- | :---: | :---: | :--- |
| **Branch 1: Raw 2,000 CNN** | Sequential Conv1D (k=11, 11, 7) | **Parallel Multi-Scale Conv1D (k=5, 15, 31)** | Local, medium, broad receptive fields + fusion conv |
| **Branch 1 Parameters** | $129,792$ | **$279,360$** | $+149,568$ params (+115.2%) |
| **Branch 2: Engineered Block** | 36 $	o$ 64 $	o$ 32 | 36 $	o$ 64 $	o$ 32 | Identical ($4,640$ params) |
| **Branch 3: Parametric/Spatial** | 519 $	o$ 256 $	o$ 128 | 519 $	o$ 256 $	o$ 128 | Identical ($167,168$ params) |
| **Fusion Prediction Head** | 416 $	o$ 128 $	o$ 1 | 416 $	o$ 128 $	o$ 1 | Identical ($53,377$ params) |
| **TOTAL PARAMETERS** | **$306,081$** | **$504,545$** | **$+198,464$ params (+64.8%)** |

---

## 3. Dev-Val Benchmark Leaderboard

| Model | Architecture | AUC-PR | ROC-AUC | Tuned F1 | Precision | Recall | Specificity | Tuned Thresh | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B + C1 Ensemble** | GBDT + Triple-Branch CNN (alpha*=0.855) | **0.57861** | **0.89203** | **0.55342** | 0.7672 | 0.4328 | 0.9947 | 0.900 | N/A (blend) |
| **Tri-Blend: C2 + LGB + C1** | 3-Way Probability Blend | **0.57945** | **0.89335** | **0.55530** | 0.7426 | 0.4435 | 0.9938 | 0.890 | N/A (blend) |
| **Model C2 + LightGBM B** | Probability Blend (alpha*=0.800) | **0.57378** | 0.89021 | 0.55188 | 0.8129 | 0.4177 | 0.9961 | 0.885 | N/A (blend) |
| **Model C2** | Multi-Scale 1D CNN + Tabular MLP | **0.57173** | 0.88932 | 0.55085 | 0.8064 | 0.4183 | 0.9959 | 0.9150 | 496.6 s |
| **Model C1** | Triple-Branch 1D CNN + Tabular MLP | **0.57658** | 0.89131 | **0.55038** | 0.7429 | 0.4371 | 0.9939 | 0.925 | 84.8 s |
| **LightGBM B** | Gradient Boosted Trees (555 feats) | **0.55432** | 0.87598 | **0.54045** | 0.8217 | 0.4026 | 0.9965 | 0.815 | ~12.0 s |

---

## 4. Pairwise Deltas vs. Baselines

- **Model C2 vs. Model C1**:
  - Delta AUC-PR: **-0.00485** (-0.84%)
  - Delta ROC-AUC: **-0.00199**
  - Delta F1: **+0.00047**
- **Model C2 vs. LightGBM B**:
  - Delta AUC-PR: **+0.01741** (+3.14%)
  - Delta ROC-AUC: **+0.01334**
- **Model C2 vs. Champion (B + C1)**:
  - Delta AUC-PR: **-0.00688**

---

## 5. Complementarity Analysis

```csv
 model_x reference_model  pearson_corr  spearman_corr  disagreement_count  disagreement_rate  defects_both_caught  defects_only_c2_caught  defects_only_ref_caught  defects_both_missed  total_unique_defects_caught  unique_defect_coverage_pct
Model C2        Model C1        0.9247         0.9135                 682             0.0050                 2203                      42                      143                 2979                         2388                       44.49
Model C2      LightGBM B        0.8630         0.8419                 784             0.0057                 2079                     166                       82                 3040                         2327                       43.36
```

- **Correlation with C1**: r = 0.9247
- **Correlation with LightGBM B**: r = 0.8630

---

## 6. Targeted Ensemble Sweeps

```csv
                                   ensemble_name  optimal_alpha_c2      partner_weight   AUC-PR  ROC-AUC  Tuned F1  Precision   Recall  Optimal Threshold
                           Model C2 + LightGBM B              0.80                 0.2 0.573782 0.890210  0.551877   0.812908 0.417738              0.885
                             Model C2 + Model C1              0.31                0.69 0.578705 0.893128  0.553496   0.821415 0.417365              0.930
                 Tri-Blend: C2 + LightGBM B + C1              0.27 C1=0.630, LGB=0.100 0.579452 0.893351  0.555296   0.742590 0.443451              0.890
Benchmark: Model B + Model C1 (Current Champion)               NaN C1=0.855, LGB=0.145 0.578610 0.892030  0.553420   0.767200 0.432800              0.900
```

---

## 7. Zero-Retraining Branch Attribution Ablation

| Ablated Configuration | Dev-Val AUC-PR | Relative Drop from Full C2 | Interpretation |
| :--- | :---: | :---: | :--- |
| **Full Model C2** | **0.57173** | **0.00%** | Full multi-modal representation |
| Mask Branch 1 (Raw Multi-Scale CNN) | **0.55169** | **-3.51%** | Loss of all raw temporal waveforms |
| Mask Branch 2 (Engineered Block MLP) | **0.55296** | **-3.28%** | Loss of 36 statistical block summaries |
| Mask Branch 3 (Parametric/Spatial MLP) | **0.12903** | **-77.43%** | Loss of die-level and spatial context |

---

## 8. Final Decision & Strategic Recommendation

CASE B: C2 standalone is comparable, but Tri-Blend: C2 + LightGBM B + C1 clearly beats benchmark champion B+C1. -> Tri-Blend: C2 + LightGBM B + C1 becomes finalist.

- **Final Champion to Freeze**: **Ensemble: Tri-Blend: C2 + LightGBM B + C1**
- **Next Step**: Per guidelines, model development is complete. We immediately freeze the champion pipeline and advance directly to 5-fold grouped validation and interpretability (SHAP, spatial defect heatmaps, block wave traces).
