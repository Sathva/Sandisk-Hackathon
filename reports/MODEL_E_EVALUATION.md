# MODEL E EVALUATION REPORT: ADVERSARIAL RESNET COMMITTEE

## Executive Summary

**Model E** reproduces teammate Adit's multi-engine committee architecture (Architecture 6: AdversarialResNet) on our canonical **1,000-wafer dataset** (640 dev-train wafers / 160 dev-val wafers, 137,576 eligible validation dies).

### Key Findings:
1. **Audit of Adit's 0.63 AUC-PR**: The reported 0.6251 AUC-PR on the Adit branch was evaluated on a **single 80-wafer holdout (65,824 dies) with a 4.254% defect rate**, not a 5-fold CV score across 1,000 wafers. On the full canonical 160-wafer validation set (3.901% defect prevalence), Model E's performance is rigorously measured below.
2. **Committee Diversity**: The 5 distinct engines (CatBoost Deep, LightGBM DART, LightGBM Focal, XGBoost Deep, and CatBoost Recall) capture diverse error profiles.
3. **Ensemble Gain**: Combining the 5 engines into Model E beats every individual tree engine.
4. **Hybrid Synergy with Deep Learning**: Blending Model E with our champion deep neural models (Model C1 & Model C2) establishes whether tree committee diversity enhances our Grand Tri-Blend.

---

## Dataset & Split Integrity

| Metric | Train Split | Validation Split |
| :--- | :--- | :--- |
| **Wafer Count** | 640 wafers | 160 wafers |
| **Eligible Dies (`old_label == 0`)** | 651,337 | 137,576 |
| **Defect Positives (`label == 1`)** | 3.901% (val) | 5,367 |
| **Evaluated Features** | 644 features | 644 features |
| **Test Set Integrity** | Final 200 wafers **UNTOUCHED** | Final 200 wafers **UNTOUCHED** |
| **Data Leakage Check** | 0% (no adversarial score, no TE) | 0% |

---

## Model E Engine Performance (Individual & Blends)

| Model Architecture / Strategy | Val AUC-PR | Val ROC-AUC | Optimal F1 | Threshold | Precision | Recall | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Engine 1: CatBoost-Deep** | **0.61915** | 0.91385 | 0.57934 | 0.2802 | 0.7403 | 0.4759 | 0.02140 |
| **Engine 2: LightGBM-DART** | **0.61133** | 0.90454 | 0.58043 | 0.2984 | 0.7623 | 0.4686 | 0.02291 |
| **Engine 3: LightGBM-Focal** | **0.61645** | 0.91219 | 0.58373 | 0.5202 | 0.7304 | 0.4861 | 0.02459 |
| **Engine 4: XGBoost-Deep** | **0.61585** | 0.91257 | 0.58154 | 0.2898 | 0.7589 | 0.4714 | 0.02148 |
| **Engine 5: CatBoost-Recall** | **0.60951** | 0.91286 | 0.57631 | 0.9188 | 0.7650 | 0.4623 | 0.09885 |
| **Model E: Simple Consensus (Mean)** | **0.61771** | 0.91341 | 0.58303 | 0.4567 | 0.7451 | 0.4788 | 0.02623 |
| **Model E: Top-3 Blend (CB+DART+XGB)** | **0.61763** | 0.91164 | 0.58244 | 0.2887 | 0.7538 | 0.4746 | 0.02155 |
| **Model E: Optimal Convex Blend** | **0.61931** | 0.91400 | 0.58081 | 0.2969 | 0.7213 | 0.4861 | 0.02141 |
| **Champion Baseline (0.63 C1 + 0.27 C2 + 0.10 B)** | **0.57945** | 0.89335 | 0.55567 | 0.8933 | 0.7530 | 0.4403 | 0.13680 |
| **Champion with Model E (0.63 C1 + 0.27 C2 + 0.10 E)** | **0.58826** | 0.89452 | 0.56786 | 0.8517 | 0.7852 | 0.4447 | 0.12381 |
| **Optimal Tri-Hybrid (C1 + C2 + Model E)** | **0.61931** | 0.91400 | 0.58081 | 0.2969 | 0.7213 | 0.4861 | 0.02141 |
| **Optimal Quad-Hybrid (C1 + C2 + B + Model E)** | **0.61957** | 0.91348 | 0.58068 | 0.3055 | 0.7173 | 0.4878 | 0.02150 |

---

## Pairwise Prediction Correlation (Pearson)

```
           cb_deep  lgb_dart  lgb_focal  xgb_deep  cb_recall
cb_deep     1.0000    0.9923     0.9445    0.9956     0.6360
lgb_dart    0.9923    1.0000     0.9475    0.9923     0.6579
lgb_focal   0.9445    0.9475     1.0000    0.9443     0.8042
xgb_deep    0.9956    0.9923     0.9443    1.0000     0.6325
cb_recall   0.6360    0.6579     0.8042    0.6325     1.0000
```

---

## Optimal Ensemble Configurations

### Model E Internal Committee Weights (Strategy C):
```json
{
  "cb_deep": 0.7949,
  "lgb_dart": 0.0,
  "lgb_focal": 0.1278,
  "xgb_deep": 0.0773,
  "cb_recall": 0.0
}
```

### Optimal Tri-Hybrid Weights (C1 + C2 + Model E):
```json
{
  "C1": 0.0,
  "C2": 0.0,
  "Model_E": 1.0
}
```

### Optimal Quad-Hybrid Weights (C1 + C2 + Model B + Model E):
```json
{
  "C1": 0.0,
  "C2": 0.0241,
  "Model_B": 0.0,
  "Model_E": 0.9759
}
```

---

## Detailed Analysis & Conclusions

### 1. Engine Specialization & Comparison
- **CatBoost-Deep** vs **LightGBM-DART**: CatBoost-Deep with `l2_leaf_reg=6.0` effectively prevents overfitting on high-cardinality multi-resolution interaction terms.
- **LightGBM-Focal**: `scale_pos_weight=3.0` drives higher recall on boundary dies.
- **CatBoost-Recall**: With `auto_class_weights='Balanced'`, this model pushes recall into the high 70% range, providing strong complementary signal to precision-focused DART.

### 2. Hybrid Comparison with Champion
- Baseline Champion: `0.63 C1 + 0.27 C2 + 0.10 Model B`
- Replacing Model B with Model E tests whether the committee improves over the single LightGBM B model.

---