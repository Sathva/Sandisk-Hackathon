# SanDisk Hackathon: Multi-Resolution Die Yield Prediction with Interpretable Spatial Context

An end-to-end, high-performance machine learning pipeline for semiconductor wafer yield prediction. The system models multi-resolution test signals—fusing wafer-scale spatial topography, die-level parametric measurements, and sub-die block readings—to detect marginal defect signatures while maintaining full interpretability for process engineers.

---

## 📌 Executive Summary & Core Objective

In semiconductor manufacturing, each silicon wafer contains thousands of dies tested sequentially. While traditional testing relies solely on die-level parametric features, failures are heavily driven by:
1. **Neighborhood effects**: Pre-existing defective clusters.
2. **Wafer gradients**: Thermal and chemical process chamber variations (radial and linear).
3. **Sub-die block readings**: High-dimensional internal signals ($2,000$ readings per die) capturing fine-grained defects.

### The Competition Benchmark & Hypotheses:
* **Model A (Die-Level + Spatial)**: Predicts die pass/fail probability using $500$ parametric test measurements + $19$ spatial context features ($519$ features total).
* **Model B (Die + Spatial + Block)**: Predicts die pass/fail probability using everything in Model A **plus** $36$ sub-die block anomaly features ($555$ features total).
* **Model B-Without-Spatial (Controlled Ablation)**: Predicts pass/fail using $500$ parametric + $36$ block features ($536$ features total), strictly excluding spatial context to isolate the independent predictive value of high-dimensional block readings.
* **Goal**: Quantify and demonstrate how the block-level signal resolves marginal failures that are indistinguishable using die-level features alone.

---

## 🔬 Key Data Semantics & Evaluation Golden Rule

| Column | Phase | Values | Semiconductor Meaning | Modeling Role |
| :--- | :---: | :---: | :--- | :--- |
| **`old_label`** | **Pre-Test** | `0` = Healthy<br>`1` = Defect | Die status **before** the new electrical test (from base WM-811K wafer map). | **Known pre-test context.** Used to compute neighborhood defect densities and distance metrics. |
| **`label`** | **Post-Test** | `0` = Pass<br>`1` = Total Fail | Cumulative die status **after** the test (pre-test defects + newly failed dies). | **Prediction Target.** Ground truth post-test outcome. |

### The Golden Rule:
* **Eligible Population (`old_label == 0`)**: Dies that were healthy before the test.
  * `old_label = 0` AND `label = 0` $\to$ **Stayed Healthy (Pass)** (~$96.3\%$).
  * `old_label = 0` AND `label = 1` $\to$ **Newly Failed (Target)** (~$3.7\%$). **This is our positive minority target class.**
* **Ineligible Population (`old_label == 1`)**: Dies that were already broken before testing.
  * In the submission file, any die with `old_label == 1` is **hardcoded to `predicted_label = 1`**.
  * Model loss functions and decision thresholds are trained and tuned **exclusively on eligible dies (`old_label == 0`)**.
* **Zero Leakage**: `label` is strictly an output. It is never used to construct any input feature.
* **Wafer ID Safety**: `wafer_id` is an identifier, strictly excluded from predictive features.

---

## 🏆 Model Benchmark & Ablation Results

All models were trained on **$651,337$ eligible training dies** (640 wafers) and evaluated strictly on **$137,576$ eligible validation dies** (160 wafers) across the canonical wafer-disjoint split.

### 1. Master Performance Comparison Table

| Model Architecture | Features / Input | Block Representation | AUC-PR 🥇 | Tuned F1 🥈 | Precision (Fail) | Recall (Fail) | ROC-AUC | Overall Accuracy | Optimal Threshold ($T^*$) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No-Skill Baseline** (Prevalence) | — | None | 0.0390 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 96.10% | N/A | — |
| **Model A** (LightGBM Param + Spatial) | 519 | None | 0.4937 | 0.5200 | **0.9441** | 0.3589 | 0.8318 | **97.42%** | 0.8127 | 109.1 s |
| **Model B-no-spatial** (LightGBM Param + Block) | 536 | 36 Summary Stats | 0.5517 | 0.5392 | 0.7933 | 0.4084 | 0.8732 | 97.28% | 0.8033 | 136.2 s |
| **Model B** (LightGBM Full Fusion) | 555 | 36 Summary Stats | 0.5543 | 0.5397 | 0.8266 | 0.4006 | 0.8760 | 97.33% | 0.8176 | 142.7 s |
| **Model C (Multi-Res 1D CNN)** 🏆 | **519 + Raw 2,000 Seq** | **Learned 1D CNN (256-dim)** | **0.5721** | **0.5505** | 0.7677 | **0.4291** | **0.8891** | 97.27% | 0.8250 | **84.4 s** |

---

### 2. Pairwise Incremental Predictive Value (Deltas & Improvements)

| Pairwise Comparison | Research Question Answered | $\Delta$ AUC-PR | Rel. AUC-PR | $\Delta$ F1 | Rel. F1 | $\Delta$ Recall | $\Delta$ Precision | $\Delta$ ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model C vs. Model B** | **Learned 1D CNN vs. 36 summary features** | **$+0.0178$** | **$+3.21\%$** | **$+0.0108$** | **$+2.00\%$** | **$+0.0285$** | $-0.0589$ | **$+0.0131$** |
| **Model C vs. Model A** | **Learned block sequence + spatial vs. spatial alone** | **$+0.0784$** | **$+15.88\%$** | **$+0.0305$** | **$+5.86\%$** | **$+0.0702$** | $-0.1764$ | **$+0.0573$** |
| **Model B vs. Model A** | Incremental value of adding 36 block features to spatial model | **$+0.0607$** | **$+12.29\%$** | **$+0.0196$** | **$+3.77\%$** | **$+0.0417$** | $-0.1175$ | **$+0.0442$** |
| **Model B-no-spatial vs. Model A** | Block context vs. Spatial context on top of parametric | **$+0.0581$** | **$+11.76\%$** | **$+0.0192$** | **$+3.69\%$** | **$+0.0496$** | $-0.1508$ | **$+0.0433$** |
| **Model B vs. Model B-no-spatial** | Incremental value of spatial context when block features exist | **$+0.0026$** | **$+0.47\%$** | **$+0.0004$** | **$+0.08\%$** | $-0.0078$ | **$+0.0333$** | **$+0.0010$** |

### Key Engineering Takeaways:
1. **Raw 2,000-Reading Sequence Outperforms Hand-Crafted Summaries**: Learning directly from the raw 2,000 block sequence with a 1D CNN (**Model C**) achieves **AUC-PR = 0.5721**, surpassing the 36 engineered summary features in Model B by **$+3.21\%$ relative lift** ($+0.0178$ absolute).
2. **Defect Catch Rate Reaches New High**: Model C catches **$2,303$ defects out of $5,367$** ($42.91\%$ recall)—detecting **$153$ more defective dies** than Model B and **$377$ more** than Model A, while maintaining a $99.47\%$ specificity ($131,512$ clean pass classifications).
3. **Multi-Scale Convolutional Receptive Fields Capture Local Gradients**: By using convolutional filters ($k=11, 11, 7$) followed by dual global pooling (`AdaptiveAvgPool1d` + `AdaptiveMaxPool1d`), Model C preserves both baseline voltage/timing shifts and localized micro-defects simultaneously.
4. **Hardware-Accelerated Efficiency**: Model C trains in only **$84.4$ seconds** on the RTX 4500 Ada GPU using mixed precision (AMP) with zero host memory spikes due to memory-mapped binary caching.

---

### 3. Confusion Matrices (at Tuned F1 Thresholds)

#### Model C ($T^* = 0.8250$) — 🏆 Current Leader
```text
                    Pred Fail      Pred Pass         Metric                     Value
Actual Fail             2,303          3,064         Fail Accuracy (Recall)     0.429104
Actual Pass               697        131,512         Pass Accuracy (Specificity)0.994728
```

#### Model B ($T^* = 0.8176$)
```text
                    Pred Fail      Pred Pass         Metric                     Value
Actual Fail             2,150          3,217         Fail Accuracy (Recall)     0.400596
Actual Pass               451        131,758         Pass Accuracy (Specificity)0.996589
```

#### Model B-no-spatial ($T^* = 0.8033$)
```text
                    Pred Fail      Pred Pass         Metric                     Value
Actual Fail             2,192          3,175         Fail Accuracy (Recall)     0.408422
Actual Pass               571        131,638         Pass Accuracy (Specificity)0.995681
```

#### Model A ($T^* = 0.8127$)
```text
                    Pred Fail      Pred Pass         Metric                     Value
Actual Fail             1,926          3,441         Fail Accuracy (Recall)     0.358860
Actual Pass               114        132,095         Pass Accuracy (Specificity)0.999138
```

* **Defect Detection Lift**: Model B captures **224 more defective dies** than Model A while preserving a **$99.66\%$ pass accuracy** (only 451 false alarms out of 132,209 healthy dies).

---

### 4. Feature Group Attribution & Importance Breakdown

#### Feature Group Gain Summary (Model B):
```text
Group           Features   Total Gain         Mean Gain       Gain %     Total Splits
parametric      500        8,633,762.43       17,267.52       78.56%     27,223
block           36         1,754,707.60       48,741.88       15.97%      1,095
spatial         19           601,333.27       31,649.12        5.47%      1,202
```

* **Highest Information Density**: Block features averaged **$48,741.88$ gain per feature** ($2.8\times$ higher than parametric and $1.5\times$ higher than spatial).
* **Top Features by Decision Tree Gain (Model B)**:
  1. `max_rolling_mean_400` (**Block**) — Gain: **$950,354.11$** (Rank #1 by a landslide, $>4.5\times$ higher than any spatial feature)
  2. `max_rolling_mean_200` (**Block**) — Gain: **$227,567.46$** (Rank #2)
  3. `wafer_die_count` (**Spatial**) — Gain: **$209,768.26$** (Rank #3)
  4. `block_mean_top200` (**Block**) — Gain: **$203,134.64$** (Rank #4)
  5. `block_mean` (**Block**) — Gain: **$188,893.02$** (Rank #5)
  6. `wafer_old_fail_rate` (**Spatial**) — Gain: **$170,384.20$** (Rank #6)
  7. `wafer_old_fail_count` (**Spatial**) — Gain: **$119,416.18$** (Rank #7)
  8. `feature_365` (**Parametric**) — Gain: **$67,630.83$** (Rank #8)

---

## 🚀 Milestones Completed

1. **WM-811K Source Sampling Audit & Provenance Tracking**:
   - Audited the full $811,457$ WM-811K source population ($25,519$ defect-pattern wafers and $147,431$ none-type wafers).
   - Preserved full source provenance for every generated wafer in [`datasources/input/wafer_provenance.csv`](datasources/input/wafer_provenance.csv).
2. **Memory-Safe 1,000-Wafer Generation**:
   - Fixed an OOM memory bottleneck in `generate_data.py`.
   - Generated the final **1,000-wafer dataset** ($1,096,761$ total dies: $800$ train wafers / $200$ test wafers).
3. **Streaming Data Inspection (`src/inspect_data.py`)**:
   - Validated all $1,096,761$ dies: **0 missing values**, **0 duplicate coordinates**, **0 malformed block strings**, all sequences verified at length $2,000$.
4. **Multi-Scale Feature Engineering (`src/features/`)**:
   - **19 Spatial Features** ([`spatial.py`](src/features/spatial.py)): Normalized coordinates, radial distance, radial squared, distance to border, multi-scale old-defect density ($3\times3, 5\times5, 7\times7, 9\times9, 11\times11$), Euclidean distance to nearest defect, wafer die count, and defect rates.
   - **36 Block-Level Anomaly Features** ([`block.py`](src/features/block.py)): Multi-scale rolling means & stds ($W \in \{50, 100, 200, 400\}$), top/bottom tail means, quantiles, extreme outlier counts, robust MAD deviations, and longest contiguous anomaly runs.
5. **High-Performance Parquet Preprocessing (`src/preprocess.py`)**:
   - Extracted all 560 features across $1.1\text{M}$ dies in **$4.45\text{ minutes}$** using parallel multiprocessing. Reduced footprint from ~25 GB CSV down to compact `float32` Parquet files in `processed/`.
6. **Canonical Development Split (`src/make_dev_split.py`)**:
   - Partitioned the 800 train wafers strictly at the wafer level into **640 dev-train wafers** ($734,137$ dies) and **160 dev-val wafers** ($154,360$ dies) with **zero wafer overlap** and balanced positive rates ($3.69\%$ vs $3.90\%$).
   - Saved canonical metadata to [`reports/development_split.json`](reports/development_split.json).
7. **Model A Implementation & Evaluation (`src/models/train_model_a.py`)**:
   - Trained LightGBM baseline on 519 parametric + spatial features with `scale_pos_weight = 26.068` and threshold optimization.
   - Achieved **AUC-PR = 0.4937**, **Tuned F1 = 0.5200**, **Precision = 0.9441**, and **Accuracy = 97.42%**.
8. **Model B Implementation & Evaluation (`src/models/train_model_b.py`)**:
   - Trained full multi-resolution model on 555 features under identical hyperparameters.
   - Achieved **AUC-PR = 0.5543 ($+12.29\%$ lift)**, **Tuned F1 = 0.5397**, **ROC-AUC = 0.8760**, and captured **224 more defects**.
9. **Controlled Ablation Study (`src/models/train_model_b_no_spatial.py`)**:
   - Evaluated Model B-without-spatial (536 features), proving that block features independently drive a **$+11.76\%$ AUC-PR lift** over Model A, while spatial context refines precision.
10. **Model C Implementation & Deep Learning Benchmark (`src/models/train_model_c.py`)**:
   - Designed and trained a dual-branch hybrid neural network (**Multi-Resolution 1D CNN + Tabular MLP**) fusing the raw 2,000-element sub-die sequence with 519 die-level parametric and spatial features.
   - Built a high-speed memory-mapped binary cache (`processed/cache/`) enabling zero host RAM spikes and mixed precision GPU training in **84.4 seconds** on NVIDIA RTX 4500 Ada.
   - Set the **new benchmark record: AUC-PR = 0.5721 (+15.88% over Model A, +3.21% over Model B)**, **ROC-AUC = 0.8891**, **Tuned F1 = 0.5505**, catching **2,303 defects** (+153 defects over Model B).

---

## 🛠️ How to Run All Commands (Step-by-Step)

Ensure your virtual environment is active:

```bash
cd /home/user/Vinay/san
source datasources/venv/bin/activate
pip install -r datasources/requirements.txt lightgbm torch
```

### Step 1: Generate the 1,000-Wafer Dataset
```bash
cd /home/user/Vinay/san/datasources
python generate_data.py --num_wafers 1000 --csv
```

### Step 2: Audit and Inspect Dataset Integrity
```bash
cd /home/user/Vinay/san
python src/inspect_data.py
```

### Step 3: Run Preprocessing & Feature Extraction
```bash
cd /home/user/Vinay/san
python src/preprocess.py
```

### Step 4: Generate Canonical Development Train/Validation Split
```bash
cd /home/user/Vinay/san
python src/make_dev_split.py
```

### Step 5: Run Statistical Effect Size Diagnostics
```bash
cd /home/user/Vinay/san
python src/diagnostics.py
```

### Step 6: Generate Diagnostic Plots
```bash
cd /home/user/Vinay/san
python src/visualize.py
```

### Step 7: Train & Evaluate Model A (LightGBM Baseline)
```bash
cd /home/user/Vinay/san
nohup python -u src/models/train_model_a.py > train_model_a.log 2>&1 &
# Watch output: tail -f train_model_a.log
```

### Step 8: Train & Evaluate Model B (Multi-Resolution Fusion)
```bash
cd /home/user/Vinay/san
nohup python -u src/models/train_model_b.py > train_model_b.log 2>&1 &
# Watch output: tail -f train_model_b.log
```

### Step 9: Run Controlled Ablation (Model B-Without-Spatial)
```bash
cd /home/user/Vinay/san
nohup python -u src/models/train_model_b_no_spatial.py > train_model_b_no_spatial.log 2>&1 &
# Watch output: tail -f train_model_b_no_spatial.log
```

### Step 10: Train & Evaluate Model C (Multi-Resolution 1D CNN)
```bash
cd /home/user/Vinay/san
nohup python -u src/models/train_model_c.py > train_model_c.log 2>&1 &
# Watch output: tail -f train_model_c.log
```

---

## 📂 Repository Structure

```text
/home/user/Vinay/san/
├── README.md                                       # Main documentation & run guide
├── PREPROCESSING_NOTES.md                          # Detailed mathematical formulas & leakage rules
├── datasources/
│   ├── generate_data.py                            # Data generator with provenance tracking
│   ├── config.yaml                                 # Generator configuration (seed=42)
│   ├── requirements.txt                            # Project dependencies
│   ├── data/
│   │   └── LSWMD.pkl                               # WM-811K wafer dataset
│   └── input/
│       ├── train.csv                               # 800 wafers (888,497 dies)
│       ├── test.csv                                # 200 wafers (208,264 dies)
│       ├── validation.csv                          # 200 wafers without label
│       └── wafer_provenance.csv                    # 1,000-wafer WM-811K source mapping
├── src/
│   ├── config.py                                   # Centralized dynamic path manager
│   ├── inspect_data.py                             # Streaming data inspection & schema validator
│   ├── preprocess.py                               # Multi-threaded feature extraction pipeline
│   ├── make_dev_split.py                           # Wafer-isolated 80/20 train/val partitioner
│   ├── diagnostics.py                              # Cohen's d effect size evaluation
│   ├── visualize.py                                # Diagnostic plotting script
│   ├── features/
│   │   ├── spatial.py                              # 19 multi-scale spatial context features
│   │   └── block.py                                # 36 vectorized block anomaly features
│   └── models/
│       ├── common.py                               # Model feature sets, data loaders & metric routines
│       ├── model_c_architecture.py                 # Multi-Resolution 1D CNN + Tabular MLP (PyTorch)
│       ├── prepare_cnn_data.py                     # High-speed memory-mapped cache builder
│       ├── train_model_a.py                        # Model A training & evaluation pipeline
│       ├── train_model_b.py                        # Model B training & evaluation pipeline
│       ├── train_model_b_no_spatial.py             # Controlled ablation training & evaluation pipeline
│       └── train_model_c.py                        # Model C end-to-end training & benchmarking pipeline
├── models/
│   ├── model_a.joblib / model_a.txt                # Model A trained artifacts (3.44 MB)
│   ├── model_b.joblib / model_b.txt                # Model B trained artifacts (3.45 MB)
│   ├── model_b_no_spatial.joblib / .txt            # Model B-no-spatial trained artifacts (3.50 MB)
│   ├── model_c_cnn.pt                              # Model C PyTorch best checkpoint (3.45 MB)
│   ├── model_c_cnn_config.json                     # Model C architecture & hyperparameters
│   └── model_c_normalization.json                  # Model C zero-leakage normalization parameters
├── processed/
│   ├── train_features.parquet                      # 888,497 rows x 560 cols (2.03 GB)
│   ├── test_features.parquet                       # 208,264 rows x 560 cols (616.7 MB)
│   ├── validation_features.parquet                 # 208,264 rows x 559 cols (616.7 MB)
│   ├── dev_train_features.parquet                  # 734,137 rows x 560 cols (1.73 GB)
│   ├── dev_val_features.parquet                    # 154,360 rows x 560 cols (457.1 MB)
│   └── cache/                                      # Model C memory-mapped cache
│       ├── dev_train_raw_blocks.dat                # 651,337 x 2,000 float32 memmap (4.85 GB)
│       ├── dev_val_raw_blocks.dat                  # 137,576 x 2,000 float32 memmap (1.03 GB)
│       ├── dev_train_tabular_norm.npy              # 651,337 x 519 normalized tabular features
│       ├── dev_val_tabular_norm.npy                # 137,576 x 519 normalized tabular features
│       ├── dev_train_labels.npy / dev_val_labels.npy
│       └── dev_val_meta.parquet                    # Validation metadata for die mapping
├── reports/
│   ├── development_split.json                      # Canonical split definition & wafer IDs
│   ├── model_a_metrics.json / .csv                 # Model A metrics & baseline comparisons
│   ├── model_a_feature_importance.csv              # Model A gain & split rankings
│   ├── model_a_dev_val_predictions.parquet         # Model A per-die validation predictions
│   ├── model_b_metrics.json / .csv                 # Model B metrics
│   ├── model_b_feature_importance.csv              # Model B gain & split rankings
│   ├── model_b_dev_val_predictions.parquet         # Model B per-die validation predictions
│   ├── model_comparison_a_vs_b.json / .csv         # Direct Model A vs. Model B deltas
│   ├── model_b_no_spatial_metrics.json / .csv      # Ablation metrics
│   ├── model_b_no_spatial_feature_importance.csv   # Ablation feature rankings
│   ├── model_b_no_spatial_dev_val_predictions.parquet # Ablation predictions
│   ├── model_ablation_comparison.json / .csv       # Full 3-way ablation comparison summary
│   ├── model_c_metrics.json / .csv                 # Model C metrics
│   ├── model_c_training_history.csv                # Model C epoch-by-epoch loss & validation AUC-PR
│   ├── model_c_dev_val_predictions.parquet         # Model C per-die validation predictions
│   ├── model_comparison_a_b_c.json / .csv          # 4-way benchmark comparison (A, B-no-spatial, B, C)
│   └── figures/
│       ├── model_c_training_curve.png              # Model C train/val loss & AUC-PR progression
│       └── model_c_pr_comparison.png               # Precision-Recall curves: Model A vs B-no-spatial vs B vs C
└── plots/
    ├── 1_target_distribution.png                   # Eligible class imbalance breakdown
    ├── 2_spatial_feature_distributions.png         # Spatial feature shifts (healthy vs fail)
    ├── 3_block_feature_distributions.png           # Block anomaly feature divergence
    ├── 4_new_failure_block_traces.png              # 2,000 block traces for failing dies
    ├── 5_healthy_block_traces.png                  # 2,000 block traces for healthy dies
    └── 6_sample_wafer_map.png                      # Wafer map with Pre-Test and Target Fails
```
