# SanDisk Hackathon: Multi-Resolution Die Yield Prediction with Interpretable Spatial Context

An end-to-end, high-performance machine learning pipeline for semiconductor wafer yield prediction. The system models multi-resolution test signals—fusing wafer-scale spatial topography, die-level parametric measurements, and sub-die block readings—to detect marginal defect signatures while maintaining full interpretability for process engineers.

---

## 📌 Executive Summary & Core Objective

In semiconductor manufacturing, each silicon wafer contains thousands of dies tested sequentially. While traditional testing relies solely on die-level parametric features, failures are heavily driven by:
1. **Neighborhood effects**: Pre-existing defective clusters.
2. **Wafer gradients**: Thermal and chemical process chamber variations (radial and linear).
3. **Sub-die block readings**: High-dimensional internal signals ($2,000$ readings per die) capturing fine-grained defects.

### The Competition Benchmark:
* **Model A (Die-Level Only)**: Predicts die pass/fail probability using $500$ parametric test measurements + spatial context ($m \times m$ neighborhood, radial/edge geometry).
* **Model B (Die + Block-Level)**: Predicts die pass/fail probability using everything in Model A **plus** the $2,000$-dimensional sub-die block readings.
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

---

## 🚀 Milestones Completed

1. **WM-811K Source Sampling Audit & Provenance Tracking**:
   - Audited the full $811,457$ WM-811K source population ($25,519$ defect-pattern wafers and $147,431$ none-type wafers).
   - Modified the generator to preserve full source provenance for every generated wafer, saving [`input/wafer_provenance.csv`](datasources/input/wafer_provenance.csv) with original WM-811K row index, defect pattern, lot name, and wafer index.
2. **Memory-Safe 1,000-Wafer Generation**:
   - Fixed an OOM memory bottleneck in `generate_data.py` (preventing string duplication during summary calculations and freeing train memory before test generation).
   - Successfully generated the final **1,000-wafer dataset** ($1,096,761$ total dies: $800$ train wafers / $200$ test wafers).
3. **Streaming Data Inspection (`src/inspect_data.py`)**:
   - Validated all $1,096,761$ dies: **0 missing values**, **0 duplicate coordinates**, **0 malformed block strings**, all sequences verified at length $2,000$.
4. **Multi-Scale Feature Engineering (`src/features/`)**:
   - **19 Spatial Features** ([`spatial.py`](src/features/spatial.py)): Normalized coordinates, radial distance, radial squared, distance to border, multi-scale old-defect density ($3\times3, 5\times5, 7\times7, 9\times9, 11\times11$), Euclidean distance to nearest defect, wafer die count, and defect rates.
   - **36 Block-Level Anomaly Features** ([`block.py`](src/features/block.py)): Multi-scale rolling means & stds ($W \in \{50, 100, 200, 400\}$), top/bottom tail means, quantiles, extreme outlier counts, robust MAD deviations, and longest contiguous anomaly runs.
5. **High-Performance Parquet Preprocessing (`src/preprocess.py`)**:
   - Extracted all 560 features across $1.1\text{M}$ dies in **$4.45\text{ minutes}$** using parallel multiprocessing. Reduced data footprint from ~25 GB of CSV down to compact `float32` Parquet files in `processed/`.
6. **Canonical Development Split (`src/make_dev_split.py`)**:
   - Partitioned the 800 train wafers strictly at the wafer level into **640 dev-train wafers** ($734,137$ dies) and **160 dev-val wafers** ($154,360$ dies) with **zero wafer overlap** and balanced positive rates ($3.69\%$ vs $3.90\%$).
   - Saved canonical metadata to [`reports/development_split.json`](reports/development_split.json).
7. **Statistical Effect Size Validation (`src/diagnostics.py`)**:
   - Proved that block anomaly features achieve **Cohen's $d = 0.8075$** ($>3.4\times$ stronger than the best parametric feature at $d = 0.2360$), confirming that block readings resolve marginal failures.
8. **Diagnostic Visualizations (`src/visualize.py`)**:
   - Generated 6 high-resolution diagnostic plots in `plots/`.

---

## 🛠️ How to Run All Commands (Step-by-Step)

Ensure your virtual environment is created and active:

```bash
cd /home/user/Vinay/san/datasources
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Generate the 1,000-Wafer Dataset
Generates `train.csv`, `test.csv`, `validation.csv`, and `wafer_provenance.csv`:
```bash
cd /home/user/Vinay/san/datasources
python generate_data.py --num_wafers 1000 --csv
```

### Step 2: Audit and Inspect Dataset Integrity
Streams through the raw CSV files, verifying coordinates, shapes, distributions, and schema:
```bash
cd /home/user/Vinay/san
python src/inspect_data.py
```

### Step 3: Run Preprocessing & Feature Extraction
Extracts 19 spatial features and 36 block anomaly features, saving to `processed/*.parquet`:
```bash
cd /home/user/Vinay/san
python src/preprocess.py
```

### Step 4: Generate Canonical Development Train/Validation Split
Partitions the 800 train wafers into 640 dev-train and 160 dev-validation with zero leakage:
```bash
cd /home/user/Vinay/san
python src/make_dev_split.py
```

### Step 5: Run Statistical Effect Size Diagnostics
Computes Cohen's $d$ effect sizes on eligible dies (`old_label == 0`) and ranks the top features:
```bash
cd /home/user/Vinay/san
python src/diagnostics.py
```

### Step 6: Generate Diagnostic Plots
Generates all 6 diagnostic figures in `plots/`:
```bash
cd /home/user/Vinay/san
python src/visualize.py
```

---

## 📊 Feature Effect Size Benchmark

Statistical effect sizes computed on **$788,913$ eligible training dies**:

```text
Domain                 Top Feature                   |Cohen's d|  Impact
---------------------------------------------------------------------------------------------
Die Parametric Tests   feature_336                      0.2360    Marginal (High distribution overlap)
Spatial Topography     old_fail_density_5x5             0.2081    Defect clustering effect
Block Test Readings    max_rolling_mean_400             0.8075    VERY LARGE (>3.4x signal boost!)
```

---

## 📂 Repository Structure

```text
/home/user/Vinay/san/
├── README.md                              # Main documentation & run guide
├── PREPROCESSING_NOTES.md                 # Detailed mathematical formulas & leakage rules
├── datasources/
│   ├── generate_data.py                   # Data generator with provenance tracking
│   ├── config.yaml                        # Generator configuration (seed=42)
│   ├── requirements.txt                   # Project dependencies
│   ├── audit_sampling.py                  # Source sampling audit script
│   ├── data/
│   │   └── LSWMD.pkl                      # WM-811K wafer dataset
│   └── input/
│       ├── train.csv                      # 800 wafers (888,497 dies)
│       ├── test.csv                       # 200 wafers (208,264 dies)
│       ├── validation.csv                 # 200 wafers without label
│       └── wafer_provenance.csv           # 1,000-wafer WM-811K source mapping
├── src/
│   ├── config.py                          # Centralized dynamic path manager
│   ├── inspect_data.py                    # Streaming data inspection & schema validator
│   ├── preprocess.py                      # Multi-threaded feature extraction pipeline
│   ├── make_dev_split.py                  # Wafer-isolated 80/20 train/val partitioner
│   ├── diagnostics.py                     # Cohen's d effect size evaluation
│   ├── visualize.py                       # Diagnostic plotting script
│   └── features/
│       ├── spatial.py                     # 19 multi-scale spatial context features
│       └── block.py                       # 36 vectorized block anomaly features
├── processed/
│   ├── train_features.parquet             # 888,497 rows x 560 cols (2.03 GB)
│   ├── test_features.parquet              # 208,264 rows x 560 cols (616.7 MB)
│   ├── validation_features.parquet        # 208,264 rows x 559 cols (616.7 MB)
│   ├── dev_train_features.parquet         # 734,137 rows x 560 cols (1.73 GB)
│   └── dev_val_features.parquet           # 154,360 rows x 560 cols (457.1 MB)
├── reports/
│   ├── development_split.json             # Canonical split definition & wafer IDs
│   └── source_wafer_mapping.csv           # Audited source provenance mapping
└── plots/
    ├── 1_target_distribution.png          # Eligible class imbalance breakdown
    ├── 2_spatial_feature_distributions.png# Spatial feature shifts (healthy vs fail)
    ├── 3_block_feature_distributions.png  # Block anomaly feature divergence
    ├── 4_new_failure_block_traces.png     # 2,000 block traces for failing dies
    ├── 5_healthy_block_traces.png         # 2,000 block traces for healthy dies
    └── 6_sample_wafer_map.png             # Wafer map with Pre-Test and Target Fails
```
