# Preprocessing & Data Understanding Guide
## SanDisk Hackathon: Multi-Resolution Die Yield Prediction with Interpretable Spatial Context

---

### 1. Label Semantics & Target Population

| Column | Pre/Post Test | Values | Semiconductor Meaning | Role in Modeling |
| :--- | :---: | :---: | :--- | :--- |
| **`old_label`** | **Pre-Test** | `0` = Healthy<br>`1` = Defect | Die status **before** the new electrical test (from base WM-811K wafer map). | **Known pre-test context.** Used to compute spatial neighborhood defect densities and distance metrics. |
| **`label`** | **Post-Test** | `0` = Pass<br>`1` = Total Fail | Cumulative die status **after** the test (contains pre-test defects + newly failed dies). | **Prediction Target.** Ground truth post-test outcome. |

#### The Evaluation Golden Rule
* **Eligible Population (`old_label == 0`)**:
  Dies that were healthy before test. Only these dies are at risk of newly failing.
  - `old_label = 0` AND `label = 0` $\to$ **Stayed Healthy (Pass)** (~95.8% of eligible dies).
  - `old_label = 0` AND `label = 1` $\to$ **Newly Failed (NEW FAIL)** (~4.2% of eligible dies). **This is our positive minority target class.**
* **Ineligible Population (`old_label == 1`)**:
  Dies that were already broken before testing. They trivially remain failed (`label = 1`).
  - **In submission CSV**: For any die with `old_label == 1`, we can trivially output `predicted_label = 1`.
  - **In model training & validation**: Evaluating on `old_label == 1` artificially inflates accuracy and F1 score. Models and decision thresholds must be trained and evaluated **strictly on eligible dies (`old_label == 0`)**.

---

### 2. Leakage Prevention Rules

1. **`label` is Strictly an Output**:
   - `label` is **NEVER** used to engineer any feature.
   - Any statistic derived from `label` (such as post-test wafer yield or post-test defect density) is an illegal data leak.
2. **Strict Spatial Isolation Per Wafer**:
   - All spatial transformations (grid coordinates, Euclidean Distance Transform, multi-scale boxed convolutions) are computed strictly **within each individual wafer**.
   - No cross-wafer spatial contamination can occur.
3. **Pre-Test Spatial Information Only**:
   - Neighborhood defect density features (`old_fail_density_3x3`, `old_fail_density_5x5`, etc.) and distance to defects (`distance_to_nearest_old_failure`) use **exclusively `old_label`**.
4. **Validation Set Purity**:
   - `validation.csv` contains no `label` column. The pipeline processes `validation_features.parquet` with the exact same transformations without requiring `label`.

---

### 3. Spatial Context Feature Dictionary

All spatial features are computed per wafer using `(die_row, die_col)` and `old_label`:

| Feature Name | Type | Formula / Description | Physical Meaning |
| :--- | :---: | :--- | :--- |
| `normalized_row` | Geometric | $(r - r_{center}) / \Delta r$ | Normalized row coordinate in $[-1, 1]$. |
| `normalized_col` | Geometric | $(c - c_{center}) / \Delta c$ | Normalized column coordinate in $[-1, 1]$. |
| `radius` | Geometric | $\sqrt{\text{norm\_row}^2 + \text{norm\_col}^2}$ | Distance from wafer center (0=center, 1=perimeter). Captures chamber radial variation. |
| `radius_squared` | Geometric | $\text{radius}^2$ | Non-linear quadratic radial edge effect. |
| `distance_to_edge` | Geometric | Normalized EDT to wafer perimeter | Direct distance to wafer edge (0=boundary, 1=center). |
| `old_fail_count_KxK` | Neighborhood | Sum of `old_label == 1` dies in $K \times K$ window | Raw count of pre-test failed dies in $3\times3, 5\times5, 7\times7, 9\times9, 11\times11$ neighborhood. |
| `old_fail_density_KxK`| Neighborhood | $\frac{\text{old\_fail\_count}}{\text{valid\_dies\_in\_window}}$ | Density of pre-test failed dies (normalized by actual dies present on grid). |
| `distance_to_nearest_old_failure` | Proximity | Normalized EDT to closest `old_label == 1` | Continuous distance to nearest pre-existing defect. If no defects exist, safely set to diagonal. |
| `wafer_die_count` | Wafer Context | Total dies on wafer | Wafer scale / die resolution. |
| `wafer_old_fail_count` | Wafer Context | Count of `old_label == 1` dies on wafer | Pre-test wafer defect burden. |
| `wafer_old_fail_rate` | Wafer Context | `wafer_old_fail_count / wafer_die_count` | Overall baseline defect rate of the wafer. |

---

### 4. Block Reading Feature Dictionary

Each die contains 2,000 sub-die block test readings.
- **Passing dies**: Gaussian noise ($\mu=100.0, \sigma=15.0$) with 1D smoothing.
- **Failing dies**: A sparse, localized cluster of anomalous readings (~100 blocks, 5% of sequence).

The pipeline extracts high-signal statistical and anomaly metrics:

| Category | Feature Name | Description | Why It Works |
| :--- | :--- | :--- | :--- |
| **Global** | `block_mean`, `block_std`, `block_min`, `block_max`, `block_range`, `block_median` | Overall distribution moments | Identifies gross die-wide shifts. |
| **Quantiles** | `block_q01`, `block_q05`, `block_q25`, `block_q50`, `block_q75`, `block_q95`, `block_q99` | Quantile profile from sorted sequence | Captures distribution asymmetry and skewness. |
| **Tail** | `block_mean_top{10,50,100,200}`<br>`block_mean_bottom{10,50,100,200}` | Mean of extreme top-k and bottom-k values | **High effect size:** anomalous cluster elevates top-k means without being washed out by 2,000 values. |
| **Anomaly** | `block_count_z_gt_{2,3,4}`<br>`block_max_z` | Counts and maximum of standardized z-scores | Detects presence of statistical outliers ($>2\sigma, >3\sigma, >4\sigma$). |
| **Robust Anomaly** | `block_count_mad_gt_3`<br>`block_max_mad_deviation` | Median Absolute Deviation (MAD) metrics | Outlier metrics immune to extreme point corruption. |
| **Cluster / Rolling** | `max_rolling_mean_{50,100,200,400}` | Maximum rolling mean across window sizes | **The Primary Discriminator:** Captures the ~100-block localized defect burst. |
| **Cluster Index** | `max_rolling_mean_100_start_idx` | Block index where window 100 is maximized | Root-cause interpretability: pinpoint where the defect cluster is located. |
| **Cluster Variance**| `max_rolling_std_{50,100,200}` | Maximum rolling standard deviation | Detects localized signal instability. |
| **Contiguous Run** | `largest_contiguous_anomaly_run` | Longest uninterrupted sequence of blocks with $\|z\| > 2$ | Normal noise rarely exceeds 3-4 consecutive outliers; defect clusters produce runs of 20-80+. |

---

### 5. Output Parquet Schemas

The processed files in `processed/` are:
* `processed/train_features.parquet`: ~173,099 rows $\times$ 546 columns.
* `processed/test_features.parquet`: ~39,351 rows $\times$ 546 columns.
* `processed/validation_features.parquet`: ~39,351 rows $\times$ 545 columns (no `label`).

All floating-point feature columns are stored as `float32` with snappy compression to optimize I/O and RAM usage during model training.
