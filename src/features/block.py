"""
Block Reading Parsing and High-Speed Feature Extraction Module.
Processes the 2000-dimensional sub-die test reading sequence per die.

Extracts:
1. Global distribution statistics (mean, std, min, max, range, quantiles)
2. Tail statistics (top-k and bottom-k means)
3. Anomaly statistics (z-score and robust MAD-based deviations)
4. Clustered / Localized anomaly metrics (multi-scale rolling mean/std and contiguous run lengths)

Performance:
- Employs fast vectorized NumPy operations.
- Direct quantile indexing from sorted arrays.
- Fast 1D convolution/cumsum for rolling window detection.
"""

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def extract_single_die_block_features(readings_str):
    """
    Extracts all block-level engineered features from a single space-separated string.
    Returns (feature_dict, is_valid) tuple.
    """
    if not isinstance(readings_str, str) or not readings_str.strip():
        return None, False

    try:
        x = np.fromstring(readings_str, dtype=np.float32, sep=' ')
    except Exception:
        return None, False

    n = len(x)
    if n != 2000:
        return None, False

    # -------------------------------------------------------------
    # 1. Global Features
    # -------------------------------------------------------------
    mean_val = float(np.mean(x))
    std_val = float(np.std(x))
    min_val = float(x.min())
    max_val = float(x.max())
    range_val = max_val - min_val

    # Fast quantiles and tail means via single sort
    x_sorted = np.sort(x)
    q01 = float(x_sorted[20])
    q05 = float(x_sorted[100])
    q25 = float(x_sorted[500])
    q50 = float(x_sorted[1000])  # median
    q75 = float(x_sorted[1500])
    q95 = float(x_sorted[1900])
    q99 = float(x_sorted[1980])

    # -------------------------------------------------------------
    # 2. Tail Features
    # -------------------------------------------------------------
    top10_mean = float(np.mean(x_sorted[-10:]))
    top50_mean = float(np.mean(x_sorted[-50:]))
    top100_mean = float(np.mean(x_sorted[-100:]))
    top200_mean = float(np.mean(x_sorted[-200:]))

    bottom10_mean = float(np.mean(x_sorted[:10]))
    bottom50_mean = float(np.mean(x_sorted[:50]))
    bottom100_mean = float(np.mean(x_sorted[:100]))
    bottom200_mean = float(np.mean(x_sorted[:200]))

    # -------------------------------------------------------------
    # 3. Anomaly Features (Z-score & Robust MAD)
    # -------------------------------------------------------------
    denom_std = std_val if std_val > 1e-8 else 1e-8
    abs_z = np.abs(x - mean_val) / denom_std
    count_z_gt_2 = int(np.sum(abs_z > 2.0))
    count_z_gt_3 = int(np.sum(abs_z > 3.0))
    count_z_gt_4 = int(np.sum(abs_z > 4.0))
    max_z = float(np.max(abs_z))

    # Robust MAD anomaly metrics
    abs_dev = np.abs(x - q50)
    mad = float(np.median(abs_dev) * 1.4826)
    denom_mad = mad if mad > 1e-8 else 1e-8
    mad_z = abs_dev / denom_mad
    count_mad_gt_3 = int(np.sum(mad_z > 3.0))
    max_mad_deviation = float(np.max(mad_z))

    # -------------------------------------------------------------
    # 4. Cluster / Localized Anomaly Metrics (Rolling Window)
    # -------------------------------------------------------------
    # Fast rolling means using prefix sums (cumsum)
    cumsum = np.empty(n + 1, dtype=np.float64)
    cumsum[0] = 0.0
    np.cumsum(x, out=cumsum[1:])

    cumsum2 = np.empty(n + 1, dtype=np.float64)
    cumsum2[0] = 0.0
    np.cumsum(x.astype(np.float64) ** 2, out=cumsum2[1:])

    # Window 50
    w50 = 50
    rm50 = (cumsum[w50:] - cumsum[:-w50]) / w50
    max_rm50 = float(np.max(rm50))
    rv50 = np.maximum(0.0, (cumsum2[w50:] - cumsum2[:-w50]) / w50 - rm50 ** 2)
    max_rstd50 = float(np.sqrt(np.max(rv50)))

    # Window 100
    w100 = 100
    rm100 = (cumsum[w100:] - cumsum[:-w100]) / w100
    max_rm100 = float(np.max(rm100))
    max_rm100_start_idx = int(np.argmax(rm100))
    rv100 = np.maximum(0.0, (cumsum2[w100:] - cumsum2[:-w100]) / w100 - rm100 ** 2)
    max_rstd100 = float(np.sqrt(np.max(rv100)))

    # Window 200
    w200 = 200
    rm200 = (cumsum[w200:] - cumsum[:-w200]) / w200
    max_rm200 = float(np.max(rm200))
    rv200 = np.maximum(0.0, (cumsum2[w200:] - cumsum2[:-w200]) / w200 - rm200 ** 2)
    max_rstd200 = float(np.sqrt(np.max(rv200)))

    # Window 400
    w400 = 400
    rm400 = (cumsum[w400:] - cumsum[:-w400]) / w400
    max_rm400 = float(np.max(rm400))

    # -------------------------------------------------------------
    # 5. Largest Contiguous Anomaly Run (Vectorized RLE)
    # -------------------------------------------------------------
    anomaly_mask = abs_z > 2.0
    if not np.any(anomaly_mask):
        largest_run = 0
    else:
        padded = np.concatenate(([False], anomaly_mask, [False]))
        diffs = np.diff(padded.astype(np.int8))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        largest_run = int(np.max(ends - starts))

    feat = {
        # Global
        "block_mean": np.float32(mean_val),
        "block_std": np.float32(std_val),
        "block_min": np.float32(min_val),
        "block_max": np.float32(max_val),
        "block_range": np.float32(range_val),
        "block_median": np.float32(q50),
        "block_q01": np.float32(q01),
        "block_q05": np.float32(q05),
        "block_q25": np.float32(q25),
        "block_q50": np.float32(q50),
        "block_q75": np.float32(q75),
        "block_q95": np.float32(q95),
        "block_q99": np.float32(q99),
        # Tail
        "block_mean_top10": np.float32(top10_mean),
        "block_mean_top50": np.float32(top50_mean),
        "block_mean_top100": np.float32(top100_mean),
        "block_mean_top200": np.float32(top200_mean),
        "block_mean_bottom10": np.float32(bottom10_mean),
        "block_mean_bottom50": np.float32(bottom50_mean),
        "block_mean_bottom100": np.float32(bottom100_mean),
        "block_mean_bottom200": np.float32(bottom200_mean),
        # Anomaly
        "block_count_z_gt_2": np.float32(count_z_gt_2),
        "block_count_z_gt_3": np.float32(count_z_gt_3),
        "block_count_z_gt_4": np.float32(count_z_gt_4),
        "block_max_z": np.float32(max_z),
        "block_count_mad_gt_3": np.float32(count_mad_gt_3),
        "block_max_mad_deviation": np.float32(max_mad_deviation),
        # Cluster / Localized Anomaly
        "max_rolling_mean_50": np.float32(max_rm50),
        "max_rolling_mean_100": np.float32(max_rm100),
        "max_rolling_mean_200": np.float32(max_rm200),
        "max_rolling_mean_400": np.float32(max_rm400),
        "max_rolling_mean_100_start_idx": np.float32(max_rm100_start_idx),
        "max_rolling_std_50": np.float32(max_rstd50),
        "max_rolling_std_100": np.float32(max_rstd100),
        "max_rolling_std_200": np.float32(max_rstd200),
        "largest_contiguous_anomaly_run": np.float32(largest_run),
    }

    return feat, True


def _process_batch(strings):
    """Processes a batch of string representations."""
    features = []
    malformed_count = 0
    for s in strings:
        feat, is_valid = extract_single_die_block_features(s)
        if is_valid:
            features.append(feat)
        else:
            malformed_count += 1
            # Fill with NaN placeholders in case of malformed row
            features.append({})
    return features, malformed_count


def extract_block_features_series(series, num_workers=None):
    """
    Extracts block features for a Pandas Series of block_readings strings.
    Utilizes process-level parallelism for maximum throughput across available CPU cores.
    """
    if num_workers is None:
        num_workers = max(1, min(mp.cpu_count() - 1, 8))

    strings = series.tolist()
    total = len(strings)

    if num_workers <= 1 or total < 5000:
        feats, malformed = _process_batch(strings)
    else:
        chunk_size = max(1000, total // (num_workers * 4))
        chunks = [strings[i:i + chunk_size] for i in range(0, total, chunk_size)]
        
        feats = []
        malformed = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(_process_batch, chunks)
            for batch_feats, batch_malformed in results:
                feats.extend(batch_feats)
                malformed += batch_malformed

    df_feats = pd.DataFrame(feats, index=series.index)
    return df_feats, malformed
