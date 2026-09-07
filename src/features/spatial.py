"""
Spatial Feature Engineering Module for Semiconductor Wafer Maps.
Calculates geometric, multi-scale neighborhood, and wafer-level context
features strictly per wafer using only pre-test information (old_label).

LEAKAGE SAFETY RULE:
- NEVER uses the 'label' column.
- All spatial computations are strictly contained within individual wafers.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, distance_transform_edt, convolve


def extract_wafer_spatial_features(wafer_df):
    """
    Computes spatial features for all dies belonging to a single wafer.

    Parameters:
    -----------
    wafer_df : pd.DataFrame
        DataFrame containing dies for a SINGLE wafer with columns:
        ['die_row', 'die_col', 'old_label']

    Returns:
    --------
    pd.DataFrame: DataFrame of engineered spatial features aligned with wafer_df.
    """
    rows = wafer_df["die_row"].values
    cols = wafer_df["die_col"].values
    old_labels = wafer_df["old_label"].values

    # Determine wafer grid bounding box
    max_r = int(rows.max()) + 1
    max_c = int(cols.max()) + 1
    
    # Grid center coordinates
    center_r = (rows.max() + rows.min()) / 2.0
    center_c = (cols.max() + cols.min()) / 2.0
    half_span_r = max((rows.max() - rows.min()) / 2.0, 1.0)
    half_span_c = max((cols.max() - cols.min()) / 2.0, 1.0)

    # Reconstruct 2D wafer masks
    valid_mask = np.zeros((max_r, max_c), dtype=bool)
    old_fail_mask = np.zeros((max_r, max_c), dtype=bool)

    valid_mask[rows, cols] = True
    old_fail_mask[rows, cols] = (old_labels == 1)

    # -------------------------------------------------------------
    # A. Geometry Features
    # -------------------------------------------------------------
    norm_r = (rows - center_r) / half_span_r
    norm_c = (cols - center_c) / half_span_c
    radius = np.sqrt(norm_r ** 2 + norm_c ** 2)
    radius_sq = radius ** 2

    # Distance to wafer edge:
    # Use distance transform from outside the valid die region to the inside
    # Points on the boundary will have small distance; center dies will have large distance.
    # Invert so distance_to_edge = 0 at border, or direct edge proximity.
    dist_from_border = distance_transform_edt(valid_mask)[rows, cols]
    max_dist_from_border = dist_from_border.max() if len(dist_from_border) > 0 and dist_from_border.max() > 0 else 1.0
    # distance_to_edge: 0 at outer edge, 1 at wafer center
    norm_dist_from_edge = dist_from_border / max_dist_from_border

    # -------------------------------------------------------------
    # B. Multi-Scale Neighborhood Old-Failure Features
    # -------------------------------------------------------------
    # Window sizes: 3x3, 5x5, 7x7, 9x9, 11x11
    window_sizes = [3, 5, 7, 9, 11]
    neighborhood_features = {}

    float_old_fail = old_fail_mask.astype(float)
    float_valid = valid_mask.astype(float)

    for w in window_sizes:
        kernel = np.ones((w, w), dtype=float)
        
        # Convolve using mode='constant' (padding with 0 outside wafer bounds)
        fail_count_grid = convolve(float_old_fail, kernel, mode='constant', cval=0.0)
        valid_count_grid = convolve(float_valid, kernel, mode='constant', cval=0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            density_grid = np.where(valid_count_grid > 0, fail_count_grid / valid_count_grid, 0.0)

        neighborhood_features[f"old_fail_count_{w}x{w}"] = fail_count_grid[rows, cols].astype(np.float32)
        neighborhood_features[f"old_fail_density_{w}x{w}"] = density_grid[rows, cols].astype(np.float32)

    # -------------------------------------------------------------
    # C. Distance to Nearest Old Failure
    # -------------------------------------------------------------
    n_old_fails = int(old_fail_mask.sum())
    diagonal = np.sqrt(max_r ** 2 + max_c ** 2)

    if n_old_fails > 0:
        # Distance transform to nearest True in old_fail_mask:
        # EDT of (~old_fail_mask) gives distance to the nearest old failure
        dist_to_fail_grid = distance_transform_edt(~old_fail_mask)
        dist_to_nearest_old = dist_to_fail_grid[rows, cols]
    else:
        # If wafer has 0 old failures, distance is set safely to wafer diagonal
        dist_to_nearest_old = np.full(len(rows), diagonal, dtype=float)

    # Normalize by wafer diagonal for scale invariance across different wafer resolutions
    norm_dist_to_nearest_old = dist_to_nearest_old / max(diagonal, 1.0)

    # -------------------------------------------------------------
    # D. Wafer-Level Context Features (Using strictly old_label)
    # -------------------------------------------------------------
    wafer_die_count = float(len(wafer_df))
    wafer_old_fail_count = float(n_old_fails)
    wafer_old_fail_rate = wafer_old_fail_count / max(wafer_die_count, 1.0)

    # Assemble all spatial features into dictionary
    feat_dict = {
        "normalized_row": norm_r.astype(np.float32),
        "normalized_col": norm_c.astype(np.float32),
        "radius": radius.astype(np.float32),
        "radius_squared": radius_sq.astype(np.float32),
        "distance_to_edge": norm_dist_from_edge.astype(np.float32),
        "distance_to_nearest_old_failure": norm_dist_to_nearest_old.astype(np.float32),
        "wafer_die_count": np.full(len(rows), wafer_die_count, dtype=np.float32),
        "wafer_old_fail_count": np.full(len(rows), wafer_old_fail_count, dtype=np.float32),
        "wafer_old_fail_rate": np.full(len(rows), wafer_old_fail_rate, dtype=np.float32),
    }

    # Add neighborhood features
    feat_dict.update(neighborhood_features)

    return pd.DataFrame(feat_dict, index=wafer_df.index)


def compute_spatial_features(df):
    """
    Computes spatial features for an entire dataset by grouping by wafer_id.
    Guarantees no spatial information leaks across wafers.
    """
    spatial_dfs = []
    # Preserve order of original dataframe
    grouped = df.groupby("wafer_id", sort=False)
    for wafer_id, wafer_df in grouped:
        sp_df = extract_wafer_spatial_features(wafer_df)
        spatial_dfs.append(sp_df)

    return pd.concat(spatial_dfs, axis=0)
