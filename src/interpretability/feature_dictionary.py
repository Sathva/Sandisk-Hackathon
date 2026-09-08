"""
Feature Dictionary and Domain Classification for Interpretability.

Translates technical feature names into plain-English, engineer-readable descriptions
and categorizes them into seven physical/methodological domains:
1. Parametric (raw die-level electrical test measurements)
2. Spatial Context (neighborhood defect density, wafer radius, perimeter)
3. Block-Level Dynamics (sub-die reading statistics, rolling bursts, wavelet energy)
4. Wafer-Relative & Detrended (detrended residuals, within-wafer ranks and Z-scores)
5. PCA & Manifold Projections (shrinkage LDA direction, detrended principal components)
6. Cross-Resolution Interactions (bilinear products of parametric drift and block bursts)
7. Geometry & Coordinates (grid positions, polynomial coordinate terms)
"""

import re
from typing import Tuple

# Domain Names
DOMAIN_PARAMETRIC = "Parametric (Die-Level)"
DOMAIN_SPATIAL = "Spatial Context"
DOMAIN_BLOCK = "Block Dynamics (Sub-Die)"
DOMAIN_WAFER_RELATIVE = "Wafer-Relative / Detrended"
DOMAIN_MANIFOLD = "Manifold / Projections (LDA & PCA)"
DOMAIN_INTERACTIONS = "Cross-Resolution Interactions"
DOMAIN_GEOMETRY = "Coordinates & Geometry"

DOMAINS_ORDER = [
    DOMAIN_PARAMETRIC,
    DOMAIN_SPATIAL,
    DOMAIN_BLOCK,
    DOMAIN_WAFER_RELATIVE,
    DOMAIN_MANIFOLD,
    DOMAIN_INTERACTIONS,
    DOMAIN_GEOMETRY,
]


def classify_and_translate_feature(feat_name: str) -> Tuple[str, str]:
    """
    Given a technical feature name, return (domain_category, readable_description).
    Strictly avoids opaque jargon and explains the physical/statistical meaning.
    """
    name = feat_name.strip()

    # 1. Cross-Resolution Interactions (e.g., contains '*' or '_x_' or 'wdev_ldadt_x_')
    if "_x_" in name or "*" in name or ("wdev_" in name and "_wdev_" in name):
        parts = name.split("_x_") if "_x_" in name else name.split("*")
        sub_descs = [classify_and_translate_feature(p.strip())[1] for p in parts]
        return (
            DOMAIN_INTERACTIONS,
            f"Bilinear interaction: [{' × '.join(sub_descs)}]"
        )

    # 2. PCA & Manifold Projections (LDA, PCA)
    if "lda" in name or "ldadt" in name:
        if name.startswith("wdev_"):
            return (
                DOMAIN_MANIFOLD,
                "Wafer-relative deviation of Bayes-optimal shrinkage LDA discriminant direction"
            )
        if name.startswith("wzscore_"):
            return (
                DOMAIN_MANIFOLD,
                "Within-wafer standardized Z-score of shrinkage LDA discriminant direction"
            )
        if name.startswith("wrank_"):
            return (
                DOMAIN_MANIFOLD,
                "Within-wafer percentile rank of shrinkage LDA discriminant direction"
            )
        return (
            DOMAIN_MANIFOLD,
            "Shrinkage LDA discriminant projection (Bayes-optimal linear separation axis)"
        )

    if re.search(r"(?:dt_)?pca_\d+", name) or re.search(r"PC\d+", name):
        pca_num = re.findall(r"\d+", name)[-1]
        if name.startswith("dt_"):
            return (
                DOMAIN_MANIFOLD,
                f"Principal Component {pca_num} of wafer-detrended parametric measurement space"
            )
        if name.startswith("wdev_"):
            return (
                DOMAIN_MANIFOLD,
                f"Wafer-relative deviation of Principal Component {pca_num}"
            )
        return (
            DOMAIN_MANIFOLD,
            f"Principal Component {pca_num} of electrical parametric measurement space"
        )

    # 3. Spatial Context & Neighborhood
    spatial_keywords = [
        "neighbor", "density", "dist_to", "cluster", "radius", "radial", "is_edge",
        "edge_dist", "top_hat", "gradient", "zernike", "quadrant"
    ]
    if any(k in name for k in spatial_keywords) or name in [
        "r_norm", "r_norm_sq", "theta", "radial_dist", "is_edge", "dist_to_nearest_fail",
        "neighbor_fail_count_3x3", "neighbor_fail_rate_3x3",
        "neighbor_fail_count_5x5", "neighbor_fail_rate_5x5",
        "neighbor_fail_count_7x7", "neighbor_fail_rate_7x7",
    ]:
        if "neighbor_fail_rate" in name:
            win = re.findall(r"\d+x\d+", name)
            win_str = win[0] if win else "local"
            return (
                DOMAIN_SPATIAL,
                f"Pre-test defective die rate in surrounding {win_str} window"
            )
        if "neighbor_fail_count" in name:
            win = re.findall(r"\d+x\d+", name)
            win_str = win[0] if win else "local"
            return (
                DOMAIN_SPATIAL,
                f"Number of pre-test defective dies in surrounding {win_str} window"
            )
        if "dist_to_nearest_fail" in name:
            return (
                DOMAIN_SPATIAL,
                "Euclidean distance to closest known pre-test defective die"
            )
        if name in ["r_norm", "radial_dist"]:
            return (
                DOMAIN_SPATIAL,
                "Normalized radial distance from wafer center to die position (0=center, 1=edge)"
            )
        if name == "r_norm_sq":
            return (
                DOMAIN_SPATIAL,
                "Squared radial distance (quadratic edge roll-off model term)"
            )
        if name == "is_edge":
            return (
                DOMAIN_SPATIAL,
                "Binary perimeter indicator (die located on outermost wafer ring)"
            )
        if "zernike" in name:
            return (
                DOMAIN_SPATIAL,
                f"Zernike polynomial circular harmonic term: {name}"
            )
        if "top_hat" in name:
            return (
                DOMAIN_SPATIAL,
                f"Morphological top-hat filter for local spatial anomaly extraction ({name})"
            )
        return (
            DOMAIN_SPATIAL,
            f"Wafer-scale spatial context feature ({name})"
        )

    # 4. Geometry & Coordinates
    if name in ["die_row", "die_col", "x_norm", "y_norm", "xy_norm", "row_norm", "col_norm"]:
        desc_map = {
            "die_row": "Vertical die coordinate on wafer grid",
            "die_col": "Horizontal die coordinate on wafer grid",
            "x_norm": "Normalized horizontal coordinate centered at wafer origin",
            "y_norm": "Normalized vertical coordinate centered at wafer origin",
            "xy_norm": "Product of normalized horizontal and vertical coordinates (hyperbolic term)",
        }
        return (DOMAIN_GEOMETRY, desc_map.get(name, f"Wafer grid coordinate term ({name})"))

    # 5. Block-Level Dynamics (Sub-Die Readings)
    block_keywords = [
        "block_", "roll", "rolling", "burst", "haar", "wavelet", "curv", "grad_",
        "subdie", "reading"
    ]
    if any(k in name for k in block_keywords):
        if "rolling_mean" in name or "roll" in name:
            win = re.findall(r"\d+", name)
            win_str = f"W={win[0]}" if win else "rolling"
            if "start_idx" in name:
                return (
                    DOMAIN_BLOCK,
                    f"Sub-die reading sequence index where peak burst window ({win_str}) begins"
                )
            return (
                DOMAIN_BLOCK,
                f"Maximum sub-die reading amplitude within a {win_str}-reading rolling window"
            )
        if "burst_center_idx" in name:
            win = re.findall(r"\d+", name)
            win_str = f"W={win[0]}" if win else "burst"
            return (
                DOMAIN_BLOCK,
                f"Center position index of the maximum detected reading burst ({win_str})"
            )
        if "burst_excess" in name:
            win = re.findall(r"\d+", name)
            win_str = f"W={win[0]}" if win else "burst"
            return (
                DOMAIN_BLOCK,
                f"Burst elevation excess: peak burst amplitude minus die reference baseline ({win_str})"
            )
        if "burst_peak_ratio" in name:
            win = re.findall(r"\d+", name)
            win_str = f"W={win[0]}" if win else "burst"
            return (
                DOMAIN_BLOCK,
                f"Ratio of maximum reading burst to die median level ({win_str})"
            )
        if "haar" in name or "wavelet" in name:
            return (
                DOMAIN_BLOCK,
                f"Haar wavelet sub-band energy across sub-die readings ({name})"
            )
        if "grad_" in name or "curv" in name:
            return (
                DOMAIN_BLOCK,
                f"Numerical derivative / curvature dynamic across sub-die readings ({name})"
            )
        if name.startswith("block_"):
            stat = name.replace("block_", "")
            stat_map = {
                "mean": "Mean amplitude across all 2,000 sub-die readings",
                "std": "Standard deviation / signal volatility across 2,000 sub-die readings",
                "min": "Minimum reading level across 2,000 sub-die readings",
                "max": "Peak reading amplitude across 2,000 sub-die readings",
                "q25": "25th percentile reading amplitude across 2,000 sub-die readings",
                "q50": "Median reading amplitude across 2,000 sub-die readings",
                "q75": "75th percentile reading amplitude across 2,000 sub-die readings",
                "skew": "Reading distribution skewness across 2,000 sub-die readings",
                "kurt": "Reading distribution kurtosis (outlier tail weight)",
                "range": "Peak-to-peak reading swing (max minus min)",
                "iqr": "Interquartile range of reading amplitudes",
                "mean_top200": "Mean amplitude across top 200 highest sub-die readings",
            }
            return (DOMAIN_BLOCK, stat_map.get(stat, f"Statistical summary of sub-die block readings ({stat})"))
        return (DOMAIN_BLOCK, f"Sub-die block reading dynamic feature ({name})")

    # 6. Wafer-Relative & Detrended Residuals
    if name.startswith("dt_feature_"):
        f_num = name.replace("dt_feature_", "")
        return (
            DOMAIN_WAFER_RELATIVE,
            f"Wafer-detrended residual of parametric test #{f_num} (spatial field subtracted)"
        )
    if name.startswith("wdev_"):
        sub = name.replace("wdev_", "")
        _, sub_desc = classify_and_translate_feature(sub)
        return (
            DOMAIN_WAFER_RELATIVE,
            f"Within-wafer deviation: {sub_desc} minus wafer baseline"
        )
    if name.startswith("wrank_"):
        sub = name.replace("wrank_", "")
        _, sub_desc = classify_and_translate_feature(sub)
        return (
            DOMAIN_WAFER_RELATIVE,
            f"Within-wafer percentile rank of: {sub_desc}"
        )
    if name.startswith("wzscore_"):
        sub = name.replace("wzscore_", "")
        _, sub_desc = classify_and_translate_feature(sub)
        return (
            DOMAIN_WAFER_RELATIVE,
            f"Within-wafer standardized Z-score of: {sub_desc}"
        )

    # 7. Raw Electrical Parametric Tests (feature_1 through feature_500)
    if re.match(r"^feature_\d+$", name):
        f_num = name.replace("feature_", "")
        return (
            DOMAIN_PARAMETRIC,
            f"Electrical parametric test measurement #{f_num} (raw die-level reading)"
        )

    # Default Fallback
    return (DOMAIN_PARAMETRIC, f"Die measurement feature ({name})")
