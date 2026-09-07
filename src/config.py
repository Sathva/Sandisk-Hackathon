"""
Centralized Configuration and Path Management for Hackathon Pipeline.
Ensures paths resolve correctly whether scripts are run from /san or /san/datasources.
"""

import os
from pathlib import Path

# Identify project roots
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent  # /home/user/Vinay/san

# Locate input directory
possible_input_dirs = [
    REPO_ROOT / "datasources" / "input",
    REPO_ROOT / "input",
    Path("datasources/input").resolve(),
    Path("input").resolve(),
]

INPUT_DIR = None
for p in possible_input_dirs:
    if p.exists() and (p / "train.csv").exists():
        INPUT_DIR = p
        break

if INPUT_DIR is None:
    # Fallback default
    INPUT_DIR = REPO_ROOT / "datasources" / "input"

# Set output directories
PROCESSED_DIR = REPO_ROOT / "processed"
PLOTS_DIR = REPO_ROOT / "plots"
REPORTS_DIR = REPO_ROOT / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset file paths
TRAIN_CSV = INPUT_DIR / "train.csv"
TEST_CSV = INPUT_DIR / "test.csv"
VALIDATION_CSV = INPUT_DIR / "validation.csv"

TRAIN_PARQUET = PROCESSED_DIR / "train_features.parquet"
TEST_PARQUET = PROCESSED_DIR / "test_features.parquet"
VALIDATION_PARQUET = PROCESSED_DIR / "validation_features.parquet"

# Development split file paths
DEV_TRAIN_PARQUET = PROCESSED_DIR / "dev_train_features.parquet"
DEV_VAL_PARQUET = PROCESSED_DIR / "dev_val_features.parquet"
DEV_SPLIT_JSON = REPORTS_DIR / "development_split.json"

# Feature constants
NUM_FEATURES = 500
FEATURE_COLS = [f"feature_{i}" for i in range(1, NUM_FEATURES + 1)]
NUM_BLOCK_READINGS = 2000

# Random seed
SEED = 42
