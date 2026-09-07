# Synthetic Die-Level Feature Generator for Hackathon

## Input Structure

Files are generated in `input/` directory:

| File | Description |
|------|-------------|
| `train.csv` / `test.csv` |Labeled train/test data for model development |
| `validation.csv` | Unlabeled data for model validation |

## Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `wafer_id` | Identifier | Unique wafer identifier |
| `die_row`, `die_col` | Identifier | Die position on wafer grid |
| `feature_1` ... `feature_n` | Feature | Parametric test measurements |
| `block_readings` | Feature | Space-separated array of k float values (sub-die block test signal, default k=2000) |
| `old_label` | Feature | **Pre-test** die status: 0=pass, 1=fail (from WM-811K map) |
| `label` | Label (target) | **Post-test** die status: 0=pass, 1=fail (old fails + new fails) |

## Evaluation Logic

The evaluation should **only consider eligible dies** (those with `old_label=0`). Dies that were already failed (`old_label=1`) should be excluded because they trivially remain failed and would inflate accuracy numbers.

**Metrics:**
- **Overall Accuracy**: Correct predictions / eligible dies
- **Pass Accuracy (Recall)**: Of dies that stayed pass, fraction correctly predicted as pass
- **Fail Accuracy (Recall)**: Of dies that newly failed, fraction correctly predicted as fail
- **Pass Precision**: Of dies predicted as pass, fraction that actually passed
- **Fail Precision**: Of dies predicted as fail, fraction that actually failed
- **Pass F1 Score**: Harmonic mean of pass precision and pass accuracy (Recall)
- **Fail F1 Score**: Harmonic mean of fail precision and fail accuracy (Recall)

**Expected Confusion Matrix Output:**
```
                      Pred Fail  Pred Pass       Metric               Value
  Actual Fail            1330       1164       Fail Accuracy        0.533280
  Actual Pass             118      23917       Pass Accuracy        0.995090
```

## Prediction Submission Format

Predictions file must be a CSV with columns:

```csv
wafer_id,die_row,die_col,predicted_label
W_F_0001,5,12,0
W_F_0001,5,13,1
...
```

- `predicted_label`: 0=pass, 1=fail (for ALL dies, including old fails)
