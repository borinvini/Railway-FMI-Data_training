# XGBoost Regression Metrics Upgrade — Design Spec

**Date:** 2026-06-24
**Stage:** Stage 6 — `train_xgboost_with_randomized_search_cv`
**Files touched:** `src/training_pipeline.py`, `config/const_training.py`

---

## Problem Being Solved

The regression path currently reports RMSE, MAE, R², and MAPE. MAPE is broken for this dataset because the delay target contains zero or near-zero values (division by zero produces inf/nan). Additionally, the user wants to evaluate whether the model can correctly classify trains as "late" (delay > 5 min) or "on time", which requires binary classification metrics (Precision, Recall, F1, Accuracy) derived from the continuous predictions via a threshold.

---

## Design

### New config constant

Add to `config/const_training.py` (after existing regression-related constants):

```python
DELAY_THRESHOLD_MINUTES = 5  # delay > 5 min = late (1), else on time (0)
```

Import `DELAY_THRESHOLD_MINUTES` in `src/training_pipeline.py` alongside the existing `const_training` imports.

### WMAPE replaces MAPE

WMAPE (Weighted Mean Absolute Percentage Error) divides by the sum of absolute actuals rather than each individual value, so zeros in the target do not cause division-by-zero:

```
WMAPE = sum(|y_true - y_pred|) / sum(|y_true|) * 100
```

Implementation:

```python
denom = np.sum(np.abs(y_test))
test_wmape = (np.sum(np.abs(y_test - y_pred)) / denom * 100) if denom > 0 else 0.0
```

MAPE is removed from all locations it currently appears. WMAPE takes its place with the same variable naming pattern (`test_wmape`, `final_test_wmape`, `test_wmape_scores`, etc.).

### Binary classification metrics on regression output

After computing `y_pred` for each iteration (and for the final best model), binarize both actuals and predictions:

```python
y_test_binary = (y_test > DELAY_THRESHOLD_MINUTES).astype(int)
y_pred_binary = (y_pred > DELAY_THRESHOLD_MINUTES).astype(int)
```

Then compute:

```python
test_bin_precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
test_bin_recall    = recall_score(y_test_binary, y_pred_binary, zero_division=0)
test_bin_f1        = f1_score(y_test_binary, y_pred_binary, zero_division=0)
test_bin_accuracy  = accuracy_score(y_test_binary, y_pred_binary)
```

No new imports are required — `precision_score`, `recall_score`, `f1_score`, and `accuracy_score` are already imported for the classification path.

---

## Scope of Changes

### `config/const_training.py`
- Add `DELAY_THRESHOLD_MINUTES = 5`

### `src/training_pipeline.py` — iteration loop (regression branch only)

Per iteration, after `y_pred = np.expm1(y_pred) - y_shift`:

1. Replace `test_mape` computation with `test_wmape` (WMAPE formula above)
2. Compute binarized actuals and predictions using `DELAY_THRESHOLD_MINUTES`
3. Compute `test_bin_precision`, `test_bin_recall`, `test_bin_f1`, `test_bin_accuracy`
4. Per-iteration print line becomes:
   ```
   Iteration N: CV Score = X, RMSE = X, R² = X, MAE = X, WMAPE = X%, Bin F1 = X, Bin Acc = X
   ```
5. Replace `test_mape_scores` list with `test_wmape_scores`; add `test_bin_precision_scores`, `test_bin_recall_scores`, `test_bin_f1_scores`, `test_bin_accuracy_scores`
6. `iteration_results` dict: replace `test_mape` key with `test_wmape`; add `test_bin_precision`, `test_bin_recall`, `test_bin_f1`, `test_bin_accuracy`

### `src/training_pipeline.py` — final evaluation (regression branch only)

After `final_y_pred = np.expm1(final_y_pred) - y_shift`:

1. Replace `final_test_mape` with `final_test_wmape`
2. Compute `final_bin_precision`, `final_bin_recall`, `final_bin_f1`, `final_bin_accuracy` using the same binarization

### `src/training_pipeline.py` — results dict and return value

`results["final_metrics"]` for regression:
```python
{
    "test_rmse": float(final_test_rmse),
    "test_mae":  float(final_test_mae),
    "test_wmape": float(final_test_wmape),
    "test_r2":   float(final_test_r2),
    "delay_threshold_minutes": DELAY_THRESHOLD_MINUTES,
    "test_bin_precision": float(final_bin_precision),
    "test_bin_recall":    float(final_bin_recall),
    "test_bin_f1":        float(final_bin_f1),
    "test_bin_accuracy":  float(final_bin_accuracy)
}
```

`results["iteration_metrics_summary"]` for regression:
- Replace `mape_values`/`best_mape`/`average_mape` with `wmape_values`/`best_wmape`/`average_wmape`
- Add `bin_f1_values`, `bin_accuracy_values`, `bin_precision_values`, `bin_recall_values`

Return dict for regression path: replace `test_mape` with `test_wmape`; add `test_bin_precision`, `test_bin_recall`, `test_bin_f1`, `test_bin_accuracy`.

### `src/training_pipeline.py` — summary print

Replace MAPE line with WMAPE; add binary metrics block:
```
WMAPE: X%
Binary metrics (threshold > 5 min):
  Precision: X  Recall: X  F1: X  Accuracy: X
```

---

## Out of Scope

- The classification path (`is_classification`) — unchanged
- The iteration curve plot — unchanged (still shows RMSE + CV score)
- Feature importance plot and CSV — unchanged
- `train_xgboost_selected_features` — separate follow-up
- Any change to CV scoring metric or model selection logic
