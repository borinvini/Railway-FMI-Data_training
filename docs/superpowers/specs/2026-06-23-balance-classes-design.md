---
name: balance-classes-design
description: Design spec for the balance_classes pipeline stage — applies SMOTE-Tomek to address 3:1 class imbalance before train/test split
metadata:
  type: project
---

# Design: `balance_classes` Pipeline Stage

**Date:** 2026-06-23  
**Status:** Approved

## Problem

The merged training dataset has a 3.1:1 class imbalance (75.8% Punctual vs 24.2% Delayed at the 5-minute threshold). Left unaddressed this biases classifiers toward the majority class. The imbalance is moderate — not extreme — but enough to warrant resampling before the train/test split.

Class balance report (pre-fix):

| Class              | Count  | Share |
|--------------------|--------|-------|
| Punctual (≤ 5 min) | 76,681 | 75.8% |
| Delayed (> 5 min)  | 24,462 | 24.2% |
| Ratio              | 3.1:1  |       |

## Approach

Approach A (chosen): new `balance_classes` stage saves balanced data to a new folder `503-merged_balanced`. The `split_dataset` call in the state machine routes to this folder when `balance_classes` is enabled, otherwise falls back to `501-merged_selected_training_ready`. Mirrors the existing `scale_weather_features` routing pattern used by `train_xgboost_with_randomized_search_cv`.

## Architecture

### Stage placement in state machine

```
merge_data_files
  → filter_delay_outliers      (502-merged_outlier_filtered)
  → balance_classes            (504-merged_balanced)  ← NEW
  → select_training_cols       (501-merged_selected_training_ready)
  → split_dataset              (reads 503 if balance_classes enabled, else 501)
  → train_xgboost_with_randomized_search_cv
```

### Data flow

- Input: `result["data"]` (in-memory DataFrame from `filter_delay_outliers`)
- Output: balanced DataFrame passed through `result["data"]` + saved to disk
- Output folder: `data/output/504-merged_balanced`

## Implementation Details

### New constant (`const_training.py`)

```python
MERGED_BALANCED_OUTPUT_FOLDER = "data/output/504-merged_balanced"
```

`RESAMPLING_METHOD`, `IMBALANCE_THRESHOLD`, and `SMOTE_RANDOM_STATE` are already defined and will be reused.

### `TRAINING_STATE_MACHINE` update (`const_training.py`)

```python
TRAINING_STATE_MACHINE = {
    "merge_data_files": True,
    "filter_delay_outliers": True,
    "balance_classes": True,       # ← new
    "select_training_cols": True,
    "split_dataset": True,
    ...
}
```

### New method: `balance_classes(data)` in `TrainingPipeline`

**Steps:**

1. Guard: if `data is None`, return `{"success": False, "error": "data is None"}`.
2. Guard: if `differenceInMinutes` column is absent, return `{"success": True, "data": data, "skipped": True}` with a warning.
3. Derive binary label on-the-fly: `y = (data["differenceInMinutes"] > TRAIN_DELAY_MINUTES).astype(int)`.
4. Compute minority share from `y`. Print a "before" class balance table.
5. If minority share ≥ `IMBALANCE_THRESHOLD` (30%): log that balance is acceptable, set `skipped=True`, skip resampling, save data unchanged to disk.
6. Separate numeric and non-numeric columns. SMOTE input `X` = numeric columns only (excluding the derived `y`). Non-numeric columns are dropped from the resampled output (they cannot be interpolated).
7. Apply `SMOTETomek(random_state=SMOTE_RANDOM_STATE)` → `X_res, y_res`.
8. Reconstruct DataFrame from `X_res`. Print "after" class balance table.
9. Save to `MERGED_BALANCED_OUTPUT_FOLDER` via `save_dataframe_to_parquet`.

**Return dict:**

```python
{
    "success": bool,
    "data": pd.DataFrame,
    "rows_before": int,
    "rows_after": int,
    "minority_share_before": float,
    "minority_share_after": float,
    "resampling_method": str,        # "SMOTE_TOMEK", "NONE", etc.
    "skipped": bool,                 # True if minority_share >= IMBALANCE_THRESHOLD
    "dropped_non_numeric_cols": list[str],
}
```

### State machine wiring (`execute_training_pipeline_steps`)

**1. New `balance_classes` block** inserted after the `filter_delay_outliers` block, before `select_training_cols`. Follows the exact same try/except/print pattern as all other stages.

**2. `split_dataset` routing** — one new routing block before the existing `split_dataset` call:

```python
_split_data_dir = (
    os.path.join(self.project_root, MERGED_BALANCED_OUTPUT_FOLDER)
    if state_machine.get("balance_classes", False)
    else os.path.join(self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)
)
split_result = self.split_dataset(data_dir=_split_data_dir)
```

**3. `split_dataset` signature change** — add `data_dir` parameter with default:

```python
def split_dataset(self, csv_files=None, data_dir=None, test_size=TEST_SIZE, ...):
    merged_training_ready_dir = data_dir or os.path.join(
        self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
    )
```

### Non-numeric column handling

SMOTE synthesizes samples by interpolating numeric feature vectors. Columns that cannot be interpolated (string categoricals like `causes`) are excluded from `X` before applying SMOTE, and are absent from the balanced output. The `dropped_non_numeric_cols` key in the return dict documents which columns were lost. Weather scenario one-hot columns are already numeric (0/1) and are included in `X`.

### Dependency

`imbalanced-learn` must be available — `BorderlineSMOTE` is already imported in `training_pipeline.py`, confirming the package is installed. `SMOTETomek` comes from the same package (`imblearn.combine`).

## Files Changed

| File | Change |
|------|--------|
| `config/const_training.py` | Add `MERGED_BALANCED_OUTPUT_FOLDER`; add `"balance_classes": True` to `TRAINING_STATE_MACHINE`; import `RESAMPLING_METHOD`, `IMBALANCE_THRESHOLD`, `SMOTE_RANDOM_STATE` in pipeline |
| `src/training_pipeline.py` | Add `balance_classes()` method; wire into `execute_training_pipeline_steps`; add `data_dir` param to `split_dataset`; import `SMOTETomek` from `imblearn.combine` |

## Out of Scope

- Fixing the existing issue where `select_training_cols` reads from folder 500 (raw merged) rather than the filtered/balanced data.
- Adding SMOTE support for the regression target (`differenceInMinutes`). The stage balances using the binary label but produces a balanced feature set that benefits both classification and regression.
- Persistence of the derived binary label column — it is used only internally by this stage and not added to the saved DataFrame.
