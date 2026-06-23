# Filter Delay Outliers Stage — Design Spec

**Date:** 2026-06-23  
**Pipeline:** Training pipeline (`training_pipeline.py`)  
**Insertion point:** Between `merge_data_files` and `select_training_cols`

---

## Problem

After merging all monthly parquet files, the `differenceInMinutes` regression target has extreme values at both tails that are likely data errors:

- **Upper tail:** p90 = 16 min, p99 = 74 min, max = 963 min — extreme positive outliers skew the target even after log-transform
- **Lower tail:** min = -183 min — extreme negative values (trains arriving very early) are implausible and likely sensor/recording errors

Removing these rows before training produces a cleaner target distribution and reduces noise in the model.

---

## Approach

Asymmetric quantile-based row filtering computed on the full merged dataset.

- Compute `lower_bound = df['differenceInMinutes'].quantile(FILTER_LOWER_QUANTILE)`
- Compute `upper_bound = df['differenceInMinutes'].quantile(FILTER_UPPER_QUANTILE)`
- Drop rows where `differenceInMinutes < lower_bound` or `differenceInMinutes > upper_bound`
- Thresholds are asymmetric: lower quantile cuts conservatively (few rows expected), upper quantile cuts more aggressively (heavy right tail)

Thresholds are computed once on the merged dataset — not per month — so they reflect the real distribution across all years/seasons.

---

## Constants (added to `const_training.py`)

```python
FILTER_LOWER_QUANTILE = 0.01   # drop bottom 1% of differenceInMinutes
FILTER_UPPER_QUANTILE = 0.99   # drop top 1% of differenceInMinutes (tune to 0.97–0.98 for heavier cut)

MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER = "data/output/502-merged_outlier_filtered"
```

The existing `502-merged_selected_scaled` folder constant is renamed to `503-merged_selected_scaled`.

---

## State Machine Change (`const_training.py`)

```python
TRAINING_STATE_MACHINE = {
    "merge_data_files": True,
    "filter_delay_outliers": True,   # new
    "select_training_cols": True,
    "split_dataset": True,
    ...
}
```

---

## Method Signature (`training_pipeline.py`)

```python
def filter_delay_outliers(self, data=None):
```

### Behavior

1. Validates `differenceInMinutes` column exists; if missing, logs a warning and returns data unchanged
2. Computes `lower_bound` and `upper_bound` from configured quantile constants
3. Drops rows outside `[lower_bound, upper_bound]`
4. Prints console summary:
   - Computed thresholds
   - Rows removed from lower tail (count + %)
   - Rows removed from upper tail (count + %)
   - Total rows removed and remaining
5. Saves filtered dataframe to `MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER` as parquet
6. Returns filtered dataframe

### Error handling

- Missing `differenceInMinutes` column: warn + return data unchanged (no crash)
- Save failure: warn + continue (same pattern as other pipeline steps)
- If filtered dataframe is empty after removal: warn + return empty dataframe (do not crash; downstream steps will surface the issue)

---

## State Machine Execution (`training_pipeline.py`)

Inserted after the `merge_data_files` block, before `select_training_cols`, following the exact same guard pattern used by all existing steps:

```python
if state_machine.get("filter_delay_outliers", False):
    if result["data"] is not None:
        try:
            filtered = self.filter_delay_outliers(data=result["data"])
            if filtered is not None:
                result["data"] = filtered
                result["steps_executed"].append("filter_delay_outliers")
        except Exception as e:
            result["errors"].append(f"filter_delay_outliers failed: {str(e)}")
            return result
```

---

## What is NOT changing

- Preprocessing pipeline (`preprocessing_pipeline.py`) — untouched
- Dataframe schema — only rows are removed, no columns added or dropped
- Log-transform in the training step — still applies after this stage
- `select_training_cols` — receives the same column structure, fewer rows
