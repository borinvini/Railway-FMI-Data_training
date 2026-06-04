# Pipeline Data Quality Fixes — Design Spec

**Date:** 2026-06-04  
**Status:** Approved

---

## Problem Summary

Three data quality issues were identified in the preprocessing pipeline that affect the merged training-ready output:

1. **907 rows with `month=0`** — rows where `actualTime` failed datetime parsing receive fill value `0`, producing invalid temporal features that should never reach training.
2. **`differenceInMinutes` and `differenceInMinutes_offset` stored as object type** — numeric delay columns contain string representations (e.g., `"1.0"`) and in the 907 bad rows, ISO timestamp strings, causing dtype mismatches at training time.
3. **78 rolling window columns silently dropped** — the source parquet files (122 cols each) contain pre-computed rolling windows (`Air temperature (12h max)`, `Snow depth (72h min)`, etc.) but `filter_columns` only keeps columns from explicit constant lists, which never included the window columns.

---

## Root Causes

| # | Root Cause | File | Location |
|---|-----------|------|----------|
| 1 | `fillna(0)` used as fallback for unparseable dates instead of row drop | `preprocessing_pipeline.py` | `process_actual_time_column`, after temporal extraction |
| 2 | No numeric coercion before `dropna` on target columns | `preprocessing_pipeline.py` | `handle_missing_values`, before `dropna(subset=required_cols)` |
| 3 | `filter_columns` uses include-lists from constants; no list ever included window columns | `preprocessing_pipeline.py` | `filter_columns`, column-keep logic |

---

## Fixes

### Fix 1 — Drop rows with unparseable `actualTime` (Approach 1A)

**File:** `src/preprocessing_pipeline.py`  
**Method:** `process_actual_time_column`  
**Location:** Immediately after the block that fills `month`, `day_of_week`, `day_of_month` with `fillna(0).astype(int)`

Drop rows where `month == 0` (i.e., datetime parsing failed):

```python
unparseable = (df['month'] == 0)
if unparseable.any():
    count = unparseable.sum()
    df = df[~unparseable].copy()
    print(f"Dropped {count} rows with unparseable actualTime (month=0)")
    logger.info(f"Dropped {count} rows with unparseable actualTime (month=0)")
```

**No changes** to state machine or constants.

---

### Fix 2 — Coerce target columns to numeric before dropna (Approach 2A)

**File:** `src/preprocessing_pipeline.py`  
**Method:** `handle_missing_values`  
**Location:** After the categorical fill loop, before the `dropna(subset=required_cols)` block

```python
for col in VALID_TARGET_FEATURES:
    if col in df.columns and df[col].dtype == object:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        after = df[col].isna().sum()
        coerced_to_nan = after - before
        if coerced_to_nan > 0:
            print(f"Coerced '{col}' to numeric: {coerced_to_nan} non-numeric values became NaN")
            logger.info(f"Coerced '{col}' to numeric: {coerced_to_nan} non-numeric values → NaN")
```

The existing `df = df.dropna(subset=required_cols)` immediately after handles any NaN produced by coercion.

**No changes** to constants.

---

### Fix 3 — Pattern-based rolling window column retention (Approach 3B)

**File:** `src/preprocessing_pipeline.py`  
**Method:** `filter_columns`  
**Location:** After the `weather_scenario_cols_found` block, before deduplication

```python
window_patterns = ('(12h', '(24h', '(72h')
window_cols_found = [col for col in df.columns if any(p in col for p in window_patterns)]
columns_to_keep.extend(window_cols_found)
print(f"Rolling window columns found: {len(window_cols_found)}")
logger.info(f"Rolling window columns found: {window_cols_found}")
```

**Patterns match** (from verified source data):
- `Air temperature (12h max/min/mean)`, `(24h ...)`, `(72h ...)`
- `Wind speed`, `Relative humidity`, `Precipitation intensity`, `Snow depth`, `Pressure (msl)`, `Horizontal visibility`, `Cloud amount` — same 3-window × 3-stat pattern
- `Precipitation amount (12h mean)`, `(12h cumulative)`, `(24h ...)`, `(72h ...)`

Total: **78 window columns** automatically detected.

**`handle_missing_values` — no changes needed.** The existing weather imputation loop uses substring matching:
```python
any(weather_condition in col for weather_condition in IMPORTANT_WEATHER_FEATURES)
```
For example, `"Air temperature" in "Air temperature (12h max)"` → `True`. All window columns are automatically imputed by the existing logic.

**No changes** to constants.

---

## Interaction Between Fixes

Fixes 1 and 2 address overlapping rows (the 907 bad rows): Fix 1 removes them at `process_actual_time_column`; Fix 2 catches any residual non-numeric values in later runs or edge cases. Both are needed independently — Fix 2 also corrects `differenceInMinutes_offset` which stores all its values as string objects regardless of the 907 rows.

Fix 3 is independent of the other two.

---

## Expected Output After All Fixes

| Metric | Before | After |
|--------|--------|-------|
| Rows in merged CSV | 90,575 | ~89,668 (−907) |
| Columns in merged CSV | 39 | ~117 (+78 window cols) |
| `month=0` rows | 907 | 0 |
| `differenceInMinutes` dtype | object | float64 |
| `differenceInMinutes_offset` dtype | object | float64 |
| Rolling window cols present | 0 | 78 |

---

## Files Changed

| File | Change |
|------|--------|
| `src/preprocessing_pipeline.py` | 3 targeted edits in 3 separate methods |
| `config/const_preprocessing.py` | No changes |
| `main.py` | No changes |

---

## Re-run Scope

All 84 year-month parquet files must be reprocessed through the full pipeline (from the existing source parquets in `data/output/2-filter_by_target_station/`) to produce updated `training_ready_*.parquet` files. The merge step must then be re-run to produce a new `merged_data_2018-01_to_2024-12.csv`.
