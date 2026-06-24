# select_training_cols — Data Flow & Reproducibility Fixes

**Date:** 2026-06-24  
**Stage:** Stage 3 — `select_training_cols`  
**Files touched:** `src/training_pipeline.py`, `config/const_training.py`

---

## Problems Being Solved

### Problem 1 & 2 — Outlier filtering silently bypassed; `data` parameter ignored

`select_training_cols` ignores the `data` DataFrame passed by the dispatcher and always auto-discovers from `500-merge_data_files/`. This means when `filter_delay_outliers=True` and `select_training_cols=True`, the filtered output in `501/` is a dead end — 502/ is populated from raw 500/ data, and the model trains on unfiltered rows.

### Problem 3 — Blocking `input()` in a pipeline

The interactive `input()` call prevents CI, remote server, and scheduled runs when `select_training_cols=True`. There is also no config constant storing the selected columns, making reruns non-reproducible.

### Problem 4 — `result["data"]` never updated in dispatcher

The dispatcher does not assign `result["data"] = cols_selection_result.get("data")` after success, and carries a wrong comment claiming the stage "doesn't modify data." Any future in-memory stage after this one would operate on the un-column-selected DataFrame.

---

## Design

### Section 1 — Data flow fix (`select_training_cols` input priority)

The method uses a three-level priority chain for its source data:

1. **`data` is not None** (in-memory DataFrame from dispatcher): use it directly, no disk discovery.
2. **`data` is None, discover from disk**: check `501-filter_delay_outliers/` first (pattern `merged_data_*.parquet`, excluding `_train`/`_test`), then fall back to `500-merge_data_files/`.

Output always goes to `502-select_training_cols/` (unchanged). No changes to `split_dataset` routing — it already reads from 502/ when `select_training_cols=True`, which will now correctly hold filtered + column-selected data.

### Section 2 — `SELECTED_COLUMNS` config bypass

Add to `const_training.py`:

```python
SELECTED_COLUMNS = []  # Empty = interactive prompt; populate to skip prompt
```

Inside `select_training_cols`, before the `while True:` loop:

- **If `SELECTED_COLUMNS` is non-empty:** validate all listed columns exist in the DataFrame, log which columns are being applied, proceed directly to column selection — no `input()` called.
- **If `SELECTED_COLUMNS` is empty:** run the existing interactive prompt unchanged.

The user manually edits `const_training.py` after an interactive run to persist the selection. No auto-write-back.

### Section 3 — Dispatcher fixes

In `execute_training_pipeline_steps`, at the `select_training_cols` success block:

1. Add `result["data"] = cols_selection_result.get("data")` so downstream in-memory stages receive the column-selected DataFrame.
2. Remove the wrong comment `# Note: This stage doesn't modify data, just displays column info`.

---

## Data Flow After Fix

With `filter_delay_outliers=True, select_training_cols=True`:

```
500/ (raw merged)
  ↓ filter_delay_outliers → saves to 501/, passes in-memory DataFrame
  ↓ select_training_cols receives data= (filtered DataFrame)
      → if SELECTED_COLUMNS set: applies config columns
      → if SELECTED_COLUMNS empty: prompts user interactively
      → saves to 502/
  ↓ split_dataset reads 502/ (filtered + column-selected) ✓
  ↓ balance_classes reads 503/
  ↓ XGBoost
```

---

## Out of Scope

- Auto-write-back of selected columns to `const_training.py` after interactive run (user handles manually).
- Changes to `split_dataset` routing logic.
- Any refactor of the interactive display/formatting.
