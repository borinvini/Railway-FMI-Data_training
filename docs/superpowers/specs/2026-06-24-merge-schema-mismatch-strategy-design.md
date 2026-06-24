# merge_data_files — Schema Mismatch Strategy

**Date:** 2026-06-24  
**Stage:** Stage 1 — `merge_data_files`  
**Files touched:** `src/training_pipeline.py`, `config/const_training.py`, `tests/test_merge_data_files.py`

---

## Problem Being Solved

The schema check added in the previous session hard-fails on any column mismatch. In practice, a file may have extra columns (e.g., "Cloud amount" columns appeared in 2019-06 files) that are acceptable to drop rather than block the entire merge. The user needs an interactive prompt — and a config bypass for non-interactive runs — to choose between dropping mismatched columns (intersection) or failing.

---

## Design

### New config constant

Add to `config/const_training.py`:

```python
SCHEMA_MISMATCH_STRATEGY = ''  # '' = ask interactively; 'intersect' = auto-drop; 'fail' = auto-fail
```

Import `SCHEMA_MISMATCH_STRATEGY` in `src/training_pipeline.py` alongside the existing `const_training` imports.

### Modified schema check behavior

When a column mismatch is detected, the method now branches on `SCHEMA_MISMATCH_STRATEGY`:

- **`'fail'`** (or any unrecognized value): return the error immediately — same as current behavior.
- **`'intersect'`**: automatically proceed with the column intersection, no prompt.
- **`''`** (empty string, default): print the mismatch details and prompt:
  ```
  Proceed with the column intersection? Extra/missing columns will be dropped. (y/n):
  ```
  - `y` → proceed with intersection
  - anything else → return the error

### Intersection logic (shared by 'intersect' and 'y' response)

```python
common_cols = sorted(reference_cols & current_cols)
all_dataframes = [df_prev[common_cols] for df_prev in all_dataframes]
df = df[common_cols]
reference_cols = set(common_cols)
```

`all_dataframes` is reassigned in-place so previously loaded DataFrames also lose the extra columns. The current file `df` is reselected before being appended. `reference_cols` is narrowed to the intersection so subsequent files are compared against the narrower schema.

### Test impact

The two existing mismatch tests (`test_schema_check_fails_on_missing_column`, `test_schema_check_fails_on_extra_column`) currently rely on the hard-fail path. After this change the default strategy is `''` (interactive), which would call `input()` in tests. Fix: patch `src.training_pipeline.SCHEMA_MISMATCH_STRATEGY` to `'fail'` in those two tests via `unittest.mock.patch`.

### New tests

- `test_schema_mismatch_constant_exists` — `SCHEMA_MISMATCH_STRATEGY` is importable from `config.const_training` and equals `''`
- `test_intersect_strategy_drops_extra_columns` — with strategy `'intersect'`, a second file with extra columns → success, merged df has only the common columns
- `test_intersect_strategy_drops_missing_columns` — with strategy `'intersect'`, a second file missing columns → success, merged df has only the common columns
- `test_intersect_strategy_reselects_already_loaded_dfs` — with strategy `'intersect'` and 3 files (first two match, third has extras) → all DataFrames in the merge have only the common columns

---

## Out of Scope

- Dtype reconciliation.
- Logging the intersection decision to the summary file.
- Any changes outside `merge_data_files` and its config constant.
