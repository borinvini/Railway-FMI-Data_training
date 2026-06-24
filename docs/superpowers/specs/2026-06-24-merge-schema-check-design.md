# merge_data_files — Column Schema Consistency Check

**Date:** 2026-06-24  
**Stage:** Stage 1 — `merge_data_files`  
**Files touched:** `src/training_pipeline.py`, `tests/test_merge_data_files.py` (new)

---

## Problem Being Solved

`pd.concat` on DataFrames with mismatched column sets silently fills missing columns with `NaN` rather than raising an error. If two source files have different columns (e.g., one was preprocessed with a different feature set), the merged output is silently corrupted — NaN-filled columns propagate through every downstream stage without any warning.

---

## Design

### Check placement

Inside the existing `for file_path in training_ready_files:` loop in `merge_data_files`, immediately after a file is loaded into `df` and confirmed non-empty:

- **First valid file:** capture `reference_cols = set(df.columns)` and `reference_filename = filename`.
- **Each subsequent file:** compare `set(df.columns)` against `reference_cols`. If they differ, return immediately with a structured error (no remaining files are loaded).

The check uses column names as a set — order is irrelevant because `pd.concat` aligns by name anyway. Dtypes are out of scope; minor type differences (float32 vs float64) are benign and not worth false-alarming on.

### Error return

On mismatch, the method returns:

```python
{
    "success": False,
    "error": (
        f"Schema mismatch: {filename} differs from {reference_filename}.\n"
        f"  Missing columns (in reference, not in this file): {sorted(missing)}\n"
        f"  Extra columns (in this file, not in reference): {sorted(extra)}\n"
        f"Fix source files so all have identical columns before merging."
    ),
    "processed_files": 0
}
```

where `missing = reference_cols - current_cols` and `extra = current_cols - reference_cols`. Both lists are sorted for deterministic output.

### Single-file case

When only one file is found, `reference_cols` is set on the first (and only) iteration and no comparison ever runs. The merge proceeds normally.

---

## Out of Scope

- Dtype consistency checks.
- Auto-aligning to the column intersection (the opposite of failing early).
- Changes to any other stage or method.

---

## Tests

New file: `tests/test_merge_data_files.py`

Four cases:

1. **Identical schemas** — two files with the same columns → `success: True`.
2. **Missing column in second file** — second file lacks one column the reference has → `success: False`, error names the missing column and both filenames.
3. **Extra column in second file** — second file has one column the reference does not → `success: False`, error names the extra column and both filenames.
4. **Single file** — one file only → `success: True` (no comparison runs).

Tests use `tmp_path` fixtures and write real parquet files to disk, following the `_make_pipeline(tmp_path)` helper pattern from existing test files.
