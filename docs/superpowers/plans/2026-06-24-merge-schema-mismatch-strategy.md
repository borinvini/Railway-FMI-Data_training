# Merge Schema Mismatch Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-fail schema check in `merge_data_files` with a strategy-driven approach: `''` (default) asks the user interactively, `'intersect'` auto-drops mismatched columns, and `'fail'` preserves the current hard-fail behavior.

**Architecture:** Add `SCHEMA_MISMATCH_STRATEGY = ''` to `config/const_training.py` and import it into `src/training_pipeline.py`. The existing schema check block is extended to branch on the strategy before returning an error: on `'intersect'` or a `'y'` response, compute the column intersection, reselect all already-loaded DataFrames, reselect the current file, and narrow `reference_cols` before continuing. Existing mismatch tests gain a `@patch` decorator to pin the strategy to `'fail'`; new tests cover the `'intersect'` path.

**Tech Stack:** Python, pandas, pytest, unittest.mock

## Global Constraints

- `SCHEMA_MISMATCH_STRATEGY = ''` added to `config/const_training.py` immediately after `SELECTED_COLUMNS`
- Import `SCHEMA_MISMATCH_STRATEGY` in `src/training_pipeline.py` alongside `SELECTED_COLUMNS` at line ~115
- Strategy values: `''` = interactive `input()`, `'intersect'` = auto-proceed, `'fail'` = auto-error; any other value treated as `'fail'`
- Intersection logic: `common_cols = sorted(reference_cols & current_cols)`; reselect `all_dataframes` in-place; reselect `df`; set `reference_cols = set(common_cols)`
- Existing mismatch tests must be patched with `@patch('src.training_pipeline.SCHEMA_MISMATCH_STRATEGY', 'fail')` — do NOT change their assertion logic
- New tests use `@patch('src.training_pipeline.SCHEMA_MISMATCH_STRATEGY', 'intersect')`
- All tests in `tests/test_merge_data_files.py`

---

### Task 1: Strategy constant, import, logic change, and tests

**Files:**
- Modify: `config/const_training.py` (add constant after `SELECTED_COLUMNS` at line 39)
- Modify: `src/training_pipeline.py` (~line 115 for import; ~lines 1234–1254 for logic)
- Modify: `tests/test_merge_data_files.py` (add import, patch existing tests, add 4 new tests)

**Interfaces:**
- Consumes: `SCHEMA_MISMATCH_STRATEGY: str` from `config.const_training`; existing schema check variables `reference_cols`, `reference_filename`, `current_cols`, `missing`, `extra`, `all_dataframes`, `df`, `filename`
- Produces: no interface change — same return dict shape; merged `result["data"]` now reflects the intersected column set when strategy is `'intersect'`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_merge_data_files.py`. At the top of the file, add `from unittest.mock import patch` to the existing imports block:

```python
import os
from unittest.mock import patch

import pandas as pd

from src.training_pipeline import TrainingPipeline
```

Then append these tests at the bottom of the file:

```python
def test_schema_mismatch_constant_exists():
    from config.const_training import SCHEMA_MISMATCH_STRATEGY
    assert SCHEMA_MISMATCH_STRATEGY == ''


@patch('src.training_pipeline.SCHEMA_MISMATCH_STRATEGY', 'intersect')
def test_intersect_strategy_drops_extra_columns(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8], "extra": [9, 10]})
    df1.to_parquet(src / "training_ready_2024_01.parquet", index=False)
    df2.to_parquet(src / "training_ready_2024_02.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is True
    assert "extra" not in result["data"].columns
    assert sorted(result["data"].columns) == ["a", "b"]


@patch('src.training_pipeline.SCHEMA_MISMATCH_STRATEGY', 'intersect')
def test_intersect_strategy_drops_missing_columns(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6]})  # missing "b"
    df1.to_parquet(src / "training_ready_2024_01.parquet", index=False)
    df2.to_parquet(src / "training_ready_2024_02.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is True
    assert "b" not in result["data"].columns
    assert list(result["data"].columns) == ["a"]


@patch('src.training_pipeline.SCHEMA_MISMATCH_STRATEGY', 'intersect')
def test_intersect_strategy_reselects_already_loaded_dfs(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df1 = pd.DataFrame({"a": [1], "b": [2]})
    df2 = pd.DataFrame({"a": [3], "b": [4]})
    df3 = pd.DataFrame({"a": [5], "b": [6], "extra": [7]})
    df1.to_parquet(src / "training_ready_2024_01.parquet", index=False)
    df2.to_parquet(src / "training_ready_2024_02.parquet", index=False)
    df3.to_parquet(src / "training_ready_2024_03.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is True
    assert "extra" not in result["data"].columns
    assert result.get("files_merged") == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_merge_data_files.py::test_schema_mismatch_constant_exists tests/test_merge_data_files.py::test_intersect_strategy_drops_extra_columns tests/test_merge_data_files.py::test_intersect_strategy_drops_missing_columns tests/test_merge_data_files.py::test_intersect_strategy_reselects_already_loaded_dfs -v
```

Expected: `test_schema_mismatch_constant_exists` FAILS (`ImportError` on `SCHEMA_MISMATCH_STRATEGY`); the three `intersect` tests FAIL (no such attribute to patch, or method still returns error).

- [ ] **Step 3: Add `SCHEMA_MISMATCH_STRATEGY` to `config/const_training.py`**

In `config/const_training.py`, after line 39 (`SELECTED_COLUMNS = []  # ...`), add:

```python
SCHEMA_MISMATCH_STRATEGY = ''  # '' = ask interactively; 'intersect' = auto-drop; 'fail' = auto-fail
```

- [ ] **Step 4: Import `SCHEMA_MISMATCH_STRATEGY` in `src/training_pipeline.py`**

In the `from config.const_training import (` block (around line 113), find:

```python
    SELECTED_COLUMNS,
```

Replace with:

```python
    SELECTED_COLUMNS,
    SCHEMA_MISMATCH_STRATEGY,
```

- [ ] **Step 5: Replace the schema check block in `merge_data_files`**

In `src/training_pipeline.py`, find the schema consistency check block (lines ~1234–1254):

```python
                    # Schema consistency check
                    if not all_dataframes:
                        reference_cols = set(df.columns)
                        reference_filename = filename
                    else:
                        current_cols = set(df.columns)
                        missing = reference_cols - current_cols
                        extra = current_cols - reference_cols
                        if missing or extra:
                            error_msg = (
                                f"Schema mismatch: {filename} differs from {reference_filename}.\n"
                                f"  Missing columns (in reference, not in this file): {sorted(missing)}\n"
                                f"  Extra columns (in this file, not in reference): {sorted(extra)}\n"
                                f"Fix source files so all have identical columns before merging."
                            )
                            print(f"    merge_data_files: {error_msg}")
                            return {
                                "success": False,
                                "error": error_msg,
                                "processed_files": 0
                            }
```

Replace with:

```python
                    # Schema consistency check
                    if not all_dataframes:
                        reference_cols = set(df.columns)
                        reference_filename = filename
                    else:
                        current_cols = set(df.columns)
                        missing = reference_cols - current_cols
                        extra = current_cols - reference_cols
                        if missing or extra:
                            mismatch_msg = (
                                f"Schema mismatch: {filename} differs from {reference_filename}.\n"
                                f"  Missing columns (in reference, not in this file): {sorted(missing)}\n"
                                f"  Extra columns (in this file, not in reference): {sorted(extra)}\n"
                                f"Fix source files so all have identical columns before merging."
                            )
                            print(f"    merge_data_files: {mismatch_msg}")
                            if SCHEMA_MISMATCH_STRATEGY == 'intersect':
                                proceed = True
                            elif SCHEMA_MISMATCH_STRATEGY == 'fail':
                                proceed = False
                            else:
                                response = input("    Proceed with the column intersection? Extra/missing columns will be dropped. (y/n): ").strip().lower()
                                proceed = response == 'y'
                            if proceed:
                                common_cols = sorted(reference_cols & current_cols)
                                all_dataframes = [df_prev[common_cols] for df_prev in all_dataframes]
                                df = df[common_cols]
                                reference_cols = set(common_cols)
                                print(f"    merge_data_files: Proceeding with {len(common_cols)} common columns.")
                            else:
                                return {
                                    "success": False,
                                    "error": mismatch_msg,
                                    "processed_files": 0
                                }
```

- [ ] **Step 6: Patch the two existing mismatch tests**

In `tests/test_merge_data_files.py`, add `@patch('src.training_pipeline.SCHEMA_MISMATCH_STRATEGY', 'fail')` immediately above `def test_schema_check_fails_on_missing_column` and above `def test_schema_check_fails_on_extra_column`:

```python
@patch('src.training_pipeline.SCHEMA_MISMATCH_STRATEGY', 'fail')
def test_schema_check_fails_on_missing_column(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6]})          # missing "b"
    df1.to_parquet(src / "training_ready_2024_01.parquet", index=False)
    df2.to_parquet(src / "training_ready_2024_02.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is False
    assert "b" in result["error"]
    assert "training_ready_2024_02.parquet" in result["error"]
    assert "training_ready_2024_01.parquet" in result["error"]


@patch('src.training_pipeline.SCHEMA_MISMATCH_STRATEGY', 'fail')
def test_schema_check_fails_on_extra_column(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [5, 6], "extra": [7, 8]})   # extra column
    df1.to_parquet(src / "training_ready_2024_01.parquet", index=False)
    df2.to_parquet(src / "training_ready_2024_02.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is False
    assert "extra" in result["error"]
    assert "training_ready_2024_02.parquet" in result["error"]
```

- [ ] **Step 7: Run all merge tests to verify they pass**

```
pytest tests/test_merge_data_files.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 8: Run full test suite to check for regressions**

```
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```
git add config/const_training.py src/training_pipeline.py tests/test_merge_data_files.py
git commit -m "feat: add SCHEMA_MISMATCH_STRATEGY — interactive/auto-intersect/fail on column mismatch"
```

---

## Self-Review

**Spec coverage:**
- `SCHEMA_MISMATCH_STRATEGY = ''` in const_training → Step 3
- Import alongside `SELECTED_COLUMNS` → Step 4
- Strategy branching (`'intersect'` / `'fail'` / interactive) → Step 5
- Intersection: reselect `all_dataframes`, reselect `df`, narrow `reference_cols` → Step 5
- Existing mismatch tests patched with `'fail'` → Step 6
- New constant-exists test → Step 1
- New intersect tests (extra, missing, three-file reselect) → Step 1

**Placeholder scan:** None. All code blocks are complete and exact.

**Type consistency:** `SCHEMA_MISMATCH_STRATEGY` is `str` throughout. `common_cols` is `list[str]` from `sorted(...)`. `reference_cols` is reassigned to `set(common_cols)` — consistent with its original type `set`. `all_dataframes` list comprehension produces `list[pd.DataFrame]` — same type as before.
