# Merge Schema Consistency Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an incremental column schema check to `merge_data_files` that fails immediately with a clear error when source files have mismatched columns, preventing silent NaN corruption in the merged output.

**Architecture:** Single change inside the existing file-load loop in `merge_data_files`. The first successfully loaded file sets the reference column set; each subsequent file is compared against it before being appended to `all_dataframes`. On mismatch, the method returns immediately — no files are left to load. No new config constants or imports are needed.

**Tech Stack:** Python, pandas, pytest

## Global Constraints

- Check runs inside the existing `for file_path in training_ready_files:` loop, after the `df.empty` guard and before `all_dataframes.append(df)`
- Error return shape: `{"success": False, "error": "<message>", "processed_files": 0}` — matches existing error returns in the method
- Column comparison uses sets (order-independent); missing and extra column lists are `sorted()` for deterministic output
- Single-file case must pass without error (no comparison runs)
- All tests in `tests/test_merge_data_files.py` (new file)
- Source files live in `data/output/101-preprocessed_training_ready/` relative to `project_root`; filename pattern is `training_ready_YYYY_MM.parquet`

---

### Task 1: Schema check in merge_data_files

**Files:**
- Modify: `src/training_pipeline.py` (inside `merge_data_files`, after line ~1232)
- Create: `tests/test_merge_data_files.py`

**Interfaces:**
- Consumes: `merge_data_files(self, csv_files)` — `csv_files` is accepted but unused; the method discovers files itself via `glob` on `TRAINING_READY_OUTPUT_FOLDER`
- `pipeline.project_root` is the only instance attribute the method uses for path construction
- Produces: no interface change — same return dict shape as before

- [ ] **Step 1: Write the failing tests**

Create `tests/test_merge_data_files.py`:

```python
import os

import pandas as pd

from src.training_pipeline import TrainingPipeline


def _make_pipeline(tmp_path):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.script_dir = str(tmp_path)
    pipeline.project_root = str(tmp_path)
    pipeline.output_dir = str(tmp_path / "output")
    pipeline.preprocessed_dir = str(tmp_path / "preprocessed")
    pipeline.randomized_search_dir = str(tmp_path / "rs")
    pipeline.random_forest_dir = str(tmp_path / "rf")
    pipeline.important_features_randomized_search_dir = str(tmp_path / "if")
    pipeline.xgboost_rs_dir = str(tmp_path / "xgb_rs")
    pipeline.regularized_regression_dir = str(tmp_path / "rr")
    pipeline.log_dir = str(tmp_path / "log")
    pipeline.DATA_FILE_PREFIX_FOR_TRAINING = "preprocessed_data_"
    os.makedirs(pipeline.log_dir, exist_ok=True)
    return pipeline


def _make_source_dir(tmp_path):
    src = tmp_path / "data/output/101-preprocessed_training_ready"
    src.mkdir(parents=True)
    return src


def test_schema_check_passes_when_all_files_match(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    df1.to_parquet(src / "training_ready_2024_01.parquet", index=False)
    df2.to_parquet(src / "training_ready_2024_02.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is True


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


def test_schema_check_single_file_passes(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_parquet(src / "training_ready_2024_01.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_merge_data_files.py -v
```

Expected: `test_schema_check_passes_when_all_files_match` and `test_schema_check_single_file_passes` PASS (no check exists yet so concat succeeds); `test_schema_check_fails_on_missing_column` and `test_schema_check_fails_on_extra_column` FAIL (method currently returns `success: True` on mismatched schemas).

- [ ] **Step 3: Add the schema check inside the load loop**

In `src/training_pipeline.py`, inside `merge_data_files`, find the block after the empty-file guard (around line 1232). The current code is:

```python
                    print(f"      Loaded {len(df):,} rows, {len(df.columns)} columns")
                    
                    # Store the dataframe and file info
                    all_dataframes.append(df)
```

Replace with:

```python
                    print(f"      Loaded {len(df):,} rows, {len(df.columns)} columns")

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

                    # Store the dataframe and file info
                    all_dataframes.append(df)
```

- [ ] **Step 4: Run the new tests to verify they pass**

```
pytest tests/test_merge_data_files.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full test suite to check for regressions**

```
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```
git add src/training_pipeline.py tests/test_merge_data_files.py
git commit -m "feat: add column schema consistency check to merge_data_files"
```

---

## Self-Review

**Spec coverage:**
- Incremental check inside load loop → Step 3
- Fails immediately on first mismatch → Step 3 (returns before appending)
- Error message names missing columns, extra columns, and both filenames → Step 3
- Single-file case passes → test_schema_check_single_file_passes + implementation uses `if not all_dataframes` as the first-file guard
- Column comparison is set-based (order-independent) → `set(df.columns)` throughout
- Sorted output for determinism → `sorted(missing)`, `sorted(extra)`
- Return shape matches existing errors → `{"success": False, "error": ..., "processed_files": 0}`

**Placeholder scan:** None. All code blocks are complete.

**Type consistency:** `reference_cols` and `current_cols` are both `set[str]`. `missing` and `extra` are `set[str]`, passed to `sorted()` which returns `list[str]`. Consistent throughout.
