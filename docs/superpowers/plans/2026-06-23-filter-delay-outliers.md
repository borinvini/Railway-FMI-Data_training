# Filter Delay Outliers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert a `filter_delay_outliers` stage in the training pipeline between `merge_data_files` and `select_training_cols` that removes rows where `differenceInMinutes` falls outside asymmetric quantile bounds.

**Architecture:** The method filters the in-memory merged dataframe using configurable lower/upper quantile thresholds, prints a console summary of what was removed, and saves the filtered dataframe as a parquet snapshot for offline inspection. The state machine controls whether the step runs. The folder `502` (previously scaled data) shifts to `503` to accommodate the new `502-merged_outlier_filtered` folder.

**Tech Stack:** Python, pandas, pytest, parquet (via pandas `to_parquet`/`read_parquet`)

---

## File Map

| Action | File | What changes |
|--------|------|-------------|
| Modify | `config/const_training.py` | Add 2 quantile constants + new folder constant, rename old `502` constant to `503` |
| Modify | `src/training_pipeline.py` | Add import of new constants, add `filter_delay_outliers` method, insert state machine block |
| Create | `tests/test_filter_delay_outliers.py` | Unit tests for the new method and state machine wiring |

---

## Task 1: Add constants to `config/const_training.py`

**Files:**
- Modify: `config/const_training.py`

- [ ] **Step 1: Write the failing test for constants**

```python
# tests/test_filter_delay_outliers.py
import pytest

def test_quantile_constants_exist():
    from config.const_training import FILTER_LOWER_QUANTILE, FILTER_UPPER_QUANTILE
    assert 0.0 < FILTER_LOWER_QUANTILE < 0.1
    assert 0.9 < FILTER_UPPER_QUANTILE < 1.0
    assert FILTER_LOWER_QUANTILE < FILTER_UPPER_QUANTILE

def test_outlier_folder_constant_exists():
    from config.const_training import MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
    assert "502" in MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
    assert "outlier" in MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER

def test_scaled_folder_renamed_to_503():
    from config.const_training import MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
    assert "503" in MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_filter_delay_outliers.py::test_quantile_constants_exist tests/test_filter_delay_outliers.py::test_outlier_folder_constant_exists tests/test_filter_delay_outliers.py::test_scaled_folder_renamed_to_503 -v
```

Expected: 3 FAILED (ImportError or AssertionError)

- [ ] **Step 3: Add constants to `config/const_training.py`**

In `config/const_training.py`, replace the existing folder constant lines:

```python
MERGED_TRAINING_READY_OUTPUT_FOLDER = "data/output/500-merged_training_ready"
MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER = "data/output/501-merged_selected_training_ready"
MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER = "data/output/502-merged_selected_scaled_training_ready"
```

with:

```python
MERGED_TRAINING_READY_OUTPUT_FOLDER = "data/output/500-merged_training_ready"
MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER = "data/output/501-merged_selected_training_ready"
MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER = "data/output/502-merged_outlier_filtered"
MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER = "data/output/503-merged_selected_scaled_training_ready"

# Asymmetric quantile thresholds for delay outlier removal
# Lower tail: conservative cut (few implausibly-early arrivals)
# Upper tail: more aggressive cut (heavy right tail has likely data errors)
FILTER_LOWER_QUANTILE = 0.01
FILTER_UPPER_QUANTILE = 0.99
```

Also add `filter_delay_outliers` entry to `TRAINING_STATE_MACHINE`:

```python
TRAINING_STATE_MACHINE = {
    "merge_data_files": True,
    "filter_delay_outliers": True,   # new
    "select_training_cols": True,
    "split_dataset": True,
    "scale_weather_features": False,
    "numeric_correlation_analysis": False,
    "data_distribution_analysis": False,
    "target_feature_analysis": False,
    "train_xgboost_with_randomized_search_cv": True,
}
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_filter_delay_outliers.py::test_quantile_constants_exist tests/test_filter_delay_outliers.py::test_outlier_folder_constant_exists tests/test_filter_delay_outliers.py::test_scaled_folder_renamed_to_503 -v
```

Expected: 3 PASSED

- [ ] **Step 5: Run existing tests to confirm nothing broke**

```
pytest tests/ -v
```

Expected: all previously-passing tests still PASS. The only test that references `502` is `test_xgboost_data_dir_routing.py` which checks `MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER` — it will still pass because it imports the constant (which now points to `503`) rather than hardcoding `"502"`.

- [ ] **Step 6: Commit**

```bash
git add config/const_training.py tests/test_filter_delay_outliers.py
git commit -m "feat: add filter_delay_outliers constants and state machine entry"
```

---

## Task 2: Implement `filter_delay_outliers` method in `training_pipeline.py`

**Files:**
- Modify: `src/training_pipeline.py`
- Test: `tests/test_filter_delay_outliers.py`

- [ ] **Step 1: Add tests for the method**

Append to `tests/test_filter_delay_outliers.py`:

```python
import numpy as np
import pandas as pd
import os
from unittest.mock import patch

from src.training_pipeline import TrainingPipeline


def _make_pipeline(tmp_path):
    """Instantiate TrainingPipeline without filesystem side-effects."""
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


def _make_df(n=100, low=-50, high=500):
    """Build a dataframe with differenceInMinutes spread from low to high."""
    rng = np.random.default_rng(42)
    values = np.concatenate([
        rng.uniform(low, -10, 5),    # extreme negatives (lower tail)
        rng.uniform(-5, 30, n - 10), # normal range
        rng.uniform(200, high, 5),   # extreme positives (upper tail)
    ])
    rng.shuffle(values)
    return pd.DataFrame({"differenceInMinutes": values, "feature_a": range(n)})


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/outlier_filtered.parquet")
def test_filter_removes_extreme_rows(mock_save, tmp_path):
    """Rows outside [lower_q, upper_q] bounds must be dropped."""
    pipeline = _make_pipeline(tmp_path)
    df = _make_df(n=100)
    rows_before = len(df)

    result = pipeline.filter_delay_outliers(data=df)

    assert result["success"] is True
    assert result["data"] is not None
    assert len(result["data"]) < rows_before, "Expected some rows to be removed"
    col = result["data"]["differenceInMinutes"]
    assert col.min() >= result["lower_bound"], "No row should be below lower_bound"
    assert col.max() <= result["upper_bound"], "No row should be above upper_bound"


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/outlier_filtered.parquet")
def test_filter_result_contains_removal_counts(mock_save, tmp_path):
    """Result dict must expose removal statistics."""
    pipeline = _make_pipeline(tmp_path)
    df = _make_df(n=100)

    result = pipeline.filter_delay_outliers(data=df)

    assert "rows_before" in result
    assert "rows_removed_lower" in result
    assert "rows_removed_upper" in result
    assert result["rows_removed_lower"] + result["rows_removed_upper"] == result["rows_before"] - len(result["data"])


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/outlier_filtered.parquet")
def test_filter_missing_column_returns_data_unchanged(mock_save, tmp_path):
    """If differenceInMinutes is absent, return data unchanged with success=True."""
    pipeline = _make_pipeline(tmp_path)
    df = pd.DataFrame({"some_other_col": [1, 2, 3]})

    result = pipeline.filter_delay_outliers(data=df)

    assert result["success"] is True
    assert len(result["data"]) == 3


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/outlier_filtered.parquet")
def test_filter_none_data_returns_failure(mock_save, tmp_path):
    """If data=None, return success=False."""
    pipeline = _make_pipeline(tmp_path)

    result = pipeline.filter_delay_outliers(data=None)

    assert result["success"] is False
    assert "error" in result


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/outlier_filtered.parquet")
def test_filter_saves_parquet(mock_save, tmp_path):
    """filter_delay_outliers must call save_dataframe_to_parquet exactly once."""
    pipeline = _make_pipeline(tmp_path)
    df = _make_df(n=100)

    pipeline.filter_delay_outliers(data=df)

    mock_save.assert_called_once()
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_filter_delay_outliers.py -k "test_filter" -v
```

Expected: 5 FAILED (AttributeError: TrainingPipeline has no attribute 'filter_delay_outliers')

- [ ] **Step 3: Add import to `src/training_pipeline.py`**

At the top of `src/training_pipeline.py`, in the `from config.const_training import (...)` block, add the two new names:

```python
from config.const_training import (
    MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER,
    MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER,   # add this
    FILTER_LOWER_QUANTILE,                   # add this
    FILTER_UPPER_QUANTILE,                   # add this
    RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER,
    # ... rest unchanged
)
```

- [ ] **Step 4: Add `filter_delay_outliers` method to `TrainingPipeline`**

Insert the following method in `src/training_pipeline.py`, just before `def merge_data_files(self, csv_files):` (line 835):

```python
def filter_delay_outliers(self, data=None):
    """
    Remove rows where differenceInMinutes falls outside asymmetric quantile bounds.

    Uses FILTER_LOWER_QUANTILE and FILTER_UPPER_QUANTILE from const_training to
    compute bounds on the full merged dataset, then drops rows outside those bounds.
    Saves the filtered dataframe to MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER for inspection.

    Parameters
    ----------
    data : pd.DataFrame or None
        Merged dataframe from merge_data_files. Required.

    Returns
    -------
    dict
        {
          "success": bool,
          "data": pd.DataFrame | None,
          "rows_before": int,
          "rows_removed_lower": int,
          "rows_removed_upper": int,
          "lower_bound": float,
          "upper_bound": float,
        }
        On missing column: success=True, data unchanged, removal counts are 0.
        On data=None: success=False, error key present.
    """
    if data is None:
        print("    filter_delay_outliers: data is None — skipping")
        return {"success": False, "error": "data is None", "data": None}

    target_col = "differenceInMinutes"

    if target_col not in data.columns:
        print(f"    filter_delay_outliers: '{target_col}' not found — returning data unchanged")
        return {
            "success": True,
            "data": data,
            "rows_before": len(data),
            "rows_removed_lower": 0,
            "rows_removed_upper": 0,
            "lower_bound": None,
            "upper_bound": None,
        }

    df = data.copy()
    rows_before = len(df)

    lower_bound = float(df[target_col].quantile(FILTER_LOWER_QUANTILE))
    upper_bound = float(df[target_col].quantile(FILTER_UPPER_QUANTILE))

    lower_mask = df[target_col] < lower_bound
    upper_mask = df[target_col] > upper_bound

    rows_removed_lower = int(lower_mask.sum())
    rows_removed_upper = int(upper_mask.sum())

    df = df[~lower_mask & ~upper_mask].copy()
    rows_after = len(df)

    print(f"\n{'='*60}")
    print(f"FILTER DELAY OUTLIERS")
    print(f"{'='*60}")
    print(f"  Lower bound (q={FILTER_LOWER_QUANTILE}): {lower_bound:.2f} min")
    print(f"  Upper bound (q={FILTER_UPPER_QUANTILE}): {upper_bound:.2f} min")
    print(f"  Rows before : {rows_before:,}")
    print(f"  Removed (lower tail): {rows_removed_lower:,} ({rows_removed_lower / rows_before * 100:.2f}%)")
    print(f"  Removed (upper tail): {rows_removed_upper:,} ({rows_removed_upper / rows_before * 100:.2f}%)")
    print(f"  Rows after  : {rows_after:,} ({rows_after / rows_before * 100:.2f}% kept)")
    print(f"{'='*60}\n")

    print(f"--- SAVING filter_delay_outliers DATA ---")
    try:
        saved_path = save_dataframe_to_parquet(
            folder_path=os.path.join(self.project_root, MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER),
            month_id="outlier_filtered",
            df=df,
            file_prefix="merged_data",
        )
        print(f"✓ Saved filtered data to: {saved_path}")
    except Exception as save_error:
        print(f"⚠️  Warning: Failed to save filtered data: {save_error}")
        print("Continuing with in-memory filtered data.")

    if df.empty:
        print("⚠️  Warning: All rows removed after outlier filtering — check quantile thresholds.")

    return {
        "success": True,
        "data": df,
        "rows_before": rows_before,
        "rows_removed_lower": rows_removed_lower,
        "rows_removed_upper": rows_removed_upper,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }
```

- [ ] **Step 5: Run tests to confirm they pass**

```
pytest tests/test_filter_delay_outliers.py -k "test_filter" -v
```

Expected: 5 PASSED

- [ ] **Step 6: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/training_pipeline.py tests/test_filter_delay_outliers.py
git commit -m "feat: implement filter_delay_outliers method in TrainingPipeline"
```

---

## Task 3: Wire `filter_delay_outliers` into the state machine execution

**Files:**
- Modify: `src/training_pipeline.py` (the `execute_training_pipeline_steps` method)
- Test: `tests/test_filter_delay_outliers.py`

- [ ] **Step 1: Add state machine wiring test**

Append to `tests/test_filter_delay_outliers.py`:

```python
from unittest.mock import MagicMock


_MERGE_SUCCESS = {
    "success": True,
    "data": pd.DataFrame({
        "differenceInMinutes": list(range(-10, 90)),
        "feature_a": list(range(100)),
    }),
    "processed_files": 1,
    "total_rows": 100,
    "total_columns": 2,
}

_SELECT_COLS_SUCCESS = {
    "success": True,
    "total_columns": 2,
    "file_path": "/fake/merged.parquet",
    "dataset_shape": (100, 2),
    "column_types": {},
}


def _make_state_machine_with_filter(filter_enabled: bool) -> dict:
    return {
        "merge_data_files": True,
        "filter_delay_outliers": filter_enabled,
        "select_training_cols": False,
        "split_dataset": False,
        "scale_weather_features": False,
        "numeric_correlation_analysis": False,
        "data_distribution_analysis": False,
        "target_feature_analysis": False,
        "train_xgboost_with_randomized_search_cv": False,
    }


@patch.object(TrainingPipeline, "filter_delay_outliers")
@patch.object(TrainingPipeline, "merge_data_files", return_value=_MERGE_SUCCESS)
def test_state_machine_calls_filter_when_enabled(mock_merge, mock_filter, tmp_path):
    """When filter_delay_outliers=True in state machine, method must be called."""
    pipeline = _make_pipeline(tmp_path)
    mock_filter.return_value = {
        "success": True,
        "data": _MERGE_SUCCESS["data"],
        "rows_before": 100,
        "rows_removed_lower": 1,
        "rows_removed_upper": 1,
        "lower_bound": -9.0,
        "upper_bound": 88.0,
    }

    pipeline.execute_training_pipeline_steps([], state_machine=_make_state_machine_with_filter(True))

    mock_filter.assert_called_once()
    _, kwargs = mock_filter.call_args
    assert kwargs.get("data") is not None


@patch.object(TrainingPipeline, "filter_delay_outliers")
@patch.object(TrainingPipeline, "merge_data_files", return_value=_MERGE_SUCCESS)
def test_state_machine_skips_filter_when_disabled(mock_merge, mock_filter, tmp_path):
    """When filter_delay_outliers=False in state machine, method must NOT be called."""
    pipeline = _make_pipeline(tmp_path)

    pipeline.execute_training_pipeline_steps([], state_machine=_make_state_machine_with_filter(False))

    mock_filter.assert_not_called()
```

- [ ] **Step 2: Run these two tests to confirm they fail**

```
pytest tests/test_filter_delay_outliers.py::test_state_machine_calls_filter_when_enabled tests/test_filter_delay_outliers.py::test_state_machine_skips_filter_when_disabled -v
```

Expected: both FAILED (filter method never called / always called wrong)

- [ ] **Step 3: Insert the state machine block in `execute_training_pipeline_steps`**

In `src/training_pipeline.py`, locate the block starting at line ~190:

```python
        if state_machine.get("select_training_cols", False):
```

Insert the following block immediately BEFORE that line (after the closing `else` of the `merge_data_files` block):

```python
        if state_machine.get("filter_delay_outliers", False):
            if result["data"] is not None:
                try:
                    print(f"    → filter_delay_outliers")
                    filter_result = self.filter_delay_outliers(data=result["data"])

                    if filter_result and filter_result.get("success", False):
                        result["data"] = filter_result.get("data")
                        result["steps_executed"].append("filter_delay_outliers")
                        print(f"      ✓ Removed {filter_result.get('rows_removed_lower', 0) + filter_result.get('rows_removed_upper', 0):,} outlier rows")
                        print(f"      ✓ Rows remaining: {len(result['data']):,}")
                    else:
                        error_msg = filter_result.get("error", "filter_delay_outliers returned unsuccessful result")
                        result["errors"].append(error_msg)
                        print(f"      ✗ Failed - {error_msg}")
                        return result

                except Exception as e:
                    result["errors"].append(f"filter_delay_outliers failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ filter_delay_outliers (no data available)")
        else:
            print(f"    ⊝ filter_delay_outliers (disabled)")
```

- [ ] **Step 4: Run the wiring tests to confirm they pass**

```
pytest tests/test_filter_delay_outliers.py::test_state_machine_calls_filter_when_enabled tests/test_filter_delay_outliers.py::test_state_machine_skips_filter_when_disabled -v
```

Expected: 2 PASSED

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/training_pipeline.py tests/test_filter_delay_outliers.py
git commit -m "feat: wire filter_delay_outliers into training pipeline state machine"
```

---

## Self-Review

**Spec coverage:**
- ✅ Stage inserted between `merge_data_files` and `select_training_cols` (Task 3)
- ✅ Asymmetric quantile bounds via `FILTER_LOWER_QUANTILE` / `FILTER_UPPER_QUANTILE` (Task 1)
- ✅ Console summary with thresholds, per-tail removal counts, % kept (Task 2, method body)
- ✅ Saves filtered parquet to `502-merged_outlier_filtered` (Task 2, method body)
- ✅ Missing column: warn + return unchanged (Task 2, method body)
- ✅ Save failure: warn + continue (Task 2, method body)
- ✅ Empty result after filtering: warn + continue (Task 2, method body)
- ✅ `MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER` shifted from `502` to `503` (Task 1)
- ✅ State machine entry `filter_delay_outliers: True` (Task 1)

**Method name consistency:** `filter_delay_outliers` used consistently across const, method def, state machine block, and tests.

**Import:** `MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER`, `FILTER_LOWER_QUANTILE`, `FILTER_UPPER_QUANTILE` added to the import block in `training_pipeline.py` (Task 2, Step 3).

**Folder renaming note:** The physical folder `data/output/502-merged_selected_scaled_training_ready/` on disk is not renamed automatically — only the constant changes. If that folder already has data on disk, the next `scale_weather_features` run will write to the new `503` path instead.
