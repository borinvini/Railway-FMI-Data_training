# Pipeline Data Quality Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 3 data quality bugs in `PreprocessingPipeline` — drop rows with unparseable `actualTime`, coerce target columns to numeric, and preserve rolling window columns through `filter_columns`.

**Architecture:** All 3 changes are targeted single-block insertions in 3 methods inside `src/preprocessing_pipeline.py`. No constant changes. Tests go in a new `tests/test_preprocessing_fixes.py`. After all code changes pass tests, re-run `main.py` (preprocessing only) to regenerate 84 training-ready parquets, then run the merge step.

**Tech Stack:** Python 3, pandas, pytest, unittest.mock — run all commands with `conda run -n venv_rail_fmi_training`

---

### Task 1: Fix 1 — Drop month=0 rows in `process_actual_time_column`

**Files:**
- Create: `tests/test_preprocessing_fixes.py`
- Modify: `src/preprocessing_pipeline.py` — insert 5 lines after line 1748

---

- [ ] **Step 1: Create test file with failing test for Fix 1**

Create `tests/__init__.py` (empty) and `tests/test_preprocessing_fixes.py`:

```python
import numpy as np
import pandas as pd
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from src.preprocessing_pipeline import PreprocessingPipeline


@contextmanager
def _null_logger(*args, **kwargs):
    yield MagicMock()


def _make_pipeline(tmp_path):
    """Instantiate pipeline without running __init__ (avoids filesystem setup)."""
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.log_dir = str(tmp_path)
    pipeline.current_file_id = "2023_01"
    return pipeline


# ---------------------------------------------------------------------------
# Fix 1
# ---------------------------------------------------------------------------

@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_drop_month_zero_rows(mock_save, tmp_path):
    """Rows where actualTime cannot be parsed must be dropped (month=0 → removed)."""
    pipeline = _make_pipeline(tmp_path)

    df = pd.DataFrame({
        "scheduledTime": [
            "2023-01-15T10:00:00.000Z",
            "2023-01-16T08:00:00.000Z",
            "2023-01-17T09:00:00.000Z",
        ],
        "actualTime": [
            "2023-01-15T10:05:00.000Z",  # valid → month=1
            "2023-01-16T08:03:00.000Z",  # valid → month=1
            "NOT_A_DATE",                 # invalid → month=0 → must be dropped
        ],
    })

    with patch.object(pipeline, "get_logger", _null_logger):
        result = pipeline.process_actual_time_column(dataframe=df, month_id="2023_01")

    assert result is not None
    assert len(result) == 2, f"Expected 2 rows after dropping bad date row, got {len(result)}"
    assert (result["month"] != 0).all(), "No row should have month=0 after fix"
```

- [ ] **Step 2: Run test to confirm it fails**

```
cd "D:\OneDrive - University of Oulu and Oamk\Railway-FMI-Data_training"
conda run -n venv_rail_fmi_training pytest tests/test_preprocessing_fixes.py::test_drop_month_zero_rows -v
```

Expected: **FAIL** — `AssertionError: Expected 2 rows after dropping bad date row, got 3`

- [ ] **Step 3: Apply Fix 1**

In `src/preprocessing_pipeline.py`, locate this block (lines 1745–1748):

```python
                # Convert new columns to appropriate data types
                for col in ['month', 'day_of_week', 'day_of_month']:
                    if col in df.columns:
                        # Fill NaN values with 0 for rows that couldn't be parsed, then convert to int
                        df[col] = df[col].fillna(0).astype(int)
```

Insert the following **immediately after** line 1748, before the `# Log statistics if we have valid data` comment:

```python
                # Drop rows where actualTime could not be parsed (month stays 0 after fillna)
                unparseable_mask = (df['month'] == 0)
                if unparseable_mask.any():
                    count = int(unparseable_mask.sum())
                    df = df[~unparseable_mask].copy()
                    print(f"Dropped {count} rows with unparseable actualTime (month=0)")
                    logger.info(f"Dropped {count} rows with unparseable actualTime (month=0)")
```

- [ ] **Step 4: Run test to confirm it passes**

```
conda run -n venv_rail_fmi_training pytest tests/test_preprocessing_fixes.py::test_drop_month_zero_rows -v
```

Expected: **PASS**

- [ ] **Step 5: Commit**

```
git add src/preprocessing_pipeline.py tests/__init__.py tests/test_preprocessing_fixes.py
git commit -m "fix: drop month=0 rows from unparseable actualTime in process_actual_time_column"
```

---

### Task 2: Fix 2 — Coerce target columns to numeric in `handle_missing_values`

**Files:**
- Modify: `src/preprocessing_pipeline.py` — insert 7 lines before line 2185 (`# Check required columns`)
- Modify: `tests/test_preprocessing_fixes.py` — append new test

---

- [ ] **Step 1: Add failing test for Fix 2**

Append to `tests/test_preprocessing_fixes.py`:

```python
# ---------------------------------------------------------------------------
# Fix 2
# ---------------------------------------------------------------------------

@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_coerce_target_columns_to_numeric(mock_save, tmp_path):
    """differenceInMinutes (object with timestamps) and differenceInMinutes_offset
    (object with string floats) must be coerced to float64. Rows where coercion
    produces NaN must be dropped by the existing dropna step."""
    pipeline = _make_pipeline(tmp_path)

    df = pd.DataFrame({
        # Two numeric strings + one ISO timestamp that will coerce to NaN
        "differenceInMinutes": ["5.0", "3.0", "2018-03-28T02:28:20.000Z"],
        "differenceInMinutes_offset": ["1.0", "-2.0", "0.0"],   # all numeric strings
        "differenceInMinutes_eachStation_offset": [5.0, -2.0, 0.0],
        "trainDelayed": [True, False, False],
        "cancelled": [False, False, False],
        # Temporal (required by imputation path)
        "month": [1, 1, 1],
        # Minimum weather columns to pass the weather-check in handle_missing_values
        "Air temperature": [2.0, 3.0, 1.0],
        "Wind speed": [5.0, 4.0, 6.0],
        "Gust speed": [8.0, 7.0, 9.0],
        "Wind direction": [180.0, 90.0, 270.0],
        "Relative humidity": [75.0, 80.0, 70.0],
        "Dew-point temperature": [0.0, 1.0, -1.0],
        "Precipitation intensity": [0.0, 0.1, 0.0],
        "Snow depth": [0.0, 0.0, 5.0],
        "Pressure (msl)": [1013.0, 1010.0, 1015.0],
        "Horizontal visibility": [10000.0, 8000.0, 9000.0],
        "Cloud amount": [2.0, 4.0, 6.0],
    })

    with patch.object(pipeline, "get_logger", _null_logger):
        result = pipeline.handle_missing_values(dataframe=df, month_id="2023_01")

    assert result is not None
    assert result["differenceInMinutes"].dtype in (np.float64, np.float32), (
        f"Expected float dtype for differenceInMinutes, got {result['differenceInMinutes'].dtype}"
    )
    assert result["differenceInMinutes_offset"].dtype in (np.float64, np.float32), (
        f"Expected float dtype for differenceInMinutes_offset, got {result['differenceInMinutes_offset'].dtype}"
    )
    # Row 3 had a timestamp in differenceInMinutes → coerced to NaN → dropped
    assert len(result) == 2, f"Expected 2 rows after dropping non-numeric target row, got {len(result)}"
```

- [ ] **Step 2: Run test to confirm it fails**

```
conda run -n venv_rail_fmi_training pytest tests/test_preprocessing_fixes.py::test_coerce_target_columns_to_numeric -v
```

Expected: **FAIL** — dtype assertion error (`object` instead of `float64`)

- [ ] **Step 3: Apply Fix 2**

In `src/preprocessing_pipeline.py`, locate this comment (line 2185):

```python
            # Check required columns
            required_cols = [col for col in VALID_TARGET_FEATURES if col in df.columns]
```

Insert the following **immediately before** that comment:

```python
            # Coerce target feature columns from object/string to numeric.
            # Handles string-encoded floats ("1.0") and timestamp contamination.
            # Rows where coercion produces NaN are removed by the dropna below.
            for col in VALID_TARGET_FEATURES:
                if col in df.columns and df[col].dtype == object:
                    before_nulls = int(df[col].isna().sum())
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    coerced_to_nan = int(df[col].isna().sum()) - before_nulls
                    if coerced_to_nan > 0:
                        print(f"Coerced '{col}' to numeric: {coerced_to_nan} non-numeric values became NaN")
                        logger.info(f"Coerced '{col}' to numeric: {coerced_to_nan} non-numeric values → NaN")

```

- [ ] **Step 4: Run test to confirm it passes**

```
conda run -n venv_rail_fmi_training pytest tests/test_preprocessing_fixes.py::test_coerce_target_columns_to_numeric -v
```

Expected: **PASS**

- [ ] **Step 5: Commit**

```
git add src/preprocessing_pipeline.py tests/test_preprocessing_fixes.py
git commit -m "fix: coerce VALID_TARGET_FEATURES to numeric before dropna in handle_missing_values"
```

---

### Task 3: Fix 3 — Pattern-based rolling window column detection in `filter_columns`

**Files:**
- Modify: `src/preprocessing_pipeline.py` — insert 4 lines after line 1880 (after `weather_scenario_cols_found` block)
- Modify: `tests/test_preprocessing_fixes.py` — append new test

---

- [ ] **Step 1: Add failing test for Fix 3**

Append to `tests/test_preprocessing_fixes.py`:

```python
# ---------------------------------------------------------------------------
# Fix 3
# ---------------------------------------------------------------------------

@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_filter_columns_keeps_rolling_windows(mock_save, tmp_path):
    """filter_columns must retain columns whose names contain (12h, (24h, or (72h."""
    pipeline = _make_pipeline(tmp_path)

    df = pd.DataFrame({
        # Target features
        "differenceInMinutes_eachStation_offset": [1.0, 2.0],
        "differenceInMinutes": [1.0, 2.0],
        "differenceInMinutes_offset": [1.0, 2.0],
        "trainDelayed": [1, 0],
        "cancelled": [0, 0],
        # Train prediction features
        "trainStopping": [1, 1],
        "commercialStop": [1, 1],
        "month": [1, 2],
        "hour": ["10:00", "11:00"],
        "day_of_week": [2, 3],
        "day_of_month": [15, 16],
        "causes_related_to_weather": [0, 1],
        # Basic weather
        "Air temperature": [2.0, 3.0],
        "Wind speed": [5.0, 4.0],
        # Rolling window columns — these MUST be kept after the fix
        "Air temperature (12h max)": [4.0, 5.0],
        "Air temperature (24h mean)": [3.0, 4.0],
        "Snow depth (72h max)": [10.0, 15.0],
        "Precipitation amount (12h cumulative)": [0.5, 0.0],
        # Columns that must be dropped
        "stationShortCode": ["OL", "OL"],
        "trainNumber": [123, 456],
    })

    with patch.object(pipeline, "get_logger", _null_logger):
        result = pipeline.filter_columns(dataframe=df, month_id="2023_01")

    assert result is not None
    assert "Air temperature (12h max)" in result.columns, "12h window column must be kept"
    assert "Air temperature (24h mean)" in result.columns, "24h window column must be kept"
    assert "Snow depth (72h max)" in result.columns, "72h window column must be kept"
    assert "Precipitation amount (12h cumulative)" in result.columns, "12h cumulative must be kept"
    assert "stationShortCode" not in result.columns, "stationShortCode must be dropped"
    assert "trainNumber" not in result.columns, "trainNumber must be dropped"
```

- [ ] **Step 2: Run test to confirm it fails**

```
conda run -n venv_rail_fmi_training pytest tests/test_preprocessing_fixes.py::test_filter_columns_keeps_rolling_windows -v
```

Expected: **FAIL** — `AssertionError: 12h window column must be kept`

- [ ] **Step 3: Apply Fix 3**

In `src/preprocessing_pipeline.py`, locate these two lines (1879–1880):

```python
                print(f"Weather scenario columns found: {weather_scenario_cols_found}")
                logger.info(f"Weather scenario columns found: {weather_scenario_cols_found}")
```

Insert the following **immediately after** line 1880:

```python

                # Add rolling window columns matching (12h, (24h, (72h patterns
                window_patterns = ('(12h', '(24h', '(72h')
                window_cols_found = [col for col in df.columns if any(p in col for p in window_patterns)]
                columns_to_keep.extend(window_cols_found)
                print(f"Rolling window columns found: {len(window_cols_found)}")
                logger.info(f"Rolling window columns found: {window_cols_found}")
```

- [ ] **Step 4: Run test to confirm it passes**

```
conda run -n venv_rail_fmi_training pytest tests/test_preprocessing_fixes.py::test_filter_columns_keeps_rolling_windows -v
```

Expected: **PASS**

- [ ] **Step 5: Run all 3 tests**

```
conda run -n venv_rail_fmi_training pytest tests/test_preprocessing_fixes.py -v
```

Expected: **3 passed, 0 failed**

- [ ] **Step 6: Commit**

```
git add src/preprocessing_pipeline.py tests/test_preprocessing_fixes.py
git commit -m "fix: add pattern-based rolling window column detection in filter_columns"
```

---

### Task 4: Re-run preprocessing pipeline on all 84 year-month files

**Files:**
- Modify: `config/const.py` — toggle execution flags

---

- [ ] **Step 1: Enable preprocessing only**

In `config/const.py`, set:

```python
EXECUTE_PREPROCESSING_DATA_PIPELINE = True
EXECUTE_TRAINING_PIPELINE = False
```

- [ ] **Step 2: Run preprocessing**

```
cd "D:\OneDrive - University of Oulu and Oamk\Railway-FMI-Data_training"
conda run -n venv_rail_fmi_training python main.py
```

Expected: `84/84` files processed. Watch for these lines per file in the output:
- `Dropped N rows with unparseable actualTime (month=0)` → Fix 1 active
- `Coerced 'differenceInMinutes' to numeric: N non-numeric values became NaN` → Fix 2 active
- `Rolling window columns found: 78` → Fix 3 active

- [ ] **Step 3: Run the merge step**

In `config/const.py`, set:

```python
EXECUTE_PREPROCESSING_DATA_PIPELINE = False
EXECUTE_TRAINING_PIPELINE = True
```

```
conda run -n venv_rail_fmi_training python main.py
```

Expected: `merge_data_files` step produces a new `merged_data_2018-01_to_2024-12.csv` in `data/output/500-merged_training_ready/`.

- [ ] **Step 4: Verify merged output**

```
conda run -n venv_rail_fmi_training python -c "
import pandas as pd
df = pd.read_csv(r'data/output/500-merged_training_ready/merged_data_2018-01_to_2024-12.csv', low_memory=False)
print('Shape:', df.shape)
print('month=0 rows:', (df['month'] == 0).sum())
print('differenceInMinutes dtype:', df['differenceInMinutes'].dtype)
window_cols = [c for c in df.columns if any(p in c for p in ('(12h', '(24h', '(72h'))]
print('Window columns present:', len(window_cols))
"
```

Expected output:
```
Shape: (~89668, ~117)
month=0 rows: 0
differenceInMinutes dtype: float64
Window columns present: 78
```

- [ ] **Step 5: Restore config and commit**

In `config/const.py`, restore to intended defaults:

```python
EXECUTE_PREPROCESSING_DATA_PIPELINE = False
EXECUTE_TRAINING_PIPELINE = True
```

```
git add config/const.py
git commit -m "chore: restore pipeline execution flags after re-run"
```
