# balance_classes Pipeline Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `balance_classes` stage after `filter_delay_outliers` that applies SMOTE-Tomek to the in-memory DataFrame, saves the balanced result to `data/output/504-merged_balanced`, and routes `split_dataset` to read from that folder when the stage is enabled.

**Architecture:** The stage follows the exact same try/except/print pattern as all other stages in `execute_training_pipeline_steps`. It derives the binary label on-the-fly from `differenceInMinutes > TRAIN_DELAY_MINUTES`, applies `SMOTETomek` to numeric columns only, and saves to a new numbered output folder. The `split_dataset` method gains a `data_dir` parameter; the state machine call passes `504-merged_balanced` when `balance_classes` is enabled, mirroring the existing `scale_weather_features` routing pattern used by `train_xgboost_with_randomized_search_cv`.

**Tech Stack:** Python, pandas, numpy, imbalanced-learn (`SMOTETomek` from `imblearn.combine`), pytest

## Global Constraints

- `imblearn` is already installed (confirmed by existing `BorderlineSMOTE` import in `training_pipeline.py`)
- All stage methods return `{"success": bool, "data": pd.DataFrame | None, ...}`
- `save_dataframe_to_parquet` from `src.file_utils` is used for all disk writes (already imported)
- `TRAIN_DELAY_MINUTES` is already imported from `config.const_preprocessing` in `training_pipeline.py`
- Output folder numbering: 500 raw merge, 501 selected cols, 502 outlier filtered, 503 scaled, **504 balanced** (new)
- Tests use `TrainingPipeline.__new__(TrainingPipeline)` + manual attribute assignment — never call `__init__`
- All parquet saves in tests must be mocked with `@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/balanced.parquet")`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `config/const_training.py` | Modify | Add `MERGED_BALANCED_OUTPUT_FOLDER`; add `"balance_classes": True` to state machine |
| `src/training_pipeline.py` | Modify | Import `SMOTETomek` + new constants; add `balance_classes()` method; wire state machine block; add `data_dir` param to `split_dataset` |
| `tests/test_balance_classes.py` | Create | All tests for `balance_classes()` method and state machine wiring |

---

## Task 1: Add constant and state machine key

**Files:**
- Modify: `config/const_training.py`

**Interfaces:**
- Produces: `MERGED_BALANCED_OUTPUT_FOLDER = "data/output/504-merged_balanced"` — used by Tasks 2 and 3

- [ ] **Step 1: Write the failing test**

Create `tests/test_balance_classes.py`:

```python
def test_balanced_folder_constant_exists():
    from config.const_training import MERGED_BALANCED_OUTPUT_FOLDER
    assert "504" in MERGED_BALANCED_OUTPUT_FOLDER
    assert "balanced" in MERGED_BALANCED_OUTPUT_FOLDER


def test_balance_classes_in_state_machine():
    from config.const_training import TRAINING_STATE_MACHINE
    assert "balance_classes" in TRAINING_STATE_MACHINE
    assert isinstance(TRAINING_STATE_MACHINE["balance_classes"], bool)


def test_balance_classes_after_filter_delay_outliers_in_state_machine():
    from config.const_training import TRAINING_STATE_MACHINE
    keys = list(TRAINING_STATE_MACHINE.keys())
    assert keys.index("balance_classes") == keys.index("filter_delay_outliers") + 1
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_balance_classes.py -v
```

Expected: 3 FAILs — `MERGED_BALANCED_OUTPUT_FOLDER` not found, `balance_classes` not in state machine.

- [ ] **Step 3: Add constant and state machine key to `config/const_training.py`**

After the line `MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER = "data/output/503-merged_selected_scaled_training_ready"`, add:

```python
MERGED_BALANCED_OUTPUT_FOLDER = "data/output/504-merged_balanced"
```

In `TRAINING_STATE_MACHINE`, insert `"balance_classes": True` immediately after `"filter_delay_outliers": True`:

```python
TRAINING_STATE_MACHINE = {
    "merge_data_files": True,
    "filter_delay_outliers": True,
    "balance_classes": True,
    "select_training_cols": True,
    "split_dataset": True,
    "scale_weather_features": False,
    "numeric_correlation_analysis": False,
    "data_distribution_analysis": False,
    "target_feature_analysis": False,
    "train_xgboost_with_randomized_search_cv": True,
}
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_balance_classes.py::test_balanced_folder_constant_exists tests/test_balance_classes.py::test_balance_classes_in_state_machine tests/test_balance_classes.py::test_balance_classes_after_filter_delay_outliers_in_state_machine -v
```

Expected: 3 PASSes.

- [ ] **Step 5: Commit**

```bash
git add config/const_training.py tests/test_balance_classes.py
git commit -m "feat: add MERGED_BALANCED_OUTPUT_FOLDER constant and balance_classes state machine key"
```

---

## Task 2: Implement `balance_classes()` method

**Files:**
- Modify: `src/training_pipeline.py`
- Test: `tests/test_balance_classes.py`

**Interfaces:**
- Consumes: `result["data"]` (pd.DataFrame with `differenceInMinutes` column); `TRAIN_DELAY_MINUTES` (int, already imported from `const_preprocessing`); `RESAMPLING_METHOD`, `IMBALANCE_THRESHOLD`, `SMOTE_RANDOM_STATE`, `MERGED_BALANCED_OUTPUT_FOLDER` from `const_training`
- Produces: `TrainingPipeline.balance_classes(data: pd.DataFrame | None) -> dict` with keys `success`, `data`, `rows_before`, `rows_after`, `minority_share_before`, `minority_share_after`, `resampling_method`, `skipped`, `dropped_non_numeric_cols`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_balance_classes.py`:

```python
import numpy as np
import pandas as pd
import os
from unittest.mock import patch

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


def _make_imbalanced_df(n_punctual=300, n_delayed=100, seed=42):
    """3:1 imbalanced DataFrame with differenceInMinutes and two numeric features."""
    rng = np.random.default_rng(seed)
    punctual = rng.uniform(-4, 5, n_punctual)
    delayed = rng.uniform(6, 60, n_delayed)
    diff = np.concatenate([punctual, delayed])
    feat_a = rng.normal(0, 1, n_punctual + n_delayed)
    feat_b = rng.normal(5, 2, n_punctual + n_delayed)
    return pd.DataFrame({
        "differenceInMinutes": diff,
        "feature_a": feat_a,
        "feature_b": feat_b,
    })


def _make_balanced_df(n=200, seed=42):
    """50/50 split — minority share will exceed IMBALANCE_THRESHOLD (30%)."""
    rng = np.random.default_rng(seed)
    diff = np.concatenate([rng.uniform(-4, 5, n // 2), rng.uniform(6, 60, n // 2)])
    return pd.DataFrame({
        "differenceInMinutes": diff,
        "feature_a": rng.normal(0, 1, n),
    })


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/balanced.parquet")
def test_none_data_returns_failure(mock_save, tmp_path):
    pipeline = _make_pipeline(tmp_path)
    result = pipeline.balance_classes(data=None)
    assert result["success"] is False
    assert "error" in result


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/balanced.parquet")
def test_missing_column_returns_data_unchanged(mock_save, tmp_path):
    pipeline = _make_pipeline(tmp_path)
    df = pd.DataFrame({"feature_a": [1, 2, 3]})
    result = pipeline.balance_classes(data=df)
    assert result["success"] is True
    assert result["skipped"] is True
    assert len(result["data"]) == 3


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/balanced.parquet")
def test_skips_when_already_balanced(mock_save, tmp_path):
    pipeline = _make_pipeline(tmp_path)
    df = _make_balanced_df()
    result = pipeline.balance_classes(data=df)
    assert result["success"] is True
    assert result["skipped"] is True
    assert result["resampling_method"] == "NONE"


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/balanced.parquet")
def test_smote_tomek_increases_minority_count(mock_save, tmp_path):
    pipeline = _make_pipeline(tmp_path)
    df = _make_imbalanced_df(n_punctual=300, n_delayed=100)
    result = pipeline.balance_classes(data=df)
    assert result["success"] is True
    assert result["skipped"] is False
    assert result["resampling_method"] == "SMOTE_TOMEK"
    assert result["minority_share_after"] > result["minority_share_before"]


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/balanced.parquet")
def test_rows_after_is_in_result(mock_save, tmp_path):
    pipeline = _make_pipeline(tmp_path)
    df = _make_imbalanced_df()
    result = pipeline.balance_classes(data=df)
    assert result["rows_before"] == len(df)
    assert result["rows_after"] == len(result["data"])


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/balanced.parquet")
def test_non_numeric_columns_are_dropped(mock_save, tmp_path):
    pipeline = _make_pipeline(tmp_path)
    df = _make_imbalanced_df()
    df["causes"] = "weather"  # non-numeric column
    result = pipeline.balance_classes(data=df)
    assert result["success"] is True
    assert "causes" not in result["data"].columns
    assert "causes" in result["dropped_non_numeric_cols"]


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/balanced.parquet")
def test_saves_parquet_once(mock_save, tmp_path):
    pipeline = _make_pipeline(tmp_path)
    df = _make_imbalanced_df()
    pipeline.balance_classes(data=df)
    mock_save.assert_called_once()


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/balanced.parquet")
def test_result_contains_required_keys(mock_save, tmp_path):
    pipeline = _make_pipeline(tmp_path)
    df = _make_imbalanced_df()
    result = pipeline.balance_classes(data=df)
    for key in ("success", "data", "rows_before", "rows_after",
                 "minority_share_before", "minority_share_after",
                 "resampling_method", "skipped", "dropped_non_numeric_cols"):
        assert key in result, f"Missing key: {key}"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_balance_classes.py -v -k "not state_machine and not constant and not folder"
```

Expected: 8 FAILs — `AttributeError: 'TrainingPipeline' object has no attribute 'balance_classes'`.

- [ ] **Step 3: Add imports to `src/training_pipeline.py`**

In the existing `from imblearn.over_sampling import BorderlineSMOTE` line (line 21), add `SMOTETomek` on the next line:

```python
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTETomek
```

In the existing `from config.const_training import (...)` block (lines 80–108), add these four names to the import list:

```python
    MERGED_BALANCED_OUTPUT_FOLDER,
    RESAMPLING_METHOD,
    IMBALANCE_THRESHOLD,
    SMOTE_RANDOM_STATE,
```

- [ ] **Step 4: Add `balance_classes()` method to `TrainingPipeline`**

Insert the method immediately after `filter_delay_outliers` (after line 979, before `def merge_data_files`):

```python
def balance_classes(self, data=None):
    """
    Apply SMOTE-Tomek resampling to address class imbalance before train/test split.

    Derives the binary class label on-the-fly from differenceInMinutes >
    TRAIN_DELAY_MINUTES. Only numeric columns are passed to SMOTE; non-numeric
    columns are dropped from the output and listed in dropped_non_numeric_cols.
    Skips resampling if the minority share already meets IMBALANCE_THRESHOLD.

    Parameters
    ----------
    data : pd.DataFrame or None
        Filtered dataframe from filter_delay_outliers. Required.

    Returns
    -------
    dict
        {
          "success": bool,
          "data": pd.DataFrame | None,
          "rows_before": int,
          "rows_after": int,
          "minority_share_before": float | None,
          "minority_share_after": float | None,
          "resampling_method": str,
          "skipped": bool,
          "dropped_non_numeric_cols": list[str],
        }
    """
    if data is None:
        print("    balance_classes: data is None — skipping")
        return {"success": False, "error": "data is None", "data": None}

    target_col = "differenceInMinutes"

    if target_col not in data.columns:
        print(f"    balance_classes: '{target_col}' not found — returning data unchanged")
        return {
            "success": True,
            "data": data,
            "rows_before": len(data),
            "rows_after": len(data),
            "minority_share_before": None,
            "minority_share_after": None,
            "resampling_method": "NONE",
            "skipped": True,
            "dropped_non_numeric_cols": [],
        }

    df = data.copy()
    rows_before = len(df)

    y = (df[target_col] > TRAIN_DELAY_MINUTES).astype(int)
    class_counts = y.value_counts()
    total = len(y)
    minority_count = int(class_counts.min())
    minority_share = minority_count / total * 100

    print(f"\n    balance_classes: Class balance before resampling:")
    print(f"      Punctual (≤ {TRAIN_DELAY_MINUTES} min): {int(class_counts.get(0, 0)):,} ({int(class_counts.get(0, 0)) / total * 100:.1f}%)")
    print(f"      Delayed  (> {TRAIN_DELAY_MINUTES} min): {int(class_counts.get(1, 0)):,} ({int(class_counts.get(1, 0)) / total * 100:.1f}%)")
    print(f"      Minority share: {minority_share:.1f}% (threshold: {IMBALANCE_THRESHOLD}%)")

    output_folder = os.path.join(self.project_root, MERGED_BALANCED_OUTPUT_FOLDER)

    if minority_share >= IMBALANCE_THRESHOLD:
        print(f"    balance_classes: Balance acceptable — skipping resampling")
        try:
            saved_path = save_dataframe_to_parquet(
                folder_path=output_folder,
                month_id="balanced",
                df=df,
                file_prefix="merged_data",
            )
            print(f"      ✓ Saved (unchanged) data to: {saved_path}")
        except Exception as save_error:
            print(f"      ⚠️  Warning: Failed to save data: {save_error}")
        return {
            "success": True,
            "data": df,
            "rows_before": rows_before,
            "rows_after": rows_before,
            "minority_share_before": minority_share,
            "minority_share_after": minority_share,
            "resampling_method": "NONE",
            "skipped": True,
            "dropped_non_numeric_cols": [],
        }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if non_numeric_cols:
        print(f"    balance_classes: Dropping {len(non_numeric_cols)} non-numeric column(s): {non_numeric_cols}")

    X = df[numeric_cols].copy()

    if RESAMPLING_METHOD == "SMOTE_TOMEK":
        print(f"    balance_classes: Applying SMOTETomek (random_state={SMOTE_RANDOM_STATE})...")
        resampler = SMOTETomek(random_state=SMOTE_RANDOM_STATE)
        X_res, y_res = resampler.fit_resample(X, y)
        used_method = "SMOTE_TOMEK"
    else:
        print(f"    balance_classes: RESAMPLING_METHOD='{RESAMPLING_METHOD}' not handled — returning data unchanged")
        X_res, y_res = X.values, y.values
        used_method = "NONE"

    df_balanced = pd.DataFrame(X_res, columns=numeric_cols)
    rows_after = len(df_balanced)

    y_after = pd.Series(y_res)
    counts_after = y_after.value_counts()
    minority_share_after = int(counts_after.min()) / len(y_after) * 100

    print(f"\n    balance_classes: Class balance after resampling:")
    print(f"      Punctual (≤ {TRAIN_DELAY_MINUTES} min): {int(counts_after.get(0, 0)):,} ({int(counts_after.get(0, 0)) / len(y_after) * 100:.1f}%)")
    print(f"      Delayed  (> {TRAIN_DELAY_MINUTES} min): {int(counts_after.get(1, 0)):,} ({int(counts_after.get(1, 0)) / len(y_after) * 100:.1f}%)")
    print(f"      Rows: {rows_before:,} → {rows_after:,}")

    try:
        saved_path = save_dataframe_to_parquet(
            folder_path=output_folder,
            month_id="balanced",
            df=df_balanced,
            file_prefix="merged_data",
        )
        print(f"      ✓ Saved balanced data to: {saved_path}")
    except Exception as save_error:
        print(f"      ⚠️  Warning: Failed to save balanced data: {save_error}")
        print("      Continuing with in-memory balanced data.")

    return {
        "success": True,
        "data": df_balanced,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "minority_share_before": minority_share,
        "minority_share_after": minority_share_after,
        "resampling_method": used_method,
        "skipped": False,
        "dropped_non_numeric_cols": non_numeric_cols,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_balance_classes.py -v -k "not state_machine and not constant and not folder"
```

Expected: 8 PASSes.

- [ ] **Step 6: Run full test suite to check for regressions**

```
pytest tests/ -v
```

Expected: all previously-passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/training_pipeline.py tests/test_balance_classes.py
git commit -m "feat: implement balance_classes method with SMOTE-Tomek resampling"
```

---

## Task 3: Wire into state machine and route `split_dataset`

**Files:**
- Modify: `src/training_pipeline.py` — `execute_training_pipeline_steps` and `split_dataset`
- Test: `tests/test_balance_classes.py`

**Interfaces:**
- Consumes: `balance_classes(data)` from Task 2; `MERGED_BALANCED_OUTPUT_FOLDER` from Task 1
- Produces: `split_dataset(csv_files=None, data_dir=None, ...)` — callers pass `data_dir` to override the default read folder

- [ ] **Step 1: Write the failing state machine tests**

Append to `tests/test_balance_classes.py` (imports `pd`, `np`, `os`, `patch`, `TrainingPipeline`, and `_make_pipeline` are already in the file from Task 2 — do not duplicate them):

```python
_MERGE_SUCCESS = {
    "success": True,
    "data": pd.DataFrame({
        "differenceInMinutes": np.concatenate([
            np.random.default_rng(0).uniform(-4, 5, 300),
            np.random.default_rng(0).uniform(6, 60, 100),
        ]),
        "feature_a": np.arange(400, dtype=float),
    }),
    "processed_files": 1,
    "total_rows": 400,
    "total_columns": 2,
}

_BALANCE_SUCCESS = {
    "success": True,
    "data": _MERGE_SUCCESS["data"],
    "rows_before": 400,
    "rows_after": 420,
    "minority_share_before": 25.0,
    "minority_share_after": 48.0,
    "resampling_method": "SMOTE_TOMEK",
    "skipped": False,
    "dropped_non_numeric_cols": [],
}


def _base_state_machine(**overrides):
    sm = {
        "merge_data_files": True,
        "filter_delay_outliers": False,
        "balance_classes": True,
        "select_training_cols": False,
        "split_dataset": False,
        "scale_weather_features": False,
        "numeric_correlation_analysis": False,
        "data_distribution_analysis": False,
        "target_feature_analysis": False,
        "train_xgboost_with_randomized_search_cv": False,
    }
    sm.update(overrides)
    return sm


@patch.object(TrainingPipeline, "balance_classes")
@patch.object(TrainingPipeline, "merge_data_files", return_value=_MERGE_SUCCESS)
def test_state_machine_calls_balance_classes_when_enabled(mock_merge, mock_balance, tmp_path):
    pipeline = _make_pipeline(tmp_path)
    mock_balance.return_value = _BALANCE_SUCCESS

    result = pipeline.execute_training_pipeline_steps([], state_machine=_base_state_machine())

    mock_balance.assert_called_once()
    _, kwargs = mock_balance.call_args
    assert kwargs.get("data") is not None
    assert "balance_classes" in result.get("steps_executed", [])


@patch.object(TrainingPipeline, "balance_classes")
@patch.object(TrainingPipeline, "merge_data_files", return_value=_MERGE_SUCCESS)
def test_state_machine_skips_balance_classes_when_disabled(mock_merge, mock_balance, tmp_path):
    pipeline = _make_pipeline(tmp_path)

    pipeline.execute_training_pipeline_steps(
        [], state_machine=_base_state_machine(balance_classes=False)
    )

    mock_balance.assert_not_called()


@patch.object(TrainingPipeline, "split_dataset")
@patch.object(TrainingPipeline, "balance_classes", return_value=_BALANCE_SUCCESS)
@patch.object(TrainingPipeline, "merge_data_files", return_value=_MERGE_SUCCESS)
def test_split_dataset_receives_balanced_folder_when_balance_enabled(mock_merge, mock_balance, mock_split, tmp_path):
    from config.const_training import MERGED_BALANCED_OUTPUT_FOLDER
    pipeline = _make_pipeline(tmp_path)
    mock_split.return_value = {"success": True, "processed_files": 1, "total_train_rows": 336, "total_test_rows": 84}

    pipeline.execute_training_pipeline_steps(
        [], state_machine=_base_state_machine(split_dataset=True)
    )

    mock_split.assert_called_once()
    _, kwargs = mock_split.call_args
    assert kwargs.get("data_dir") is not None
    assert MERGED_BALANCED_OUTPUT_FOLDER in kwargs["data_dir"]


@patch.object(TrainingPipeline, "split_dataset")
@patch.object(TrainingPipeline, "merge_data_files", return_value=_MERGE_SUCCESS)
def test_split_dataset_receives_default_folder_when_balance_disabled(mock_merge, mock_split, tmp_path):
    from config.const_training import MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
    pipeline = _make_pipeline(tmp_path)
    mock_split.return_value = {"success": True, "processed_files": 1, "total_train_rows": 320, "total_test_rows": 80}

    pipeline.execute_training_pipeline_steps(
        [], state_machine=_base_state_machine(balance_classes=False, split_dataset=True)
    )

    mock_split.assert_called_once()
    _, kwargs = mock_split.call_args
    data_dir = kwargs.get("data_dir", "")
    assert data_dir is None or MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER in str(data_dir)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_balance_classes.py -v -k "state_machine"
```

Expected: 4 FAILs — `balance_classes` not called by state machine, `split_dataset` doesn't accept `data_dir`.

- [ ] **Step 3: Add `balance_classes` block to `execute_training_pipeline_steps`**

In `src/training_pipeline.py`, locate the `filter_delay_outliers` block (ends around line 220 with `print(f"    ⊝ filter_delay_outliers (disabled)")`). Immediately after it, insert:

```python
        if state_machine.get("balance_classes", False):
            if result["data"] is not None:
                try:
                    print(f"    → balance_classes")
                    balance_result = self.balance_classes(data=result["data"])

                    if balance_result and balance_result.get("success", False):
                        result["data"] = balance_result.get("data")
                        result["steps_executed"].append("balance_classes")
                        result["success"] = True
                        if balance_result.get("skipped", False):
                            print(f"      ✓ Skipped — minority share already meets threshold")
                        else:
                            print(f"      ✓ Rows before: {balance_result.get('rows_before', 0):,}")
                            print(f"      ✓ Rows after:  {balance_result.get('rows_after', 0):,}")
                            print(f"      ✓ Method: {balance_result.get('resampling_method', 'N/A')}")
                    else:
                        error_msg = balance_result.get("error", "balance_classes returned unsuccessful result")
                        result["errors"].append(error_msg)
                        print(f"      ✗ Failed - {error_msg}")
                        return result

                except Exception as e:
                    result["errors"].append(f"balance_classes failed: {str(e)}")
                    print(f"      ✗ Failed - {str(e)}")
                    return result
            else:
                print(f"    ⊝ balance_classes (no data available)")
        else:
            print(f"    ⊝ balance_classes (disabled)")
```

- [ ] **Step 4: Update `split_dataset` signature to accept `data_dir`**

In `src/training_pipeline.py`, find the `split_dataset` definition (around line 1447). Change the signature from:

```python
def split_dataset(self, csv_files=None, test_size=TEST_SIZE, random_state=42, stratify_column=None):
```

to:

```python
def split_dataset(self, csv_files=None, data_dir=None, test_size=TEST_SIZE, random_state=42, stratify_column=None):
```

Then find the line inside `split_dataset` that sets `merged_training_ready_dir` from the constant (around line 1475):

```python
merged_training_ready_dir = os.path.join(self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)
```

Replace it with:

```python
merged_training_ready_dir = data_dir if data_dir is not None else os.path.join(self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)
```

- [ ] **Step 5: Add routing before the `split_dataset` call in `execute_training_pipeline_steps`**

Find the `split_dataset` block in `execute_training_pipeline_steps` (around line 313). Replace the existing call line:

```python
                split_result = self.split_dataset(csv_files)
```

with:

```python
                _split_dir = (
                    os.path.join(self.project_root, MERGED_BALANCED_OUTPUT_FOLDER)
                    if state_machine.get("balance_classes", False)
                    else None
                )
                split_result = self.split_dataset(csv_files, data_dir=_split_dir)
```

- [ ] **Step 6: Run state machine tests to verify they pass**

```
pytest tests/test_balance_classes.py -v -k "state_machine"
```

Expected: 4 PASSes.

- [ ] **Step 7: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests pass. Watch specifically for any test that calls `split_dataset` directly — the new `data_dir=None` default means existing callers are unaffected.

- [ ] **Step 8: Commit**

```bash
git add src/training_pipeline.py tests/test_balance_classes.py
git commit -m "feat: wire balance_classes into training pipeline state machine and route split_dataset to 504-merged_balanced"
```
