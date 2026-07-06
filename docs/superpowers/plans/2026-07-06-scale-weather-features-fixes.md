# scale_weather_features Routing & Skewed-Feature Scaling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `scale_weather_features` so it reads from wherever the real train/test data
currently lives (instead of a hardcoded folder that never contains it), make the
training-stage routing prefer scaled output over balanced output when both are enabled,
and add a `log1p` pre-transform for zero-inflated weather features so scaling is
meaningful ahead of any future scale-sensitive model.

**Architecture:** All changes are in `src/training_pipeline.py` (one method +
two call-site routing blocks) and `config/const_preprocessing.py` (one new constant).
No new files, no schema changes to on-disk parquet layout — only the `weather_scaler.joblib`
artifact's internal shape changes (bare scaler → dict).

**Tech Stack:** Python, pandas, scikit-learn (`RobustScaler`), joblib, pytest.

## Global Constraints

- No back-compat shim needed for `weather_scaler.joblib`'s format — verified no code
  outside `scale_weather_features` itself loads it.
- Wind direction stays on plain `RobustScaler` — no sin/cos conversion in this work.
- Boolean (`trainStopping`, `commercialStop`) and cyclical sin/cos temporal features
  are already excluded from scaling — do not touch that behavior.
- Tree-based models (XGBoost, LightGBM) are scale-invariant — this work will not change
  any existing training metrics; it only fixes correctness/plumbing.
- Reference spec: `docs/superpowers/specs/2026-07-06-scale-weather-features-fixes-design.md`

---

### Task 1: Fix `scale_weather_features` input-directory routing

**Files:**
- Modify: `src/training_pipeline.py:389-413` (call site inside `execute_training_pipeline_steps`)
- Modify: `src/training_pipeline.py:2039` (method signature) and `:2069-2078` (source dir resolution)
- Modify: `src/training_pipeline.py:2205` and `:2307` (summary text "Source directory" lines)
- Test: `tests/test_scale_weather_features_routing.py` (new)

**Interfaces:**
- Consumes: existing constants already imported in `src/training_pipeline.py`:
  `MERGED_BALANCED_OUTPUT_FOLDER`, `SPLIT_DATASET_OUTPUT_FOLDER`,
  `MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER`, `MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER`.
- Produces: `TrainingPipeline.scale_weather_features(self, csv_files=None, data_dir=None)`
  — new `data_dir` keyword parameter. When provided, it is used verbatim as the source
  directory to glob `merged_data_*_train.parquet` / `_test.parquet` from. When `None`,
  falls back to the old default (`MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER`). Later
  tasks (2, 3) call this method and rely on this parameter existing.

- [ ] **Step 1: Write the failing routing tests**

Create `tests/test_scale_weather_features_routing.py`:

```python
import os
from unittest.mock import patch

from src.training_pipeline import TrainingPipeline
from config.const_training import (
    MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER,
    MERGED_BALANCED_OUTPUT_FOLDER,
    SPLIT_DATASET_OUTPUT_FOLDER,
)


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


_SCALE_SUCCESS = {
    "success": True,
    "processed_files": 1,
    "train_rows": 10,
    "test_rows": 5,
    "weather_features_scaled": [],
}


def _make_state_machine(**overrides):
    sm = {
        "merge_data_files": False,
        "filter_delay_outliers": False,
        "select_training_cols": False,
        "split_dataset": False,
        "balance_classes": False,
        "scale_weather_features": True,
        "numeric_correlation_analysis": False,
        "data_distribution_analysis": False,
        "target_feature_analysis": False,
        "train_xgboost_with_randomized_search_cv": False,
    }
    sm.update(overrides)
    return sm


@patch.object(TrainingPipeline, "scale_weather_features")
def test_scale_reads_from_selected_dir_by_default(mock_scale, tmp_path):
    """No split/balance/filter enabled -> scale reads from 502-select_training_cols."""
    pipeline = _make_pipeline(tmp_path)
    mock_scale.return_value = _SCALE_SUCCESS

    pipeline.execute_training_pipeline_steps([], state_machine=_make_state_machine())

    mock_scale.assert_called_once()
    _, kwargs = mock_scale.call_args
    expected = os.path.join(str(tmp_path), MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )


@patch.object(TrainingPipeline, "split_dataset")
@patch.object(TrainingPipeline, "scale_weather_features")
def test_scale_reads_from_split_dir_when_split_enabled(mock_scale, mock_split, tmp_path):
    """split_dataset=True -> scale reads from 503-split_dataset."""
    pipeline = _make_pipeline(tmp_path)
    mock_scale.return_value = _SCALE_SUCCESS
    mock_split.return_value = {
        "success": True, "processed_files": 1,
        "total_train_rows": 1, "total_test_rows": 1,
    }

    pipeline.execute_training_pipeline_steps(
        [], state_machine=_make_state_machine(split_dataset=True)
    )

    mock_scale.assert_called_once()
    _, kwargs = mock_scale.call_args
    expected = os.path.join(str(tmp_path), SPLIT_DATASET_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )


@patch.object(TrainingPipeline, "balance_classes")
@patch.object(TrainingPipeline, "split_dataset")
@patch.object(TrainingPipeline, "scale_weather_features")
def test_scale_reads_from_balanced_dir_when_balance_enabled(mock_scale, mock_split, mock_balance, tmp_path):
    """split_dataset=True and balance_classes=True -> scale reads from 504-balance_classes."""
    pipeline = _make_pipeline(tmp_path)
    mock_scale.return_value = _SCALE_SUCCESS
    mock_split.return_value = {
        "success": True, "processed_files": 1,
        "total_train_rows": 1, "total_test_rows": 1,
    }
    mock_balance.return_value = {
        "success": True, "rows_before": 10, "rows_after": 12,
        "resampling_method": "SMOTE_TOMEK", "skipped": False,
    }

    pipeline.execute_training_pipeline_steps(
        [], state_machine=_make_state_machine(split_dataset=True, balance_classes=True)
    )

    mock_scale.assert_called_once()
    _, kwargs = mock_scale.call_args
    expected = os.path.join(str(tmp_path), MERGED_BALANCED_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scale_weather_features_routing.py -v`
Expected: all three FAIL — `scale_weather_features` is called with no `data_dir` kwarg
today (`kwargs.get("data_dir")` is `None`, not the expected path).

- [ ] **Step 3: Update the call site in `execute_training_pipeline_steps`**

In `src/training_pipeline.py`, replace the `scale_weather_features` block (currently
lines 389-413):

```python
        if state_machine.get("scale_weather_features", False):
            try:
                print(f"    → scale_weather_features")
                scaling_result = self.scale_weather_features(csv_files)
```

with:

```python
        if state_machine.get("scale_weather_features", False):
            try:
                print(f"    → scale_weather_features")
                _use_filtered_for_scale = (
                    state_machine.get("filter_delay_outliers", False)
                    and not state_machine.get("select_training_cols", False)
                )
                _scale_data_folder = (
                    MERGED_BALANCED_OUTPUT_FOLDER
                    if state_machine.get("balance_classes", False)
                    else SPLIT_DATASET_OUTPUT_FOLDER
                    if state_machine.get("split_dataset", False)
                    else MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
                    if _use_filtered_for_scale
                    else MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
                )
                scaling_result = self.scale_weather_features(
                    csv_files,
                    data_dir=os.path.join(self.project_root, _scale_data_folder),
                )
```

(The rest of the block — the `if scaling_result and scaling_result.get(...)` handling —
is unchanged.)

- [ ] **Step 4: Update `scale_weather_features`'s signature and source-directory resolution**

In `src/training_pipeline.py`, change the method signature (currently line 2039):

```python
    def scale_weather_features(self, csv_files=None):
```

to:

```python
    def scale_weather_features(self, csv_files=None, data_dir=None):
```

Then replace the source-directory resolution (currently lines 2069-2078):

```python
            # Create output directory
            scaled_training_ready_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
            os.makedirs(scaled_training_ready_dir, exist_ok=True)
            
            merged_selected_training_ready_dir = os.path.join(self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)
            train_pattern = os.path.join(merged_selected_training_ready_dir, "merged_data_*_train.parquet")
            test_pattern = os.path.join(merged_selected_training_ready_dir, "merged_data_*_test.parquet")
```

with:

```python
            # Create output directory
            scaled_training_ready_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
            os.makedirs(scaled_training_ready_dir, exist_ok=True)
            
            merged_selected_training_ready_dir = (
                data_dir if data_dir is not None
                else os.path.join(self.project_root, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)
            )
            train_pattern = os.path.join(merged_selected_training_ready_dir, "merged_data_*_train.parquet")
            test_pattern = os.path.join(merged_selected_training_ready_dir, "merged_data_*_test.parquet")
```

- [ ] **Step 5: Fix the two "Source directory" summary lines to reflect the real source**

In the "no weather features found" branch (currently line 2205):

```python
                    f.write(f"Source directory: {MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER}\n")
```

replace with:

```python
                    f.write(f"Source directory: {merged_selected_training_ready_dir}\n")
```

In the normal scaling branch (currently line 2307):

```python
                f.write(f"Source directory: {MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER}\n")
```

replace with:

```python
                f.write(f"Source directory: {merged_selected_training_ready_dir}\n")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_scale_weather_features_routing.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Run the full existing test suite to check for regressions**

Run: `pytest tests/test_xgboost_data_dir_routing.py tests/test_balance_classes.py -v`
Expected: PASS (no regressions — these tests mock `scale_weather_features` entirely or
don't call it, so the signature change doesn't affect them)

- [ ] **Step 8: Commit**

```bash
git add src/training_pipeline.py tests/test_scale_weather_features_routing.py
git commit -m "fix: route scale_weather_features to the actual train/test split location"
```

---

### Task 2: Prefer scaled output over balanced output in training-stage routing

**Files:**
- Modify: `src/training_pipeline.py:802-823` (`train_xgboost_with_randomized_search_cv` call site)
- Modify: `src/training_pipeline.py:857-878` (`train_lightgbm_with_randomized_search_cv` call site)
- Modify: `tests/test_xgboost_data_dir_routing.py` (add one test)
- Test: `tests/test_lightgbm_data_dir_routing.py` (new)

**Interfaces:**
- Consumes: `TrainingPipeline.scale_weather_features(self, csv_files=None, data_dir=None)`
  from Task 1 (only via state-machine flag check, not a direct call).
- Produces: no new interfaces — this task only reorders which folder constant is chosen
  inside the existing `_data_folder` ternary chains.

- [ ] **Step 1: Write the failing routing-priority test for xgboost**

Append to `tests/test_xgboost_data_dir_routing.py`:

```python
@patch.object(TrainingPipeline, "train_xgboost_with_randomized_search_cv")
@patch.object(TrainingPipeline, "scale_weather_features")
@patch.object(TrainingPipeline, "balance_classes")
def test_xgboost_uses_scaled_dir_when_scale_and_balance_both_enabled(
    mock_balance, mock_scale, mock_xgb, tmp_path
):
    """When both balance_classes and scale_weather_features are enabled, XGBoost
    should receive the 505 (scaled) directory, not the 504 (balanced) one."""
    pipeline = _make_pipeline(tmp_path)
    mock_xgb.return_value = _XGBOOST_SUCCESS
    mock_balance.return_value = {
        "success": True, "rows_before": 400, "rows_after": 450,
        "minority_share_before": 25.0, "minority_share_after": 45.0,
        "resampling_method": "SMOTE_TOMEK", "skipped": False,
        "dropped_non_numeric_cols": [], "dropped_counterpart_col": "differenceInMinutes",
        "train_output_path": "/fake/train.parquet", "test_output_path": "/fake/test.parquet",
    }
    mock_scale.return_value = {
        "success": True, "processed_files": 1,
        "total_train_rows": 450, "total_test_rows": 100,
        "weather_features_scaled": ["Air temperature"],
    }

    sm = _make_state_machine(scale=True)
    sm["balance_classes"] = True

    pipeline.execute_training_pipeline_steps([], state_machine=sm)

    mock_xgb.assert_called_once()
    _, kwargs = mock_xgb.call_args
    expected = os.path.join(str(tmp_path), MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )
```

- [ ] **Step 2: Write the failing routing tests for lightgbm**

Create `tests/test_lightgbm_data_dir_routing.py`:

```python
import os
from unittest.mock import patch

from src.training_pipeline import TrainingPipeline
from config.const_training import (
    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER,
    MERGED_BALANCED_OUTPUT_FOLDER,
)


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


_LIGHTGBM_SUCCESS = {
    "success": True,
    "models_trained": 1,
    "problem_type": "classification",
    "target_feature": "trainDelayed",
    "cv_score": 0.8,
    "test_f1": 0.75,
    "output_directory": "/fake/output",
}


def _make_state_machine(scale: bool, balance: bool) -> dict:
    return {
        "merge_data_files": False,
        "select_training_cols": False,
        "split_dataset": False,
        "balance_classes": balance,
        "scale_weather_features": scale,
        "numeric_correlation_analysis": False,
        "data_distribution_analysis": False,
        "target_feature_analysis": False,
        "train_lightgbm_with_randomized_search_cv": True,
    }


@patch.object(TrainingPipeline, "train_lightgbm_with_randomized_search_cv")
@patch.object(TrainingPipeline, "scale_weather_features")
def test_lightgbm_uses_scaled_dir_when_scale_enabled(mock_scale, mock_lgb, tmp_path):
    """When scale_weather_features=True, LightGBM should receive the 505 (scaled) directory."""
    pipeline = _make_pipeline(tmp_path)
    mock_lgb.return_value = _LIGHTGBM_SUCCESS
    mock_scale.return_value = {
        "success": True, "processed_files": 1,
        "total_train_rows": 1, "total_test_rows": 1,
        "weather_features_scaled": [],
    }

    pipeline.execute_training_pipeline_steps(
        [], state_machine=_make_state_machine(scale=True, balance=False)
    )

    mock_lgb.assert_called_once()
    _, kwargs = mock_lgb.call_args
    expected = os.path.join(str(tmp_path), MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )


@patch.object(TrainingPipeline, "train_lightgbm_with_randomized_search_cv")
@patch.object(TrainingPipeline, "scale_weather_features")
@patch.object(TrainingPipeline, "balance_classes")
def test_lightgbm_uses_scaled_dir_when_scale_and_balance_both_enabled(
    mock_balance, mock_scale, mock_lgb, tmp_path
):
    """When both balance_classes and scale_weather_features are enabled, LightGBM
    should receive the 505 (scaled) directory, not the 504 (balanced) one."""
    pipeline = _make_pipeline(tmp_path)
    mock_lgb.return_value = _LIGHTGBM_SUCCESS
    mock_balance.return_value = {
        "success": True, "rows_before": 400, "rows_after": 450,
        "minority_share_before": 25.0, "minority_share_after": 45.0,
        "resampling_method": "SMOTE_TOMEK", "skipped": False,
        "dropped_non_numeric_cols": [], "dropped_counterpart_col": "differenceInMinutes",
        "train_output_path": "/fake/train.parquet", "test_output_path": "/fake/test.parquet",
    }
    mock_scale.return_value = {
        "success": True, "processed_files": 1,
        "total_train_rows": 450, "total_test_rows": 100,
        "weather_features_scaled": ["Air temperature"],
    }

    pipeline.execute_training_pipeline_steps(
        [], state_machine=_make_state_machine(scale=True, balance=True)
    )

    mock_lgb.assert_called_once()
    _, kwargs = mock_lgb.call_args
    expected = os.path.join(str(tmp_path), MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )


@patch.object(TrainingPipeline, "train_lightgbm_with_randomized_search_cv")
@patch.object(TrainingPipeline, "balance_classes")
def test_lightgbm_uses_balanced_dir_when_balance_enabled_and_scale_disabled(mock_balance, mock_lgb, tmp_path):
    """When balance_classes=True and scale_weather_features=False, LightGBM should
    receive the 504 (balanced) directory."""
    pipeline = _make_pipeline(tmp_path)
    mock_lgb.return_value = _LIGHTGBM_SUCCESS
    mock_balance.return_value = {
        "success": True, "rows_before": 400, "rows_after": 450,
        "minority_share_before": 25.0, "minority_share_after": 45.0,
        "resampling_method": "SMOTE_TOMEK", "skipped": False,
        "dropped_non_numeric_cols": [], "dropped_counterpart_col": "differenceInMinutes",
        "train_output_path": "/fake/train.parquet", "test_output_path": "/fake/test.parquet",
    }

    pipeline.execute_training_pipeline_steps(
        [], state_machine=_make_state_machine(scale=False, balance=True)
    )

    mock_lgb.assert_called_once()
    _, kwargs = mock_lgb.call_args
    expected = os.path.join(str(tmp_path), MERGED_BALANCED_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_xgboost_data_dir_routing.py::test_xgboost_uses_scaled_dir_when_scale_and_balance_both_enabled tests/test_lightgbm_data_dir_routing.py -v`
Expected: `test_xgboost_uses_scaled_dir_when_scale_and_balance_both_enabled` and
`test_lightgbm_uses_scaled_dir_when_scale_and_balance_both_enabled` FAIL (both currently
get the balanced dir, not the scaled dir). The other two lightgbm tests PASS already
(they don't hit the bug).

- [ ] **Step 4: Fix the xgboost call-site priority order**

In `src/training_pipeline.py`, inside the `train_xgboost_with_randomized_search_cv`
block (currently lines 810-820), replace:

```python
                _data_folder = (
                    MERGED_BALANCED_OUTPUT_FOLDER
                    if state_machine.get("balance_classes", False)
                    else SPLIT_DATASET_OUTPUT_FOLDER
                    if state_machine.get("split_dataset", False)
                    else MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
                    if _use_scaled
                    else MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
                    if _use_filtered
                    else MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
                )
                xgboost_result = self.train_xgboost_with_randomized_search_cv(
                    data_dir=os.path.join(self.project_root, _data_folder)
                )
```

with:

```python
                _data_folder = (
                    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
                    if _use_scaled
                    else MERGED_BALANCED_OUTPUT_FOLDER
                    if state_machine.get("balance_classes", False)
                    else SPLIT_DATASET_OUTPUT_FOLDER
                    if state_machine.get("split_dataset", False)
                    else MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
                    if _use_filtered
                    else MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
                )
                xgboost_result = self.train_xgboost_with_randomized_search_cv(
                    data_dir=os.path.join(self.project_root, _data_folder)
                )
```

- [ ] **Step 5: Fix the lightgbm call-site priority order**

In the same file, inside the `train_lightgbm_with_randomized_search_cv` block
(currently lines 865-875), replace:

```python
                _data_folder = (
                    MERGED_BALANCED_OUTPUT_FOLDER
                    if state_machine.get("balance_classes", False)
                    else SPLIT_DATASET_OUTPUT_FOLDER
                    if state_machine.get("split_dataset", False)
                    else MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
                    if _use_scaled
                    else MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
                    if _use_filtered
                    else MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
                )
                lightgbm_result = self.train_lightgbm_with_randomized_search_cv(
                    data_dir=os.path.join(self.project_root, _data_folder)
                )
```

with:

```python
                _data_folder = (
                    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
                    if _use_scaled
                    else MERGED_BALANCED_OUTPUT_FOLDER
                    if state_machine.get("balance_classes", False)
                    else SPLIT_DATASET_OUTPUT_FOLDER
                    if state_machine.get("split_dataset", False)
                    else MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
                    if _use_filtered
                    else MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
                )
                lightgbm_result = self.train_lightgbm_with_randomized_search_cv(
                    data_dir=os.path.join(self.project_root, _data_folder)
                )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_xgboost_data_dir_routing.py tests/test_lightgbm_data_dir_routing.py -v`
Expected: PASS (all tests, including the two new ones)

- [ ] **Step 7: Commit**

```bash
git add src/training_pipeline.py tests/test_xgboost_data_dir_routing.py tests/test_lightgbm_data_dir_routing.py
git commit -m "fix: prefer scaled data directory over balanced directory in training routing"
```

---

### Task 3: log1p pre-transform for zero-inflated weather features

**Files:**
- Modify: `config/const_preprocessing.py` (new `SKEWED_WEATHER_FEATURES` constant, after
  `ALL_WEATHER_FEATURES` at line 146)
- Modify: `src/training_pipeline.py:73` (import), `:2145-2151` (feature partitioning),
  `:2242-2295` (scaling block + result dict), `:2297-2322` (summary text)
- Test: `tests/test_scale_weather_features_skewed.py` (new)

**Interfaces:**
- Consumes: `TrainingPipeline.scale_weather_features(self, csv_files=None, data_dir=None)`
  from Task 1.
- Produces: `scale_weather_features` return dict gains a new key
  `"skewed_features_log1p": list[str]` (columns that had `log1p` applied before
  `RobustScaler`). The `weather_scaler.joblib` artifact changes from a bare
  `RobustScaler` to `{"scaler": RobustScaler, "weather_features": list[str],
  "skewed_features": list[str]}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scale_weather_features_skewed.py`:

```python
import os
import numpy as np
import pandas as pd
import pytest
import joblib

from src.training_pipeline import TrainingPipeline
from config.const_preprocessing import SKEWED_WEATHER_FEATURES
from config.const_training import MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER


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


def _make_zero_inflated_df(n=200, seed=0):
    """~95% zeros with occasional large rain events, mimicking Precipitation amount."""
    rng = np.random.default_rng(seed)
    precip = np.zeros(n)
    event_idx = rng.choice(n, size=max(n // 20, 1), replace=False)
    precip[event_idx] = rng.uniform(5, 50, size=len(event_idx))
    return pd.DataFrame({
        "Precipitation amount": precip,
        "Air temperature": rng.normal(5, 8, n),
    })


def test_skewed_weather_features_constant_exists():
    assert SKEWED_WEATHER_FEATURES == [
        "Precipitation amount", "Precipitation intensity", "Snow depth",
    ]


def test_scale_weather_features_applies_log1p_to_skewed_columns_only(tmp_path):
    pipeline = _make_pipeline(tmp_path)

    source_dir = tmp_path / "input_data"
    source_dir.mkdir(parents=True)
    train_df = _make_zero_inflated_df(seed=1)
    test_df = _make_zero_inflated_df(n=50, seed=2)
    train_df.to_parquet(source_dir / "merged_data_x_train.parquet", index=False)
    test_df.to_parquet(source_dir / "merged_data_x_test.parquet", index=False)

    result = pipeline.scale_weather_features(data_dir=str(source_dir))

    assert result["success"] is True
    assert result["skewed_features_log1p"] == ["Precipitation amount"]

    scaled_dir = os.path.join(str(tmp_path), MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
    artifact = joblib.load(os.path.join(scaled_dir, "weather_scaler.joblib"))
    assert artifact["skewed_features"] == ["Precipitation amount"]
    assert "Air temperature" in artifact["weather_features"]

    scaled_train = pd.read_parquet(os.path.join(scaled_dir, "merged_data_x_train.parquet"))

    # This column is ~95% zero, so RobustScaler's IQR is 0 either way (center_=0,
    # scale_ falls back to 1) -- the scaled value equals the (log1p-transformed or raw)
    # input directly. That makes the log1p pre-transform directly observable: with it,
    # the scaled max equals log1p(raw max); without it, the scaled max would equal the
    # raw max unchanged.
    raw_train_max = train_df["Precipitation amount"].max()
    scaled_max = scaled_train["Precipitation amount"].max()
    assert scaled_max == pytest.approx(np.log1p(raw_train_max), abs=1e-6)
    assert scaled_max < raw_train_max
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scale_weather_features_skewed.py -v`
Expected: FAIL — `test_skewed_weather_features_constant_exists` fails with
`ImportError` (constant doesn't exist yet); the second test fails with `KeyError:
'skewed_features_log1p'`.

- [ ] **Step 3: Add the `SKEWED_WEATHER_FEATURES` constant**

In `config/const_preprocessing.py`, after the `ALL_WEATHER_FEATURES` list (currently
ends at line 146), insert:

```python
SKEWED_WEATHER_FEATURES = [
    'Precipitation amount',
    'Precipitation intensity',
    'Snow depth',
]
```

- [ ] **Step 4: Import the new constant in `training_pipeline.py`**

In `src/training_pipeline.py`, in the `from config.const_preprocessing import (...)`
block (currently lines 69-81), add `SKEWED_WEATHER_FEATURES,` next to
`ALL_WEATHER_FEATURES,`:

```python
from config.const_preprocessing import (
    PREPROCESSED_OUTPUT_FOLDER,
    DATA_FILE_PREFIX_FOR_TRAINING,
    DEFAULT_TARGET_FEATURE,
    ALL_WEATHER_FEATURES,
    SKEWED_WEATHER_FEATURES,
    TRAINING_READY_OUTPUT_FOLDER,
    USE_SIN_COS_APPROACH,
    VALID_TARGET_FEATURES,
    REGRESSION_PROBLEM,
    CLASSIFICATION_PROBLEM,
    TRAIN_DELAY_MINUTES,
    SELECTED_WEATHER_FEATURES,
)
```

- [ ] **Step 5: Partition scaled columns into skewed vs. non-skewed**

In `src/training_pipeline.py`, after the existing weather/window column discovery
(currently lines 2145-2151):

```python
            # Identify weather features that exist in the dataset
            available_weather_features = [col for col in ALL_WEATHER_FEATURES if col in train_df.columns]

            # Also include rolling window columns (12h, 24h, 72h) — same pattern as filter_columns
            window_patterns = ('(12h', '(24h', '(72h')
            available_window_features = [col for col in train_df.columns if any(p in col for p in window_patterns)]
            available_weather_features = available_weather_features + available_window_features
```

add immediately after (keep everything above unchanged):

```python

            # Zero-inflated/right-skewed features (and their rolling-window derivatives,
            # e.g. "Precipitation amount (24h cumulative)") get a log1p pre-transform
            # before RobustScaler so the heavy right tail doesn't dominate the scale.
            skewed_cols = [
                col for col in available_weather_features
                if any(col == base or col.startswith(base) for base in SKEWED_WEATHER_FEATURES)
            ]
```

- [ ] **Step 6: Apply log1p before fitting/transforming, and persist it in the artifact**

Replace the scaling block (currently lines 2242-2268):

```python
            # Create and fit scaler on training data only
            scaler = RobustScaler()

            # Extract weather features for scaling
            train_weather_features = train_df[available_weather_features]
            test_weather_features = test_df[available_weather_features]

            # Fit scaler on training data only
            scaler.fit(train_weather_features)

            # Save the fitted scaler so it can be applied to new data at inference time
            scaler_filename = "weather_scaler.joblib"
            scaler_path = os.path.join(scaled_training_ready_dir, scaler_filename)
            joblib.dump(scaler, scaler_path)
            print(f"    scale_weather_features: ✓ Scaler saved to: {scaler_filename}")

            # Transform both train and test sets using training parameters
            train_weather_scaled = scaler.transform(train_weather_features)
            test_weather_scaled = scaler.transform(test_weather_features)
            
            # Create scaled DataFrames
            train_scaled_df = train_df.copy()
            test_scaled_df = test_df.copy()
            
            # Replace weather feature columns with scaled versions
            train_scaled_df[available_weather_features] = train_weather_scaled
            test_scaled_df[available_weather_features] = test_weather_scaled
```

with:

```python
            # Extract weather features for scaling
            train_weather_features = train_df[available_weather_features].copy()
            test_weather_features = test_df[available_weather_features].copy()

            # Zero-inflated/right-skewed features get a log1p pre-transform (deterministic,
            # not fit on data) before RobustScaler
            if skewed_cols:
                print(f"    scale_weather_features: Applying log1p to {len(skewed_cols)} skewed feature(s): {skewed_cols}")
                train_weather_features[skewed_cols] = np.log1p(train_weather_features[skewed_cols])
                test_weather_features[skewed_cols] = np.log1p(test_weather_features[skewed_cols])

            # Create and fit scaler on training data only
            scaler = RobustScaler()
            scaler.fit(train_weather_features)

            # Save the fitted scaler (plus which columns need log1p) so it can be
            # applied to new data at inference time
            scaler_filename = "weather_scaler.joblib"
            scaler_path = os.path.join(scaled_training_ready_dir, scaler_filename)
            joblib.dump(
                {
                    "scaler": scaler,
                    "weather_features": available_weather_features,
                    "skewed_features": skewed_cols,
                },
                scaler_path,
            )
            print(f"    scale_weather_features: ✓ Scaler saved to: {scaler_filename}")

            # Transform both train and test sets using training parameters
            train_weather_scaled = scaler.transform(train_weather_features)
            test_weather_scaled = scaler.transform(test_weather_features)
            
            # Create scaled DataFrames
            train_scaled_df = train_df.copy()
            test_scaled_df = test_df.copy()
            
            # Replace weather feature columns with scaled versions
            train_scaled_df[available_weather_features] = train_weather_scaled
            test_scaled_df[available_weather_features] = test_weather_scaled
```

- [ ] **Step 7: Add `skewed_features_log1p` to the returned result dict**

Replace the scaling result dict (currently lines 2284-2295):

```python
            # Prepare scaling result
            scaling_result = {
                "original_train_file": train_filename,
                "original_test_file": test_filename,
                "scaled_train_file": scaled_train_filename,
                "scaled_test_file": scaled_test_filename,
                "scaler_file": scaler_filename,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "weather_features_scaled": available_weather_features,
                "scaling_method": "RobustScaler"
            }
```

with:

```python
            # Prepare scaling result
            scaling_result = {
                "original_train_file": train_filename,
                "original_test_file": test_filename,
                "scaled_train_file": scaled_train_filename,
                "scaled_test_file": scaled_test_filename,
                "scaler_file": scaler_filename,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "weather_features_scaled": available_weather_features,
                "skewed_features_log1p": skewed_cols,
                "scaling_method": "RobustScaler"
            }
```

Then in the final return dict of the method (currently lines 2325-2336):

```python
            # Return successful result
            return {
                "success": True,
                "processed_files": 1,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "weather_features_scaled": available_weather_features,
                "scaler_path": scaler_path,
                "scaling_summary": scaling_result,
                "output_directory": scaled_training_ready_dir,
                "message": f"Successfully scaled weather features for single train/test file pair: {train_filename}, {test_filename}",
                "scaling_applied": True
            }
```

replace with:

```python
            # Return successful result
            return {
                "success": True,
                "processed_files": 1,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "weather_features_scaled": available_weather_features,
                "skewed_features_log1p": skewed_cols,
                "scaler_path": scaler_path,
                "scaling_summary": scaling_result,
                "output_directory": scaled_training_ready_dir,
                "message": f"Successfully scaled weather features for single train/test file pair: {train_filename}, {test_filename}",
                "scaling_applied": True
            }
```

- [ ] **Step 8: Add the skewed-features section to `scaling_summary.txt`**

Replace the summary-writing block (currently lines 2297-2322... the tail after the
existing "Weather features scaled" loop):

```python
                f.write("Weather features scaled:\n")
                f.write("-" * 25 + "\n")
                for feature in ALL_WEATHER_FEATURES:
                    status = "✓ Scaled" if feature in available_weather_features else "✗ Not found"
                    f.write(f"  {feature}: {status}\n")
```

with:

```python
                f.write("Weather features scaled:\n")
                f.write("-" * 25 + "\n")
                for feature in ALL_WEATHER_FEATURES:
                    status = "✓ Scaled" if feature in available_weather_features else "✗ Not found"
                    f.write(f"  {feature}: {status}\n")

                f.write("\nSkewed features (log1p applied before RobustScaler):\n")
                f.write("-" * 25 + "\n")
                if skewed_cols:
                    for feature in skewed_cols:
                        f.write(f"  {feature}\n")
                else:
                    f.write("  None\n")
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/test_scale_weather_features_skewed.py -v`
Expected: PASS (2 passed)

- [ ] **Step 10: Run the full scale_weather_features-related test suite to check for regressions**

Run: `pytest tests/test_scale_weather_features_routing.py tests/test_scale_weather_features_skewed.py tests/test_xgboost_data_dir_routing.py tests/test_lightgbm_data_dir_routing.py tests/test_balance_classes.py -v`
Expected: PASS (all tests)

- [ ] **Step 11: Commit**

```bash
git add config/const_preprocessing.py src/training_pipeline.py tests/test_scale_weather_features_skewed.py
git commit -m "feat: apply log1p to zero-inflated weather features before RobustScaler"
```

---

## Final Verification

- [ ] **Run the complete test suite**

Run: `pytest tests/ -v`
Expected: PASS, no regressions anywhere.

- [ ] **Manual sanity check of scaling_summary.txt**

Run the scaling stage against a real preprocessed dataset (or use the test fixtures
above) and read the generated `data/output/505-scale_weather_features/scaling_summary.txt`
to confirm:
- "Source directory" shows the actual resolved input folder (not always `502/`).
- The new "Skewed features (log1p applied before RobustScaler)" section lists
  `Precipitation amount`, `Precipitation intensity`, `Snow depth`, and any of their
  rolling-window derivatives that were present in the dataset.
