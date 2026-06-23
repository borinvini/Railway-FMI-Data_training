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

def test_quantile_constant_exact_values():
    from config.const_training import FILTER_LOWER_QUANTILE, FILTER_UPPER_QUANTILE
    assert FILTER_LOWER_QUANTILE == 0.01
    assert FILTER_UPPER_QUANTILE == 0.99


import numpy as np
import pandas as pd
import os
from unittest.mock import MagicMock, patch

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


def test_filter_missing_column_returns_data_unchanged(tmp_path):
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


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/outlier_filtered.parquet")
def test_filter_empty_dataframe_does_not_crash(mock_save, tmp_path):
    """Empty DataFrame with the target column present must not crash (ZeroDivisionError guard)."""
    pipeline = _make_pipeline(tmp_path)
    df = pd.DataFrame({"differenceInMinutes": pd.Series([], dtype=float), "feature_a": pd.Series([], dtype=int)})

    result = pipeline.filter_delay_outliers(data=df)

    assert result["success"] is True
    assert result["rows_before"] == 0
    assert result["rows_removed_lower"] == 0
    assert result["rows_removed_upper"] == 0


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
