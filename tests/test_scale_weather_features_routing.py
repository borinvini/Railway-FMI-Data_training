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
