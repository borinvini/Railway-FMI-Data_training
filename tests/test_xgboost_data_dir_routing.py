import os
import pytest
from unittest.mock import MagicMock, patch

from src.training_pipeline import TrainingPipeline
from config.const_training import (
    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER,
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


_SUCCESS = {
    "success": True,
    "data": None,
    "processed_files": 0,
    "total_train_rows": 0,
    "total_test_rows": 0,
    "weather_features_scaled": [],
}
_XGBOOST_SUCCESS = {
    "success": True,
    "models_trained": 1,
    "problem_type": "regression",
    "target_feature": "differenceInMinutes",
    "cv_score": -5.0,
    "test_rmse": 14.0,
    "test_r2": 0.31,
    "output_directory": "/fake/output",
}


def _make_state_machine(scale: bool) -> dict:
    return {
        "merge_data_files": False,
        "select_training_cols": False,
        "split_dataset": False,
        "scale_weather_features": scale,
        "numeric_correlation_analysis": False,
        "data_distribution_analysis": False,
        "target_feature_analysis": False,
        "train_xgboost_with_randomized_search_cv": True,
    }


@patch.object(TrainingPipeline, "scale_weather_features", return_value=_SUCCESS)
@patch.object(TrainingPipeline, "train_xgboost_with_randomized_search_cv")
def test_xgboost_uses_scaled_dir_when_scale_enabled(mock_xgb, mock_scale, tmp_path):
    """When scale_weather_features=True, XGBoost should receive the 505 (scaled) directory."""
    pipeline = _make_pipeline(tmp_path)
    mock_xgb.return_value = _XGBOOST_SUCCESS

    pipeline.execute_training_pipeline_steps([], state_machine=_make_state_machine(scale=True))

    mock_xgb.assert_called_once()
    _, kwargs = mock_xgb.call_args
    expected = os.path.join(str(tmp_path), MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )


@patch.object(TrainingPipeline, "train_xgboost_with_randomized_search_cv")
def test_xgboost_uses_selected_dir_when_scale_disabled(mock_xgb, tmp_path):
    """When scale_weather_features=False, XGBoost should receive the 502 (selected) directory."""
    pipeline = _make_pipeline(tmp_path)
    mock_xgb.return_value = _XGBOOST_SUCCESS

    pipeline.execute_training_pipeline_steps([], state_machine=_make_state_machine(scale=False))

    mock_xgb.assert_called_once()
    _, kwargs = mock_xgb.call_args
    expected = os.path.join(str(tmp_path), MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )


@patch.object(TrainingPipeline, "train_xgboost_with_randomized_search_cv")
@patch.object(TrainingPipeline, "balance_classes")
def test_xgboost_uses_balanced_dir_when_balance_enabled_and_scale_disabled(mock_balance, mock_xgb, tmp_path):
    """When balance_classes=True and scale_weather_features=False, XGBoost should receive the 504 (balanced) directory."""
    pipeline = _make_pipeline(tmp_path)
    mock_xgb.return_value = _XGBOOST_SUCCESS
    mock_balance.return_value = {
        "success": True,
        "rows_before": 400,
        "rows_after": 450,
        "minority_share_before": 25.0,
        "minority_share_after": 45.0,
        "resampling_method": "SMOTE_TOMEK",
        "skipped": False,
        "dropped_non_numeric_cols": [],
        "dropped_counterpart_col": "differenceInMinutes",
        "train_output_path": "/fake/train.parquet",
        "test_output_path": "/fake/test.parquet",
    }

    sm = _make_state_machine(scale=False)
    sm["balance_classes"] = True

    pipeline.execute_training_pipeline_steps([], state_machine=sm)

    mock_xgb.assert_called_once()
    _, kwargs = mock_xgb.call_args
    expected = os.path.join(str(tmp_path), MERGED_BALANCED_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )


@patch.object(TrainingPipeline, "train_xgboost_with_randomized_search_cv")
def test_xgboost_uses_outlier_filtered_dir_when_filter_enabled_and_balance_scale_select_disabled(mock_xgb, tmp_path):
    """When filter_delay_outliers=True, balance=False, scale=False, select=False, XGBoost reads from 501."""
    from config.const_training import MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
    pipeline = _make_pipeline(tmp_path)
    mock_xgb.return_value = _XGBOOST_SUCCESS

    sm = _make_state_machine(scale=False)
    sm["filter_delay_outliers"] = True
    sm["select_training_cols"] = False
    sm["balance_classes"] = False

    pipeline.execute_training_pipeline_steps([], state_machine=sm)

    mock_xgb.assert_called_once()
    _, kwargs = mock_xgb.call_args
    expected = os.path.join(str(tmp_path), MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )


@patch.object(TrainingPipeline, "train_xgboost_with_randomized_search_cv")
@patch.object(TrainingPipeline, "split_dataset")
def test_xgboost_reads_from_split_dataset_folder_when_split_enabled(mock_split, mock_xgb, tmp_path):
    """When split_dataset=True, XGBoost should read from 503-split_dataset regardless of other flags."""
    pipeline = _make_pipeline(tmp_path)
    mock_split.return_value = {"success": True, "processed_files": 1, "total_train_rows": 400, "total_test_rows": 100}
    mock_xgb.return_value = _XGBOOST_SUCCESS

    sm = _make_state_machine(scale=False)
    sm["split_dataset"] = True

    pipeline.execute_training_pipeline_steps([], state_machine=sm)

    mock_xgb.assert_called_once()
    _, kwargs = mock_xgb.call_args
    expected = os.path.join(str(tmp_path), SPLIT_DATASET_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )


@patch.object(TrainingPipeline, "split_dataset")
def test_split_dataset_writes_to_503_folder(mock_split, tmp_path):
    """split_dataset state machine block should pass output_dir pointing to 503-split_dataset."""
    pipeline = _make_pipeline(tmp_path)
    mock_split.return_value = {"success": True, "processed_files": 1, "total_train_rows": 400, "total_test_rows": 100}

    sm = _make_state_machine(scale=False)
    sm["split_dataset"] = True

    pipeline.execute_training_pipeline_steps([], state_machine=sm)

    mock_split.assert_called_once()
    _, kwargs = mock_split.call_args
    expected = os.path.join(str(tmp_path), SPLIT_DATASET_OUTPUT_FOLDER)
    assert kwargs.get("output_dir") == expected, (
        f"Expected output_dir={expected!r}, got {kwargs.get('output_dir')!r}"
    )


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
