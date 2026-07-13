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
