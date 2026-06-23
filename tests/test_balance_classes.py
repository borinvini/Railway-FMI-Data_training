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
def test_nan_rows_are_dropped_before_resampling(mock_save, tmp_path):
    pipeline = _make_pipeline(tmp_path)
    df = _make_imbalanced_df()
    df.loc[df.index[:10], "feature_a"] = np.nan  # inject NaN into 10 rows
    result = pipeline.balance_classes(data=df)
    assert result["success"] is True
    assert result["skipped"] is False


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


@patch.object(TrainingPipeline, "split_dataset")
@patch.object(TrainingPipeline, "merge_data_files", return_value=_MERGE_SUCCESS)
def test_split_dataset_receives_outlier_filtered_folder_when_filter_enabled_and_balance_and_select_disabled(mock_merge, mock_split, tmp_path):
    from config.const_training import MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
    pipeline = _make_pipeline(tmp_path)
    mock_split.return_value = {"success": True, "processed_files": 1, "total_train_rows": 320, "total_test_rows": 80}

    pipeline.execute_training_pipeline_steps(
        [],
        state_machine=_base_state_machine(
            balance_classes=False,
            filter_delay_outliers=True,
            select_training_cols=False,
            split_dataset=True,
        ),
    )

    mock_split.assert_called_once()
    _, kwargs = mock_split.call_args
    assert kwargs.get("data_dir") is not None
    assert MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER in kwargs["data_dir"]
