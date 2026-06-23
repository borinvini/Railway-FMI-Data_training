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
