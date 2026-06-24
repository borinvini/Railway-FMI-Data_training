def test_selected_columns_constant_exists():
    from config.const_training import SELECTED_COLUMNS
    assert isinstance(SELECTED_COLUMNS, list)


import os
import numpy as np
import pandas as pd
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


def test_uses_inmemory_data_when_provided(tmp_path):
    """When data= is given, no disk access occurs — success even with empty filesystem."""
    pipeline = _make_pipeline(tmp_path)
    # No parquet files anywhere on disk; disk access would return an error
    df_in = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})

    with patch("src.training_pipeline.SELECTED_COLUMNS", ["col_a"]):
        result = pipeline.select_training_cols(data=df_in)

    assert result["success"] is True
    assert list(result["data"].columns) == ["col_a"]


def test_disk_discovery_prefers_501_over_500(tmp_path):
    """When data=None, 501/ is checked before 500/."""
    pipeline = _make_pipeline(tmp_path)

    folder_501 = tmp_path / "data/output/501-filter_delay_outliers"
    folder_501.mkdir(parents=True)
    df_501 = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    df_501.to_parquet(folder_501 / "merged_data_filtered.parquet", index=False)

    folder_500 = tmp_path / "data/output/500-merge_data_files"
    folder_500.mkdir(parents=True)
    df_500 = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
    df_500.to_parquet(folder_500 / "merged_data_raw.parquet", index=False)

    with patch("src.training_pipeline.SELECTED_COLUMNS", ["col_a"]):
        result = pipeline.select_training_cols()

    assert result["success"] is True
    # 501/ has 2 rows; 500/ has 3 — confirms 501/ was used
    assert len(result["data"]) == 2


def test_disk_discovery_falls_back_to_500_when_501_absent(tmp_path):
    """When 501/ does not exist, falls back to 500/."""
    pipeline = _make_pipeline(tmp_path)

    folder_500 = tmp_path / "data/output/500-merge_data_files"
    folder_500.mkdir(parents=True)
    df_500 = pd.DataFrame({"col_a": [10, 20, 30], "col_b": [4, 5, 6]})
    df_500.to_parquet(folder_500 / "merged_data_raw.parquet", index=False)

    with patch("src.training_pipeline.SELECTED_COLUMNS", ["col_a"]):
        result = pipeline.select_training_cols()

    assert result["success"] is True
    assert len(result["data"]) == 3


def test_config_bypass_rejects_missing_column(tmp_path):
    """SELECTED_COLUMNS referencing a column not in the DataFrame returns an error."""
    pipeline = _make_pipeline(tmp_path)
    df_in = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})

    with patch("src.training_pipeline.SELECTED_COLUMNS", ["col_a", "nonexistent_col"]):
        result = pipeline.select_training_cols(data=df_in)

    assert result["success"] is False
    assert "nonexistent_col" in result["error"]
