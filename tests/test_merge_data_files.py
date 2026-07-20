import os
from unittest.mock import patch

import pandas as pd

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


def _make_source_dir(tmp_path):
    src = tmp_path / "data/output/101-preprocessed_training_ready"
    src.mkdir(parents=True)
    return src


def test_schema_check_passes_when_all_files_match(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    df1.to_parquet(src / "training_ready_2024_01.parquet", index=False)
    df2.to_parquet(src / "training_ready_2024_02.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is True
    assert result.get("files_merged") == 2


@patch('src.training_pipeline.SCHEMA_MISMATCH_STRATEGY', 'fail')
def test_schema_check_fails_on_missing_column(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6]})          # missing "b"
    df1.to_parquet(src / "training_ready_2024_01.parquet", index=False)
    df2.to_parquet(src / "training_ready_2024_02.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is False
    assert "b" in result["error"]


@patch('src.training_pipeline.SCHEMA_MISMATCH_STRATEGY', 'fail')
def test_schema_check_fails_on_extra_column(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [5, 6], "extra": [7, 8]})   # extra column
    df1.to_parquet(src / "training_ready_2024_01.parquet", index=False)
    df2.to_parquet(src / "training_ready_2024_02.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is False
    assert "extra" in result["error"]


def test_schema_check_single_file_passes(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_parquet(src / "training_ready_2024_01.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is True
    assert result.get("files_merged") == 1


def test_schema_mismatch_constant_exists():
    from config.const_training import SCHEMA_MISMATCH_STRATEGY
    assert SCHEMA_MISMATCH_STRATEGY == ''


@patch('src.training_pipeline.SCHEMA_MISMATCH_STRATEGY', 'intersect')
def test_intersect_strategy_drops_extra_columns(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8], "extra": [9, 10]})
    df1.to_parquet(src / "training_ready_2024_01.parquet", index=False)
    df2.to_parquet(src / "training_ready_2024_02.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is True
    assert "extra" not in result["data"].columns
    # 'year' is stamped onto every row by merge_data_files (parsed from each source
    # file's YYYY_MM name) so split_dataset can carve out an out-of-time hold-out —
    # it is always present/common across files, unlike the schema-mismatch columns.
    assert sorted(result["data"].columns) == ["a", "b", "year"]


@patch('src.training_pipeline.SCHEMA_MISMATCH_STRATEGY', 'intersect')
def test_intersect_strategy_drops_missing_columns(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6]})  # missing "b"
    df1.to_parquet(src / "training_ready_2024_01.parquet", index=False)
    df2.to_parquet(src / "training_ready_2024_02.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is True
    assert "b" not in result["data"].columns
    # 'year' is stamped onto every row by merge_data_files (see comment above) — it
    # survives the intersect strategy just like 'a' does.
    assert list(result["data"].columns) == ["a", "year"]


@patch('src.training_pipeline.SCHEMA_MISMATCH_STRATEGY', 'intersect')
def test_intersect_strategy_reselects_already_loaded_dfs(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    src = _make_source_dir(tmp_path)
    df1 = pd.DataFrame({"a": [1], "b": [2]})
    df2 = pd.DataFrame({"a": [3], "b": [4]})
    df3 = pd.DataFrame({"a": [5], "b": [6], "extra": [7]})
    df1.to_parquet(src / "training_ready_2024_01.parquet", index=False)
    df2.to_parquet(src / "training_ready_2024_02.parquet", index=False)
    df3.to_parquet(src / "training_ready_2024_03.parquet", index=False)

    result = pipeline.merge_data_files([])

    assert result["success"] is True
    assert "extra" not in result["data"].columns
    assert result.get("files_merged") == 3
