"""The weather-column drop list is decided once over all months, not per month.

handle_missing_values applied the 30% missing rule to each month alone, so a
single bad month dropped a column from that month's file only. merge_data_files
keeps the column intersection (SCHEMA_MISMATCH_STRATEGY='intersect'), so the
column then vanished from all 96 months without a word. The pre-pass reads only
the target-station rows and the weather columns (about 10 s and 3 MB for the 96
months at OL) and aborts when a month disagrees with the pooled decision.
"""
import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.const_preprocessing import FOLDER_HANDLE_MISSING_VALUES
from src.preprocessing_pipeline import PreprocessingPipeline

N = 20


@contextmanager
def _null_logger(*args, **kwargs):
    yield MagicMock()


def _pipeline(tmp_path):
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.project_root = str(tmp_path)
    pipeline.log_dir = str(tmp_path)
    pipeline.current_file_id = "2023_01"
    return pipeline


def _month(path, missing=None, drop_columns=(), other_station_rows=0):
    """Write a month with N OL rows; `missing` maps column -> number of NaN rows."""
    frame = pd.DataFrame({
        "stationShortCode": ["OL"] * N,
        "differenceInMinutes": [1.0] * N,
        "Air temperature": [1.0] * N,
        "Precipitation amount": [0.0] * N,
        "Snow depth": [0.0] * N,
    })
    for col, k in (missing or {}).items():
        frame.loc[:k - 1, col] = np.nan
    if other_station_rows:
        extra = frame.iloc[:other_station_rows].copy()
        extra["stationShortCode"] = "HKI"
        extra[["Air temperature", "Precipitation amount", "Snow depth"]] = np.nan
        frame = pd.concat([frame, extra], ignore_index=True)
    frame = frame.drop(columns=list(drop_columns))
    frame.to_parquet(path, index=False)
    return str(path)


def _months(tmp_path, specs):
    return [_month(tmp_path / f"matched_data_flat_2023_{i + 1:02d}.parquet", **spec)
            for i, spec in enumerate(specs)]


def test_a_column_missing_in_every_month_is_dropped_and_the_rest_kept(tmp_path):
    files = _months(tmp_path, [{"missing": {"Precipitation amount": 18, "Snow depth": 1}}] * 3)
    dropped = _pipeline(tmp_path).decide_weather_columns_to_drop(files)
    assert dropped == ["Precipitation amount"]


def test_one_bad_month_aborts_instead_of_dropping_the_column_everywhere(tmp_path):
    specs = [{}, {"missing": {"Air temperature": 14}}, {}]   # 70% in 2023_02 only
    files = _months(tmp_path, specs)
    with pytest.raises(ValueError) as err:
        _pipeline(tmp_path).decide_weather_columns_to_drop(files)
    assert "Air temperature" in str(err.value)
    assert "2023_02" in str(err.value)


def test_a_column_absent_from_one_month_aborts(tmp_path):
    specs = [{}, {"drop_columns": ["Snow depth"]}, {}]
    files = _months(tmp_path, specs)
    with pytest.raises(ValueError) as err:
        _pipeline(tmp_path).decide_weather_columns_to_drop(files)
    assert "Snow depth" in str(err.value)
    assert "2023_02" in str(err.value)


def test_other_stations_do_not_change_the_decision(tmp_path):
    # Pins the filter_by_target_station=True behaviour: only OL rows are counted.
    # With the stage off every station counts (tests/test_station_filter_off.py).
    files = _months(tmp_path, [{"other_station_rows": 200}] * 2)
    with patch("src.preprocessing_pipeline.PREPROCESSING_STATE_MACHINE",
               {"filter_by_target_station": True}):
        assert _pipeline(tmp_path).decide_weather_columns_to_drop(files) == []


def test_the_decision_is_written_for_audit(tmp_path):
    files = _months(tmp_path, [{"missing": {"Precipitation amount": 18}}] * 2)
    _pipeline(tmp_path).decide_weather_columns_to_drop(files)
    report = tmp_path / FOLDER_HANDLE_MISSING_VALUES / "weather_columns_to_drop.json"
    assert json.loads(report.read_text(encoding="utf-8"))["columns_to_drop"] == ["Precipitation amount"]


@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_stage_12_drops_exactly_the_decided_list(_save, tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline.weather_columns_to_drop = ["Precipitation amount"]
    df = pd.DataFrame({
        "differenceInMinutes": [1.0] * N,
        "month": [1] * N,
        # 90% missing in this month, but not on the decided list: it must stay
        "Air temperature": [np.nan] * 18 + [1.0] * 2,
        "Precipitation amount": [0.0] * N,
        "Snow depth": [0.0] * N,
    })
    with patch.object(pipeline, "get_logger", _null_logger):
        out = pipeline.handle_missing_values(dataframe=df, month_id="2023_01")
    assert "Precipitation amount" not in out.columns
    assert "Air temperature" in out.columns


@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_stage_12_leaves_weather_nan_for_the_post_split_imputer(_save, tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline.weather_columns_to_drop = []
    df = pd.DataFrame({
        "differenceInMinutes": [1.0] * N,
        "month": [1] * N,
        "Air temperature": [np.nan] + [1.0] * (N - 1),
        "Snow depth": [0.0] * N,
    })
    with patch.object(pipeline, "get_logger", _null_logger):
        out = pipeline.handle_missing_values(dataframe=df, month_id="2023_01")
    assert out["Air temperature"].isna().sum() == 1


def test_run_pipeline_aborts_before_any_month_when_the_decision_fails(tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline.execute_preprocessing_pipeline_steps = MagicMock()
    with patch.object(pipeline, "decide_weather_columns_to_drop",
                      side_effect=ValueError("2023_02 disagrees")), \
         patch("src.preprocessing_pipeline.PREPROCESSING_STATE_MACHINE",
               {"handle_missing_values": True}):
        result = pipeline.run_pipeline(["data/input/matched_data_flat_2023_01.parquet"])
    pipeline.execute_preprocessing_pipeline_steps.assert_not_called()
    assert "2023_02 disagrees" in result["error"]
    assert result["successful_preprocessing"] == 0
