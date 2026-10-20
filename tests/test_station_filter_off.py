"""Preprocessing must work with filter_by_target_station = False.

With the station filter off a month holds every station (about 507k rows and 380
stations for 2023_01, against 1,385 rows at OL). Two stages silently assumed OL:

* decide_weather_columns_to_drop always counted target-station rows, so stage 12
  (which then sees every station) dropped columns on the strength of a different
  population. It happened to give the same answer on the real data, by chance.
* filter_columns dropped stationShortCode and stationUICCode, so rows from 380
  stations became indistinguishable. stationUICCode (int64, no nulls, 1:1 with the
  short code) is now kept when the filter is off, and only then.

Stage output files are also opt-in (SAVE_STAGE_OUTPUTS): at 2.5-3.0 GB per stage
for 96 months, writing all 17 folders cost about 48 GB for files nothing reads.
"""
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.const_preprocessing import (
    FOLDER_PROCESS_ACTUAL_TIME_COLUMN,
    FOLDER_REMOVE_DUPLICATES,
    PREPROCESSING_STATE_MACHINE,
    STATION_ID_COLUMN,
    TRAINING_READY_OUTPUT_FOLDER,
    PREPROCESSED_OUTPUT_FOLDER,
)
from src.preprocessing_pipeline import PreprocessingPipeline

N = 20


@contextmanager
def _null_logger(*args, **kwargs):
    yield MagicMock()


def _pipeline(tmp_path):
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.project_root = str(tmp_path)
    pipeline.log_dir = str(tmp_path)
    pipeline.training_ready_dir = os.path.join(str(tmp_path), TRAINING_READY_OUTPUT_FOLDER)
    pipeline.preprocessed_dir = os.path.join(str(tmp_path), PREPROCESSED_OUTPUT_FOLDER)
    pipeline.current_file_id = "2023_01"
    pipeline.weather_columns_to_drop = []
    return pipeline


def _machine(station_filter):
    machine = dict(PREPROCESSING_STATE_MACHINE)
    machine["filter_by_target_station"] = station_filter
    return machine


def _flag(value):
    return patch("src.preprocessing_pipeline.PREPROCESSING_STATE_MACHINE", _machine(value))


# --- weather drop list ------------------------------------------------------

def _month_with_other_station_gaps(path):
    """OL is complete; HKI misses Air temperature in every row."""
    frame = pd.DataFrame({
        "stationShortCode": ["OL"] * N + ["HKI"] * N,
        "differenceInMinutes": [1.0] * (2 * N),
        "Air temperature": [1.0] * N + [np.nan] * N,
        "Snow depth": [0.0] * (2 * N),
    })
    frame.to_parquet(path, index=False)
    return str(path)


def test_drop_list_counts_every_station_when_the_filter_is_off(tmp_path):
    files = [_month_with_other_station_gaps(tmp_path / f"matched_data_flat_2023_{i:02d}.parquet")
             for i in (1, 2)]
    with _flag(False):
        dropped = _pipeline(tmp_path).decide_weather_columns_to_drop(files)
    # 50% of all-station rows lack Air temperature (> 30%): stage 12 sees them
    assert dropped == ["Air temperature"]


def test_drop_list_counts_only_the_target_station_when_the_filter_is_on(tmp_path):
    files = [_month_with_other_station_gaps(tmp_path / f"matched_data_flat_2023_{i:02d}.parquet")
             for i in (1, 2)]
    with _flag(True):
        dropped = _pipeline(tmp_path).decide_weather_columns_to_drop(files)
    assert dropped == []


# --- filter_columns ---------------------------------------------------------

def _frame():
    return pd.DataFrame({
        "trainDelayed": [True, False],
        "differenceInMinutes": [7.0, 1.0],
        "Air temperature": [2.0, 3.0],
        "stationShortCode": ["OL", "HKI"],
        STATION_ID_COLUMN: [1200, 1],
        "trainNumber": [123, 456],
    })


def test_filter_columns_keeps_the_station_id_when_the_filter_is_off(tmp_path):
    pipeline = _pipeline(tmp_path)
    with _flag(False), patch.object(pipeline, "get_logger", _null_logger):
        out = pipeline.filter_columns(dataframe=_frame(), month_id="2023_01")
    assert STATION_ID_COLUMN in out.columns
    assert "stationShortCode" not in out.columns
    assert "trainNumber" in out.columns   # a join key, kept whatever the station filter says


def test_filter_columns_drops_the_station_id_when_the_filter_is_on(tmp_path):
    pipeline = _pipeline(tmp_path)
    with _flag(True), patch.object(pipeline, "get_logger", _null_logger):
        out = pipeline.filter_columns(dataframe=_frame(), month_id="2023_01")
    assert STATION_ID_COLUMN not in out.columns


# --- stage output saving ----------------------------------------------------

def test_a_stage_outside_save_stage_outputs_writes_no_file(tmp_path):
    pipeline = _pipeline(tmp_path)
    with patch("src.preprocessing_pipeline.SAVE_STAGE_OUTPUTS", set()):
        path = pipeline._save_stage("remove_duplicates", FOLDER_REMOVE_DUPLICATES,
                                    "2023_01", pd.DataFrame({"a": [1]}), "remove_duplicates")
    assert path is None
    assert not (tmp_path / "data").exists()


def test_a_stage_in_save_stage_outputs_writes_under_project_root(tmp_path):
    pipeline = _pipeline(tmp_path)
    with patch("src.preprocessing_pipeline.SAVE_STAGE_OUTPUTS", {"process_actual_time_column"}):
        path = pipeline._save_stage("process_actual_time_column", FOLDER_PROCESS_ACTUAL_TIME_COLUMN,
                                    "2023_01", pd.DataFrame({"a": [1]}), "process_actual_time_column")
    assert path is not None
    assert list((tmp_path / FOLDER_PROCESS_ACTUAL_TIME_COLUMN).glob("*.parquet"))


def test_a_failed_save_is_a_warning_not_a_failure(tmp_path):
    pipeline = _pipeline(tmp_path)
    with patch("src.preprocessing_pipeline.SAVE_STAGE_OUTPUTS", {"remove_duplicates"}), \
         patch("src.preprocessing_pipeline.save_dataframe_to_parquet", side_effect=OSError("disk full")):
        path = pipeline._save_stage("remove_duplicates", FOLDER_REMOVE_DUPLICATES,
                                    "2023_01", pd.DataFrame({"a": [1]}), "remove_duplicates")
    assert path is None


# --- end to end -------------------------------------------------------------

def _two_station_month(path, rows_per_station=40):
    """A small two-station month with the columns the real state machine needs."""
    rng = np.random.default_rng(0)
    frames = []
    for code, uic in (("OL", 1200), ("HKI", 1)):
        n = rows_per_station
        times = pd.date_range("2023-01-02 06:00", periods=n, freq="37min", tz="UTC")
        frames.append(pd.DataFrame({
            "trainNumber": np.arange(n) + (0 if code == "OL" else 1000),
            "departureDate": times.strftime("%Y-%m-%d"),
            "cancelled": [False] * n,
            "type": ["ARRIVAL"] * n,
            "scheduledTime": times.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "actualTime": times.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "differenceInMinutes": rng.integers(-2, 15, n).astype(float),
            "differenceInMinutes_offset": rng.integers(-2, 15, n).astype(float),
            "differenceInMinutes_eachStation_offset": rng.integers(-2, 15, n).astype(float),
            "commercialStop": [True] * n,
            "trainStopping": [True] * n,
            "causes": ["[]"] * n,
            "stationShortCode": code,
            "stationUICCode": uic,
            "Air temperature": rng.normal(0, 5, n),
            "Wind speed": rng.uniform(0, 8, n),
            "Gust speed": rng.uniform(0, 12, n),
            "Wind direction": rng.uniform(0, 360, n),
            "Relative humidity": rng.uniform(60, 100, n),
            "Dew-point temperature": rng.normal(-1, 4, n),
            "Precipitation intensity": rng.uniform(0, 2, n),
            "Snow depth": rng.uniform(0, 30, n),
            "Pressure (msl)": rng.uniform(990, 1030, n),
            "Horizontal visibility": rng.uniform(2000, 30000, n),
            "Cloud amount": rng.integers(0, 8, n).astype(float),
            "Present weather (auto)": [0.0] * n,
        }))
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return str(path)


def test_the_real_state_machine_runs_end_to_end_with_the_filter_off(tmp_path):
    source = _two_station_month(tmp_path / "matched_data_flat_2023_01.parquet")
    root = tmp_path / "root"
    root.mkdir()
    pipeline = _pipeline(root)
    machine = _machine(False)
    with _flag(False), patch.object(pipeline, "get_logger", _null_logger):
        result = pipeline.execute_preprocessing_pipeline_steps(
            input_file_path=source, file_id="2023_01", year="2023", state_machine=machine)

    assert result["errors"] == []
    assert result["success"] is True
    assert "filter_by_target_station" not in result["steps_executed"]

    ready = pd.read_parquet(root / TRAINING_READY_OUTPUT_FOLDER / "training_ready_2023_01.parquet")
    assert set(ready[STATION_ID_COLUMN]) == {1200, 1}
    assert "stationShortCode" not in ready.columns
    assert "trainDelayed" in ready.columns
