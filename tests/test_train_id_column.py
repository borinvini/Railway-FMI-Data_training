"""trainNumber is carried into the training-ready files as a join key, and is never a feature.

The 101 files had `departureDate` but no train identifier, so nothing could be joined per
train (previous-station delay, per-train analysis). A train number is a recurring timetable
slot, not a unique id (303 numbers but 6,141 runs in 2024_01): with `departureDate` it
identifies one run.

Three things must hold, and each was a way to get it wrong:

1. filter_columns keeps the column, with the station filter on and off. It used to drop it
   because it was in no keep-list.
2. remove_duplicates ignores it. Counting it would keep rows of different trains that were
   collapsed before, and change the row count of every month (and every result after it).
3. Stage 502 drops it even when a features file lists it. The trainers use every non-target
   column as a feature, so a listed trainNumber would train on a train id.

A frame without the column (old fixtures, old files) passes through all of them unchanged.
"""
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd

from config.const_preprocessing import (
    GROUP_KEY_COLUMN,
    PREPROCESSING_STATE_MACHINE,
    PREPROCESSED_OUTPUT_FOLDER,
    TRAIN_ID_COLUMN,
    TRAINING_READY_OUTPUT_FOLDER,
)
from src.preprocessing_pipeline import PreprocessingPipeline
from src.training_pipeline import TrainingPipeline

TARGET = "differenceInMinutes"


@contextmanager
def _null_logger(*args, **kwargs):
    yield MagicMock()


def _preprocessing(tmp_path):
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.project_root = str(tmp_path)
    pipeline.log_dir = str(tmp_path)
    pipeline.training_ready_dir = os.path.join(str(tmp_path), TRAINING_READY_OUTPUT_FOLDER)
    pipeline.preprocessed_dir = os.path.join(str(tmp_path), PREPROCESSED_OUTPUT_FOLDER)
    pipeline.current_file_id = "2023_01"
    pipeline.weather_columns_to_drop = []
    return pipeline


def _training(tmp_path):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.project_root = str(tmp_path)
    pipeline.script_dir = str(tmp_path)
    pipeline.log_dir = str(tmp_path / "log")
    os.makedirs(pipeline.log_dir, exist_ok=True)
    return pipeline


def _station_filter(value):
    machine = dict(PREPROCESSING_STATE_MACHINE)
    machine["filter_by_target_station"] = value
    return patch("src.preprocessing_pipeline.PREPROCESSING_STATE_MACHINE", machine)


def _raw_like():
    return pd.DataFrame({
        TARGET: [7.0, 1.0],
        "Air temperature": [2.0, 3.0],
        GROUP_KEY_COLUMN: ["2023-01-02", "2023-01-02"],
        TRAIN_ID_COLUMN: [123, 456],
        "stationShortCode": ["OL", "HKI"],
    })


# --- 1. filter_columns ---------------------------------------------------------

def test_the_train_id_is_kept_whatever_the_station_filter_says(tmp_path):
    for station_filter in (True, False):
        pipeline = _preprocessing(tmp_path)
        with _station_filter(station_filter), patch.object(pipeline, "get_logger", _null_logger):
            out = pipeline.filter_columns(dataframe=_raw_like(), month_id="2023_01")
        assert TRAIN_ID_COLUMN in out.columns, f"station filter {station_filter}"
        assert out[TRAIN_ID_COLUMN].tolist() == [123, 456]
        assert "stationShortCode" not in out.columns


def test_a_frame_without_the_train_id_still_filters(tmp_path):
    pipeline = _preprocessing(tmp_path)
    frame = _raw_like().drop(columns=[TRAIN_ID_COLUMN])
    with _station_filter(True), patch.object(pipeline, "get_logger", _null_logger):
        out = pipeline.filter_columns(dataframe=frame, month_id="2023_01")
    assert TRAIN_ID_COLUMN not in out.columns
    assert GROUP_KEY_COLUMN in out.columns


# --- 2. remove_duplicates ------------------------------------------------------

def _twins():
    """Rows 0 and 1 are identical except for the train number; row 2 is different."""
    return pd.DataFrame({
        TARGET: [4.0, 4.0, 9.0],
        "Air temperature": [1.5, 1.5, 0.5],
        GROUP_KEY_COLUMN: ["2023-01-02"] * 3,
        TRAIN_ID_COLUMN: [10, 20, 30],
    })


def test_rows_that_differ_only_in_the_train_id_are_still_duplicates(tmp_path):
    pipeline = _preprocessing(tmp_path)
    with patch.object(pipeline, "get_logger", _null_logger), \
         patch("src.preprocessing_pipeline.SAVE_STAGE_OUTPUTS", set()):
        with_id = pipeline.remove_duplicates(dataframe=_twins(), month_id="2023_01")
        without_id = pipeline.remove_duplicates(dataframe=_twins().drop(columns=[TRAIN_ID_COLUMN]),
                                                month_id="2023_01")
    assert len(with_id) == len(without_id) == 2, "the row count must not depend on the train id"
    assert TRAIN_ID_COLUMN in with_id.columns
    assert with_id[TRAIN_ID_COLUMN].tolist() == [10, 30], "the first of the twins survives"


def test_genuinely_different_rows_are_kept_with_their_train_ids(tmp_path):
    pipeline = _preprocessing(tmp_path)
    frame = _twins()
    frame.loc[1, "Air temperature"] = 2.5   # now row 1 differs in a real column
    with patch.object(pipeline, "get_logger", _null_logger), \
         patch("src.preprocessing_pipeline.SAVE_STAGE_OUTPUTS", set()):
        out = pipeline.remove_duplicates(dataframe=frame, month_id="2023_01")
    assert out[TRAIN_ID_COLUMN].tolist() == [10, 20, 30]


# --- 3. stage 502 --------------------------------------------------------------

def test_stage_502_drops_a_listed_train_id_but_keeps_the_keys_and_the_target(tmp_path):
    frame = pd.DataFrame({
        TARGET: [1.0, 2.0], "f1": [0.1, 0.2],
        GROUP_KEY_COLUMN: ["2023-01-02", "2023-01-03"], TRAIN_ID_COLUMN: [123, 456],
    })
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET):
        kept = _training(tmp_path).apply_column_selection(frame, ["f1", TRAIN_ID_COLUMN, TARGET])
    assert TRAIN_ID_COLUMN not in kept.columns
    assert set(kept.columns) == {"f1", TARGET, GROUP_KEY_COLUMN}


def test_stage_502_is_unchanged_for_a_frame_without_the_train_id(tmp_path):
    frame = pd.DataFrame({TARGET: [1.0], "f1": [0.1], GROUP_KEY_COLUMN: ["2023-01-02"]})
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET):
        kept = _training(tmp_path).apply_column_selection(frame, ["f1"])
    assert set(kept.columns) == {"f1", TARGET, GROUP_KEY_COLUMN}
