"""Cancelled trains stay in the data, and are labelled by `trainDisrupted`.

A cancelled train has no delay value. Stage 12 used to drop every row without one, so the
models never saw a cancellation (0.63% of arrivals at OL, 1.11% over all stations, and 14% of
the trains of 2023-03 and 2024-02), and `trainDelayed` (`NaN > threshold` is False) would have
called a cancelled train punctual had one slipped through. The label is now
`trainDisrupted = delay > TRAIN_DELAY_MINUTES or cancelled`, built in stage 4; stage 12 keeps
cancelled rows; stage 502 keeps only the target among the label columns, because the trainers
use every non-target column as a feature and a second label would hand the model its answer.
The threshold lives in TRAIN_DELAY_MINUTES only.
"""
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.const_preprocessing import (CATEGORIAL_TARGET_FEATURES, CLASSIFICATION_PROBLEM,
                                        DEFAULT_TARGET_FEATURE, LABEL_COLUMNS,
                                        TRAIN_DELAY_MINUTES as CONFIG_TRAIN_DELAY_MINUTES,
                                        VALID_TARGET_FEATURES)
from src.preprocessing_pipeline import PreprocessingPipeline
from src.training_pipeline import TrainingPipeline

SAVE = "src.preprocessing_pipeline.save_dataframe_to_parquet"

# The sample delays below (10, 2, 3, 9, 20, ...) are chosen around a 5-minute threshold, whatever
# TRAIN_DELAY_MINUTES is set to in the config (0 for the Oulu run). Only the test that checks the
# config constant itself reads CONFIG_TRAIN_DELAY_MINUTES.
TRAIN_DELAY_MINUTES = 5


@pytest.fixture(autouse=True)
def _five_minute_threshold():
    with patch("src.preprocessing_pipeline.TRAIN_DELAY_MINUTES", TRAIN_DELAY_MINUTES), \
            patch("src.training_pipeline.TRAIN_DELAY_MINUTES", TRAIN_DELAY_MINUTES):
        yield


@contextmanager
def _null_logger(*args, **kwargs):
    yield MagicMock()


def _pre(tmp_path):
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.project_root = str(tmp_path)
    pipeline.log_dir = str(tmp_path)
    pipeline.current_file_id = "2023_03"
    pipeline.weather_columns_to_drop = []
    return pipeline


def _trained(tmp_path):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.project_root = str(tmp_path)
    pipeline.log_dir = str(tmp_path / "log")
    os.makedirs(pipeline.log_dir, exist_ok=True)
    return pipeline


def _trains():
    return pd.DataFrame({
        "differenceInMinutes": [10.0, 2.0, np.nan, np.nan, 3.0],
        "cancelled":           [False, False, True, False, True],
    })


# --- config --------------------------------------------------------------------------------

def test_the_label_is_a_registered_classification_target():
    assert "trainDisrupted" in VALID_TARGET_FEATURES
    assert "trainDisrupted" in CLASSIFICATION_PROBLEM
    assert "trainDisrupted" in CATEGORIAL_TARGET_FEATURES      # stage 12 must not to_numeric it
    assert DEFAULT_TARGET_FEATURE == "trainDisrupted"
    assert set(LABEL_COLUMNS) == {"trainDelayed", "trainDisrupted", "cancelled"}


def test_the_delay_threshold_has_one_source():
    from config.const_training import DELAY_THRESHOLD_MINUTES
    assert DELAY_THRESHOLD_MINUTES == CONFIG_TRAIN_DELAY_MINUTES


# --- stage 4: the label -----------------------------------------------------------------------

@patch(SAVE, return_value="/tmp/fake.parquet")
def test_disrupted_is_late_or_cancelled(_save, tmp_path):
    out = _pre(tmp_path).add_train_delayed_feature(_trains())
    assert out["trainDelayed"].tolist() == [True, False, False, False, False]
    assert out["trainDisrupted"].tolist() == [True, False, True, False, True]


@patch(SAVE, return_value="/tmp/fake.parquet")
def test_the_threshold_comes_from_config_not_a_literal_five(_save, tmp_path):
    with patch("src.preprocessing_pipeline.TRAIN_DELAY_MINUTES", 12):
        high = _pre(tmp_path).add_train_delayed_feature(_trains())
    with patch("src.preprocessing_pipeline.TRAIN_DELAY_MINUTES", 1):
        low = _pre(tmp_path).add_train_delayed_feature(_trains())
    assert high["trainDisrupted"].tolist() == [False, False, True, False, True]    # 10 min is not late at 12
    assert low["trainDisrupted"].tolist() == [True, True, True, False, True]       # 2 min is late at 1


@patch(SAVE, return_value="/tmp/fake.parquet")
def test_a_cancelled_train_with_a_small_delay_is_still_disrupted(_save, tmp_path):
    df = pd.DataFrame({"differenceInMinutes": [1.0], "cancelled": [True]})
    assert _pre(tmp_path).add_train_delayed_feature(df)["trainDisrupted"].tolist() == [True]


@patch(SAVE, return_value="/tmp/fake.parquet")
def test_string_encoded_cancelled_flags_are_understood(_save, tmp_path):
    df = pd.DataFrame({"differenceInMinutes": [np.nan, np.nan, 1.0],
                       "cancelled": pd.Series(["True", "False", None], dtype=object)})
    assert _pre(tmp_path).add_train_delayed_feature(df)["trainDisrupted"].tolist() == [True, False, False]


def test_a_missing_cancelled_column_is_an_error(tmp_path):
    with pytest.raises(ValueError) as err:
        _pre(tmp_path).add_train_delayed_feature(pd.DataFrame({"differenceInMinutes": [1.0]}))
    assert "cancelled" in str(err.value)


# --- stage 12: the rows survive ---------------------------------------------------------------------

def _stage12_frame():
    n = 4
    return pd.DataFrame({
        "differenceInMinutes": [3.0, np.nan, np.nan, 20.0],
        "differenceInMinutes_offset": [3.0, np.nan, np.nan, 20.0],
        "differenceInMinutes_eachStation_offset": [3.0, np.nan, np.nan, 20.0],
        "cancelled": [False, True, False, False],          # row 1 cancelled, row 2 unlabelled
        "trainDisrupted": [False, True, False, True],
        "month": [3] * n,
        "Air temperature": [1.0, 2.0, 3.0, 4.0],
        "Wind speed": [5.0, 5.0, 5.0, 5.0],
    })


@patch(SAVE, return_value="/tmp/fake.parquet")
def test_stage_12_keeps_cancelled_trains_and_drops_unlabelled_ones(_save, tmp_path):
    pipeline = _pre(tmp_path)
    with patch.object(pipeline, "get_logger", _null_logger):
        out = pipeline.handle_missing_values(dataframe=_stage12_frame(), month_id="2023_03")
    assert out["cancelled"].tolist() == [False, True, False]        # row 2 (no delay, not cancelled) is gone
    assert out["trainDisrupted"].tolist() == [False, True, True]
    assert out["differenceInMinutes"].isna().tolist() == [False, True, False]   # the delay stays missing


@patch(SAVE, return_value="/tmp/fake.parquet")
def test_the_orchestrator_carries_a_cancelled_train_through_stages_4_to_12(_save, tmp_path):
    n = 3
    raw = pd.DataFrame({
        "differenceInMinutes": [2.0, np.nan, 9.0],
        "differenceInMinutes_offset": [2.0, np.nan, 9.0],
        "differenceInMinutes_eachStation_offset": [2.0, np.nan, 9.0],
        "cancelled": [False, True, False],
        "stationShortCode": ["OL"] * n, "trainNumber": [1, 2, 3],
        "month": [3] * n, "Air temperature": [1.0, 2.0, 3.0], "Wind speed": [4.0, 4.0, 4.0],
        "Gust speed": [6.0] * n, "Relative humidity": [80.0] * n, "Pressure (msl)": [1000.0] * n,
        "Snow depth": [0.0] * n, "Horizontal visibility": [9000.0] * n, "Cloud amount": [3.0] * n,
    })
    source = tmp_path / "matched_data_flat_2023_03.parquet"
    raw.to_parquet(source, index=False)
    machine = {"add_train_delayed_feature": True, "filter_columns": True, "handle_missing_values": True}
    pipeline = _pre(tmp_path)
    with patch.object(pipeline, "get_logger", _null_logger):
        result = pipeline.execute_preprocessing_pipeline_steps(str(source), "2023_03", "2023", machine)
    assert result["success"] is True, result["errors"]
    out = result["data"]
    assert len(out) == 3
    assert out["trainDisrupted"].tolist() == [False, True, True]
    assert "cancelled" in out.columns and out["cancelled"].tolist() == [False, True, False]


# --- stage 502: only the target survives among the labels ----------------------------------------------

def _labelled():
    return pd.DataFrame({"feature_a": [1.0, 2.0], "trainDelayed": [True, False],
                         "trainDisrupted": [True, True], "cancelled": [False, True]})


def test_column_selection_keeps_the_target_and_drops_every_other_label(tmp_path):
    pipeline = _trained(tmp_path)
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDisrupted"):
        out = pipeline.apply_column_selection(_labelled(), ["feature_a", "trainDelayed", "cancelled"])
    assert list(out.columns) == ["feature_a", "trainDisrupted"]


def test_column_selection_follows_the_configured_target(tmp_path):
    pipeline = _trained(tmp_path)
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed"):
        out = pipeline.apply_column_selection(_labelled(), ["feature_a", "trainDelayed", "trainDisrupted"])
    assert list(out.columns) == ["feature_a", "trainDelayed"]


def test_column_selection_leaves_labels_alone_when_the_target_is_absent(tmp_path):
    pipeline = _trained(tmp_path)
    frame = _labelled().drop(columns=["trainDisrupted"])
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDisrupted"):
        out = pipeline.apply_column_selection(frame, ["feature_a", "trainDelayed"])
    assert list(out.columns) == ["feature_a", "trainDelayed"]


# --- stage 504: the leakage guard knows the new target -----------------------------------------------------

@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDisrupted")
def test_balance_classes_drops_the_delay_counterpart_for_the_new_target(tmp_path):
    rng = np.random.default_rng(0)
    delay = np.concatenate([rng.uniform(-4, 5, 300), rng.uniform(6, 60, 100)])

    def frame(n=None):
        d = pd.DataFrame({"differenceInMinutes": delay, "trainDisrupted": delay > TRAIN_DELAY_MINUTES,
                          "feature_a": rng.normal(0, 1, len(delay)), "feature_b": rng.normal(5, 2, len(delay))})
        return d if n is None else d.iloc[:n]

    data_dir = tmp_path / "split"
    data_dir.mkdir()
    frame().to_parquet(data_dir / "merged_data_x_train.parquet", index=False)
    frame(60).to_parquet(data_dir / "merged_data_x_test.parquet", index=False)
    result = _trained(tmp_path).balance_classes(data_dir=str(data_dir))
    assert result["success"] is True
    assert result["dropped_counterpart_col"] == "differenceInMinutes"
