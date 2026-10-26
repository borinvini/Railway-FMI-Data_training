"""`delayClass`: a four-class delay label next to the binary ones.

The binary labels (`trainDelayed`, `trainDisrupted`) say only whether a train was late. `delayClass`
splits the delay into 0 = on time or early (delay <= 0), 1 = 0-5 min, 2 = 5-15 min,
3 = more than 15 min **or cancelled**, with the upper edges in DELAY_CLASS_BOUNDS (edge included).
A cancelled train has no delay value, so it needs its own route into class 3 (the same reason
`trainDisrupted` has one); a train with no delay that was not cancelled has no class and is dropped
by stage 12 like it is for the numeric targets. The edges are independent of TRAIN_DELAY_MINUTES,
which keeps driving only the binary labels, and the label is one more entry of the label lists, so
stage 502 drops it whenever it is not the target and the same training-ready files serve every target.
"""
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.const_preprocessing import (BINARY_CLASSIFICATION_PROBLEM, CATEGORIAL_TARGET_FEATURES,
                                        CLASSIFICATION_PROBLEM, DELAY_CLASS_BOUNDS, DELAY_CLASS_COLUMN,
                                        DELAY_CLASS_NAMES, LABEL_COLUMNS, MULTICLASS_PROBLEM,
                                        VALID_TARGET_FEATURES)
from src.preprocessing_pipeline import PreprocessingPipeline, delay_classes
from src.training_pipeline import TrainingPipeline

SAVE = "src.preprocessing_pipeline.save_dataframe_to_parquet"


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


# --- config ----------------------------------------------------------------------------------------

def test_the_label_is_registered_as_a_multiclass_target():
    assert DELAY_CLASS_COLUMN == "delayClass"
    assert DELAY_CLASS_BOUNDS == (0, 5, 15)
    assert len(DELAY_CLASS_NAMES) == len(DELAY_CLASS_BOUNDS) + 1
    assert MULTICLASS_PROBLEM == ["delayClass"]
    assert "delayClass" in CLASSIFICATION_PROBLEM
    assert "delayClass" in VALID_TARGET_FEATURES            # stage 502 / 503 treat it as a label
    assert "delayClass" in CATEGORIAL_TARGET_FEATURES       # stage 12 must not to_numeric it
    assert "delayClass" in LABEL_COLUMNS


def test_the_binary_targets_are_unchanged_and_come_first():
    assert BINARY_CLASSIFICATION_PROBLEM == ["trainDelayed", "cancelled", "trainDisrupted"]
    assert CLASSIFICATION_PROBLEM[:3] == BINARY_CLASSIFICATION_PROBLEM    # list order is pinned elsewhere
    assert "delayClass" not in BINARY_CLASSIFICATION_PROBLEM


# --- the function --------------------------------------------------------------------------------------

def test_class_edges_include_the_upper_bound():
    delay = pd.Series([-3.0, 0.0, 0.5, 5.0, 5.1, 15.0, 15.1, 60.0])
    out = delay_classes(delay, [False] * len(delay))
    assert out.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_a_cancelled_train_is_class_3_whatever_its_delay():
    delay = pd.Series([np.nan, 1.0, -2.0])
    assert delay_classes(delay, [True, True, True]).tolist() == [3, 3, 3]


def test_a_missing_delay_that_is_not_cancelled_has_no_class():
    out = delay_classes(pd.Series([np.nan, 2.0]), [False, False])
    assert out.isna().tolist() == [True, False]
    assert out.iloc[1] == 1


def test_string_delays_are_coerced():
    out = delay_classes(pd.Series(["3", "20", "x"], dtype=object), [False, False, False])
    assert out.tolist()[:2] == [1, 3] and bool(out.isna().iloc[2])


# --- stage 4 -----------------------------------------------------------------------------------------------

def _trains():
    return pd.DataFrame({
        "differenceInMinutes": [-1.0, 3.0, 10.0, 40.0, np.nan, np.nan],
        "cancelled":           [False, False, False, False, True, False],
    })


@patch(SAVE, return_value="/tmp/fake.parquet")
@pytest.mark.parametrize("threshold", [0, 5, 12])
def test_stage_4_adds_the_class_and_ignores_train_delay_minutes(_save, tmp_path, threshold):
    with patch("src.preprocessing_pipeline.TRAIN_DELAY_MINUTES", threshold):
        out = _pre(tmp_path).add_train_delayed_feature(_trains())
    assert out["delayClass"].tolist()[:5] == [0, 1, 2, 3, 3]       # same classes at every threshold
    assert bool(out["delayClass"].isna().iloc[5])                  # no delay, not cancelled: no class yet
    assert "trainDisrupted" in out.columns and "trainDelayed" in out.columns


@patch(SAVE, return_value="/tmp/fake.parquet")
def test_a_zero_threshold_binary_label_agrees_with_the_classes(_save, tmp_path):
    with patch("src.preprocessing_pipeline.TRAIN_DELAY_MINUTES", 0):
        out = _pre(tmp_path).add_train_delayed_feature(_trains()).iloc[:5]
    assert (out["trainDisrupted"] == (out["delayClass"] >= 1)).all()


# --- stage 12 ----------------------------------------------------------------------------------------------

@patch(SAVE, return_value="/tmp/fake.parquet")
def test_stage_12_drops_the_unlabelled_row_and_leaves_a_plain_int_label(_save, tmp_path):
    pipeline = _pre(tmp_path)
    with patch("src.preprocessing_pipeline.TRAIN_DELAY_MINUTES", 0):
        labelled = pipeline.add_train_delayed_feature(_trains())
    n = len(labelled)
    labelled["differenceInMinutes_offset"] = labelled["differenceInMinutes"]
    labelled["differenceInMinutes_eachStation_offset"] = labelled["differenceInMinutes"]
    labelled["month"] = [3] * n
    labelled["Air temperature"] = np.arange(n, dtype=float)
    labelled["Wind speed"] = [5.0] * n
    with patch.object(pipeline, "get_logger", _null_logger):
        out = pipeline.handle_missing_values(dataframe=labelled, month_id="2023_03")
    assert len(out) == 5                                           # the unlabelled row is gone
    assert out["delayClass"].tolist() == [0, 1, 2, 3, 3]
    assert out["delayClass"].dtype == np.int8


@patch(SAVE, return_value="/tmp/fake.parquet")
def test_stage_12_refuses_a_label_that_still_has_gaps(_save, tmp_path):
    pipeline = _pre(tmp_path)
    frame = pd.DataFrame({
        "differenceInMinutes": [3.0, 4.0], "differenceInMinutes_offset": [3.0, 4.0],
        "differenceInMinutes_eachStation_offset": [3.0, 4.0], "cancelled": [False, False],
        "delayClass": pd.Series([1, pd.NA], dtype="Int8"),
        "month": [3, 3], "Air temperature": [1.0, 2.0], "Wind speed": [5.0, 5.0],
    })
    with patch.object(pipeline, "get_logger", _null_logger), pytest.raises(ValueError, match="delayClass"):
        pipeline.handle_missing_values(dataframe=frame, month_id="2023_03")


# --- stage 502 ---------------------------------------------------------------------------------------------

def _labelled():
    return pd.DataFrame({"feature_a": [1.0, 2.0], "trainDelayed": [True, False],
                         "trainDisrupted": [True, True], "cancelled": [False, True],
                         "delayClass": [1, 3]})


def test_a_binary_run_drops_the_class_label():
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDisrupted"):
        out = pipeline.apply_column_selection(_labelled(), ["feature_a", "delayClass", "trainDelayed"])
    assert list(out.columns) == ["feature_a", "trainDisrupted"]


def test_a_class_run_keeps_only_the_class_label():
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "delayClass"):
        out = pipeline.apply_column_selection(_labelled(), ["feature_a", "trainDisrupted", "cancelled"])
    assert list(out.columns) == ["feature_a", "delayClass"]
