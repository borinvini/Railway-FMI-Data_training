"""day_of_month gets a sin/cos pair like month, hour and day_of_week.

The raw 1-31 integer put the 31st and the 1st at opposite ends of the scale although
they are neighbours on the calendar; the other three calendar features already had a
cyclical encoding. The pair is day_month_sin / day_month_cos on a fixed 31-position
circle, sin/cos(2*pi*(day-1)/31), the raw column is kept (as day_of_week is), and FCLL
derives the same values from the date, so the two paths cannot drift apart.
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.const_preprocessing import (DAY_OF_MONTH_CYCLE, PREPROCESSING_STATE_MACHINE,
                                        TEMPORAL_SINCOS_FEATURES)
from src.fcll_sequences import _add_temporal_features
from src.preprocessing_pipeline import PreprocessingPipeline

STAGE = "convert_dayofmonth_to_sincos"


@contextmanager
def _null_logger(*args, **kwargs):
    yield MagicMock()


@pytest.fixture(autouse=True)
def _no_parquet_writes():
    with patch("src.preprocessing_pipeline.save_dataframe_to_parquet",
               return_value="/tmp/fake.parquet"):
        yield


def _run(tmp_path, frame):
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.project_root = str(tmp_path)
    pipeline.log_dir = str(tmp_path)
    pipeline.current_file_id = "2023_01"
    pipeline.training_ready_dir = str(tmp_path / "training_ready")
    pipeline.preprocessed_dir = str(tmp_path / "preprocessed")
    with patch.object(pipeline, "get_logger", _null_logger):
        return pipeline.convert_dayofmonth_to_sincos(dataframe=frame, month_id="2023_01")


def test_the_pair_is_on_a_31_position_circle(tmp_path):
    days = list(range(1, 32))
    out = _run(tmp_path, pd.DataFrame({"day_of_month": days}))

    angle = 2 * np.pi * (np.array(days) - 1) / 31.0
    np.testing.assert_allclose(out["day_month_sin"], np.sin(angle))
    np.testing.assert_allclose(out["day_month_cos"], np.cos(angle))
    assert (out["day_month_sin"].iloc[0], out["day_month_cos"].iloc[0]) == (0.0, 1.0)
    np.testing.assert_allclose(out["day_month_sin"] ** 2 + out["day_month_cos"] ** 2, 1.0)


def test_the_31st_is_as_close_to_the_1st_as_the_2nd_is(tmp_path):
    out = _run(tmp_path, pd.DataFrame({"day_of_month": [1, 2, 31]})).set_index("day_of_month")
    dist = lambda a, b: float(np.hypot(*(out.loc[a, ["day_month_sin", "day_month_cos"]]
                                         - out.loc[b, ["day_month_sin", "day_month_cos"]])))
    assert dist(31, 1) == pytest.approx(dist(1, 2))
    # the raw integers would say 30 apart
    assert dist(31, 1) < dist(2, 31) + 1e-9


def test_the_original_column_is_kept_and_the_pair_follows_it(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "day_of_month": [14, 15], "z": [3, 4]})
    out = _run(tmp_path, frame)
    assert list(out.columns) == ["a", "day_of_month", "day_month_sin", "day_month_cos", "z"]
    assert out["day_of_month"].tolist() == [14, 15]


def test_a_missing_day_gets_the_neutral_point(tmp_path):
    out = _run(tmp_path, pd.DataFrame({"day_of_month": [5.0, np.nan]}))
    assert (out["day_month_sin"].iloc[1], out["day_month_cos"].iloc[1]) == (0.0, 0.0)
    assert out["day_month_cos"].iloc[0] != 0.0


def test_the_input_frame_is_not_modified(tmp_path):
    frame = pd.DataFrame({"day_of_month": [14]})
    _run(tmp_path, frame)
    assert list(frame.columns) == ["day_of_month"]


def test_the_switch_is_on_and_runs_between_the_week_stage_and_the_drop_stage():
    keys = list(PREPROCESSING_STATE_MACHINE)
    assert PREPROCESSING_STATE_MACHINE[STAGE] is True
    assert keys.index("convert_dayofweek_to_sincos") < keys.index(STAGE) \
        < keys.index("drop_original_temporal_columns")


def test_the_new_columns_are_registered_as_temporal_sincos_features():
    assert {"day_month_sin", "day_month_cos"} <= set(TEMPORAL_SINCOS_FEATURES)
    assert DAY_OF_MONTH_CYCLE == 31


def test_dropping_the_originals_also_drops_day_of_month(tmp_path):
    frame = pd.DataFrame({"month": [3], "hour": ["10:30"], "day_of_week": [2],
                          "day_of_month": [15], "day_month_sin": [0.1]})
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.project_root = str(tmp_path)
    pipeline.log_dir = str(tmp_path)
    pipeline.current_file_id = "2023_01"
    pipeline.training_ready_dir = str(tmp_path / "training_ready")
    pipeline.preprocessed_dir = str(tmp_path / "preprocessed")
    with patch.object(pipeline, "get_logger", _null_logger):
        out = pipeline.drop_original_temporal_columns(dataframe=frame, month_id="2023_01")
    assert list(out.columns) == ["day_month_sin"]


def test_fcll_derives_the_same_pair_from_the_date():
    moment = pd.Series(pd.to_datetime(["2023-01-01T08:00:00Z", "2023-01-31T08:00:00Z",
                                       "2023-02-14T22:30:00Z"], utc=True))
    wide = _add_temporal_features(pd.DataFrame(index=moment.index), moment)
    angle = 2 * np.pi * (moment.dt.day.to_numpy() - 1) / float(DAY_OF_MONTH_CYCLE)
    np.testing.assert_allclose(wide["day_month_sin"], np.sin(angle))
    np.testing.assert_allclose(wide["day_month_cos"], np.cos(angle))
    assert wide["day_of_month"].tolist() == [1, 31, 14]
