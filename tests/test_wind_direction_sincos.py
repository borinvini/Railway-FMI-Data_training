"""Wind direction gets a sin/cos pair: 359 and 1 degrees are both north.

`Wind direction` (0-360) was fed to the models as a straight line, so 359 and 1 degrees
were 358 apart: a tree needed two splits to isolate a north wind and a linear model could
not represent it. wind_dir_sin / wind_dir_cos fix that, the raw column is kept, and a
direction of 0 - FMI's calm code, wind speed is 0.00 on every such row - maps to (0, 0)
instead of "north". FCLL derives the same pair from the raw events (it silently drops any
requested feature it cannot find), through the same function.
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.const_preprocessing import (PREPROCESSING_STATE_MACHINE, WIND_CALM_DIRECTION,
                                        WIND_SINCOS_FEATURES)
from src.circular_features import wind_direction_sincos
from src.fcll_sequences import derive_station_visits
from src.preprocessing_pipeline import PreprocessingPipeline

STAGE = "convert_winddirection_to_sincos"


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
        return pipeline.convert_winddirection_to_sincos(dataframe=frame, month_id="2023_01")


def _distance(a, b):
    s, c = wind_direction_sincos(pd.Series([a, b]))
    return float(np.hypot(s[0] - s[1], c[0] - c[1]))


def test_the_compass_points_land_where_they_should():
    s, c = wind_direction_sincos(pd.Series([90.0, 180.0, 270.0, 360.0]))
    np.testing.assert_allclose(s, [1.0, 0.0, -1.0, 0.0], atol=1e-12)   # E, S, W, N
    np.testing.assert_allclose(c, [0.0, -1.0, 0.0, 1.0], atol=1e-12)


def test_every_non_calm_direction_is_on_the_unit_circle():
    s, c = wind_direction_sincos(pd.Series(np.arange(1.0, 361.0)))
    np.testing.assert_allclose(s ** 2 + c ** 2, 1.0)


def test_359_and_1_degrees_are_as_close_as_1_and_3():
    assert _distance(359.0, 1.0) == pytest.approx(_distance(1.0, 3.0))
    # the raw numbers would put them 358 apart
    assert abs(359.0 - 1.0) == 358.0


def test_calm_is_the_no_direction_point_not_north():
    assert WIND_CALM_DIRECTION == 0
    s, c = wind_direction_sincos(pd.Series([0.0, 360.0]))
    assert (s[0], c[0]) == (0.0, 0.0)             # calm
    assert c[1] == pytest.approx(1.0)             # 360 is north


def test_a_missing_direction_stays_missing():
    s, c = wind_direction_sincos(pd.Series([90.0, np.nan]))
    assert np.isnan(s[1]) and np.isnan(c[1])
    assert s[0] == pytest.approx(1.0)


def test_the_stage_keeps_the_original_and_places_the_pair_after_it(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "Wind direction": [90.0, 270.0], "z": [3, 4]})
    out = _run(tmp_path, frame)
    assert list(out.columns) == ["a", "Wind direction", "wind_dir_sin", "wind_dir_cos", "z"]
    assert out["Wind direction"].tolist() == [90.0, 270.0]
    assert list(frame.columns) == ["a", "Wind direction", "z"], "the input frame must not change"
    assert WIND_SINCOS_FEATURES == ["wind_dir_sin", "wind_dir_cos"]


def test_the_stage_leaves_a_missing_value_for_the_split_time_imputation(tmp_path):
    out = _run(tmp_path, pd.DataFrame({"Wind direction": [45.0, np.nan]}))
    assert out["wind_dir_sin"].iloc[0] == pytest.approx(np.sin(np.pi / 4))
    assert out[["wind_dir_sin", "wind_dir_cos"]].iloc[1].isna().all()


def test_the_switch_is_on_between_the_day_of_month_stage_and_the_drop_stage():
    keys = list(PREPROCESSING_STATE_MACHINE)
    assert PREPROCESSING_STATE_MACHINE[STAGE] is True
    assert keys.index("convert_dayofmonth_to_sincos") < keys.index(STAGE) \
        < keys.index("drop_original_temporal_columns")


def _event(station, kind, wind):
    return {"trainNumber": 1, "departureDate": "2023-01-14", "stationShortCode": station,
            "type": kind, "scheduledTime": "2023-01-14T10:00:00.000Z",
            "actualTime": "2023-01-14T10:05:00.000Z", "differenceInMinutes": 5.0,
            "trainStopping": True, "commercialStop": True, "Wind direction": wind}


def test_fcll_derives_the_same_pair_so_a_requested_name_is_not_dropped():
    raw = pd.DataFrame([_event("HKI", "ARRIVAL", 359.0), _event("OL", "ARRIVAL", 0.0)])

    visits = derive_station_visits(raw, carry_columns=["wind_dir_sin", "wind_dir_cos"])

    assert {"wind_dir_sin", "wind_dir_cos"} <= set(visits.columns)
    by_station = visits.set_index("stationShortCode")
    expected_sin, expected_cos = wind_direction_sincos(pd.Series([359.0]))
    assert by_station.loc["HKI", "wind_dir_sin"] == pytest.approx(expected_sin[0])
    assert by_station.loc["HKI", "wind_dir_cos"] == pytest.approx(expected_cos[0])
    assert (by_station.loc["OL", "wind_dir_sin"], by_station.loc["OL", "wind_dir_cos"]) == (0.0, 0.0)


def test_fcll_and_the_preprocessing_stage_agree(tmp_path):
    degrees = [10.0, 123.0, 359.0]
    out = _run(tmp_path, pd.DataFrame({"Wind direction": degrees}))
    s, c = wind_direction_sincos(pd.Series(degrees))
    np.testing.assert_allclose(out["wind_dir_sin"], s)
    np.testing.assert_allclose(out["wind_dir_cos"], c)
