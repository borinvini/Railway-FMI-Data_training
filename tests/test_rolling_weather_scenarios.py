"""Rolling weather scenarios must scale with the window length.

add_rolling_weather_scenarios_col used the instant thresholds on worst-case
aggregates in every window: a cumulative amount above 10 mm is Heavy Rain whether
it fell in 12 hours or 72, the peak intensity inside the window counted as
well, and Black Ice / Freezing Rain required min and max temperature to stay
inside +-2 C for the whole window. On the 117,868 rows at OL the 72h columns
read Heavy Rain 17.3%, Blizzard 11.5%, Heavy Snow 9.2% (instant: 0.18, 0.05,
0.02%) and Black Ice 0.04% (instant: 13.5%).

Window state now comes from means and rates: precipitation is the cumulative
amount against a mm/h rate times the window length, temperature phase and band
use the window mean, and humidity uses the window mean (RH max exceeds 80 in
99% of 72h windows). Extremes (Extreme Cold / Heat, winds, visibility, snow
depth) keep their peaks.
"""
import glob
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from config.const_preprocessing import ROLLING_WINDOWS, rolling_window_hours
from src.preprocessing_pipeline import PreprocessingPipeline

QUIET = {
    "Air temperature": {"min": 8.0, "max": 12.0, "mean": 10.0},
    "Wind speed": {"max": 3.0},
    "Relative humidity": {"max": 60.0, "mean": 50.0},
    "Precipitation intensity": {"max": 0.0},
    "Snow depth": {"max": 0.0},
    "Horizontal visibility": {"min": 20000.0},
}


def _window_columns(window, overrides, amount):
    values = {k: dict(v) for k, v in QUIET.items()}
    for name, aggs in overrides.items():
        values[name].update(aggs)
    cols = {f"{name} ({window} {agg})": v
            for name, aggs in values.items() for agg, v in aggs.items()}
    cols[f"Precipitation amount ({window} cumulative)"] = amount
    return cols


def _classify(window, overrides=None, amount=0.0):
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.current_file_id = "2023_01"
    pipeline.project_root = "unused"
    cols = {}
    for w in ROLLING_WINDOWS:
        cols.update(_window_columns(w, overrides or {}, amount))
    frame = pd.DataFrame([cols])
    with patch("src.preprocessing_pipeline.save_dataframe_to_parquet",
               return_value="/tmp/fake.parquet"):
        out = pipeline.add_rolling_weather_scenarios_col(frame, month_id="2023_01")
    return out[f"weather_scenario_{window}"].iloc[0]


def test_window_hours_come_from_the_window_names():
    assert [rolling_window_hours(w) for w in ROLLING_WINDOWS] == [12, 24, 72]


def test_the_same_total_is_heavy_rain_in_12h_but_not_in_72h():
    warm = {"Air temperature": {"mean": 5.0, "min": 3.0, "max": 7.0}}
    assert _classify("12h", warm, amount=10.0) == "Heavy Rain"
    assert _classify("72h", warm, amount=10.0) == "Normal/Clear"
    assert _classify("72h", warm, amount=40.0) == "Heavy Rain"


def test_peak_intensity_alone_is_not_heavy_rain():
    warm = {"Air temperature": {"mean": 5.0, "min": 3.0, "max": 7.0},
            "Precipitation intensity": {"max": 20.0}}
    assert _classify("24h", warm, amount=1.0) == "Normal/Clear"


def test_heavy_snow_needs_a_rate_that_scales_with_the_window():
    snow = {"Air temperature": {"mean": -5.0, "min": -8.0, "max": -2.0},
            "Snow depth": {"max": 20.0}}
    assert _classify("72h", snow, amount=10.0) == "Normal/Clear"
    assert _classify("72h", snow, amount=20.0) == "Heavy Snow"


def test_band_uses_the_window_mean_not_min_and_max():
    wide = {"Air temperature": {"min": -6.0, "max": 4.0, "mean": 0.0},
            "Relative humidity": {"mean": 90.0, "max": 99.0}}
    assert _classify("72h", wide, amount=0.0) == "Black Ice"


def test_wet_band_is_freezing_rain_and_dry_band_is_black_ice():
    band = {"Air temperature": {"min": -1.0, "max": 1.0, "mean": 0.0},
            "Relative humidity": {"mean": 90.0, "max": 95.0}}
    assert _classify("24h", band, amount=24 * 0.5) == "Black Ice"
    assert _classify("24h", band, amount=24 * 0.5 + 1.0) == "Freezing Rain"


def test_humidity_peak_alone_is_neither_black_ice_nor_dense_fog():
    band = {"Air temperature": {"min": -1.0, "max": 1.0, "mean": 0.0},
            "Relative humidity": {"mean": 70.0, "max": 99.0},
            "Horizontal visibility": {"min": 400.0}}
    assert _classify("24h", band, amount=0.0) == "Normal/Clear"


def test_dense_fog_rate_scales_with_the_window():
    fog = {"Relative humidity": {"mean": 98.0, "max": 99.0},
           "Horizontal visibility": {"min": 400.0}}
    assert _classify("72h", fog, amount=72 * 0.1) == "Dense Fog"
    # 3 mm in 12h is 0.25 mm/h: too wet for fog, not yet Heavy Rain (5 mm)
    assert _classify("12h", fog, amount=3.0) == "Normal/Clear"


def test_extreme_cold_keeps_the_window_minimum():
    cold = {"Air temperature": {"min": -25.0, "max": -5.0, "mean": -10.0}}
    assert _classify("72h", cold) == "Extreme Cold"


def _ol_rows(months=("03", "11")):
    frames = []
    for month in months:
        for path in sorted(glob.glob(f"data/input/matched_data_flat_*_{month}.parquet")):
            df = pd.read_parquet(path)
            frames.append(df[df["stationShortCode"] == "OL"])
    return pd.concat(frames, ignore_index=True) if frames else None


def test_real_ol_rows_keep_the_72h_columns_close_to_the_instant_rates():
    frame = _ol_rows()
    if frame is None or frame.empty:
        pytest.skip("no parquet in data/input")
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.current_file_id = "real"
    pipeline.project_root = "unused"
    with patch("src.preprocessing_pipeline.save_dataframe_to_parquet",
               return_value="/tmp/fake.parquet"):
        out = pipeline.add_rolling_weather_scenarios_col(frame, month_id="real")
    share = out["weather_scenario_72h"].value_counts(normalize=True) * 100
    for heavy in ("Heavy Rain", "Blizzard", "Heavy Snow"):
        assert share.get(heavy, 0.0) < 5.0, (heavy, share.get(heavy))
    assert share.get("Black Ice", 0.0) > 1.0, share.get("Black Ice")
