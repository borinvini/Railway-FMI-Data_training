"""Weather values outside their physical limits are set to NaN, never kept or dropped.

The raw matched_data_flat_2019_03 holds one weather observation (Jarvenpaa Sorto,
2019-03-23 05:29 UTC) with Dew-point temperature = -84.8 while the air temperature was
1.5 C and the humidity 63%, which imply about -4.8 C. It was joined to the ARRIVAL and
DEPARTURE rows of one train, so two cells carried it through to the training-ready
files. null_out_of_bounds_weather (stage 4b) nulls such cells, and the train-only
median imputation at the split fills them. The -1 "not measured" sentinel of Snow depth
and the precipitation columns, and window means that average it (values between -1 and
0), stay inside the limits on purpose: that is a separate issue.
"""
import numpy as np
import pandas as pd
import pytest

from config.const_preprocessing import (PREPROCESSING_STATE_MACHINE, WEATHER_BOUNDS_MAX_NULLED_SHARE,
                                        weather_bounds_for)
from src.preprocessing_pipeline import PreprocessingPipeline

N = 5000    # one bad cell must stay under WEATHER_BOUNDS_MAX_NULLED_SHARE (0.1%)


def _pipeline():
    return PreprocessingPipeline.__new__(PreprocessingPipeline)


def _frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Air temperature": rng.normal(2, 8, N),
        "Dew-point temperature": rng.normal(-1, 7, N),
        "Relative humidity": rng.uniform(40, 100, N),
        "Wind direction": rng.uniform(0, 360, N),
        "Snow depth": rng.choice([-1.0, 0.0, 12.0], N),
        "Air temperature (24h max)": rng.normal(4, 8, N),
        "trainDelayed": rng.integers(0, 2, N).astype(bool),
    })


def test_the_2019_03_glitch_becomes_nan_and_nothing_else_changes():
    df = _frame()
    df.loc[10, ["Air temperature", "Relative humidity", "Dew-point temperature"]] = [1.5, 63.0, -84.8]
    out = _pipeline().null_out_of_bounds_weather(df, "2019_03")

    assert np.isnan(out.loc[10, "Dew-point temperature"])
    expected = df.copy()
    expected.loc[10, "Dew-point temperature"] = np.nan
    pd.testing.assert_frame_equal(out, expected)
    assert len(out) == len(df)


def test_the_minus_one_sentinel_and_averages_of_it_are_left_alone():
    df = _frame()
    df.loc[3, "Snow depth"] = -0.5          # a window mean that averaged the sentinel
    df["Snow depth (24h mean)"] = -0.5
    out = _pipeline().null_out_of_bounds_weather(df, "2019_03")
    assert (out["Snow depth"] == -1.0).sum() == (df["Snow depth"] == -1.0).sum()
    assert out.loc[3, "Snow depth"] == -0.5
    assert (out["Snow depth (24h mean)"] == -0.5).all()


def test_window_columns_use_the_base_limits():
    df = _frame()
    df.loc[7, "Air temperature (24h max)"] = 90.0
    out = _pipeline().null_out_of_bounds_weather(df, "2019_03")
    assert np.isnan(out.loc[7, "Air temperature (24h max)"])
    assert weather_bounds_for("Air temperature (24h max)") == weather_bounds_for("Air temperature")
    assert weather_bounds_for("Precipitation amount (72h cumulative)") == (-1, 1000)
    assert weather_bounds_for("Pressure (msl) (12h min)") == (900, 1100)
    assert weather_bounds_for("trainDelayed") is None


def test_infinity_is_nulled_and_values_on_a_limit_are_kept():
    df = _frame()
    df.loc[1, "Air temperature"] = np.inf
    df.loc[2, "Relative humidity"] = 100.0
    df.loc[3, "Relative humidity"] = 0.0
    df.loc[4, "Wind direction"] = 360.0
    out = _pipeline().null_out_of_bounds_weather(df, "2019_03")
    assert np.isnan(out.loc[1, "Air temperature"])
    assert out.loc[2, "Relative humidity"] == 100.0
    assert out.loc[3, "Relative humidity"] == 0.0
    assert out.loc[4, "Wind direction"] == 360.0


def test_too_many_out_of_bounds_cells_raise_instead_of_hiding_an_upstream_change():
    df = _frame()
    df.loc[: int(N * WEATHER_BOUNDS_MAX_NULLED_SHARE) + 5, "Air temperature"] = 300.0    # e.g. Kelvin
    with pytest.raises(ValueError) as err:
        _pipeline().null_out_of_bounds_weather(df, "2019_03")
    assert "Air temperature" in str(err.value)
    assert "2019_03" in str(err.value)


def test_a_month_without_any_weather_column_is_a_failure():
    with pytest.raises(ValueError):
        _pipeline().null_out_of_bounds_weather(pd.DataFrame({"trainDelayed": [True, False]}), "2019_03")


def test_none_and_empty_frames_are_handled_like_the_other_stages():
    assert _pipeline().null_out_of_bounds_weather(None) is None
    empty = _frame().iloc[0:0]
    assert _pipeline().null_out_of_bounds_weather(empty, "2019_03").empty


def test_the_stage_runs_after_the_label_and_before_the_scenarios():
    order = list(PREPROCESSING_STATE_MACHINE)
    assert order.index("add_train_delayed_feature") < order.index("null_out_of_bounds_weather")
    assert order.index("null_out_of_bounds_weather") < order.index("add_weather_scenarios_col")


def test_the_orchestrator_runs_the_stage_and_honours_the_switch(tmp_path):
    df = _frame()
    df.loc[10, "Dew-point temperature"] = -84.8
    source = tmp_path / "matched_data_flat_2019_03.parquet"
    df.to_parquet(source, index=False)

    on = _pipeline().execute_preprocessing_pipeline_steps(
        str(source), "2019_03", "2019", {"null_out_of_bounds_weather": True})
    assert on["success"] is True
    assert "null_out_of_bounds_weather" in on["steps_executed"]
    assert np.isnan(on["data"].loc[10, "Dew-point temperature"])

    off = _pipeline().execute_preprocessing_pipeline_steps(
        str(source), "2019_03", "2019", {"null_out_of_bounds_weather": False})
    assert "null_out_of_bounds_weather" not in off["steps_executed"]
    assert off["data"].loc[10, "Dew-point temperature"] == -84.8


def test_the_stage_failure_is_reported_by_the_orchestrator(tmp_path):
    source = tmp_path / "matched_data_flat_2019_03.parquet"
    pd.DataFrame({"trainDelayed": [True, False]}).to_parquet(source, index=False)
    result = _pipeline().execute_preprocessing_pipeline_steps(
        str(source), "2019_03", "2019", {"null_out_of_bounds_weather": True})
    assert result["success"] is False
    assert any("null_out_of_bounds_weather" in e for e in result["errors"])
