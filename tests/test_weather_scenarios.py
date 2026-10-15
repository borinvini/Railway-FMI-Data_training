"""Instant weather scenarios must not depend on "Precipitation amount".

That column is null in 87.8% of the rows at OL (81-90% in every year). Pandas
compares NaN as False, so `amount <= 0.5` (Black Ice) and `amount <= 0.1`
(Dense Fog) could only be true on the 12% of rows where the sensor reported.
Black Ice fired on 1.4% of rows instead of about 10%, Dense Fog on 0.04% instead
of 0.36%, and the OR terms in Blizzard, Heavy Snow, Heavy Rain and Freezing Rain
made those classes depend on data availability rather than weather.

"Precipitation intensity" is complete (0.0% null at OL), and on the rows that
have both the old amount cuts agree with intensity cuts on 96-98% of rows. The
scenarios now read intensity only; these tests pin that the amount column is
irrelevant.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.preprocessing_pipeline import PreprocessingPipeline

WEATHER = {
    "Air temperature": 10.0, "Wind speed": 2.0, "Gust speed": 3.0,
    "Relative humidity": 60.0, "Dew-point temperature": 2.0,
    "Precipitation amount": np.nan, "Precipitation intensity": 0.0,
    "Snow depth": 0.0, "Horizontal visibility": 20000.0,
}


def _classify(rows):
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.current_file_id = "2023_01"
    frame = pd.DataFrame([{**WEATHER, **row} for row in rows])
    with patch("src.preprocessing_pipeline.save_dataframe_to_parquet",
               return_value="/tmp/fake.parquet"):
        out = pipeline.add_weather_scenarios_col(frame, month_id="2023_01")
    return out["weather_scenario"].tolist()


BLACK_ICE = {"Air temperature": 0.5, "Relative humidity": 90.0,
             "Dew-point temperature": 0.2}
DENSE_FOG = {"Horizontal visibility": 400.0, "Relative humidity": 98.0}


def test_black_ice_fires_when_the_amount_is_missing():
    assert _classify([{**BLACK_ICE, "Precipitation amount": np.nan}]) == ["Black Ice"]


def test_dense_fog_fires_when_the_amount_is_missing():
    assert _classify([{**DENSE_FOG, "Precipitation amount": np.nan}]) == ["Dense Fog"]


def test_active_precipitation_is_not_black_ice_or_dense_fog():
    # 2 mm/h is above both cuts; at 0.5 C it is Freezing Rain, in fog it is nothing special.
    rows = [{**BLACK_ICE, "Precipitation intensity": 2.0},
            {**DENSE_FOG, "Precipitation intensity": 2.0}]
    out = _classify(rows)
    assert out[0] == "Freezing Rain"
    assert out[1] == "Normal/Clear"


@pytest.mark.parametrize("amount", [np.nan, 0.0, 5.0])
def test_the_amount_column_does_not_change_any_scenario(amount):
    rows = [
        {**BLACK_ICE},
        {**DENSE_FOG},
        {"Air temperature": -3.0, "Precipitation intensity": 1.5, "Wind speed": 12.0,
         "Horizontal visibility": 500.0},                                  # Blizzard
        {"Air temperature": -3.0, "Precipitation intensity": 2.5, "Snow depth": 10.0},
        {"Air temperature": 8.0, "Precipitation intensity": 5.0},          # Heavy Rain
        {"Air temperature": 1.0, "Precipitation intensity": 0.8},          # Freezing Rain
    ]
    reference = _classify([{**r, "Precipitation amount": np.nan} for r in rows])
    with_amount = _classify([{**r, "Precipitation amount": amount} for r in rows])

    assert with_amount == reference
    assert reference == ["Black Ice", "Dense Fog", "Blizzard", "Heavy Snow",
                         "Heavy Rain", "Freezing Rain"]


def test_a_large_amount_alone_no_longer_creates_a_scenario():
    # amount > 10 used to make Heavy Rain by itself, but only on the 12% of rows
    # that report it; with zero intensity it must stay Normal/Clear.
    out = _classify([{"Air temperature": 8.0, "Precipitation amount": 12.0,
                      "Precipitation intensity": 0.0}])
    assert out == ["Normal/Clear"]
