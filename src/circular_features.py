"""Circular encodings shared by preprocessing and FCLL-Net.

One definition, imported by both `PreprocessingPipeline.convert_winddirection_to_sincos`
and `fcll_sequences.derive_station_visits`, so the two paths cannot drift apart.
"""
import numpy as np
import pandas as pd

from config.const_preprocessing import WIND_CALM_DIRECTION


def wind_direction_sincos(direction):
    """(sin, cos) of a wind direction in degrees, as two Series with the input's index.

    North (0 / 360 degrees) is (0, 1) and east (90) is (1, 0), so 359 and 1 degrees end
    up next to each other instead of 358 apart. FMI uses 0 for calm (wind speed is 0.00 on
    every such row), so a direction of WIND_CALM_DIRECTION maps to (0, 0), the
    no-direction point, rather than to north. A missing direction stays NaN: the
    train-only imputation at the split fills it, never this function.
    """
    degrees = pd.Series(direction).astype("float64")
    angle = np.deg2rad(degrees)
    calm = degrees == WIND_CALM_DIRECTION
    return np.sin(angle).mask(calm, 0.0), np.cos(angle).mask(calm, 0.0)
