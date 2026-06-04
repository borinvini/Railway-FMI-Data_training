import numpy as np
import pandas as pd
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from src.preprocessing_pipeline import PreprocessingPipeline


@contextmanager
def _null_logger(*args, **kwargs):
    yield MagicMock()


def _make_pipeline(tmp_path):
    """Instantiate pipeline without running __init__ (avoids filesystem setup)."""
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.log_dir = str(tmp_path)
    pipeline.current_file_id = "2023_01"
    return pipeline


# ---------------------------------------------------------------------------
# Fix 1
# ---------------------------------------------------------------------------

@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_drop_month_zero_rows(mock_save, tmp_path):
    """Rows where actualTime cannot be parsed must be dropped (month=0 → removed)."""
    pipeline = _make_pipeline(tmp_path)

    df = pd.DataFrame({
        "scheduledTime": [
            "2023-01-15T10:00:00.000Z",
            "2023-01-16T08:00:00.000Z",
            "2023-01-17T09:00:00.000Z",
        ],
        "actualTime": [
            "2023-01-15T10:05:00.000Z",  # valid → month=1
            "2023-01-16T08:03:00.000Z",  # valid → month=1
            "NOT_A_DATE",                 # invalid → month=0 → must be dropped
        ],
    })

    with patch.object(pipeline, "get_logger", _null_logger):
        result = pipeline.process_actual_time_column(dataframe=df, month_id="2023_01")

    assert result is not None
    assert len(result) == 2, f"Expected 2 rows after dropping bad date row, got {len(result)}"
    assert (result["month"] != 0).all(), "No row should have month=0 after fix"


# ---------------------------------------------------------------------------
# Fix 2
# ---------------------------------------------------------------------------

@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_coerce_target_columns_to_numeric(mock_save, tmp_path):
    """differenceInMinutes (object with timestamps) and differenceInMinutes_offset
    (object with string floats) must be coerced to float64. Rows where coercion
    produces NaN must be dropped by the existing dropna step."""
    pipeline = _make_pipeline(tmp_path)

    df = pd.DataFrame({
        # Two numeric strings + one ISO timestamp that will coerce to NaN
        "differenceInMinutes": ["5.0", "3.0", "2018-03-28T02:28:20.000Z"],
        "differenceInMinutes_offset": ["1.0", "-2.0", "0.0"],   # all numeric strings
        "differenceInMinutes_eachStation_offset": [5.0, -2.0, 0.0],
        "trainDelayed": [True, False, False],
        "cancelled": [False, False, False],
        # Temporal (required by imputation path)
        "month": [1, 1, 1],
        # Minimum weather columns to pass the weather-check in handle_missing_values
        "Air temperature": [2.0, 3.0, 1.0],
        "Wind speed": [5.0, 4.0, 6.0],
        "Gust speed": [8.0, 7.0, 9.0],
        "Wind direction": [180.0, 90.0, 270.0],
        "Relative humidity": [75.0, 80.0, 70.0],
        "Dew-point temperature": [0.0, 1.0, -1.0],
        "Precipitation intensity": [0.0, 0.1, 0.0],
        "Snow depth": [0.0, 0.0, 5.0],
        "Pressure (msl)": [1013.0, 1010.0, 1015.0],
        "Horizontal visibility": [10000.0, 8000.0, 9000.0],
        "Cloud amount": [2.0, 4.0, 6.0],
    })

    with patch.object(pipeline, "get_logger", _null_logger):
        result = pipeline.handle_missing_values(dataframe=df, month_id="2023_01")

    assert result is not None
    assert result["differenceInMinutes"].dtype in (np.float64, np.float32), (
        f"Expected float dtype for differenceInMinutes, got {result['differenceInMinutes'].dtype}"
    )
    assert result["differenceInMinutes_offset"].dtype in (np.float64, np.float32), (
        f"Expected float dtype for differenceInMinutes_offset, got {result['differenceInMinutes_offset'].dtype}"
    )
    # Row 3 had a timestamp in differenceInMinutes → coerced to NaN → dropped
    assert len(result) == 2, f"Expected 2 rows after dropping non-numeric target row, got {len(result)}"


@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_coerce_does_not_drop_string_boolean_columns(mock_save, tmp_path):
    """trainDelayed and cancelled as string-encoded booleans must NOT be coerced to NaN."""
    pipeline = _make_pipeline(tmp_path)

    df = pd.DataFrame({
        "differenceInMinutes": ["5.0", "3.0"],
        "differenceInMinutes_offset": ["1.0", "-2.0"],
        "differenceInMinutes_eachStation_offset": [5.0, -2.0],
        # String-encoded booleans — realistic from CSV or some parquet reads
        "trainDelayed": pd.Series(["True", "False"], dtype=object),
        "cancelled": pd.Series(["False", "False"], dtype=object),
        "month": [1, 1],
        "Air temperature": [2.0, 3.0],
        "Wind speed": [5.0, 4.0],
        "Gust speed": [8.0, 7.0],
        "Wind direction": [180.0, 90.0],
        "Relative humidity": [75.0, 80.0],
        "Dew-point temperature": [0.0, 1.0],
        "Precipitation intensity": [0.0, 0.1],
        "Snow depth": [0.0, 0.0],
        "Pressure (msl)": [1013.0, 1010.0],
        "Horizontal visibility": [10000.0, 8000.0],
        "Cloud amount": [2.0, 4.0],
    })

    with patch.object(pipeline, "get_logger", _null_logger):
        result = pipeline.handle_missing_values(dataframe=df, month_id="2023_01")

    assert result is not None
    assert len(result) == 2, (
        f"Expected all 2 rows retained (string booleans must not be coerced to NaN), got {len(result)}"
    )


# ---------------------------------------------------------------------------
# Fix 3
# ---------------------------------------------------------------------------

@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_filter_columns_keeps_rolling_windows(mock_save, tmp_path):
    """filter_columns must retain columns whose names contain (12h, (24h, or (72h."""
    pipeline = _make_pipeline(tmp_path)

    df = pd.DataFrame({
        # Target features
        "differenceInMinutes_eachStation_offset": [1.0, 2.0],
        "differenceInMinutes": [1.0, 2.0],
        "differenceInMinutes_offset": [1.0, 2.0],
        "trainDelayed": [1, 0],
        "cancelled": [0, 0],
        # Train prediction features
        "trainStopping": [1, 1],
        "commercialStop": [1, 1],
        "month": [1, 2],
        "hour": ["10:00", "11:00"],
        "day_of_week": [2, 3],
        "day_of_month": [15, 16],
        "causes_related_to_weather": [0, 1],
        # Basic weather
        "Air temperature": [2.0, 3.0],
        "Wind speed": [5.0, 4.0],
        # Rolling window columns — these MUST be kept after the fix
        "Air temperature (12h max)": [4.0, 5.0],
        "Air temperature (24h mean)": [3.0, 4.0],
        "Snow depth (72h max)": [10.0, 15.0],
        "Precipitation amount (12h cumulative)": [0.5, 0.0],
        # Columns that must be dropped
        "stationShortCode": ["OL", "OL"],
        "trainNumber": [123, 456],
    })

    with patch.object(pipeline, "get_logger", _null_logger):
        result = pipeline.filter_columns(dataframe=df, month_id="2023_01")

    assert result is not None
    assert "Air temperature (12h max)" in result.columns, "12h window column must be kept"
    assert "Air temperature (24h mean)" in result.columns, "24h window column must be kept"
    assert "Snow depth (72h max)" in result.columns, "72h window column must be kept"
    assert "Precipitation amount (12h cumulative)" in result.columns, "12h cumulative must be kept"
    assert "stationShortCode" not in result.columns, "stationShortCode must be dropped"
    assert "trainNumber" not in result.columns, "trainNumber must be dropped"
