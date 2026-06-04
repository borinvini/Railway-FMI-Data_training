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
