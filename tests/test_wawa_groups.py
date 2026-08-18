import numpy as np
import pandas as pd
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from config.const_preprocessing import (
    WAWA_CODE_TO_GROUP,
    WAWA_EMITTED_GROUPS,
    WAWA_NON_EMITTED_GROUPS,
    WAWA_FALLBACK_GROUP,
    WAWA_SOURCE_COLUMN,
    VALID_WAWA_FEATURES,
)


@contextmanager
def _null_logger(*args, **kwargs):
    yield MagicMock()


def _make_pipeline(tmp_path):
    """Instantiate pipeline without running __init__ (avoids filesystem setup)."""
    from src.preprocessing_pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.log_dir = str(tmp_path)
    pipeline.current_file_id = "2023_01"
    return pipeline


# ---------------------------------------------------------------------------
# Mapping table
# ---------------------------------------------------------------------------

def test_every_code_0_to_99_mapped_exactly_once():
    """No WMO 4680 code may fall through unassigned, and none may be listed twice."""
    assert sorted(WAWA_CODE_TO_GROUP.keys()) == list(range(100))


def test_all_mapped_groups_are_declared():
    known = set(WAWA_EMITTED_GROUPS) | set(WAWA_NON_EMITTED_GROUPS)
    unknown = set(WAWA_CODE_TO_GROUP.values()) - known
    assert not unknown, f"mapping produced undeclared groups: {unknown}"


def test_emitted_and_non_emitted_groups_are_disjoint():
    overlap = set(WAWA_EMITTED_GROUPS) & set(WAWA_NON_EMITTED_GROUPS)
    assert not overlap, f"group declared both emitted and non-emitted: {overlap}"


def test_emitted_group_order_is_fixed():
    """Column order must be stable across month files, so this list is frozen."""
    assert WAWA_EMITTED_GROUPS == [
        'clear', 'haze', 'mist', 'fog', 'precip_unspec', 'drizzle', 'rain',
        'freezing', 'sleet', 'snow', 'ice_particles', 'rain_shower',
        'snow_shower', 'hail',
    ]
    assert len(set(WAWA_EMITTED_GROUPS)) == 14


@pytest.mark.parametrize("code,group", [
    (0, 'clear'),           # No significant weather observed
    (4, 'haze'),            # Haze/smoke/dust, visibility >= 1 km
    (10, 'mist'),           # Mist
    (11, 'ice_particles'),  # Diamond dust
    (18, 'squall'),         # Squalls
    (20, 'fog'),            # Fog during the preceding hour
    (21, 'precip_unspec'),  # PRECIPITATION during the preceding hour
    (25, 'freezing'),       # Freezing drizzle or freezing rain, preceding hour
    (28, 'blowing_snow'),   # Blowing/drifting snow, visibility >= 1 km
    (33, 'fog'),            # Fog or ice fog, no appreciable change
    (42, 'precip_unspec'),  # Precipitation, heavy
    (46, 'snow'),           # Solid precipitation, heavy
    (57, 'drizzle'),        # Drizzle and rain, slight
    (63, 'rain'),           # Rain, not freezing, heavy
    (66, 'freezing'),       # Rain, freezing, heavy
    (67, 'sleet'),          # Rain (or drizzle) and snow, slight
    (71, 'snow'),           # Snow, slight
    (77, 'ice_particles'),  # Snow grains
    (82, 'rain_shower'),    # Rain shower(s), moderate
    (86, 'snow_shower'),    # Snow shower(s), moderate
    (89, 'hail'),           # Hail
    (93, 'thunder'),        # Thunderstorm, slight or moderate, with hail
    (99, 'tornado'),        # Tornado
])
def test_spot_checks_against_wmo_table(code, group):
    assert WAWA_CODE_TO_GROUP[code] == group


def test_one_hot_column_names():
    assert VALID_WAWA_FEATURES == [f'wawa_group_{g}' for g in WAWA_EMITTED_GROUPS]
    assert VALID_WAWA_FEATURES[0] == 'wawa_group_clear'
    assert VALID_WAWA_FEATURES[-1] == 'wawa_group_hail'


def test_fallback_group_is_emitted():
    """Folding into a group that is never emitted would lose those rows entirely."""
    assert WAWA_FALLBACK_GROUP in WAWA_EMITTED_GROUPS


def test_source_column_name():
    assert WAWA_SOURCE_COLUMN == 'Present weather (auto)'


# ---------------------------------------------------------------------------
# add_wawa_group_col
# ---------------------------------------------------------------------------

@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_float_codes_map_as_integers(mock_save, tmp_path):
    """The source column is an Arrow double: codes arrive as 71.0, not 71."""
    pipeline = _make_pipeline(tmp_path)
    df = pd.DataFrame({WAWA_SOURCE_COLUMN: [71.0, 61.0, 0.0, 89.0]})

    result = pipeline.add_wawa_group_col(dataframe=df, month_id="2023_01")

    assert result is not None
    assert list(result["wawa_group"]) == ["snow", "rain", "clear", "hail"]


@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_missing_and_invalid_codes_fold_to_clear(mock_save, tmp_path):
    """NaN, non-numeric, out-of-range and non-integral values all fall back."""
    pipeline = _make_pipeline(tmp_path)
    df = pd.DataFrame({WAWA_SOURCE_COLUMN: [np.nan, 150.0, -1.0, 71.5]})

    result = pipeline.add_wawa_group_col(dataframe=df, month_id="2023_01")

    assert list(result["wawa_group"]) == ["clear"] * 4


@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_non_emitted_groups_fold_to_clear(mock_save, tmp_path):
    """thunder (93), blowing_snow (28), squall (18) and tornado (99) are mapped
    but never emitted, so they must land in the fallback group."""
    pipeline = _make_pipeline(tmp_path)
    df = pd.DataFrame({WAWA_SOURCE_COLUMN: [93.0, 28.0, 18.0, 99.0, 71.0]})

    result = pipeline.add_wawa_group_col(dataframe=df, month_id="2023_01")

    assert list(result["wawa_group"]) == ["clear", "clear", "clear", "clear", "snow"]


@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_missing_source_column_does_not_abort(mock_save, tmp_path):
    """A month file without the wawa column must not cost the whole run."""
    pipeline = _make_pipeline(tmp_path)
    df = pd.DataFrame({"Air temperature": [1.0, 2.0]})

    result = pipeline.add_wawa_group_col(dataframe=df, month_id="2023_01")

    assert result is not None
    assert list(result["wawa_group"]) == ["clear", "clear"]


@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_source_column_is_not_dropped_by_this_stage(mock_save, tmp_path):
    """filter_columns drops it later; this stage only reads it."""
    pipeline = _make_pipeline(tmp_path)
    df = pd.DataFrame({WAWA_SOURCE_COLUMN: [71.0], "Air temperature": [1.0]})

    result = pipeline.add_wawa_group_col(dataframe=df, month_id="2023_01")

    assert WAWA_SOURCE_COLUMN in result.columns
    assert "Air temperature" in result.columns


@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_input_dataframe_is_not_mutated(mock_save, tmp_path):
    pipeline = _make_pipeline(tmp_path)
    df = pd.DataFrame({WAWA_SOURCE_COLUMN: [71.0]})

    pipeline.add_wawa_group_col(dataframe=df, month_id="2023_01")

    assert "wawa_group" not in df.columns


@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_non_emitted_counts_are_reported(mock_save, tmp_path, capsys):
    """The zero-occurrence claim is inherited, not re-verified, so a station
    that does report thunder must show up in the stage output."""
    pipeline = _make_pipeline(tmp_path)
    df = pd.DataFrame({WAWA_SOURCE_COLUMN: [93.0, 93.0, 71.0]})

    pipeline.add_wawa_group_col(dataframe=df, month_id="2023_01")

    out = capsys.readouterr().out
    thunder_lines = [ln for ln in out.splitlines() if "thunder" in ln]
    assert thunder_lines, "the non-emitted thunder count must be reported"
    assert any("2" in ln for ln in thunder_lines), f"expected a count of 2, got {thunder_lines}"


@patch("src.preprocessing_pipeline.save_dataframe_to_parquet", return_value="/tmp/fake.parquet")
def test_empty_dataframe_is_handled(mock_save, tmp_path):
    """A month can filter down to zero rows upstream; that must not raise."""
    pipeline = _make_pipeline(tmp_path)
    df = pd.DataFrame({WAWA_SOURCE_COLUMN: pd.Series([], dtype=float)})

    result = pipeline.add_wawa_group_col(dataframe=df, month_id="2023_01")

    assert result is not None
    assert len(result) == 0
    assert "wawa_group" in result.columns
