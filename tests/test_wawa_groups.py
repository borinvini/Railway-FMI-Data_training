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
