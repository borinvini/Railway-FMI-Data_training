"""Calendar features must come from the SCHEDULED time, never the actual time.

actualTime is the scheduled time plus the delay. process_actual_time_column used
to derive hour (as HH:MM), weekday and day of month from it, so the clock time of
an actual arrival handed the model the label: timetable slots are fixed, so
"arrived at 14:07" reads as "seven minutes late". On the 117,868 rows at OL a
single time-of-day feature scored ROC-AUC 0.775 from actualTime against 0.706
from scheduledTime, a gap that comes from the delay and nothing else.

The optional convert_scheduled_time_to_local stage is a separate, smaller
matter: every timestamp is UTC already, but timetables follow the local clock, so
a fixed slot is 06:00 UTC in winter and 05:00 UTC in summer. The stage keeps the
slot at one hour all year; its effect on score was within +-0.008 AUC and changed
sign with the encoding, so it is optional and off by default. The FCLL path
mirrors preprocessing, so both must agree under either setting.
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.const_preprocessing import PREPROCESSING_STATE_MACHINE
from src.fcll_sequences import derive_station_visits
from src.preprocessing_pipeline import PreprocessingPipeline

STAGE = "convert_scheduled_time_to_local"


@contextmanager
def _null_logger(*args, **kwargs):
    yield MagicMock()


def _pipeline(tmp_path):
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.log_dir = str(tmp_path)
    pipeline.current_file_id = "2023_01"
    return pipeline


@pytest.fixture(autouse=True)
def _no_parquet_writes():
    with patch("src.preprocessing_pipeline.save_dataframe_to_parquet",
               return_value="/tmp/fake.parquet"):
        yield


def _features(pipeline, frame, local):
    """Run the (optional) local-time stage and the calendar stage on a frame."""
    if local:
        frame = pipeline.convert_scheduled_time_to_local(dataframe=frame, month_id="2023_01")
        assert frame is not None
    with patch.object(pipeline, "get_logger", _null_logger):
        return pipeline.process_actual_time_column(dataframe=frame, month_id="2023_01")


def test_a_late_train_gets_the_same_calendar_features_as_an_on_time_one(tmp_path):
    frame = pd.DataFrame({
        "scheduledTime": ["2023-03-14T08:50:00.000Z", "2023-03-14T08:50:00.000Z"],
        "actualTime":    ["2023-03-14T08:50:00.000Z", "2023-03-14T09:25:00.000Z"],
    })

    out = _features(_pipeline(tmp_path), frame, local=False)

    # 35 minutes of delay moves actualTime into the next hour; the schedule did not move.
    assert out["hour"].tolist() == ["08:50", "08:50"]
    assert out["day_of_month"].nunique() == 1


def test_a_missing_actual_time_no_longer_drops_the_row(tmp_path):
    frame = pd.DataFrame({
        "scheduledTime": ["2023-03-14T08:50:00.000Z", "2023-03-14T09:50:00.000Z"],
        "actualTime":    ["2023-03-14T08:55:00.000Z", None],
    })

    out = _features(_pipeline(tmp_path), frame, local=False)

    assert len(out) == 2, "the features do not depend on actualTime, so the row stays"


def test_without_the_stage_a_fixed_slot_shifts_by_an_hour_across_dst(tmp_path):
    # 08:00 Helsinki time: 06:00Z in January (UTC+2), 05:00Z in July (UTC+3).
    frame = pd.DataFrame({
        "scheduledTime": ["2023-01-16T06:00:00.000Z", "2023-07-17T05:00:00.000Z"],
        "actualTime":    ["2023-01-16T06:00:00.000Z", "2023-07-17T05:00:00.000Z"],
    })

    out = _features(_pipeline(tmp_path), frame, local=False)

    assert out["hour"].tolist() == ["06:00", "05:00"]


def test_with_the_stage_a_fixed_slot_keeps_one_hour_all_year(tmp_path):
    frame = pd.DataFrame({
        "scheduledTime": ["2023-01-16T06:00:00.000Z", "2023-07-17T05:00:00.000Z"],
        "actualTime":    ["2023-01-16T06:00:00.000Z", "2023-07-17T05:00:00.000Z"],
    })

    out = _features(_pipeline(tmp_path), frame, local=True)

    assert out["hour"].tolist() == ["08:00", "08:00"]


def test_the_stage_moves_a_late_evening_utc_time_to_the_next_local_day(tmp_path):
    # Saturday 2023-01-14 22:30Z is Sunday 2023-01-15 00:30 in Helsinki.
    frame = pd.DataFrame({
        "scheduledTime": ["2023-01-14T22:30:00.000Z"],
        "actualTime":    ["2023-01-14T22:30:00.000Z"],
    })
    pipeline = _pipeline(tmp_path)

    utc = _features(pipeline, frame, local=False).iloc[0]
    local = _features(pipeline, frame, local=True).iloc[0]

    assert (utc["day_of_week"], utc["day_of_month"], utc["hour"]) == (7, 14, "22:30")
    assert (local["day_of_week"], local["day_of_month"], local["hour"]) == (1, 15, "00:30")


def test_the_stage_adds_a_column_and_leaves_the_utc_original_alone(tmp_path):
    frame = pd.DataFrame({"scheduledTime": ["2023-07-17T05:00:00.000Z"]})

    out = _pipeline(tmp_path).convert_scheduled_time_to_local(dataframe=frame, month_id="2023_07")

    assert out["scheduledTime"].tolist() == ["2023-07-17T05:00:00.000Z"]
    assert out["scheduledTime_local"].iloc[0] == pd.Timestamp("2023-07-17 08:00:00")


def test_the_stage_fails_loudly_when_its_source_column_is_missing(tmp_path):
    frame = pd.DataFrame({"actualTime": ["2023-07-17T05:00:00.000Z"]})

    out = _pipeline(tmp_path).convert_scheduled_time_to_local(dataframe=frame, month_id="2023_07")

    assert out is None, "returning the frame would silently leave UTC features in place"


def test_the_state_machine_key_exists_and_defaults_off():
    assert PREPROCESSING_STATE_MACHINE[STAGE] is False
    keys = list(PREPROCESSING_STATE_MACHINE)
    assert keys.index(STAGE) < keys.index("process_actual_time_column"), (
        "the stage must run before the stage that reads its column")


# --- FCLL parity -----------------------------------------------------------

def _fcll_row(sched, actual):
    return {
        "trainNumber": 1, "departureDate": "2023-01-14", "stationShortCode": "OL",
        "type": "ARRIVAL", "scheduledTime": sched, "actualTime": actual,
        "differenceInMinutes": 0.0, "trainStopping": True, "commercialStop": True,
    }


@pytest.mark.parametrize("local", [False, True])
def test_fcll_and_preprocessing_derive_the_same_calendar_features(tmp_path, monkeypatch, local):
    monkeypatch.setitem(PREPROCESSING_STATE_MACHINE, STAGE, local)
    sched, actual = "2023-01-14T22:30:00.000Z", "2023-01-14T23:10:00.000Z"

    visits = derive_station_visits(pd.DataFrame([_fcll_row(sched, actual)])).iloc[0]
    pre = _features(
        _pipeline(tmp_path),
        pd.DataFrame({"scheduledTime": [sched], "actualTime": [actual]}),
        local=local,
    ).iloc[0]

    hh, mm = (int(x) for x in pre["hour"].split(":"))
    assert visits["hour_sin"] == pytest.approx(np.sin(2 * np.pi * (hh + mm / 60.0) / 24.0))
    assert visits["hour_cos"] == pytest.approx(np.cos(2 * np.pi * (hh + mm / 60.0) / 24.0))
    assert visits["day_of_month"] == pre["day_of_month"]
    assert visits["month_sin"] == pytest.approx(np.sin(2 * np.pi * (pre["month"] - 1) / 12.0))
    assert visits["day_week_sin"] == pytest.approx(
        np.sin(2 * np.pi * (pre["day_of_week"] - 1) / 7.0))


def test_fcll_calendar_features_ignore_the_delay():
    on_time = _fcll_row("2023-03-14T08:50:00.000Z", "2023-03-14T08:50:00.000Z")
    late = _fcll_row("2023-03-14T08:50:00.000Z", "2023-03-14T09:25:00.000Z")

    a = derive_station_visits(pd.DataFrame([on_time])).iloc[0]
    b = derive_station_visits(pd.DataFrame([late])).iloc[0]

    assert a["hour_sin"] == pytest.approx(b["hour_sin"])
    assert a["hour_cos"] == pytest.approx(b["hour_cos"])
