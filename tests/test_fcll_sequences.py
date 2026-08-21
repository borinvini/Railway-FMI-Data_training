"""Fixture-driven tests for the sequence builder.

A mis-sorted or mis-paired sequence trains perfectly well and means nothing,
so every derived quantity is pinned against a hand-built timetable where the
right answer is known by construction."""
import pandas as pd
import pytest

from src.fcll_sequences import derive_station_visits


def _event(train, station, kind, sched, actual, order):
    return {
        "trainNumber": train,
        "departureDate": "2018-01-01",
        "stationShortCode": station,
        "type": kind,
        "scheduledTime": f"2018-01-01T{sched}:00.000Z",
        "actualTime": f"2018-01-01T{actual}:00.000Z",
        "differenceInMinutes": 0.0,
        "trainStopping": True,
        "commercialStop": True,
    }


@pytest.fixture
def timetable():
    """Train 1: A(dep 10:00) -> B(arr 10:10, dep 10:12) -> C(arr 10:30).
    Runs two minutes late out of A, recovers one by C."""
    rows = [
        _event(1, "A", "DEPARTURE", "10:00", "10:02", 0),
        _event(1, "B", "ARRIVAL",   "10:10", "10:12", 1),
        _event(1, "B", "DEPARTURE", "10:12", "10:14", 1),
        _event(1, "C", "ARRIVAL",   "10:30", "10:31", 2),
    ]
    return pd.DataFrame(rows)


def test_dwell_is_departure_minus_arrival_at_the_same_station(timetable):
    visits = derive_station_visits(timetable)
    b = visits[visits["stationShortCode"] == "B"].iloc[0]
    assert b["dwell_scheduled"] == pytest.approx(2.0)
    assert b["dwell_actual"] == pytest.approx(2.0)


def test_running_time_is_next_arrival_minus_this_departure(timetable):
    visits = derive_station_visits(timetable)
    a = visits[visits["stationShortCode"] == "A"].iloc[0]
    assert a["run_scheduled"] == pytest.approx(10.0)
    assert a["run_actual"] == pytest.approx(10.0)


def test_station_order_follows_scheduled_time(timetable):
    visits = derive_station_visits(timetable).sort_values("station_order")
    assert list(visits["stationShortCode"]) == ["A", "B", "C"]


def test_a_terminating_visit_has_no_running_time(timetable):
    """C is the last station; run_* must be NaN, not zero. Zero would read as
    'arrived instantly' and is a real value the model would learn from."""
    visits = derive_station_visits(timetable)
    c = visits[visits["stationShortCode"] == "C"].iloc[0]
    assert pd.isna(c["run_actual"])
    assert pd.isna(c["run_scheduled"])


def test_headway_is_measured_against_the_previous_train_at_the_station():
    """Train 2 arrives at B six minutes after train 1."""
    rows = [
        _event(1, "B", "ARRIVAL", "10:10", "10:10", 0),
        _event(2, "B", "ARRIVAL", "10:16", "10:18", 0),
    ]
    visits = derive_station_visits(pd.DataFrame(rows))
    second = visits[visits["trainNumber"] == 2].iloc[0]
    assert second["headway_scheduled"] == pytest.approx(6.0)
    assert second["headway_actual"] == pytest.approx(8.0)
