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


import numpy as np

from src.fcll_sequences import OF_COLUMNS, assemble_sequences


def test_the_z_sequence_holds_the_five_stations_before_the_target():
    rows = []
    for i, st in enumerate(["A", "B", "C", "D", "E", "F", "OL"]):
        rows.append(_event(1, st, "ARRIVAL", f"10:{i*5:02d}", f"10:{i*5:02d}", i))
    visits = derive_station_visits(pd.DataFrame(rows))
    out = assemble_sequences(visits, target_station="OL", nf_columns=[], z=5, h=5)
    assert out["z_seq"].shape == (1, 5, len(OF_COLUMNS))
    assert out["z_mask"].sum() == 5, "all five slots are real, none padded"


def test_a_short_route_is_padded_and_masked_not_dropped():
    """A train originating three stations before OL is a real operating
    pattern. Dropping it would bias the set toward long-distance trains."""
    rows = [
        _event(1, "A", "ARRIVAL", "10:00", "10:00", 0),
        _event(1, "B", "ARRIVAL", "10:05", "10:05", 1),
        _event(1, "OL", "ARRIVAL", "10:10", "10:10", 2),
    ]
    visits = derive_station_visits(pd.DataFrame(rows))
    out = assemble_sequences(visits, target_station="OL", nf_columns=[], z=5, h=5)
    assert out["z_seq"].shape == (1, 5, len(OF_COLUMNS))
    assert out["z_mask"].sum() == 2, "only two real stations precede OL"
    assert np.all(out["z_seq"][0, out["z_mask"][0] == 0] == 0.0), "padding is zeros"


def test_the_z_sequence_is_ordered_oldest_to_newest():
    """LSTM-2 reads station order; reversing it trains a model that predicts
    the past from the future."""
    rows = [_event(1, st, "ARRIVAL", f"10:{i*5:02d}", f"10:{i*5+i:02d}", i)
            for i, st in enumerate(["A", "B", "OL"])]
    visits = derive_station_visits(pd.DataFrame(rows))
    out = assemble_sequences(visits, target_station="OL", nf_columns=[], z=5, h=5)
    delays = out["z_seq"][0, out["z_mask"][0] == 1, OF_COLUMNS.index("arrival_delay")]
    assert len(delays) == 2


def test_the_target_row_is_never_in_its_own_sequences():
    """The label is the target's arrival delay at OL. Including that row in
    z_seq hands the model the answer."""
    rows = [_event(1, st, "ARRIVAL", f"10:{i*5:02d}", f"10:{i*5:02d}", i)
            for i, st in enumerate(["A", "B", "OL"])]
    visits = derive_station_visits(pd.DataFrame(rows))
    out = assemble_sequences(visits, target_station="OL", nf_columns=[], z=5, h=5)
    assert out["z_mask"][0].sum() == 2, "OL itself must not appear in its own z_seq"


def test_only_target_station_arrivals_become_samples():
    """Two trains call at OL among four visits, so there are two samples.

    The two OL calls are two different trains on purpose: one train visiting a
    station twice in a day is not an operating pattern this data has, and the
    per-visit pivot collapses such a pair into one row by design.
    """
    rows = [
        _event(1, "A", "ARRIVAL", "10:00", "10:00", 0),
        _event(1, "OL", "ARRIVAL", "10:05", "10:05", 1),
        _event(2, "B", "ARRIVAL", "10:10", "10:10", 0),
        _event(2, "OL", "ARRIVAL", "10:15", "10:15", 1),
    ]
    visits = derive_station_visits(pd.DataFrame(rows))
    out = assemble_sequences(visits, target_station="OL", nf_columns=[], z=5, h=5)
    assert out["label"].shape[0] == 2
    # Train 2 follows train 1 into OL, so its H sequence has exactly one entry.
    assert out["h_mask"].sum() == 1


def test_non_operational_columns_are_carried_through_the_pivot():
    """Weather is joined onto the raw events. Without carry_columns the pivot
    drops every one of them and the FCNN branch gets a zero-width input."""
    rows = [_event(1, st, "ARRIVAL", f"10:{i*5:02d}", f"10:{i*5:02d}", i)
            for i, st in enumerate(["A", "OL"])]
    for i, row in enumerate(rows):
        row["Air temperature"] = -3.5 + i

    visits = derive_station_visits(pd.DataFrame(rows), carry_columns=["Air temperature"])
    assert "Air temperature" in visits.columns

    out = assemble_sequences(visits, target_station="OL",
                             nf_columns=["Air temperature"], z=5, h=5)
    assert out["nf"].shape == (1, 1)
    assert out["nf"][0, 0] == pytest.approx(-2.5)


def test_the_label_is_derived_when_the_raw_events_carry_none():
    """The raw parquet has no trainDelayed; preprocessing adds it downstream.
    Falling back to zeros would train the model on a constant."""
    from config.const_preprocessing import TRAIN_DELAY_MINUTES

    rows = [_event(1, st, "ARRIVAL", f"10:{i*5:02d}", f"10:{i*5:02d}", i)
            for i, st in enumerate(["A", "OL"])]
    rows[1]["differenceInMinutes"] = TRAIN_DELAY_MINUTES + 1.0

    visits = derive_station_visits(pd.DataFrame(rows))
    assert "trainDelayed" not in visits.columns
    out = assemble_sequences(visits, target_station="OL", nf_columns=[], z=5, h=5)
    assert out["label"].tolist() == [1.0]

    rows[1]["differenceInMinutes"] = float(TRAIN_DELAY_MINUTES)
    visits = derive_station_visits(pd.DataFrame(rows))
    out = assemble_sequences(visits, target_station="OL", nf_columns=[], z=5, h=5)
    assert out["label"].tolist() == [0.0], "the threshold is strictly greater-than"


def test_the_calendar_features_are_reproduced_from_the_raw_timestamp():
    """Every scenario lists month_sin/hour_sin/day_week_sin; none of the raw
    event files carry them, because preprocessing derives them in a stage
    fcll_net skips. Without this the FCNN branch is empty on scenario 2."""
    rows = [_event(1, "OL", "ARRIVAL", "10:00", "10:00", 0)]
    rows[0]["scheduledTime"] = "2018-03-14T09:30:00.000Z"
    rows[0]["actualTime"] = "2018-03-14T09:30:00.000Z"

    visits = derive_station_visits(pd.DataFrame(rows))
    row = visits.iloc[0]

    # 2018-03-14 is a Wednesday; add_temporal_features numbers 1=Sunday, so 4.
    assert row["month_sin"] == pytest.approx(np.sin(2 * np.pi * 2 / 12.0))
    assert row["hour_sin"] == pytest.approx(np.sin(2 * np.pi * 9.5 / 24.0))
    assert row["day_week_sin"] == pytest.approx(np.sin(2 * np.pi * 3 / 7.0))
    assert row["day_of_month"] == 14


# --- Prediction-boundary leakage -------------------------------------------
#
# The label is the target train's arrival delay at the target station. Any
# *_actual field whose measurement window extends past that arrival is not
# observable at prediction time, and hands the model the answer.

def _late_arrival_timetable():
    """A -> B -> OL, running five minutes late into OL."""
    rows = [
        _event(1, "A", "DEPARTURE", "10:00", "10:00", 0),
        _event(1, "B", "ARRIVAL",   "10:10", "10:10", 1),
        _event(1, "B", "DEPARTURE", "10:12", "10:12", 1),
        _event(1, "OL", "ARRIVAL",  "10:30", "10:38", 2),
    ]
    return pd.DataFrame(rows)


def test_run_actual_is_masked_at_the_last_station_before_the_target():
    """run_actual is 'this departure -> the NEXT station's arrival'. For the
    station immediately before the target, that next arrival IS the arrival
    being predicted, so the field measures the future.

    Concretely: arrival_delay_at_target == departure_delay_at_S
    + (run_actual_S - run_scheduled_S). Leaving run_actual in place makes the
    label an exact arithmetic function of z_seq's last timestep, which is
    worth about 0.999 ROC-AUC and means nothing.
    """
    visits = derive_station_visits(_late_arrival_timetable())
    out = assemble_sequences(visits, target_station="OL", nf_columns=[], z=5, h=5)
    run_actual = OF_COLUMNS.index("run_actual")

    assert out["z_mask"][0, -1] == 1.0, "the last slot must still be a real station"
    assert out["z_seq"][0, -1, run_actual] == 0.0, "run_actual crosses the boundary"


def test_run_scheduled_is_kept_at_the_last_station():
    """The timetable is known in advance, so the scheduled run is observable.
    Masking it too would throw away the baseline the delay is measured against."""
    visits = derive_station_visits(_late_arrival_timetable())
    out = assemble_sequences(visits, target_station="OL", nf_columns=[], z=5, h=5)
    run_scheduled = OF_COLUMNS.index("run_scheduled")
    assert out["z_seq"][0, -1, run_scheduled] == pytest.approx(18.0)


def test_earlier_stations_keep_their_actual_running_time():
    """Only the final hop crosses the boundary. B->OL is the future; A->B is
    not, and masking it would discard real upstream signal."""
    visits = derive_station_visits(_late_arrival_timetable())
    out = assemble_sequences(visits, target_station="OL", nf_columns=[], z=5, h=5)
    run_actual = OF_COLUMNS.index("run_actual")
    # slots -2 and -1 hold A and B; A->B ran 10:00 -> 10:10 = 10 minutes
    assert out["z_seq"][0, -2, run_actual] == pytest.approx(10.0)


def test_the_preceding_trains_post_departure_fields_are_masked():
    """h_seq rows are the preceding trains' visits AT the target station, so
    their dwell and onward run are measured after they arrived there — not
    guaranteed to have happened before the target train arrives."""
    rows = [
        _event(1, "OL", "ARRIVAL",   "10:00", "10:00", 0),
        _event(1, "OL", "DEPARTURE", "10:02", "10:09", 0),
        _event(2, "OL", "ARRIVAL",   "10:10", "10:10", 0),
    ]
    visits = derive_station_visits(pd.DataFrame(rows))
    out = assemble_sequences(visits, target_station="OL", nf_columns=[], z=5, h=5)
    for name in ("run_actual", "dwell_actual"):
        column = OF_COLUMNS.index(name)
        assert np.all(out["h_seq"][:, :, column] == 0.0), f"{name} is not observable"


def test_the_label_is_not_arithmetically_recoverable_from_the_sequences():
    """Regression guard on real data for the 0.9993-ROC-AUC leak.

    Runs the identity that reconstructed the label before masking. Agreement
    must fall well short of the near-perfect reconstruction it used to give.
    It stays meaningfully above the 68.6% all-negative baseline on purpose:
    departure_delay at the previous station really does predict arrival delay,
    is observable before the train reaches the target, and is the signal the
    model is supposed to learn. This asserts the arithmetic identity is gone,
    not that upstream delay stopped mattering.
    """
    import glob

    files = sorted(glob.glob("data/input/matched_data_flat_*.parquet"))
    if not files:
        pytest.skip("requires the local raw input parquet")

    from config.const_preprocessing import TARGET_STATION_CODE, TRAIN_DELAY_MINUTES

    out = assemble_sequences(
        derive_station_visits(pd.read_parquet(files[0])),
        target_station=TARGET_STATION_CODE, nf_columns=[],
    )
    z, mask, label = out["z_seq"], out["z_mask"], out["label"]
    idx = {c: k for k, c in enumerate(OF_COLUMNS)}

    assert np.abs(z[:, -1, idx["run_actual"]]).max() == 0.0
    assert np.abs(out["h_seq"][:, :, idx["run_actual"]]).max() == 0.0
    assert np.abs(out["h_seq"][:, :, idx["dwell_actual"]]).max() == 0.0

    has_history = mask[:, -1] == 1
    last = z[has_history, -1, :]
    implied = (last[:, idx["departure_delay"]]
               + last[:, idx["run_actual"]] - last[:, idx["run_scheduled"]])
    agreement = float(((implied > TRAIN_DELAY_MINUTES) == label[has_history]).mean())
    assert agreement < 0.90, (
        f"the label is {agreement:.1%} recoverable by arithmetic on one timestep"
    )
