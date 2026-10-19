"""Turn event-level train records into the per-visit operational features that
FCLL-Net's two LSTMs consume.

Reads the raw matched parquet rather than 101-preprocessed_training_ready,
because that file has already been filtered to one station: the sequences this
module builds no longer exist in it.

Huang et al. (2020) define nine operational variables per station visit. All
nine are derived here. The ninth, "number of stops in the section", cannot be
taken literally on this data: every station the train passes is already its own
row, so the count of stations between two consecutive rows is always zero. It is
therefore read as "does the train actually stop in the section ending here",
from trainStopping/commercialStop — the only stop information the raw events
carry at per-visit granularity.
"""
import numpy as np
import pandas as pd

from config.const_preprocessing import (
    DAY_OF_MONTH_CYCLE,
    LOCAL_TIMEZONE,
    PREPROCESSING_STATE_MACHINE,
    TIME_FEATURE_SOURCE_COLUMN,
    TRAIN_DELAYED_TARGET_COLUMN,
    TRAIN_DELAY_MINUTES,
    WAWA_CODE_TO_GROUP,
    WAWA_EMITTED_GROUPS,
    WAWA_FALLBACK_GROUP,
    WAWA_SOURCE_COLUMN,
    WIND_DIRECTION_COLUMN,
)
from src.circular_features import wind_direction_sincos

OF_COLUMNS = [
    "arrival_delay", "departure_delay",
    "dwell_actual", "dwell_scheduled",
    "run_actual", "run_scheduled",
    "headway_actual", "headway_scheduled",
    "stops_in_section",
]


def _to_datetime(series):
    return pd.to_datetime(series, format="ISO8601", utc=True)


def derive_station_visits(df, carry_columns=None):
    """Collapse ARRIVAL/DEPARTURE event pairs into one row per station visit.

    `carry_columns` names non-operational columns — weather, calendar — to keep
    alongside the derived features. They are joined onto the raw events, so the
    pivot would otherwise drop them, and without them the FCNN branch has no
    input at all. The ARRIVAL row's value is taken, because the label is about
    the arrival.
    """
    df = df.copy()
    df["_sched"] = _to_datetime(df["scheduledTime"])
    df["_actual"] = _to_datetime(df["actualTime"])

    # The sin/cos of the wind direction is a preprocessing column the raw events lack.
    # Build it before the carry filter below: build_fcll_sequences silently drops any
    # requested feature name that is not in the visits, so a missing wind_dir_* would
    # vanish from the model without a warning.
    if WIND_DIRECTION_COLUMN in df.columns:
        df["wind_dir_sin"], df["wind_dir_cos"] = wind_direction_sincos(df[WIND_DIRECTION_COLUMN])

    keys =["trainNumber", "departureDate", "stationShortCode"]
    carry_columns = [c for c in (carry_columns or []) if c in df.columns and c not in keys]
    wide = df.pivot_table(
        index=keys, columns="type",
        values=["_sched", "_actual", "differenceInMinutes"],
        aggfunc="first",
    )
    wide.columns = [f"{a}_{b}".lower() for a, b in wide.columns]
    wide = wide.reset_index()

    def col(name):
        if name in wide.columns:
            return wide[name]
        return pd.Series(pd.NaT, index=wide.index, dtype="datetime64[ns, UTC]")

    def minutes(a, b):
        return (col(a) - col(b)).dt.total_seconds() / 60.0

    wide["dwell_actual"] = minutes("_actual_departure", "_actual_arrival")
    wide["dwell_scheduled"] = minutes("_sched_departure", "_sched_arrival")
    wide["arrival_delay"] = wide.get("differenceinminutes_arrival", np.nan)
    wide["departure_delay"] = wide.get("differenceinminutes_departure", np.nan)

    # Order each train's visits by scheduled time. The visit sort key is the
    # arrival where there is one and the departure at the originating station.
    # visit_time is the scheduled moment this visit happens: the arrival where
    # there is one, the departure at the originating station. It survives into
    # the output because assemble_sequences needs it to order the trains through
    # the target station — train number is not a time order.
    wide["visit_time"] = col("_sched_arrival").fillna(col("_sched_departure"))
    wide = wide.sort_values(["trainNumber", "departureDate", "visit_time"])
    wide["station_order"] = wide.groupby(["trainNumber", "departureDate"]).cumcount()

    # Running time: this departure to the NEXT station's arrival, within one
    # train-day. shift(-1) across the group boundary would splice two journeys.
    grp = wide.groupby(["trainNumber", "departureDate"])
    for kind, name in (("_actual", "run_actual"), ("_sched", "run_scheduled")):
        arrival, departure = f"{kind}_arrival", f"{kind}_departure"
        if arrival not in wide.columns or departure not in wide.columns:
            wide[name] = np.nan
            continue
        nxt = grp[arrival].shift(-1)
        wide[name] = (nxt - wide[departure]).dt.total_seconds() / 60.0

    # Headway: gap to the previous train at the SAME station, by scheduled order.
    wide = wide.sort_values(["stationShortCode", "visit_time"])
    station = wide.groupby("stationShortCode")
    for kind, name in (("_sched_arrival", "headway_scheduled"),
                       ("_actual_arrival", "headway_actual")):
        if kind not in wide.columns:
            wide[name] = np.nan
            continue
        wide[name] = station[kind].diff().dt.total_seconds() / 60.0

    # See the module docstring: read as "the train stops in the section ending
    # at this station", which is the only stop signal the raw events carry.
    stopping = [c for c in ("trainStopping", "commercialStop") if c in df.columns]
    if stopping:
        flags = df.groupby(keys, sort=False)[stopping].max().reset_index()
        flags["stops_in_section"] = (
            flags[stopping].astype(float).min(axis=1)
        )
        wide = wide.merge(flags[keys + ["stops_in_section"]], on=keys, how="left")
    else:
        wide["stops_in_section"] = np.nan

    # Calendar features come from the SCHEDULED time, as in preprocessing: the
    # actual time carries the delay and would hand the model the label.
    calendar_kind = "_sched" if TIME_FEATURE_SOURCE_COLUMN == "scheduledTime" else "_actual"
    wide = _add_temporal_features(
        wide, col(f"{calendar_kind}_arrival").fillna(col(f"{calendar_kind}_departure"))
    )
    wide = _add_wawa_groups(wide, df, keys)

    if carry_columns:
        # Prefer the ARRIVAL row's values; fall back to the DEPARTURE row for
        # visits that only have one of the two events.
        ordered = df.sort_values("type", ascending=True)  # ARRIVAL before DEPARTURE
        carried = ordered.groupby(keys, sort=False)[carry_columns].first().reset_index()
        wide = wide.merge(carried, on=keys, how="left")

    return wide.drop(columns=[c for c in wide.columns if c.startswith("_")])


def assemble_sequences(visits, target_station, nf_columns, h=5, z=5,
                       label_column="trainDelayed"):
    """Build one sample per *arrival* at `target_station`.

    Arrival is meant strictly: a train that originates at the target station
    never arrives there and is not a sample, though it stays in the pool of
    preceding traffic. See the `targets` comment below.

    Padding is pre-padding with zeros plus an explicit mask, rather than
    dropping short routes: a train originating two stations before the target
    is a normal operating pattern, and excluding it would bias the sample
    toward long-distance services.
    """
    visits = visits.sort_values(
        ["trainNumber", "departureDate", "station_order"]
    ).reset_index(drop=True)

    of = visits[OF_COLUMNS].to_numpy(dtype=np.float32)
    of = np.nan_to_num(of, nan=0.0, posinf=0.0, neginf=0.0)

    # Position within the train's own journey. Because `visits` is sorted by
    # journey, the Z history of row p is simply the rows immediately before it,
    # and this tells us how many of those belong to the same journey.
    journey_pos = visits.groupby(
        ["trainNumber", "departureDate"], sort=False
    ).cumcount().to_numpy()

    # Every visit at the target station, in the order they call there. This is
    # the pool the H sequence draws from, and it includes trains that only
    # depart: they occupy the platform ahead of the next arrival whether or not
    # we predict them.
    at_station = np.flatnonzero((visits["stationShortCode"] == target_station).to_numpy())

    # Samples are the subset that actually arrives. A train ORIGINATING at the
    # target station has no ARRIVAL event, so arrival_delay is NaN and there is
    # nothing to predict — it never arrives anywhere to be late. Emitting it
    # would run that NaN through derive_label's fillna(0.0) and label a
    # non-event 'not delayed': 32% of the real dataset, every one a guaranteed
    # negative, and identifiable from an empty z_mask because an originating
    # train has no preceding stations either.
    arrived = visits["arrival_delay"].notna().to_numpy()
    targets = at_station[arrived[at_station]]

    z_seq = np.zeros((len(targets), z, len(OF_COLUMNS)), dtype=np.float32)
    h_seq = np.zeros((len(targets), h, len(OF_COLUMNS)), dtype=np.float32)
    z_mask = np.zeros((len(targets), z), dtype=np.float32)
    h_mask = np.zeros((len(targets), h), dtype=np.float32)

    # The trains ahead of this one are the target-station visits scheduled
    # earlier, so the H order is visit_time — not train number, and not the
    # journey order the frame happens to be sorted in.
    order = np.argsort(visits["visit_time"].to_numpy()[at_station], kind="stable")
    ranked = at_station[order]
    rank_of = {position: rank for rank, position in enumerate(ranked)}
    station_of = of[ranked]

    for row, position in enumerate(targets):
        # Z: this train's own preceding stations, oldest -> newest.
        take = int(min(journey_pos[position], z))
        if take:
            z_seq[row, z - take:] = of[position - take:position]
            z_mask[row, z - take:] = 1.0

        # H: the trains ahead of this one into the target station.
        k = rank_of[position]
        block = station_of[max(0, k - h):k]
        if len(block):
            h_seq[row, h - len(block):] = block
            h_mask[row, h - len(block):] = 1.0

    _mask_boundary_crossing_fields(z_seq, z_mask, h_seq)

    if nf_columns:
        nf = visits.loc[targets, nf_columns].to_numpy(dtype=np.float32)
    else:
        nf = np.zeros((len(targets), 0), dtype=np.float32)

    if label_column in visits.columns:
        label = visits.loc[targets, label_column].to_numpy(dtype=np.float32)
    else:
        # The raw events carry no label; preprocessing derives it downstream and
        # that path is not available here. Reproduce the same definition rather
        # than silently emitting a column of zeros.
        label = derive_label(visits.loc[targets])

    return {
        "z_seq": z_seq, "h_seq": h_seq,
        "z_mask": z_mask, "h_mask": h_mask,
        "nf": np.nan_to_num(nf, nan=0.0, posinf=0.0, neginf=0.0),
        "label": label,
        "group": visits.loc[targets, "departureDate"].to_numpy(),
    }


def derive_label(visits):
    """Binary delay label, matching preprocessing_pipeline.add_train_delayed_feature.

    That method computes `differenceInMinutes > TRAIN_DELAY_MINUTES` on the raw
    events. Here the equivalent quantity is the arrival delay of the visit.
    """
    delay = visits["arrival_delay"]
    if delay.isna().all() and TRAIN_DELAYED_TARGET_COLUMN in visits.columns:
        delay = visits[TRAIN_DELAYED_TARGET_COLUMN]
    return (delay.fillna(0.0) > TRAIN_DELAY_MINUTES).to_numpy(dtype=np.float32)


def _add_temporal_features(wide, moment):
    """Reproduce the calendar features preprocessing derives from the schedule.

    `moment` is the scheduled arrival (departure where there is none) in UTC.
    When the convert_scheduled_time_to_local stage is enabled it is converted to
    LOCAL_TIMEZONE first, exactly as that stage does, so both paths agree.

    They are in every scenario catalogue but in none of the raw event files,
    because preprocessing computes them downstream — the stage FCLL-Net skips.
    Without this the FCNN branch sees nothing at all on the operational-only
    scenarios. Formulas and the -1 offsets match convert_*_to_sincos exactly;
    day_of_week is 1=Sunday..7=Saturday as add_temporal_features defines it.
    """
    if PREPROCESSING_STATE_MACHINE.get("convert_scheduled_time_to_local", False):
        moment = moment.dt.tz_convert(LOCAL_TIMEZONE)
    month = moment.dt.month
    hour_decimal = moment.dt.hour + moment.dt.minute / 60.0
    pandas_dayofweek = moment.dt.dayofweek
    day_of_week = np.where(pandas_dayofweek == 6, 1, pandas_dayofweek + 2)

    wide["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    wide["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    wide["hour_sin"] = np.sin(2 * np.pi * hour_decimal / 24.0)
    wide["hour_cos"] = np.cos(2 * np.pi * hour_decimal / 24.0)
    wide["day_week_sin"] = np.sin(2 * np.pi * (day_of_week - 1) / 7.0)
    wide["day_week_cos"] = np.cos(2 * np.pi * (day_of_week - 1) / 7.0)
    wide["day_of_month"] = moment.dt.day
    wide["day_month_sin"] = np.sin(2 * np.pi * (moment.dt.day - 1) / float(DAY_OF_MONTH_CYCLE))
    wide["day_month_cos"] = np.cos(2 * np.pi * (moment.dt.day - 1) / float(DAY_OF_MONTH_CYCLE))

    # A neutral position on the unit circle for unparseable timestamps, as the
    # preprocessing stages do.
    for name in ("month_sin", "month_cos", "hour_sin", "hour_cos",
                 "day_week_sin", "day_week_cos", "day_of_month",
                 "day_month_sin", "day_month_cos"):
        wide[name] = wide[name].fillna(0.0)
    return wide


def _add_wawa_groups(wide, df, keys):
    """Reproduce the WMO 4680 present-weather one-hot columns.

    Same reason as _add_temporal_features: scenarios 9-13 list them, and
    preprocessing builds them in a stage fcll_net skips. The code-to-group map
    and the emitted-group list are imported rather than restated, so the two
    paths cannot drift apart.
    """
    names = [f"wawa_group_{group}" for group in WAWA_EMITTED_GROUPS]
    if WAWA_SOURCE_COLUMN not in df.columns:
        for name in names:
            wide[name] = 0.0
        return wide

    codes = pd.to_numeric(df[WAWA_SOURCE_COLUMN], errors="coerce").astype("Int64")
    groups = codes.map(WAWA_CODE_TO_GROUP).fillna(WAWA_FALLBACK_GROUP).astype(str)
    per_visit = (
        df.assign(_wawa_group=groups)
        .groupby(keys, sort=False)["_wawa_group"]
        .first()
        .reset_index()
    )
    wide = wide.merge(per_visit, on=keys, how="left")
    for group, name in zip(WAWA_EMITTED_GROUPS, names):
        wide[name] = (wide["_wawa_group"] == group).astype(float)
    return wide.drop(columns=["_wawa_group"])


# Fields whose measurement window ends at the target arrival, so they are not
# observable when the prediction is made.
_RUN_ACTUAL = OF_COLUMNS.index("run_actual")
_DWELL_ACTUAL = OF_COLUMNS.index("dwell_actual")


def _mask_boundary_crossing_fields(z_seq, z_mask, h_seq):
    """Zero the fields that measure past the moment being predicted.

    Two cases, both discovered by a model scoring 0.9993 ROC-AUC on a task
    where that is not achievable:

    1. Z, final step. run_actual is 'this departure -> the NEXT station's
       arrival'. For the station immediately before the target, that next
       arrival is the arrival being predicted, which makes the label exact
       arithmetic on this timestep:
           arrival_delay = departure_delay + (run_actual - run_scheduled)
       Measured at 98.2% agreement on 2018-01 before this masking.
       Earlier steps are untouched: their next station is still upstream.

    2. H, every step. Those rows are the preceding trains' visits AT the
       target station, so their dwell there and their onward run are both
       measured after they arrived — and nothing guarantees either finished
       before the target train arrives.

    The scheduled counterparts are deliberately kept: the timetable is known
    in advance, and they are the baseline the delays are measured against.
    """
    lengths = z_mask.sum(axis=1).astype(int)
    has_history = lengths > 0
    if has_history.any():
        rows = np.flatnonzero(has_history)
        # Pre-padded, so the last real step is always the final slot.
        z_seq[rows, -1, _RUN_ACTUAL] = 0.0

    h_seq[:, :, _RUN_ACTUAL] = 0.0
    h_seq[:, :, _DWELL_ACTUAL] = 0.0
