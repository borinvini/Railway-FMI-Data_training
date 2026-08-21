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
    TRAIN_DELAYED_TARGET_COLUMN,
    TRAIN_DELAY_MINUTES,
)

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

    keys = ["trainNumber", "departureDate", "stationShortCode"]
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

    if carry_columns:
        # Prefer the ARRIVAL row's values; fall back to the DEPARTURE row for
        # visits that only have one of the two events.
        ordered = df.sort_values("type", ascending=True)  # ARRIVAL before DEPARTURE
        carried = ordered.groupby(keys, sort=False)[carry_columns].first().reset_index()
        wide = wide.merge(carried, on=keys, how="left")

    return wide.drop(columns=[c for c in wide.columns if c.startswith("_")])


def assemble_sequences(visits, target_station, nf_columns, h=5, z=5,
                       label_column="trainDelayed"):
    """Build one sample per arrival at `target_station`.

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

    targets = np.flatnonzero((visits["stationShortCode"] == target_station).to_numpy())

    z_seq = np.zeros((len(targets), z, len(OF_COLUMNS)), dtype=np.float32)
    h_seq = np.zeros((len(targets), h, len(OF_COLUMNS)), dtype=np.float32)
    z_mask = np.zeros((len(targets), z), dtype=np.float32)
    h_mask = np.zeros((len(targets), h), dtype=np.float32)

    # The trains ahead of this one are the target-station visits scheduled
    # earlier, so the H order is visit_time — not train number, and not the
    # journey order the frame happens to be sorted in.
    order = np.argsort(visits["visit_time"].to_numpy()[targets], kind="stable")
    ranked = targets[order]
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
