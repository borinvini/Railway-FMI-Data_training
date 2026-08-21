"""Turn event-level train records into the per-visit operational features that
FCLL-Net's two LSTMs consume.

Reads the raw matched parquet rather than 101-preprocessed_training_ready,
because that file has already been filtered to one station: the sequences this
module builds no longer exist in it.

Huang et al. (2020) define nine operational variables per station visit. Eight
are derived here; the ninth (number of stops in the section) needs the route,
and is added by assemble_sequences.
"""
import numpy as np
import pandas as pd

OF_COLUMNS = [
    "arrival_delay", "departure_delay",
    "dwell_actual", "dwell_scheduled",
    "run_actual", "run_scheduled",
    "headway_actual", "headway_scheduled",
    "stops_in_section",
]


def _to_datetime(series):
    return pd.to_datetime(series, format="ISO8601", utc=True)


def derive_station_visits(df):
    """Collapse ARRIVAL/DEPARTURE event pairs into one row per station visit."""
    df = df.copy()
    df["_sched"] = _to_datetime(df["scheduledTime"])
    df["_actual"] = _to_datetime(df["actualTime"])

    keys = ["trainNumber", "departureDate", "stationShortCode"]
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
    wide["_order_key"] = col("_sched_arrival").fillna(col("_sched_departure"))
    wide = wide.sort_values(["trainNumber", "departureDate", "_order_key"])
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
    wide = wide.sort_values(["stationShortCode", "_order_key"])
    station = wide.groupby("stationShortCode")
    for kind, name in (("_sched_arrival", "headway_scheduled"),
                       ("_actual_arrival", "headway_actual")):
        if kind not in wide.columns:
            wide[name] = np.nan
            continue
        wide[name] = station[kind].diff().dt.total_seconds() / 60.0

    wide["stops_in_section"] = np.nan  # filled by assemble_sequences
    return wide.drop(columns=[c for c in wide.columns if c.startswith("_")])
