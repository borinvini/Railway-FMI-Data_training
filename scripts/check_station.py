"""Read-only check: do the preprocessed 101 files belong to the requested station?

The 101 files carry no station column, so the station is recovered by joining
(trainNumber, departureDate, differenceInMinutes) of three sampled months with the
raw matched_data_flat_<month>.parquet, which has stationShortCode. The target
station matches every row (share 1.0); any other station matches only the rows
where a train happened to have the same delay there.

Usage (repo root):
    python -X utf8 scripts/check_station.py OL [--root data/stations/OL] [--input-dir data/input]
--root is the data root holding data/output/101-preprocessed_training_ready
(default: the repo root). Exit 0 = PASS, 1 = FAIL, 2 = bad usage.
"""
import argparse
import glob
import os
import sys

import pandas as pd

TRAINING_READY = os.path.join("data", "output", "101-preprocessed_training_ready")
EXPECTED_FILES = 96
# The target station matches every row (share 1.0). A neighbouring stop can match
# many rows too, because a delay carries over to the next stop: PSL matched 70-79%
# of the HKI rows. So the rule is not a gap: the station must reach MIN_SHARE and
# no other station may, which is what makes the answer unambiguous.
MIN_SHARE = 0.95
KEYS = ["trainNumber", "departureDate", "differenceInMinutes"]


def month_of(path):
    return os.path.basename(path).replace(".parquet", "")[-7:]  # YYYY_MM


def fingerprint(month_file, input_dir):
    month = month_of(month_file)
    raw_path = os.path.join(input_dir, f"matched_data_flat_{month}.parquet")
    if not os.path.exists(raw_path):
        return None, f"{month}: raw file {raw_path} not found"
    ready = pd.read_parquet(month_file, columns=KEYS).dropna(subset=["differenceInMinutes"])
    raw = pd.read_parquet(raw_path, columns=KEYS + ["stationShortCode"]).dropna(
        subset=["differenceInMinutes"])
    for frame in (ready, raw):
        frame["departureDate"] = frame["departureDate"].astype(str).str[:10]
    ready = ready.drop_duplicates(KEYS)
    if ready.empty:
        return None, f"{month}: no rows with a delay to fingerprint"
    merged = ready.merge(raw.drop_duplicates(KEYS + ["stationShortCode"]), on=KEYS)
    share = (merged.groupby("stationShortCode").size() / len(ready)).sort_values(ascending=False)
    return share, None


def month_passes(own, runner_up):
    """True when the station clearly owns the month's rows."""
    return own >= MIN_SHARE and runner_up < MIN_SHARE


def check_station(station, root=".", input_dir="data/input", log=print):
    """Return (passed, summary dict). Prints one line per sampled month."""
    station = station.strip().upper()
    folder = os.path.join(root, TRAINING_READY)
    files = sorted(glob.glob(os.path.join(folder, "training_ready_*.parquet")))
    summary = {"station": station, "files": len(files), "rows": 0}
    if not files:
        log(f"FAIL: no files in {folder}")
        return False, summary

    problems = []
    if len(files) != EXPECTED_FILES:
        problems.append(f"{len(files)} files in {folder}, expected {EXPECTED_FILES}")

    delays = pd.concat(
        [pd.read_parquet(f, columns=["differenceInMinutes"]) for f in files], ignore_index=True
    )["differenceInMinutes"]
    summary.update(rows=int(len(delays)), median_delay=float(delays.median()),
                   mean_delay=float(delays.mean()))
    log(f"{station}: files {len(files)} ({month_of(files[0])} -> {month_of(files[-1])}) | "
        f"rows {len(delays)} | delay median {delays.median():.1f}, mean {delays.mean():.2f} min")

    for f in (files[0], files[len(files) // 2], files[-1]):
        share, err = fingerprint(f, input_dir)
        month = month_of(f)
        if err:
            problems.append(err)
            log(f"  {err}")
            continue
        own = float(share.get(station, 0.0))
        others = share.drop(station, errors="ignore")
        runner_up = float(others.max()) if len(others) else 0.0
        top = ", ".join(f"{k} {v:.3f}" for k, v in share.head(3).items())
        log(f"  {month}: best {top} | {station} {own:.3f}, runner-up {runner_up:.3f}")
        if not month_passes(own, runner_up):
            problems.append(f"{month}: {station} share {own:.3f}, runner-up {runner_up:.3f} "
                            f"(needs {station} >= {MIN_SHARE} and every other station below it)")

    if problems:
        log(f"FAIL: the 101 files do NOT belong to {station}:")
        for p in problems:
            log(f"  - {p}")
        return False, summary
    log(f"PASS: the 101 files belong to {station}.")
    return True, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("station")
    parser.add_argument("--root", default=".")
    parser.add_argument("--input-dir", default=os.path.join("data", "input"))
    args = parser.parse_args(argv)
    passed, _ = check_station(args.station, args.root, args.input_dir)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
