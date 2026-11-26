"""Preprocess every station of hpc/stations.sh, check it, and file it in the results archive.

For each station, in hpc/stations.sh order:
  1. main.py --pipeline preprocessing --station ST --data-root data/stations/ST
     --input-dir <repo>/data/input, as a subprocess (config constants bind at
     import time, so each station needs a fresh interpreter). Always all months:
     SCHEMA_MISMATCH_STRATEGY='intersect' silently drops columns missing in any month.
  2. scripts/check_station.py: the 101 files must belong to ST (96 files, >= 0.95 share).
  3. Copy 100-, 101-, 9-, 12- and log into <archive>/<N> - <ST>/, the layout of
     "AI Results/BINARY". A non-empty target folder is refused without --overwrite.

hpc/stage_data.sh --stations then sends data/stations/<ST>/data/output/101-... to Roihu.

Usage (repo root):
    python -u -X utf8 scripts/preprocess_stations.py [--stations OL,ROI|all]
        [--archive DIR] [--skip-existing] [--overwrite] [--no-archive]
Exit 1 when any station failed.
"""
import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_station import check_station  # noqa: E402
from stations import REPO_ROOT, archive_folder_name, load_stations, select_stations  # noqa: E402

DEFAULT_ARCHIVE = os.path.join(
    os.path.dirname(REPO_ROOT), "Railway-FMI-Data-CSV-Files-v2", "AI Results", "NUMERIC")
STATIONS_ROOT = os.path.join(REPO_ROOT, "data", "stations")
INPUT_DIR = os.path.join(REPO_ROOT, "data", "input")

# What "AI Results/BINARY/<N> - <ST>/" keeps of a preprocessing run.
ARCHIVED_FOLDERS = (
    "100-preprocessed",
    "101-preprocessed_training_ready",
    "9-process_actual_time_column",
    "12-handle_missing_values",
    "log",
)


def station_root(station):
    return os.path.join(STATIONS_ROOT, station)


def preprocess_command(station, python=sys.executable):
    return [
        python, "-u", "-X", "utf8", os.path.join(REPO_ROOT, "main.py"),
        "--pipeline", "preprocessing",
        "--station", station,
        "--data-root", station_root(station),
        "--input-dir", INPUT_DIR,
    ]


def run_preprocessing(station, run=subprocess.run):
    """Run main.py for one station, its console going to <root>/preprocess_<ST>.log."""
    root = station_root(station)
    os.makedirs(root, exist_ok=True)
    log_path = os.path.join(root, f"preprocess_{station}.log")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(log_path, "w", encoding="utf-8") as log:
        completed = run(preprocess_command(station), stdout=log, stderr=subprocess.STDOUT,
                        cwd=REPO_ROOT, env=env)
    return completed.returncode, log_path


def archive_conflict(station, archive, stations):
    """The first non-empty archived folder already in <archive>/<N> - <ST>/, else None."""
    target = os.path.join(archive, archive_folder_name(station, stations))
    for name in ARCHIVED_FOLDERS:
        destination = os.path.join(target, name)
        if os.path.isdir(destination) and os.listdir(destination):
            return destination
    return None


def archive_station(station, archive, stations, overwrite=False):
    """Copy the archived folders into <archive>/<N> - <ST>/. Returns the target path."""
    target = os.path.join(archive, archive_folder_name(station, stations))
    output = os.path.join(station_root(station), "data", "output")
    conflict = archive_conflict(station, archive, stations)
    if conflict and not overwrite:
        raise FileExistsError(f"{conflict} is not empty; pass --overwrite to replace it")
    os.makedirs(target, exist_ok=True)
    for name in ARCHIVED_FOLDERS:
        source = os.path.join(output, name)
        if not os.path.isdir(source):
            continue
        destination = os.path.join(target, name)
        if os.path.isdir(destination):
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    return target


def process_station(station, args, stations, run=subprocess.run, check=check_station, log=print):
    """Preprocess, check and archive one station. Returns (ok, summary line)."""
    root = station_root(station)
    # Before the slow part: a full archive folder would otherwise be found only
    # after half an hour of preprocessing.
    if not args.no_archive and not args.overwrite:
        conflict = archive_conflict(station, args.archive, stations)
        if conflict:
            return False, f"✗ {station}: {conflict} is not empty; pass --overwrite to replace it"
    if args.skip_existing:
        passed, summary = check(station, root, INPUT_DIR, log=lambda *_: None)
        if passed:
            log(f"  {station}: existing data passes the check, preprocessing skipped")
        else:
            passed = None
    else:
        passed = None

    if passed is None:
        log(f"→ {station}: preprocessing into {root}")
        code, log_path = run_preprocessing(station, run=run)
        if code != 0:
            return False, f"✗ {station}: main.py exited {code}, see {log_path}"
        passed, summary = check(station, root, INPUT_DIR, log=log)
        if not passed:
            return False, f"✗ {station}: the station check failed"

    where = "not archived"
    if not args.no_archive:
        try:
            where = archive_station(station, args.archive, stations, overwrite=args.overwrite)
        except FileExistsError as exc:
            return False, f"✗ {station}: {exc}"
    return True, (f"✓ {station}: {summary['files']} files, {summary['rows']} rows, "
                  f"median delay {summary.get('median_delay', float('nan')):.1f} min -> {where}")


def main(argv=None, run=subprocess.run, check=check_station):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stations", default="all", help="Comma list or 'all' (hpc/stations.sh).")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE, help="Results archive folder.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip preprocessing a station whose data root already passes the check.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace non-empty folders in the archive.")
    parser.add_argument("--no-archive", action="store_true", help="Do not copy into the archive.")
    args = parser.parse_args(argv)

    stations = load_stations()
    try:
        chosen = select_stations(args.stations, stations)
    except ValueError as exc:
        parser.error(str(exc))
    if not os.path.isdir(INPUT_DIR):
        parser.error(f"{INPUT_DIR} not found: the raw matched_data_flat_*.parquet files go there")

    lines, failed = [], 0
    for station in chosen:
        ok, line = process_station(station, args, stations, run=run, check=check)
        print(line, flush=True)
        lines.append(line)
        failed += not ok

    print("\n" + "=" * 60)
    print(f"STATIONS: {len(chosen) - failed}/{len(chosen)} done")
    print("=" * 60)
    for line in lines:
        print(line)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
