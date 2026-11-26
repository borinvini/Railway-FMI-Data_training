"""The station list of the multi-station grid, read from hpc/stations.sh.

The shell scripts source that file; Python parses the same line here, so the two
can never disagree about which stations exist or which archive number each has.
"""
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIONS_FILE = os.path.join(REPO_ROOT, "hpc", "stations.sh")

_ASSIGNMENT = re.compile(r"^[ \t]*STATIONS=\(([^)\n]*)\)[ \t]*$", re.MULTILINE)


def load_stations(path=STATIONS_FILE):
    """The station codes in archive order (position + 1 = archive number)."""
    with open(path, encoding="utf-8") as fh:
        match = _ASSIGNMENT.search(fh.read())
    if not match:
        raise ValueError(f"{path}: no one-line STATIONS=(...) assignment")
    stations = match.group(1).split()
    if not stations:
        raise ValueError(f"{path}: STATIONS is empty")
    if len(set(stations)) != len(stations):
        raise ValueError(f"{path}: a station is listed twice: {stations}")
    return stations


def archive_folder_name(station, stations=None):
    """'<N> - <ST>', the folder name the results archive uses for a station."""
    stations = stations or load_stations()
    return f"{stations.index(station) + 1} - {station}"


def select_stations(spec, stations=None):
    """Resolve a --stations value: None or 'all' is every station, else a comma list.

    Unknown codes raise, so a typo never silently drops a station from the run.
    """
    stations = stations or load_stations()
    if spec is None or spec.strip().lower() == "all":
        return list(stations)
    wanted = [s.strip().upper() for s in spec.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in stations]
    if unknown:
        raise ValueError(f"unknown station(s) {unknown}; hpc/stations.sh lists {stations}")
    return [s for s in stations if s in wanted]
