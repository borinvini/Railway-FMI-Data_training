#!/bin/bash
# Fetch the multi-station grid from Roihu and file each station in the results archive.
#
#   hpc/fetch-stations.sh                       # every station of hpc/stations.sh
#   hpc/fetch-stations.sh --stations OL,ROI     # only those
#   hpc/fetch-stations.sh --archive "<dir>"     # default: ../Railway-FMI-Data-CSV-Files-v2/AI Results/NUMERIC
#   hpc/fetch-stations.sh --no-archive          # fetch and sort into results/<ST>/ only
#
# Per station, in hpc/stations.sh order:
#   1. hpc/fetch-results.sh --scenarios --station ST   -> results/ST/<scenario>/...
#   2. finish the sort if Windows/OneDrive locked a folder (WinError 5): retry
#      hpc/sort_results.py until no run_sNN_<model> folder is left
#   3. hpc/grid_summary.py results/ST                   -> results/ST/grid_summary.csv
#   4. copy results/ST to <archive>/<N> - <ST>/results  (refused if that is not empty)
# and a closing table of scenario folders and result JSON per station.
#
# Run from Git Bash on the laptop with the ssh agent loaded (hpc/roihu-auth.sh).
# PYTHON=<interpreter> picks the python used for the sort and the table (stdlib only).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=hpc/stations.sh
source hpc/stations.sh

STATION_SPEC="all"
ARCHIVE="${REPO_ROOT}/../Railway-FMI-Data-CSV-Files-v2/AI Results/NUMERIC"
DO_ARCHIVE=1
while [ $# -gt 0 ]; do
    case "${1}" in
        --stations) [ $# -ge 2 ] || { echo "ERROR: --stations needs a value." >&2; exit 2; }
                    STATION_SPEC="${2}"; shift ;;
        --stations=*) STATION_SPEC="${1#*=}" ;;
        --archive)  [ $# -ge 2 ] || { echo "ERROR: --archive needs a value." >&2; exit 2; }
                    ARCHIVE="${2}"; shift ;;
        --archive=*) ARCHIVE="${1#*=}" ;;
        --no-archive) DO_ARCHIVE=0 ;;
        -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "ERROR: unknown argument: ${1}" >&2; exit 2 ;;
    esac
    shift
done

if [ "${STATION_SPEC}" = "all" ]; then
    CHOSEN=("${STATIONS[@]}")
else
    IFS=',' read -r -a CHOSEN <<< "$(printf '%s' "${STATION_SPEC}" | tr '[:lower:]' '[:upper:]')"
    for w in "${CHOSEN[@]}"; do
        found=0
        for s in "${STATIONS[@]}"; do [ "${w}" = "${s}" ] && found=1; done
        if [ "${found}" -ne 1 ]; then
            echo "ERROR: unknown station '${w}'. hpc/stations.sh lists: ${STATIONS[*]}" >&2
            exit 2
        fi
    done
fi

PY="${PYTHON:-python}"
if ! command -v "${PY}" >/dev/null 2>&1; then
    echo "ERROR: ${PY} not found; set PYTHON=<interpreter> (any python 3, stdlib only)." >&2
    exit 1
fi
SCENARIOS_FILE="${COLUMNS_FILE:-config/scenarios.txt}"

# The archive number of a station: its position in hpc/stations.sh, plus one.
archive_name() {
    local i
    for i in "${!STATIONS[@]}"; do
        if [ "${STATIONS[$i]}" = "${1}" ]; then echo "$((i + 1)) - ${1}"; return; fi
    done
}

if [ "${DO_ARCHIVE}" -eq 1 ]; then
    if [ ! -d "${ARCHIVE}" ]; then
        echo "ERROR: archive folder not found: ${ARCHIVE}" >&2
        exit 1
    fi
    # Checked for every station before anything is fetched: finding a full
    # archive folder on station five would leave four fetched and two not.
    for station in "${CHOSEN[@]}"; do
        target="${ARCHIVE}/$(archive_name "${station}")/results"
        if [ -d "${target}" ] && [ -n "$(ls -A "${target}" 2>/dev/null)" ]; then
            echo "ERROR: ${target} is not empty. Move it aside first; nothing was fetched." >&2
            exit 1
        fi
    done
fi

REPORT=()
for station in "${CHOSEN[@]}"; do
    echo
    echo "==================== ${station} ===================="
    bash hpc/fetch-results.sh --scenarios --station "${station}" || \
        echo "WARNING: fetch-results.sh exited non-zero for ${station}; checking what arrived."

    # WinError 5 under OneDrive: shutil.move fails on a folder a sync client holds,
    # leaving run_sNN_* behind. Each pass empties what it can; repeat until done.
    for _ in $(seq 1 80); do
        find "results/${station}" -mindepth 1 -depth -type d -empty \
             -path "results/${station}/run_s??_*" -delete 2>/dev/null || true
        [ "$(ls -d "results/${station}"/run_s??_* 2>/dev/null | wc -l)" -eq 0 ] && break
        "${PY}" -X utf8 hpc/sort_results.py "results/${station}" "${SCENARIOS_FILE}" >/dev/null 2>&1 || true
    done
    left="$( (ls -d "results/${station}"/run_s??_* 2>/dev/null || true) | wc -l)"

    "${PY}" -X utf8 hpc/grid_summary.py "results/${station}" || true

    n_scen="$( (ls -d "results/${station}"/[0-9]*" - "* 2>/dev/null || true) | wc -l)"
    n_json="$( (find "results/${station}" \( -path '*/100[0-4]-*/*iteration_analysis*.json' \
              -o -path '*/999-*/ridge_baseline_*.json' \) 2>/dev/null || true) | wc -l)"
    where="not archived"
    if [ "${DO_ARCHIVE}" -eq 1 ] && [ "${left}" -eq 0 ] && [ "${n_json}" -gt 0 ]; then
        target="${ARCHIVE}/$(archive_name "${station}")"
        mkdir -p "${target}/results"
        cp -r "results/${station}/." "${target}/results/"
        where="${target}/results"
    elif [ "${DO_ARCHIVE}" -eq 1 ]; then
        where="NOT archived (unsorted folders or no results)"
    fi
    REPORT+=("${station}: ${n_scen} scenario folders, ${n_json} result JSON, ${left} unsorted -> ${where}")
done

echo
echo "==================== SUMMARY ===================="
printf '%s\n' "${REPORT[@]}"
echo "A full grid of 13 scenarios x 4 models is 13 scenario folders and 52 result JSON."
