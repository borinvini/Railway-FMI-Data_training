#!/bin/bash
# Submit the scenario grid once per station of hpc/stations.sh.
#
#   hpc/submit-stations.sh --models xgboost,lightgbm,random_forest,ridge_baseline --dry-run
#   hpc/submit-stations.sh --models xgboost,lightgbm,random_forest,ridge_baseline
#   hpc/submit-stations.sh --stations OL,ROI --models ...    # only those stations
#
# Every other argument (--models, --scenarios, --dry-run, sbatch options such as
# --time=08:00:00) goes unchanged to hpc/submit-scenarios.sh --station <ST>.
#
# One array per station, not one array of stations x scenarios x models: each
# station can be watched, cancelled (scancel <jobid>) or re-submitted on its own,
# its tasks read only /scratch/.../railway-fmi/stations/<ST>/, and the task-id
# decoding of train_scenarios.sh stays the one the single-station grid uses.
# Stops at the first station that fails to submit, so a later station is never
# queued against a mistake the first one revealed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=hpc/stations.sh
source "${REPO_ROOT}/hpc/stations.sh"

STATION_SPEC="all"
PASS=()
while [ $# -gt 0 ]; do
    case "${1}" in
        --stations)
            [ $# -ge 2 ] || { echo "ERROR: --stations needs a value." >&2; exit 2; }
            STATION_SPEC="${2}"
            shift 2
            ;;
        --stations=*)
            STATION_SPEC="${1#*=}"
            shift
            ;;
        --station|--station=*)
            echo "ERROR: use --stations here; --station belongs to submit-scenarios.sh." >&2
            exit 2
            ;;
        -h|--help)
            sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            PASS+=("${1}")
            shift
            ;;
    esac
done

if [ "${STATION_SPEC}" = "all" ]; then
    CHOSEN=("${STATIONS[@]}")
else
    IFS=',' read -r -a WANTED <<< "$(printf '%s' "${STATION_SPEC}" | tr '[:lower:]' '[:upper:]')"
    CHOSEN=()
    # In hpc/stations.sh order, whatever order they were typed in.
    for s in "${STATIONS[@]}"; do
        for w in "${WANTED[@]}"; do [ "${w}" = "${s}" ] && CHOSEN+=("${s}") && break; done
    done
    for w in "${WANTED[@]}"; do
        found=0
        for s in "${STATIONS[@]}"; do [ "${w}" = "${s}" ] && found=1; done
        if [ "${found}" -ne 1 ]; then
            echo "ERROR: unknown station '${w}'. hpc/stations.sh lists: ${STATIONS[*]}" >&2
            exit 2
        fi
    done
fi

SUMMARY=()
for station in "${CHOSEN[@]}"; do
    echo "=== ${station} ==="
    if ! out="$(bash hpc/submit-scenarios.sh --station "${station}" "${PASS[@]+"${PASS[@]}"}")"; then
        printf '%s\n' "${out}"
        echo "ERROR: submitting ${station} failed; stations after it were not submitted." >&2
        exit 1
    fi
    printf '%s\n' "${out}"
    job="$(printf '%s\n' "${out}" | sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p' | tail -1)"
    SUMMARY+=("${station} ${job:-(dry run)}")
done

echo
echo "station job"
printf '%s\n' "${SUMMARY[@]}"
