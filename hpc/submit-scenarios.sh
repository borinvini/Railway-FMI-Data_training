#!/bin/bash
# Submit the scenarios x models grid, sized from the catalogue rather than a
# hardcoded header.
#
#   hpc/submit-scenarios.sh                    # every scenario, every model
#   hpc/submit-scenarios.sh --scenarios 9-13   # just those five scenarios
#   hpc/submit-scenarios.sh --scenarios 9      # just that one
#   hpc/submit-scenarios.sh --dry-run          # print the sbatch line, submit nothing
#
# Any other argument is passed through to sbatch, so a one-off override needs no
# edit here:  hpc/submit-scenarios.sh --time=02:00:00
#
# WHY THIS SCRIPT EXISTS
#
# `#SBATCH --array=` is not a runtime instruction. sbatch parses it out of the
# script text on the login node, before the job exists; the batch script's own
# code runs later, on a compute node, when the array is already fixed. A batch
# script therefore cannot size its own array, and the header goes stale silently
# every time the catalogue grows.
#
# That is not hypothetical. Run 581402 went out as --array=0-39 against a
# 13-scenario catalogue: scenarios 9-13 were never queued, every task that did
# run looked healthy, and the gap only surfaced when the finished job count came
# up 25 short. Computing the range here, from the same parser main.py uses,
# removes the header from the decision.
set -euo pipefail

# Resolved from this script's own path, not $PWD: unlike the batch scripts there
# is no SLURM_SUBMIT_DIR yet, and `main.py` and the catalogue are both relative
# to the repo root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PROJECT="project_2019266"
BATCH_SCRIPT="hpc/train_scenarios.sh"

# shellcheck source=hpc/models.sh
source "${REPO_ROOT}/hpc/models.sh"
N_MODELS="${#MODELS[@]}"

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

SCENARIO_SPEC=""
DRY_RUN=0
SBATCH_ARGS=()
while [ $# -gt 0 ]; do
    case "${1}" in
        --scenarios)
            [ $# -ge 2 ] || { echo "ERROR: --scenarios needs a value." >&2; exit 2; }
            SCENARIO_SPEC="${2}"
            shift 2
            ;;
        --scenarios=*)
            SCENARIO_SPEC="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            SBATCH_ARGS+=("${1}")
            shift
            ;;
    esac
done

# The catalogue. Gitignored and uploaded straight to this clone by
# hpc/push-features.sh, so changing a scenario needs no commit, push or pull.
COLUMNS_FILE="${COLUMNS_FILE:-config/scenarios.txt}"
if [ ! -s "${COLUMNS_FILE}" ]; then
    echo "ERROR: columns file missing or empty: ${COLUMNS_FILE}" >&2
    echo "       Upload it from your laptop with:" >&2
    echo "         hpc/push-features.sh config/scenarios.txt" >&2
    echo "       or submit with: COLUMNS_FILE=<path> hpc/$(basename "${0}")" >&2
    exit 1
fi

# Only prepended when it exists, so this script is testable off the cluster.
if [ -d "/projappl/${PROJECT}/railway-env/bin" ]; then
    export PATH="/projappl/${PROJECT}/railway-env/bin:$PATH"
fi

# Counted through main.py rather than by grepping banners here: the array must
# be sized by the same parser that will later resolve --scenario, or the two can
# disagree about what counts as a section.
if ! SCENARIO_LIST="$(python main.py --columns-file "${COLUMNS_FILE}" --list-scenarios)"; then
    echo "ERROR: could not read scenarios from ${COLUMNS_FILE}." >&2
    echo "       main.py's own error is above this line." >&2
    exit 1
fi
N_SCENARIOS="$(printf '%s\n' "${SCENARIO_LIST}" | grep -c .)"
if [ "${N_SCENARIOS}" -lt 1 ]; then
    echo "ERROR: ${COLUMNS_FILE} lists no scenarios." >&2
    exit 1
fi

# Default to the whole catalogue; --scenarios narrows it.
FIRST_SCENARIO=1
LAST_SCENARIO="${N_SCENARIOS}"
if [ -n "${SCENARIO_SPEC}" ]; then
    if [[ "${SCENARIO_SPEC}" =~ ^([0-9]+)$ ]]; then
        FIRST_SCENARIO="${BASH_REMATCH[1]}"
        LAST_SCENARIO="${BASH_REMATCH[1]}"
    elif [[ "${SCENARIO_SPEC}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        FIRST_SCENARIO="${BASH_REMATCH[1]}"
        LAST_SCENARIO="${BASH_REMATCH[2]}"
    else
        echo "ERROR: --scenarios ${SCENARIO_SPEC} is not a number or a N-M range." >&2
        exit 2
    fi
    if [ "${FIRST_SCENARIO}" -lt 1 ]; then
        echo "ERROR: --scenarios ${SCENARIO_SPEC}: scenarios are numbered from 1." >&2
        exit 2
    fi
    if [ "${LAST_SCENARIO}" -lt "${FIRST_SCENARIO}" ]; then
        echo "ERROR: --scenarios ${SCENARIO_SPEC} is a backwards range." >&2
        exit 2
    fi
    if [ "${LAST_SCENARIO}" -gt "${N_SCENARIOS}" ]; then
        echo "ERROR: --scenarios ${SCENARIO_SPEC} asks for scenario ${LAST_SCENARIO}," >&2
        echo "       but ${COLUMNS_FILE} holds only ${N_SCENARIOS}:" >&2
        printf '%s\n' "${SCENARIO_LIST}" | sed 's/^/         /' >&2
        exit 2
    fi
fi

# Scenario-major, 1-based, matching train_scenarios.sh's own
# scenario = id / N_MODELS + 1. One scenario's models stay contiguous, so
# cancelling a tail range loses whole scenarios rather than fragments of many.
FIRST_TASK=$(( (FIRST_SCENARIO - 1) * N_MODELS ))
LAST_TASK=$(( LAST_SCENARIO * N_MODELS - 1 ))
N_TASKS=$(( LAST_TASK - FIRST_TASK + 1 ))

if [ "${FIRST_SCENARIO}" -eq 1 ] && [ "${LAST_SCENARIO}" -eq "${N_SCENARIOS}" ]; then
    echo "${N_SCENARIOS} scenarios x ${N_MODELS} models = ${N_TASKS} tasks"
else
    echo "scenarios ${FIRST_SCENARIO}-${LAST_SCENARIO} of ${N_SCENARIOS}" \
         "x ${N_MODELS} models = ${N_TASKS} tasks"
fi
echo "catalogue: ${COLUMNS_FILE} (sha256 $(sha256sum "${COLUMNS_FILE}" | cut -c1-16))"

CMD=(sbatch "--array=${FIRST_TASK}-${LAST_TASK}" "${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"}" "${BATCH_SCRIPT}")

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "${CMD[*]}"
    echo "(dry run — nothing submitted)"
    exit 0
fi

echo "${CMD[*]}"
COLUMNS_FILE="${COLUMNS_FILE}" exec "${CMD[@]}"
