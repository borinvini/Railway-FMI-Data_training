#!/bin/bash
# Submit every station of hpc/stations.sh as self-continuing chains (lanes).
#
#   bash hpc/chain-stations.sh --models xgboost,lightgbm,random_forest,ridge_baseline --dry-run
#   bash hpc/chain-stations.sh --models xgboost,lightgbm,random_forest,ridge_baseline
#   bash hpc/chain-stations.sh --lanes 2 --stations OL,ROI,YV --models ...
#
# The small partition allows 200 queued-or-running tasks per user and a station is
# 52, so 7 stations (364) cannot all be queued, and dependency-held jobs count too.
# The stations are dealt round-robin into --lanes lanes (default 3: HKI,SK,ROI /
# PSL,YV / TPE,OL). Each lane's first station is submitted now, with one 1-core
# hpc/chain_next.sh link held behind it (--dependency=afterany). When the station's
# array has ended, the link submits the lane's next station and the next link.
# At most lanes x 53 tasks are queued (3 lanes: 159).
#
# Any other argument (e.g. --time=72:00:00) goes to the grid's sbatch, for every station.
# Cancel everything with: scancel -u $USER   (links included, so no lane continues).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck source=hpc/stations.sh
source hpc/stations.sh

STATION_SPEC="all"
MODELS_ARG=""
LANES=3
DRY_RUN=0
EXTRA=()
while [ $# -gt 0 ]; do
    case "${1}" in
        --stations) STATION_SPEC="${2:?--stations needs a value}"; shift ;;
        --stations=*) STATION_SPEC="${1#*=}" ;;
        --models) MODELS_ARG="${2:?--models needs a value}"; shift ;;
        --models=*) MODELS_ARG="${1#*=}" ;;
        --lanes) LANES="${2:?--lanes needs a value}"; shift ;;
        --lanes=*) LANES="${1#*=}" ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) EXTRA+=("${1}") ;;
    esac
    shift
done

if [ -z "${MODELS_ARG}" ]; then
    echo "ERROR: --models is required, e.g. --models xgboost,lightgbm,random_forest,ridge_baseline" >&2
    exit 2
fi
if ! [[ "${LANES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --lanes must be a positive number." >&2
    exit 2
fi
for arg in "${EXTRA[@]+"${EXTRA[@]}"}"; do
    case "${arg}" in
        *" "*|*,*) echo "ERROR: extra sbatch argument '${arg}' may not contain spaces or commas." >&2; exit 2 ;;
    esac
done

if [ "${STATION_SPEC}" = "all" ]; then
    CHOSEN=("${STATIONS[@]}")
else
    IFS=',' read -r -a WANTED <<< "$(printf '%s' "${STATION_SPEC}" | tr '[:lower:]' '[:upper:]')"
    CHOSEN=()
    for s in "${STATIONS[@]}"; do
        for w in "${WANTED[@]}"; do [ "${w}" = "${s}" ] && CHOSEN+=("${s}") && break; done
    done
    for w in "${WANTED[@]}"; do
        found=0
        for s in "${STATIONS[@]}"; do [ "${w}" = "${s}" ] && found=1; done
        [ "${found}" -eq 1 ] || { echo "ERROR: unknown station '${w}' (${STATIONS[*]})." >&2; exit 2; }
    done
fi

# Round-robin into lanes, in hpc/stations.sh order.
LANE_LIST=()
for i in "${!CHOSEN[@]}"; do
    l=$(( i % LANES ))
    LANE_LIST[l]="${LANE_LIST[l]:+${LANE_LIST[l]},}${CHOSEN[$i]}"
done

echo "${#CHOSEN[@]} stations in ${#LANE_LIST[@]} lanes; models ${MODELS_ARG}; extra sbatch: ${EXTRA[*]:-none}"
for l in "${!LANE_LIST[@]}"; do
    echo "  lane $((l + 1)): ${LANE_LIST[l]//,/ -> }"
done

SUMMARY=()
for l in "${!LANE_LIST[@]}"; do
    lane="${LANE_LIST[l]}"
    first="${lane%%,*}"
    rest=""
    [ "${lane}" != "${first}" ] && rest="${lane#*,}"
    echo
    echo "=== lane $((l + 1)): ${first} now${rest:+, then ${rest}} ==="
    args=(--station "${first}" --models "${MODELS_ARG}" "${EXTRA[@]+"${EXTRA[@]}"}")
    [ "${DRY_RUN}" -eq 1 ] && args+=(--dry-run)
    if ! out="$(bash hpc/submit-scenarios.sh "${args[@]}")"; then
        printf '%s\n' "${out}"
        echo "ERROR: submitting ${first} failed; lanes after this one were not started." >&2
        exit 1
    fi
    printf '%s\n' "${out}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        SUMMARY+=("lane $((l + 1)): ${first} (dry run)${rest:+, link for ${rest}}")
        continue
    fi
    job="$(printf '%s\n' "${out}" | sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p' | tail -1)"
    link="none"
    if [ -n "${rest}" ]; then
        link_out="$(RAILWAY_CHAIN_NEXT="${rest}" RAILWAY_CHAIN_MODELS="${MODELS_ARG}" \
                    RAILWAY_CHAIN_SBATCH="${EXTRA[*]:-}" \
                    sbatch --dependency="afterany:${job}" hpc/chain_next.sh)"
        link="$(printf '%s\n' "${link_out}" | sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p')"
    fi
    SUMMARY+=("lane $((l + 1)): ${first} job ${job}; link ${link} for ${rest:-nothing}")
done

echo
printf '%s\n' "${SUMMARY[@]}"
echo "Each link writes slurm-chain-<jobid>.out here. Progress: squeue -u \$USER -o '%i %j %T'"
