#!/bin/bash
#SBATCH --job-name=railway-chain
#SBATCH --account=project_2019266
#SBATCH --partition=small
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=slurm-chain-%j.out
#
# One link of a station chain (hpc/chain-stations.sh). Started by Slurm when the
# previous station's array has ended (--dependency=afterany), it submits the next
# station of its lane and queues the link after it.
#
# Why a chain: the small partition allows 200 queued-or-running tasks per user
# (sacctmgr maxsubmit), and a station is 52. Jobs waiting on a dependency count
# too, so all 7 stations (364 tasks) can never be queued at once; HKI, PSL and
# TPE were accepted and SK was refused with AssocMaxSubmitJobLimit. A lane holds
# one station plus this one-core link, so 3 lanes are at most 3 x 53 = 159.
#
# Input, from the environment (set by chain-stations.sh or the previous link):
#   RAILWAY_CHAIN_NEXT    comma list of the lane's stations still to submit, e.g. SK,ROI
#   RAILWAY_CHAIN_MODELS  the --models value, e.g. xgboost,lightgbm,random_forest,ridge_baseline
#   RAILWAY_CHAIN_SBATCH  extra sbatch arguments for the grid (space separated, may be empty)
# A refused submit (the limit) is not a failure: the link re-queues itself 30 minutes later.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f main.py ]; then
    echo "ERROR: main.py not found in $(pwd); submit from the repository root." >&2
    exit 1
fi

NEXT="${RAILWAY_CHAIN_NEXT:-}"
MODELS_ARG="${RAILWAY_CHAIN_MODELS:-}"
EXTRA="${RAILWAY_CHAIN_SBATCH:-}"
if [ -z "${NEXT}" ] || [ -z "${MODELS_ARG}" ]; then
    echo "ERROR: RAILWAY_CHAIN_NEXT and RAILWAY_CHAIN_MODELS must be set." >&2
    exit 1
fi
STATION="${NEXT%%,*}"
REST=""
[ "${NEXT}" != "${STATION}" ] && REST="${NEXT#*,}"

# This link's own SLURM_* variables would be exported into the jobs it submits
# (sbatch exports the environment), where SLURM_CPUS_PER_TASK=1 or SLURM_MEM_*
# could leak into the 40-core grid. Drop them all before submitting anything.
for var in $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "${var}"; done

echo "Chain link: submitting ${STATION}; after it: ${REST:-nothing}"
# shellcheck disable=SC2086
if out="$(bash hpc/submit-scenarios.sh --station "${STATION}" --models "${MODELS_ARG}" ${EXTRA} 2>&1)"; then
    printf '%s\n' "${out}"
    job="$(printf '%s\n' "${out}" | sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p' | tail -1)"
    if [ -z "${job}" ]; then
        echo "ERROR: no job id in the submit output above." >&2
        exit 1
    fi
    if [ -n "${REST}" ]; then
        # The lists travel in the environment (sbatch exports it), never in
        # --export=, which splits its value on the commas of the lists.
        RAILWAY_CHAIN_NEXT="${REST}" RAILWAY_CHAIN_MODELS="${MODELS_ARG}" RAILWAY_CHAIN_SBATCH="${EXTRA}" \
            sbatch --dependency="afterany:${job}" hpc/chain_next.sh
        echo "Queued the next link after ${STATION} (job ${job}) for: ${REST}"
    else
        echo "Lane finished: ${STATION} (job ${job}) was its last station."
    fi
else
    printf '%s\n' "${out}"
    # Only the submit limit is worth waiting for. Anything else (a station never
    # staged, a bad catalogue) would fail the same way every 30 minutes forever.
    if printf '%s\n' "${out}" | grep -q -e AssocMaxSubmitJobLimit -e 'job submit limit'; then
        echo "Submit of ${STATION} refused by the submit limit; retrying this link in 30 minutes."
        RAILWAY_CHAIN_NEXT="${NEXT}" RAILWAY_CHAIN_MODELS="${MODELS_ARG}" RAILWAY_CHAIN_SBATCH="${EXTRA}" \
            sbatch --begin=now+30minutes hpc/chain_next.sh
    else
        echo "ERROR: submitting ${STATION} failed for another reason; this lane stops here." >&2
        echo "       Not submitted: ${NEXT}" >&2
        exit 1
    fi
fi
