#!/bin/bash
# Fetch training-pipeline output from Roihu into ./results/.
#
# Usage: hpc/fetch-results.sh [--models] [--train-all]
#   (no flags)   hpc/train_array.sh layout: every stage, 500 through 1004
#   --models     only the model directories, 100[0-4]-*
#   --train-all  hpc/train_all.sh layout: everything under data/output/ directly
#
# The two flags are orthogonal — --models picks the subset, --train-all picks
# the layout — so `--train-all --models` is valid and means what it says.
set -euo pipefail

cd "$(dirname "${0}")/.."

REMOTE_HOST="roihu"
SCRATCH="/scratch/project_2019266/railway-fmi"

MODELS_ONLY=0
TRAIN_ALL=0
for arg in "$@"; do
    case "${arg}" in
        --models)    MODELS_ONLY=1 ;;
        --train-all) TRAIN_ALL=1 ;;
        *)
            echo "ERROR: unknown argument: ${arg}" >&2
            echo "       Usage: hpc/fetch-results.sh [--models] [--train-all]" >&2
            exit 1
            ;;
    esac
done

# Candidate remote globs, as explicit character classes. Never 10*: that also
# matches 101-preprocessed_training_ready, which is pipeline INPUT staged up by
# hpc/stage_data.sh, and would come back down once per run directory.
CANDIDATES=()
if [ "${TRAIN_ALL}" -eq 1 ]; then
    # train_all.sh is a single sequential job with no race to isolate against,
    # so it writes straight into ${SCRATCH} with no run_N roots at all.
    if [ "${MODELS_ONLY}" -eq 0 ]; then
        CANDIDATES+=("${SCRATCH}/data/output/50[0-5]-*")
        CANDIDATES+=("${SCRATCH}/data/output/700-*")
    fi
    CANDIDATES+=("${SCRATCH}/data/output/100[0-4]-*")
else
    if [ "${MODELS_ONLY}" -eq 0 ]; then
        # Prep stages from run_0 alone. train_array.sh re-runs stages 500-505 in
        # all five run roots from identical code and identical input, so the
        # other four copies are redundant and roughly five times the transfer.
        CANDIDATES+=("${SCRATCH}/run_0/data/output/50[0-5]-*")
        CANDIDATES+=("${SCRATCH}/run_0/data/output/700-*")
    fi
    # Model directories from every run root: each array task enables exactly one
    # trainer, so these five never collide.
    CANDIDATES+=("${SCRATCH}/run_*/data/output/100[0-4]-*")
fi

# Resolve first, transfer second. One ssh call asks the remote shell to expand
# each glob, tagged so an empty group can be named. Doing this before any bytes
# move is what lets a legitimately absent directory (700-* after an array run)
# be skipped with a note instead of aborting scp partway through.
REMOTE_SCRIPT=""
for glob in "${CANDIDATES[@]}"; do
    REMOTE_SCRIPT="${REMOTE_SCRIPT}echo '### ${glob}'; ls -d ${glob} 2>/dev/null || true; "
done

# stderr is discarded because Roihu's sshd prints three post-quantum warnings on
# every connection; our own message below covers the failure case.
if ! RESOLVED="$(ssh "${REMOTE_HOST}" "${REMOTE_SCRIPT}" 2>/dev/null)"; then
    echo "ERROR: could not reach ${REMOTE_HOST}." >&2
    echo "       The SSH certificate lasts 24 h and expiry looks like a broken" >&2
    echo "       setup. Run hpc/roihu-auth.sh and try again." >&2
    exit 1
fi

PATHS=()
MISSING=()
current_glob=""
current_count=0
while IFS= read -r line; do
    case "${line}" in
        "### "*)
            if [ -n "${current_glob}" ] && [ "${current_count}" -eq 0 ]; then
                MISSING+=("${current_glob}")
            fi
            current_glob="${line#\#\#\# }"
            current_count=0
            ;;
        "")
            ;;
        *)
            PATHS+=("${line}")
            current_count=$((current_count + 1))
            ;;
    esac
done <<< "${RESOLVED}"
if [ -n "${current_glob}" ] && [ "${current_count}" -eq 0 ]; then
    MISSING+=("${current_glob}")
fi

if [ "${#MISSING[@]}" -gt 0 ]; then
    for glob in "${MISSING[@]}"; do
        echo "NOTE: nothing matched ${glob} — skipping."
        case "${glob}" in
            *700-*)
                echo "      (700-shap_correlation_analysis is produced only by"
                echo "       hpc/train_all.sh; after an array run this is normal.)"
                ;;
        esac
    done
fi

if [ "${#PATHS[@]}" -eq 0 ]; then
    echo "ERROR: nothing to fetch. Searched:" >&2
    for glob in "${CANDIDATES[@]}"; do
        echo "         ${glob}" >&2
    done
    echo "       Has the job finished? Check with: ssh ${REMOTE_HOST} squeue --me" >&2
    echo "       Ran hpc/train_all.sh rather than hpc/train_array.sh? Add --train-all." >&2
    exit 1
fi

echo
echo "Fetching ${#PATHS[@]} directories:"
# Unquoted on purpose: the remote shell must see separate arguments. Output
# directory names contain no spaces. || true so an unreadable directory costs a
# size figure, not the transfer.
ssh "${REMOTE_HOST}" "du -csh ${PATHS[*]}" 2>/dev/null || echo "  (size unavailable)"

mkdir -p results
if [ -n "$(ls -A results 2>/dev/null)" ]; then
    echo
    echo "NOTE: results/ is not empty; same-named directories will be overwritten."
    echo "      Keep the previous run first with: mv results results-\$(date +%Y%m%d)"
fi

REMOTE_ARGS=()
for path in "${PATHS[@]}"; do
    REMOTE_ARGS+=("${REMOTE_HOST}:${path}")
done

echo
echo "Copying into ./results/ ..."
scp -r "${REMOTE_ARGS[@]}" ./results/

echo "Done. ${#PATHS[@]} directories under $(pwd)/results/"
echo "Scratch deletes anything untouched for 180 days, so this copy is the durable one."
