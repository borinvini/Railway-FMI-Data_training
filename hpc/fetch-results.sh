#!/bin/bash
# Fetch training-pipeline output from Roihu into ./results/.
#
# Usage: hpc/fetch-results.sh [--models] [--train-all] [--scenarios] [--stages]
#   (no flags)    hpc/train_array.sh layout: every stage, 500 through 1004
#   --models      only the model directories, 100[0-4]-*
#   --train-all   hpc/train_all.sh layout: everything under data/output/ directly
#   --scenarios   hpc/train_scenarios.sh layout: the 40 run_sNN_<model> roots,
#                 model directories only, sorted into results/<scenario name>/
#   --stages      --scenarios only: also pull the prep stages, one set per scenario
#
# --models and --train-all are orthogonal — one picks the subset, the other the
# layout — so `--train-all --models` is valid and means what it says.
#
# Note the deliberate asymmetry: the older modes SUBTRACT with --models, while
# --scenarios ADDS with --stages. With eight distinct prep sets rather than one,
# pulling them on every fetch is the wrong default; the older modes keep their
# behaviour so existing habits and docs stay correct.
set -euo pipefail

cd "$(dirname "${0}")/.."

REMOTE_HOST="roihu"
SCRATCH="/scratch/project_2019266/railway-fmi"

MODELS_ONLY=0
TRAIN_ALL=0
SCENARIOS=0
STAGES=0
for arg in "$@"; do
    case "${arg}" in
        --models)    MODELS_ONLY=1 ;;
        --train-all) TRAIN_ALL=1 ;;
        --scenarios) SCENARIOS=1 ;;
        --stages)    STAGES=1 ;;
        *)
            echo "ERROR: unknown argument: ${arg}" >&2
            echo "       Usage: hpc/fetch-results.sh [--models] [--train-all] [--scenarios] [--stages]" >&2
            exit 1
            ;;
    esac
done
if [ "${STAGES}" -eq 1 ] && [ "${SCENARIOS}" -eq 0 ]; then
    echo "ERROR: --stages applies only to --scenarios." >&2
    echo "       The other layouts fetch the prep stages by default; use --models to omit them." >&2
    exit 1
fi

# Candidate remote globs, as explicit character classes. Never 10*: that also
# matches 101-preprocessed_training_ready, which is pipeline INPUT staged up by
# hpc/stage_data.sh, and would come back down once per run directory.
CANDIDATES=()
if [ "${SCENARIOS}" -eq 1 ]; then
    if [ "${STAGES}" -eq 1 ]; then
        # One prep set per scenario, from that scenario's xgboost root alone.
        # The five models within a scenario re-run stages 500-505 from identical
        # code and identical input, so the other four copies are redundant and
        # five times the transfer.
        CANDIDATES+=("${SCRATCH}/run_s??_xgboost/data/output/50[0-5]-*")
    fi
    # Model directories from every run root: each task enables exactly one
    # trainer, so across all 40 roots these never collide.
    CANDIDATES+=("${SCRATCH}/run_s??_*/data/output/100[0-4]-*")
elif [ "${TRAIN_ALL}" -eq 1 ]; then
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
# Sizes come back over the SAME connection as the resolve, tagged with @@ so the
# parser below can tell them from paths. Every additional ssh or scp invocation
# is a full authentication, and with no ssh-agent loaded that is another
# passphrase prompt — see the transfer step for why that matters.
REMOTE_SCRIPT="${REMOTE_SCRIPT}du -csh ${CANDIDATES[*]} 2>/dev/null | sed 's/^/@@ /' || true; "

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
SIZES=""
current_glob=""
current_count=0
while IFS= read -r line; do
    case "${line}" in
        "@@ "*)
            SIZES="${SIZES}  ${line#@@ }
"
            ;;
        "### "*)
            if [ -n "${current_glob}" ] && [ "${current_count}" -eq 0 ]; then
                MISSING+=("${current_glob}")
            fi
            current_glob="${line#\#\#\# }"
            current_count=0
            ;;
        "")
            ;;
        /*)
            # Anchored to an absolute path on purpose: every real path from
            # `ls -d` is absolute, since the globs themselves are absolute.
            # A bare `*)` would also swallow a stray banner line (MOTD, a
            # maintenance notice) ahead of the first marker and hand it to
            # scp as a bogus path, aborting the whole transfer.
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
            *run_0/data/output/50\[0-5\]-*)
                echo "      (run_0 did not complete, so the prep stages are"
                echo "       unavailable; the model directories below may also be incomplete.)"
                ;;
            *run_s??_*)
                echo "      (no scenario run roots found. Either the array has"
                echo "       not finished — check with: ssh ${REMOTE_HOST} squeue --me —"
                echo "       or the run used hpc/train_array.sh, which writes run_0..run_4;"
                echo "       fetch that layout without --scenarios.)"
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
if [ -n "${SIZES}" ]; then
    printf '%s' "${SIZES}"
else
    echo "  (size unavailable)"
fi

mkdir -p results
if [ -n "$(ls -A results 2>/dev/null)" ]; then
    echo
    echo "NOTE: results/ is not empty; same-named directories will be overwritten."
    echo "      Keep the previous run first with: mv results results-\$(date +%Y%m%d)"
fi

# One connection for the whole transfer, deliberately not scp.
#
# scp opens a SEPARATE connection per remote source argument. With 40
# directories that would be 40 authentications on top of the resolve — and with
# no ssh-agent loaded, 40 passphrase prompts. A single tar stream is also
# markedly faster here, because a few hundred MB spread over many small files is
# dominated by per-file round trips rather than bandwidth.
TAR_ARGS=""
if [ "${SCENARIOS}" -eq 1 ]; then
    # Members are archived relative to ${SCRATCH}, keeping the run_sNN_<model>
    # prefix. Flattening to basenames the way the other modes do would collapse
    # all eight scenarios' 1000-xgboost_randomized_search directories onto each
    # other — the prefix is what carries the scenario home.
    # No quoting needed around either argument: run roots are slugged precisely
    # so no path on the cluster side ever contains a space.
    for path in "${PATHS[@]}"; do
        TAR_ARGS="${TAR_ARGS} -C ${SCRATCH} ${path#${SCRATCH}/}"
    done
else
    # The repeated `-C <parent> <name>` pairs are what make the sources land
    # side by side: each member is archived under its bare directory name, so
    # results/ ends up with the same flat layout scp produced, even though the
    # sources live under different run roots.
    for path in "${PATHS[@]}"; do
        TAR_ARGS="${TAR_ARGS} -C $(dirname "${path}") $(basename "${path}")"
    done
fi

echo
echo "Copying into ./results/ ..."
ssh "${REMOTE_HOST}" "tar -czf -${TAR_ARGS}" | tar -xzvf - -C results

if [ "${SCENARIOS}" -eq 1 ]; then
    # Sort run_sNN_<model>/data/output/1000-* into results/<scenario name>/1000-*.
    #
    # Done locally, and only here: scenario names contain spaces, + and
    # parentheses, and every remote path in this script goes through an
    # unquoted glob or `ls -d`. Keeping the awkward characters on this side of
    # the connection is the whole reason the cluster uses slugs.
    #
    # The slug -> name map comes from the same parser the cluster used, so the
    # two cannot drift. Without python on PATH (Git Bash on Windows often has
    # none) the transfer still succeeded; the directories simply stay slugged.
    if command -v python >/dev/null 2>&1; then
        echo
        echo "Sorting into per-scenario folders..."
        python hpc/sort_results.py results config/scenarios.txt
    else
        echo
        echo "NOTE: python not found on PATH; results stay in run_sNN_<model>/ form."
        echo "      Re-run this script with python available to sort them by scenario."
    fi
fi

echo "Done. ${#PATHS[@]} directories under $(pwd)/results/"
echo "Scratch deletes anything untouched for 180 days, so this copy is the durable one."
