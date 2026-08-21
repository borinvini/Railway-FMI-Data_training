#!/bin/bash
# Push training data to Roihu scratch over SSH.
# Run on your LOCAL machine, from the repo root. Not a batch job, and not run
# on Roihu itself: .gitignore excludes *.parquet, so a fresh clone on Roihu has
# no data/ to copy from.
#
# Two payloads, because they differ by a factor of a hundred:
#
#   default        data/output/101-preprocessed_training_ready  (~26 MB, 96 files)
#                  What the five tabular trainers read. Staged every time.
#
#   --raw          data/input                                   (~2.6 GB, 96 files)
#                  What fcll_net reads. Its sequences — the target train's
#                  previous stations, and the trains ahead of it — exist only
#                  before filter_by_target_station collapses the data to one
#                  station, so the preprocessed file cannot serve it.
#
# The raw set is opt-in rather than automatic: it changes a 26 MB re-stage into
# a 2.6 GB one, and it only needs re-sending when the raw parquet itself
# changes, which is far rarer than re-preprocessing. Stage it once, then omit
# the flag on subsequent runs.
set -euo pipefail

PROJECT="project_2019266"
REMOTE_USER="vpozzobo"
REMOTE_HOST="roihu-cpu.csc.fi"

STAGE_RAW=0
while [ $# -gt 0 ]; do
    case "$1" in
        --raw)
            STAGE_RAW=1
            ;;
        -h|--help)
            echo "Usage: hpc/stage_data.sh [--raw]"
            echo "  --raw   also stage data/input (~2.6 GB), required by fcll_net"
            exit 0
            ;;
        *)
            # Refused, never ignored: a silently dropped --raw means the user
            # believes 2.6 GB went up, and the fcll_net cells die on the cluster
            # hours later with 'no input parquet'.
            echo "ERROR: Unknown option: $1" >&2
            echo "Usage: hpc/stage_data.sh [--raw]" >&2
            exit 1
            ;;
    esac
    shift
done

if [ "${PROJECT}" = "<project>" ] || [ "${REMOTE_USER}" = "<username>" ]; then
    echo "ERROR: edit this script and set PROJECT (see MyCSC) and REMOTE_USER." >&2
    exit 1
fi
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
SCRATCH="/scratch/${PROJECT}/railway-fmi"

SRC_PREPROCESSED="data/output/101-preprocessed_training_ready"
SRC_RAW="data/input"

# transfer <local-dir> <remote-dir> <label>
# rsync when available: at 2.6 GB resumability stops being a nicety, since scp
# restarts from zero on a dropped connection. Git Bash on Windows ships no
# rsync, hence the fallback.
transfer() {
    local src="$1" dest_dir="$2" label="$3"
    if [ ! -d "${src}" ]; then
        echo "ERROR: ${src} not found. Run this from the repository root on the machine" >&2
        echo "       that holds the data (your local machine, not Roihu)." >&2
        exit 1
    fi
    echo "Staging ${label}: ${src} ($(du -sh "${src}" 2>/dev/null | cut -f1))"
    ssh "${REMOTE}" "mkdir -p '${dest_dir}'"
    if command -v rsync >/dev/null 2>&1; then
        rsync -av --progress "${src}/" "${REMOTE}:${dest_dir}/"
    else
        echo "rsync not found; falling back to scp (no resume if the link drops)."
        scp -r "${src}/." "${REMOTE}:${dest_dir}/"
    fi
}

ssh "${REMOTE}" "mkdir -p '${SCRATCH}/data/output/log'"

transfer "${SRC_PREPROCESSED}" "${SCRATCH}/${SRC_PREPROCESSED}" "preprocessed training data"

if [ "${STAGE_RAW}" -eq 1 ]; then
    transfer "${SRC_RAW}" "${SCRATCH}/${SRC_RAW}" "raw event data (fcll_net)"
else
    echo
    echo "NOTE: data/input was NOT staged (default). fcll_net cannot run without"
    echo "      it — re-run with --raw if you are training fcll_net on the grid."
fi

echo
echo "Staged to ${REMOTE}:${SCRATCH}"
ssh "${REMOTE}" "
    echo \"Preprocessed: \$(ls -1 '${SCRATCH}/${SRC_PREPROCESSED}' 2>/dev/null | wc -l) files (expected 96)\"
    echo \"Raw input:    \$(ls -1 '${SCRATCH}/${SRC_RAW}' 2>/dev/null | wc -l) files (expected 96 if staged with --raw)\"
    du -sh '${SCRATCH}'
"
