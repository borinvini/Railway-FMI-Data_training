#!/bin/bash
# Push the preprocessed training data to Roihu scratch over SSH.
# Run on your LOCAL machine, from the repo root. Not a batch job, and not run
# on Roihu itself: .gitignore excludes *.parquet, so a fresh clone on Roihu has
# no data/output/101-preprocessed_training_ready/ to copy from.
#
# Only data/output/101-preprocessed_training_ready is copied (~25 MB, 96 files):
# training reads exclusively from there (see training_pipeline.merge_data_files),
# and data/input (2.6 GB of raw parquet) is used only by preprocessing, which we
# do not run on the cluster.
set -euo pipefail

PROJECT="project_2019266"
REMOTE_USER="vpozzobo"
REMOTE_HOST="roihu.csc.fi"
if [ "${PROJECT}" = "<project>" ] || [ "${REMOTE_USER}" = "<username>" ]; then
    echo "ERROR: edit this script and set PROJECT (see MyCSC) and REMOTE_USER." >&2
    exit 1
fi
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
SCRATCH="/scratch/${PROJECT}/railway-fmi"

ssh "${REMOTE}" "mkdir -p '${SCRATCH}/data/output/101-preprocessed_training_ready' '${SCRATCH}/data/output/log'"

SRC="data/output/101-preprocessed_training_ready"
DEST="${REMOTE}:${SCRATCH}/data/output/101-preprocessed_training_ready/"

if [ ! -d "${SRC}" ]; then
    echo "ERROR: ${SRC} not found. Run this from the repository root on the machine" >&2
    echo "       that holds the preprocessed data (your local machine, not Roihu)." >&2
    exit 1
fi

# rsync is absent from Git Bash on Windows, so fall back to scp. The payload is
# ~25 MB across 96 files, so rsync's delta transfer buys us nothing here — the
# only real loss is resumability, which barely matters at this size.
if command -v rsync >/dev/null 2>&1; then
    echo "Transferring with rsync..."
    rsync -av --progress "${SRC}/" "${DEST}"
else
    echo "rsync not found; falling back to scp (fine for ~25 MB)."
    scp -r "${SRC}/." "${DEST}"
fi

echo "Staged to ${REMOTE}:${SCRATCH}"
ssh "${REMOTE}" "echo \"Files: \$(ls -1 '${SCRATCH}/data/output/101-preprocessed_training_ready' | wc -l) (expected 96)\"; du -sh '${SCRATCH}'"
