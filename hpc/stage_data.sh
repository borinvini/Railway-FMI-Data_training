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

rsync -av --progress \
    data/output/101-preprocessed_training_ready/ \
    "${REMOTE}:${SCRATCH}/data/output/101-preprocessed_training_ready/"

echo "Staged to ${REMOTE}:${SCRATCH}"
ssh "${REMOTE}" "echo \"Files: \$(ls -1 '${SCRATCH}/data/output/101-preprocessed_training_ready' | wc -l) (expected 96)\"; du -sh '${SCRATCH}'"
