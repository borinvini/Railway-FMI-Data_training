#!/bin/bash
# Copy code and the preprocessed training data to Roihu scratch.
# Run on the Roihu LOGIN node, from the repo root. Not a batch job.
#
# Only data/output/101-preprocessed_training_ready is copied (~25 MB, 96 files):
# training reads exclusively from there (see training_pipeline.merge_data_files),
# and data/input (2.6 GB of raw parquet) is used only by preprocessing, which we
# do not run on the cluster.
set -euo pipefail

PROJECT="<project>"
SCRATCH="/scratch/${PROJECT}/railway-fmi"

mkdir -p "${SCRATCH}/data/output/101-preprocessed_training_ready"
mkdir -p "${SCRATCH}/data/input"
mkdir -p "${SCRATCH}/data/output/log"

rsync -av --progress \
    data/output/101-preprocessed_training_ready/ \
    "${SCRATCH}/data/output/101-preprocessed_training_ready/"

echo "Staged to ${SCRATCH}"
echo "Files: $(ls -1 "${SCRATCH}/data/output/101-preprocessed_training_ready" | wc -l) (expected 96)"
du -sh "${SCRATCH}"
