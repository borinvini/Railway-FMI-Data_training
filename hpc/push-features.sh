#!/bin/bash
# Upload the feature selection to Roihu.
#
# config/features.txt is gitignored: the cluster reads this uploaded copy
# directly, so changing which features are trained on needs no commit, no push
# and no pull. Code changes still go through git as before.
#
# Usage: hpc/push-features.sh [local-file]
#   local-file defaults to config/features.txt and is resolved relative to the
#   repository root, not to your current directory.
set -euo pipefail

cd "$(dirname "${0}")/.."

REMOTE="roihu:/projappl/project_2019266/railway-fmi-code/config/features.txt"
LOCAL="${1:-config/features.txt}"

if [ ! -f "${LOCAL}" ]; then
    echo "ERROR: ${LOCAL} not found (paths are relative to the repo root)." >&2
    echo "       Start from the template:" >&2
    echo "         cp config/features.example.txt config/features.txt" >&2
    exit 1
fi

scp "${LOCAL}" "${REMOTE}"

echo "Uploaded $(grep -cve '^[[:space:]]*\(#.*\)\?$' "${LOCAL}") columns, sha256 $(sha256sum "${LOCAL}" | cut -c1-16)"
echo "That sha256 must match the 'Features:' line in the slurm log."
