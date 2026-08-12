#!/bin/bash
#SBATCH --job-name=railway-train-all
#SBATCH --account=project_2019266
#SBATCH --partition=small
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --output=slurm-train-all-%j.out
#
# The sequential path pays the array's per-model costs end to end, so the old
# 12h / 32G header was the least survivable of the three scripts: on run 580873
# logistic_regression alone exceeded 8h on the wider feature sets, which leaves
# nothing for the other four models. 48h covers the whole chain with margin and
# stays inside the partition's 3-day cap; 64G matches the array scripts, and at
# 64 cores the free memory entitlement is ~124G, so it costs nothing extra.
#
# All five models sequentially in one job, using the full TRAINING_STATE_MACHINE
# from config (including shap_correlation_analysis).
#
# 64 cores rather than a full 384-core node: each trainer runs
# (10+20+30+40+50) x 5 = 750 fits, which does not usefully saturate 384 cores,
# and `small` queues faster than `medium`.
set -euo pipefail

# Slurm COPIES the batch script into a spool directory before running it, so $0
# is /var/spool/slurmd/job<N>/slurm_script - NOT this file. Deriving the repo
# root from $0 therefore lands in /var/spool/slurmd and main.py is not found.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from, which is the repo
# root in normal use; it is also Slurm's default cwd, so this is belt and braces.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f main.py ]; then
    echo "ERROR: main.py not found in $(pwd)." >&2
    echo "       Submit from the repository root, e.g.:" >&2
    echo "         cd /projappl/project_2019266/railway-fmi-code" >&2
    echo "         sbatch hpc/$(basename "${0}")" >&2
    exit 1
fi

PROJECT="project_2019266"
if [ "${PROJECT}" = "<project>" ]; then
    echo "ERROR: edit this script and set PROJECT to your CSC project ID (see MyCSC)." >&2
    exit 1
fi
SCRATCH="/scratch/${PROJECT}/railway-fmi"

# Which features to train on. The file is gitignored and uploaded straight to
# this clone with hpc/push-features.sh, so changing the feature set needs no
# commit, push or pull. Checked here rather than left to Python so a forgotten
# upload costs a second instead of the whole allocation. sbatch exports the
# submitting environment, so COLUMNS_FILE=<path> sbatch ... overrides the default.
COLUMNS_FILE="${COLUMNS_FILE:-config/features.txt}"
if [ ! -s "${COLUMNS_FILE}" ]; then
    echo "ERROR: columns file missing or empty: ${COLUMNS_FILE}" >&2
    echo "       Upload it from your laptop with hpc/push-features.sh," >&2
    echo "       or submit with: COLUMNS_FILE=<path> sbatch hpc/$(basename "${0}")" >&2
    exit 1
fi
# Recorded so a finished log identifies its feature set: the sha256 proves two
# runs differ, the list printed by config/const_training.py says how.
echo "Features: $(grep -cve '^[[:space:]]*\(#.*\)\?$' "${COLUMNS_FILE}") columns, sha256 $(sha256sum "${COLUMNS_FILE}" | cut -c1-16)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONIOENCODING=utf-8

# Since Slurm 22.05, srun no longer inherits --cpus-per-task from the batch
# allocation on its own; without this a step can start with just 1 CPU and the
# 750-fit sweep runs serially, blowing the 12h walltime.
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export PATH="/projappl/${PROJECT}/railway-env/bin:$PATH"

srun python main.py \
    --data-root "${SCRATCH}" \
    --columns-file "${COLUMNS_FILE}"

echo "Finished. Check CPU efficiency with: seff ${SLURM_JOB_ID}"
