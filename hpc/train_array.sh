#!/bin/bash
#SBATCH --job-name=railway-train
#SBATCH --account=project_2019266
#SBATCH --partition=small
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --output=slurm-train-%A_%a.out
#
# Same five trainers and the same per-task shape as train_scenarios.sh, so the
# 32G / 8h header would fail here exactly the way it failed there on run 580873:
# OOM on the widest feature sets, walltime on logistic_regression. See the
# header of hpc/train_scenarios.sh for the measurements behind these numbers.
# Whichever feature set config/features.txt happens to hold, it can be the wide
# one, so this script gets the same limits rather than a guess about the input.
#
# One model per array task. Each task re-runs the shared preparation stages
# (merge, filter, select, split, balance, scale) because they are cheap relative
# to training and this keeps tasks independent — a failure in one model does not
# block the others.
#
# Those preparation stages communicate through fixed-name files (data/output/
# 500-*..505-*) whose names depend only on the data date range, so all five
# concurrent tasks would target byte-identical paths if run against the same
# data root: one task's non-atomic parquet write can be read mid-write by
# another, corrupting its input. To avoid that race, each task gets its own
# data root (a scratch subdirectory) with the shared, read-only preprocessed
# input symlinked in rather than copied. Only data/output/1000-*..1004-* (the
# final model directories) are genuinely conflict-free between tasks; nothing
# upstream of them is.
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

MODELS=(xgboost lightgbm random_forest logistic_regression naive_bayes)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# Per-task data root: isolates the 500-*..505-* intermediate files from the
# other four concurrently running tasks (see comment above).
RUNROOT="${SCRATCH}/run_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${RUNROOT}/data/output"
ln -sfn "${SCRATCH}/data/output/101-preprocessed_training_ready" \
        "${RUNROOT}/data/output/101-preprocessed_training_ready"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONIOENCODING=utf-8

# Since Slurm 22.05, srun no longer inherits --cpus-per-task from the batch
# allocation on its own; without this a step can start with just 1 CPU and the
# 750-fit sweep runs serially, blowing the 8h walltime.
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export PATH="/projappl/${PROJECT}/railway-env/bin:$PATH"

echo "Array task ${SLURM_ARRAY_TASK_ID}: training ${MODEL} in ${RUNROOT}"

srun python main.py \
    --data-root "${RUNROOT}" \
    --model "${MODEL}" \
    --columns-file "${COLUMNS_FILE}"

echo "Finished ${MODEL}. Check with: seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
