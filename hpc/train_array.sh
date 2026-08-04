#!/bin/bash
#SBATCH --job-name=railway-train
#SBATCH --account=project_2019266
#SBATCH --partition=small
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --array=0-4
#SBATCH --output=slurm-train-%A_%a.out
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

cd "$(dirname "$(readlink -f "$0")")/.."

PROJECT="project_2019266"
if [ "${PROJECT}" = "<project>" ]; then
    echo "ERROR: edit this script and set PROJECT to your CSC project ID (see MyCSC)." >&2
    exit 1
fi
SCRATCH="/scratch/${PROJECT}/railway-fmi"

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
    --model "${MODEL}"

echo "Finished ${MODEL}. Check with: seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
