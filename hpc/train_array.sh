#!/bin/bash
#SBATCH --job-name=railway-train
#SBATCH --account=<project>
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
# Each task writes to its own model directory (data/output/1000-* .. 1004-*), so
# there is no write conflict between concurrent tasks.
set -euo pipefail

PROJECT="<project>"
if [ "${PROJECT}" = "<project>" ]; then
    echo "ERROR: edit this script and set PROJECT to your CSC project ID (see MyCSC)." >&2
    exit 1
fi
SCRATCH="/scratch/${PROJECT}/railway-fmi"

MODELS=(xgboost lightgbm random_forest logistic_regression naive_bayes)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONIOENCODING=utf-8

export PATH="/projappl/${PROJECT}/railway-env/bin:$PATH"

echo "Array task ${SLURM_ARRAY_TASK_ID}: training ${MODEL}"

srun python main.py \
    --data-root "${SCRATCH}" \
    --model "${MODEL}"

echo "Finished ${MODEL}. Check with: seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
