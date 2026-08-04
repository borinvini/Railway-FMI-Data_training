#!/bin/bash
#SBATCH --job-name=railway-train-all
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --output=slurm-train-all-%j.out
#
# All five models sequentially in one job, using the full TRAINING_STATE_MACHINE
# from config (including shap_correlation_analysis).
#
# 64 cores rather than a full 384-core node: each trainer runs
# (10+20+30+40+50) x 5 = 750 fits, which does not usefully saturate 384 cores,
# and `small` queues faster than `medium`.
set -euo pipefail

PROJECT="<project>"
SCRATCH="/scratch/${PROJECT}/railway-fmi"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONIOENCODING=utf-8

export PATH="/projappl/${PROJECT}/railway-env/bin:$PATH"

srun python main.py --data-root "${SCRATCH}"

echo "Finished. Check CPU efficiency with: seff ${SLURM_JOB_ID}"
