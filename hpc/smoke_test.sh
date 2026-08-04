#!/bin/bash
#SBATCH --job-name=railway-smoke
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-smoke-%j.out
#
# Cheapest end-to-end proof that the port works, before spending real allocation.
# Runs one fast model at the minimum search budget.
#
# --search-iterations 10 is the SMALLEST valid value: the trainer iterates
# range(10, N+1, 10), which is empty below 10 and would silently do nothing.
set -euo pipefail

PROJECT="<project>"
SCRATCH="/scratch/${PROJECT}/railway-fmi"

# Inner estimator threads are pinned to 1 (MODEL_N_JOBS) so the allocated cores
# go to the outer CV workers instead of competing OpenMP threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# The pipeline prints emoji; without this a non-UTF-8 locale raises
# UnicodeEncodeError when writing this .out file.
export PYTHONIOENCODING=utf-8

export PATH="/projappl/${PROJECT}/railway-env/bin:$PATH"

srun python main.py \
    --data-root "${SCRATCH}" \
    --model naive_bayes \
    --search-iterations 10

echo "Smoke test finished. Check CPU efficiency with: seff ${SLURM_JOB_ID}"
