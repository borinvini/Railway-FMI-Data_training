#!/bin/bash
#SBATCH --job-name=railway-smoke
#SBATCH --account=project_2019266
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-smoke-%j.out
#
# Cheapest end-to-end proof that the port works, before spending real allocation.
# Runs one model at the minimum search budget, using xgboost rather than a
# trivial model so this actually exercises XGBoost's OpenMP threading and
# validates the SEARCH_N_JOBS / MODEL_N_JOBS split (see docs/CSC-SETUP.md).
#
# --search-iterations 10 is the SMALLEST valid value: the trainer iterates
# range(10, N+1, 10), which is empty below 10 and would silently do nothing.
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

# Inner estimator threads are pinned to 1 (MODEL_N_JOBS) so the allocated cores
# go to the outer CV workers instead of competing OpenMP threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# The pipeline prints emoji; without this a non-UTF-8 locale raises
# UnicodeEncodeError when writing this .out file.
export PYTHONIOENCODING=utf-8

# Since Slurm 22.05, srun no longer inherits --cpus-per-task from the batch
# allocation on its own; without this a step can start with just 1 CPU and
# never exercise the parallelism this smoke test is meant to validate.
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export PATH="/projappl/${PROJECT}/railway-env/bin:$PATH"

srun python main.py \
    --data-root "${SCRATCH}" \
    --model xgboost \
    --search-iterations 10

echo "Smoke test finished. Check CPU efficiency with: seff ${SLURM_JOB_ID}"
