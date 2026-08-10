#!/bin/bash
#SBATCH --job-name=railway-scenarios
#SBATCH --account=project_2019266
#SBATCH --partition=small
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --array=0-39
#SBATCH --output=slurm-scenarios-%A_%a.out
#
# Every feature scenario against every model: 8 x 5 = 40 independent tasks.
#
# Task id decomposes scenario-major, so one scenario's five models are
# contiguous and cancelling a tail range loses whole scenarios rather than
# fragments of many:
#   scenario = id / 5 + 1     0-4 -> s01, 5-9 -> s02, ... 35-39 -> s08
#   model    = MODELS[id % 5]
#
# Each task re-runs the shared preparation stages (merge, filter, select, split,
# balance, scale) in its own data root. Stages 502-505 genuinely differ per
# scenario because they sit downstream of select_training_cols; 500-501 do not,
# and are duplicated deliberately to keep every task independent — a failure in
# one cell blocks nothing, and any single cell can be re-run with
# `sbatch --array=<id> hpc/train_scenarios.sh`.
#
# Those stages communicate through fixed-name files (data/output/500-*..505-*)
# whose names depend only on the data date range, so concurrent tasks would
# target byte-identical paths under a shared root: one task's non-atomic parquet
# write can be read mid-write by another. Per-task roots avoid that race.
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
SCRATCH="/scratch/${PROJECT}/railway-fmi"

# The scenario catalogue. Gitignored and uploaded straight to this clone with
# hpc/push-features.sh, so changing a scenario needs no commit, push or pull.
# Checked here rather than left to Python so a forgotten upload costs a second
# instead of the whole allocation. sbatch exports the submitting environment,
# so COLUMNS_FILE=<path> sbatch ... overrides the default.
COLUMNS_FILE="${COLUMNS_FILE:-config/scenarios.txt}"
if [ ! -s "${COLUMNS_FILE}" ]; then
    echo "ERROR: columns file missing or empty: ${COLUMNS_FILE}" >&2
    echo "       Upload it from your laptop with:" >&2
    echo "         hpc/push-features.sh config/scenarios.txt" >&2
    echo "       or submit with: COLUMNS_FILE=<path> sbatch hpc/$(basename "${0}")" >&2
    exit 1
fi

export PATH="/projappl/${PROJECT}/railway-env/bin:$PATH"

MODELS=(xgboost lightgbm random_forest logistic_regression naive_bayes)
SCENARIO=$((SLURM_ARRAY_TASK_ID / 5 + 1))
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID % 5))]}"

# The array range is fixed at 0-39 in the #SBATCH header, but the catalogue is
# uploaded separately and could hold a different number of sections. Checking
# here turns a silent wrong-scenario run into an immediate, legible failure.
#
# Captured rather than piped into wc, for the diagnostic — not for safety.
# `... | wc -l` does abort correctly here: pipefail returns the rightmost
# NON-ZERO status, so main.py's 1 wins over wc's 0 and set -e stops the run.
# What it does not do is say anything. The script dies with a traceback above
# it and no line stating what was being attempted. Capturing lets the failure
# name itself in the slurm .out file, which is the only record a finished
# cluster job leaves behind.
if ! SCENARIO_LIST="$(python main.py --columns-file "${COLUMNS_FILE}" --list-scenarios)"; then
    echo "ERROR: could not read scenarios from ${COLUMNS_FILE}." >&2
    echo "       main.py's own error is above this line." >&2
    exit 1
fi
N_SCENARIOS="$(printf '%s\n' "${SCENARIO_LIST}" | grep -c .)"
EXPECTED=$((N_SCENARIOS * ${#MODELS[@]} - 1))
if [ "${SCENARIO}" -gt "${N_SCENARIOS}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} wants scenario ${SCENARIO}, but" >&2
    echo "       ${COLUMNS_FILE} holds only ${N_SCENARIOS} scenarios." >&2
    echo "       Resubmit with: sbatch --array=0-${EXPECTED} hpc/$(basename "${0}")" >&2
    exit 1
fi

SLUG="$(printf "s%02d" "${SCENARIO}")"

# Recorded so a finished log identifies its feature set: the sha256 proves two
# runs differ, the scenario name and the list printed by config/const_training.py
# say how.
echo "Features: ${COLUMNS_FILE} scenario ${SCENARIO}/${N_SCENARIOS}, sha256 $(sha256sum "${COLUMNS_FILE}" | cut -c1-16)"

# Per-task data root: isolates the 500-*..505-* intermediate files from the
# other 39 concurrently running tasks (see comment above). Slugged rather than
# named, because scenario names contain spaces, + and parentheses.
RUNROOT="${SCRATCH}/run_${SLUG}_${MODEL}"
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

echo "Array task ${SLURM_ARRAY_TASK_ID}: scenario ${SCENARIO} (${SLUG}), model ${MODEL}, root ${RUNROOT}"

srun python main.py \
    --data-root "${RUNROOT}" \
    --model "${MODEL}" \
    --columns-file "${COLUMNS_FILE}" \
    --scenario "${SCENARIO}"

echo "Finished ${SLUG}/${MODEL}. Check with: seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
