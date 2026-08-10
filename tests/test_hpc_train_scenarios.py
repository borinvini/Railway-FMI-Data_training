"""Static checks for hpc/train_scenarios.sh.

The script calls sbatch/srun and cannot run off the cluster. What these tests
protect is the arithmetic that decides which of the 40 cells a task is — get it
wrong and the run silently trains the wrong model on the wrong feature set,
which no amount of downstream analysis would reveal.
"""
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "hpc" / "train_scenarios.sh"
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")

MODELS = ["xgboost", "lightgbm", "random_forest", "logistic_regression", "naive_bayes"]

# On Windows, plain "bash" resolves via CreateProcess's System32-first search
# to C:\Windows\System32\bash.exe (a WSL1 launcher), not Git Bash - and that
# WSL bash cannot see Windows-style paths. shutil.which() walks PATH the way a
# shell would and finds Git Bash's /usr/bin/bash.exe first. See
# git-bash-windows-gotchas memory note.
BASH = shutil.which("bash") or "bash"


def test_the_script_is_valid_bash():
    result = subprocess.run(
        [BASH, "-n", str(SCRIPT_PATH)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_the_array_covers_exactly_eight_scenarios_times_five_models():
    assert "#SBATCH --array=0-39" in SCRIPT


def test_the_model_list_matches_train_array_order():
    """The index math is model = MODELS[id % 5]; a reordered list silently
    relabels every result directory."""
    assert f"MODELS=({' '.join(MODELS)})" in SCRIPT


def test_the_scenario_index_is_one_based_and_scenario_major():
    """scenario = id/5 + 1 keeps a scenario's five models contiguous, so
    cancelling a tail range loses whole scenarios rather than fragments."""
    assert "SLURM_ARRAY_TASK_ID / 5 + 1" in SCRIPT
    assert "SLURM_ARRAY_TASK_ID % 5" in SCRIPT


def test_the_run_root_is_slugged_and_zero_padded():
    """Slugs keep spaces, + and parentheses off the cluster entirely."""
    assert 'SLUG="$(printf "s%02d" "${SCENARIO}")"' in SCRIPT
    assert 'RUNROOT="${SCRATCH}/run_${SLUG}_${MODEL}"' in SCRIPT


def test_the_shared_preprocessed_input_is_symlinked_not_copied():
    assert "101-preprocessed_training_ready" in SCRIPT
    assert "ln -sfn" in SCRIPT


def test_both_the_columns_file_and_the_scenario_reach_main_py():
    assert '--columns-file "${COLUMNS_FILE}"' in SCRIPT
    assert '--scenario "${SCENARIO}"' in SCRIPT
    assert '--model "${MODEL}"' in SCRIPT
    assert '--data-root "${RUNROOT}"' in SCRIPT


def test_the_default_columns_file_is_the_scenario_catalogue():
    assert 'COLUMNS_FILE="${COLUMNS_FILE:-config/scenarios.txt}"' in SCRIPT


def test_a_failed_scenario_listing_is_reported_as_itself():
    """A bare `... --list-scenarios | wc -l` aborts correctly under pipefail —
    the rightmost non-zero status wins — but it aborts *silently*, leaving a
    traceback and no line saying what was being attempted. Capturing the
    listing and checking its status lets the failure name itself in the slurm
    .out file, which is all a finished cluster job leaves behind."""
    assert "--list-scenarios | wc -l" not in SCRIPT
    assert 'if ! SCENARIO_LIST="$(python main.py' in SCRIPT
    assert "could not read scenarios from" in SCRIPT
    failure_at = SCRIPT.index("could not read scenarios from")
    srun_at = SCRIPT.index("srun python main.py")
    assert failure_at < srun_at, "the listing guard must run before srun"


def test_a_scenario_count_mismatch_fails_before_srun():
    """A catalogue with 6 sections submitted as --array=0-39 must fail
    immediately and name the correct range, not after queueing."""
    assert "--list-scenarios" in SCRIPT
    guard_at = SCRIPT.index("--list-scenarios")
    srun_at = SCRIPT.index("srun python main.py")
    assert guard_at < srun_at, "the count check must run before srun"
    assert "--array=0-" in SCRIPT[guard_at:srun_at], "the error must name the right range"


def test_srun_cpus_per_task_is_exported():
    """Since Slurm 22.05 srun does not inherit --cpus-per-task; without this a
    step starts with 1 CPU and the 750-fit sweep blows the walltime."""
    assert "export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}" in SCRIPT


def test_thread_pinning_is_exported():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f"export {var}=1" in SCRIPT
    assert "export PYTHONIOENCODING=utf-8" in SCRIPT


def test_it_runs_from_the_submit_dir_not_the_slurm_spool():
    """Slurm copies the batch script into /var/spool/slurmd, so $0-derived paths
    do not find main.py."""
    assert 'cd "${SLURM_SUBMIT_DIR:-$(pwd)}"' in SCRIPT


def test_the_columns_file_guard_precedes_srun():
    assert 'if [ ! -s "${COLUMNS_FILE}" ]; then' in SCRIPT
    assert SCRIPT.index('if [ ! -s "${COLUMNS_FILE}" ]; then') < SCRIPT.index("srun python main.py")


def test_the_index_math_is_a_bijection_over_the_grid():
    """Mirror of the shell arithmetic, proving the 40 ids map onto the 8x5 grid
    exactly once each — no cell trained twice, none skipped."""
    cells = {(i // 5 + 1, MODELS[i % 5]) for i in range(40)}
    assert len(cells) == 40
    assert {scenario for scenario, _ in cells} == set(range(1, 9))
    assert {model for _, model in cells} == set(MODELS)


def test_the_existing_array_script_is_untouched():
    """train_array.sh must keep working throughout: the 5-job flow is the
    fallback while this is being tested."""
    body = (REPO_ROOT / "hpc" / "train_array.sh").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-4" in body
    assert 'COLUMNS_FILE="${COLUMNS_FILE:-config/features.txt}"' in body
