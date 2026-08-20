"""Static checks for hpc/train_scenarios.sh.

The script calls sbatch/srun and cannot run off the cluster. What these tests
protect is the arithmetic that decides which cell of the scenarios x models grid
a task is — get it wrong and the run silently trains the wrong model on the
wrong feature set, which no amount of downstream analysis would reveal.

The array size itself is not pinned here. It is chosen at submit time by
hpc/submit-scenarios.sh (see tests/test_hpc_submit_scenarios.py); the header
value is only a fallback for a bare sbatch.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "hpc" / "train_scenarios.sh"
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")

MODELS = ["xgboost", "lightgbm", "random_forest", "logistic_regression", "naive_bayes"]

# Mirrors config/columns_file.py's banner rule, so the header test counts
# sections the same way the job will.
BANNER_RE = re.compile(r"^#\s*(\d+)\s*-\s*\S")

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


def test_the_header_array_falls_back_to_the_current_catalogue():
    """A bare `sbatch hpc/train_scenarios.sh` still has to do something sane.
    The header is not the source of truth — submit-scenarios.sh overrides it —
    but when it is used it must at least cover the catalogue in the repo."""
    catalogue = REPO_ROOT / "config" / "scenarios.txt"
    n_scenarios = sum(
        1 for line in catalogue.read_text(encoding="utf-8-sig").splitlines()
        if BANNER_RE.match(line.strip())
    )
    assert f"#SBATCH --array=0-{n_scenarios * len(MODELS) - 1}" in SCRIPT


def test_an_undersized_array_warns_without_failing():
    """The failure that cost run 581402: --array=0-39 against 13 scenarios ran
    scenarios 1-8 and never mentioned the other five.

    Non-fatal on purpose. Re-running one cell with `sbatch --array=17` is a
    documented workflow, and it submits an array of one; a hard failure would
    break it."""
    assert "SLURM_ARRAY_TASK_COUNT" in SCRIPT
    warn_at = SCRIPT.index("SLURM_ARRAY_TASK_COUNT")
    srun_at = SCRIPT.index("srun python main.py")
    assert warn_at < srun_at, "the warning must be emitted before the work starts"
    assert "submit-scenarios.sh" in SCRIPT, "the warning must name the fix"


def test_the_model_list_comes_from_the_shared_file():
    """The index math is model = MODELS[id % n]; a reordered or re-length'd list
    silently relabels every result directory. One definition, three consumers."""
    assert "models.sh" in SCRIPT
    assert "MODELS=(" not in SCRIPT, "the batch script must source, not redefine, MODELS"
    shared = (REPO_ROOT / "hpc" / "models.sh").read_text(encoding="utf-8")
    assert f"MODELS=({' '.join(MODELS)})" in shared


def test_the_scenario_index_is_one_based_and_scenario_major():
    """scenario = id/n + 1 keeps a scenario's models contiguous, so cancelling
    a tail range loses whole scenarios rather than fragments.

    Pins the complete assignments, not fragments: asserting only the divisor
    appears somewhere still passes when the two expressions are swapped between
    SCENARIO and MODEL. The divisor is ${#MODELS[@]} rather than a literal 5 so
    adding a sixth trainer needs no edit here."""
    assert "SCENARIO=$((SLURM_ARRAY_TASK_ID / ${#MODELS[@]} + 1))" in SCRIPT
    assert 'MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID % ${#MODELS[@]}))]}"' in SCRIPT


@pytest.mark.parametrize("task_id,expected_scenario,expected_model", [
    (0, 1, "xgboost"),
    (1, 1, "lightgbm"),
    (4, 1, "naive_bayes"),
    (5, 2, "xgboost"),
    (16, 4, "lightgbm"),
    (35, 8, "xgboost"),
    (39, 8, "naive_bayes"),
])
def test_the_shells_own_arithmetic_produces_the_documented_cell(
    task_id, expected_scenario, expected_model
):
    """Runs the script's actual assignment lines under bash rather than
    asserting they appear as text.

    A substring check passes even when the two expressions are swapped between
    SCENARIO and MODEL — a mutation that trains all 40 cells wrong while every
    static test stays green. This executes them, so it cannot.
    """
    lines = [line.strip() for line in SCRIPT.splitlines()]
    shared = (REPO_ROOT / "hpc" / "models.sh").read_text(encoding="utf-8")
    models_line = next(
        l.strip() for l in shared.splitlines() if l.strip().startswith("MODELS=(")
    )
    scenario_line = next(l for l in lines if l.startswith("SCENARIO="))
    model_line = next(l for l in lines if l.startswith("MODEL="))
    snippet = "\n".join([
        f"SLURM_ARRAY_TASK_ID={task_id}",
        models_line,
        scenario_line,
        model_line,
        'printf "%s %s" "${SCENARIO}" "${MODEL}"',
    ])
    result = subprocess.run(
        [BASH, "-c", snippet], capture_output=True, text=True, check=True
    )
    assert result.stdout == f"{expected_scenario} {expected_model}"


@pytest.mark.parametrize("task_id,task_min,task_count,should_warn,label", [
    (0, 0, 65, False, "the full grid warns about nothing"),
    (0, 0, 40, True, "a stale --array=0-39 against 13 scenarios is the bug"),
    (40, 40, 25, False, "a deliberate tail subset is not a mistake"),
    (17, 17, 1, False, "re-running one cell must stay usable"),
    (1, 0, 40, False, "only the first task speaks, not all 40"),
])
def test_the_undersized_array_warning_fires_only_when_it_should(
    task_id, task_min, task_count, should_warn, label
):
    """Executes the guard rather than asserting its text.

    A warning that fires on every task buries the message 40 times over, and one
    that fires on `sbatch --array=17` would train the reader to ignore it. Both
    mutations keep every static assertion green, so this runs the real
    conditional under the environments Slurm actually sets.
    """
    block = SCRIPT[
        SCRIPT.index('if [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MIN:-0}" ]'):
        SCRIPT.index('SLUG="$(printf')
    ]
    preamble = "\n".join([
        f"SLURM_ARRAY_TASK_ID={task_id}",
        f"SLURM_ARRAY_TASK_MIN={task_min}",
        f"SLURM_ARRAY_TASK_COUNT={task_count}",
        "EXPECTED=64",
        "N_SCENARIOS=13",
        "COLUMNS_FILE=config/scenarios.txt",
        f"MODELS=({' '.join(MODELS)})",
        "",
    ])
    result = subprocess.run(
        [BASH, "-c", preamble + block], capture_output=True, text=True
    )
    assert result.returncode == 0, "the warning must never abort the task"
    assert ("WARNING" in result.stderr) is should_warn, f"{label}: {result.stderr}"
    if should_warn:
        # Names the scenarios actually lost, not just that something is wrong:
        # run 581402 silently dropped exactly these five.
        assert "Scenarios 9-13" in result.stderr, result.stderr


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
    """Asserts the attribution wrapper, nothing stronger. Both forms abort
    correctly under pipefail, and main.py's argparse already prints a specific
    message to stderr, which no pipe on stdout could hide. What this pins is
    that the listing is captured and its status checked, so a 40-task array's
    .out file names the scenario-listing step instead of leaving the reader to
    infer which command produced the argparse error."""
    assert "--list-scenarios | wc -l" not in SCRIPT
    assert 'if ! SCENARIO_LIST="$(python main.py' in SCRIPT
    assert "could not read scenarios from" in SCRIPT
    failure_at = SCRIPT.index("could not read scenarios from")
    srun_at = SCRIPT.index("srun python main.py")
    assert failure_at < srun_at, "the listing guard must run before srun"


def test_a_scenario_count_mismatch_fails_before_srun():
    """A catalogue with 6 sections submitted as --array=0-39 must fail
    immediately and name the remedy, not burn an allocation first.

    The remedy is the wrapper, not a hand-typed range: telling the reader to
    resubmit with --array=0-29 fixes today's mismatch and leaves them doing the
    same arithmetic by hand next time the catalogue changes."""
    assert "--list-scenarios" in SCRIPT
    guard_at = SCRIPT.index("--list-scenarios")
    srun_at = SCRIPT.index("srun python main.py")
    assert guard_at < srun_at, "the count check must run before srun"
    assert "submit-scenarios.sh" in SCRIPT[guard_at:srun_at], (
        "the error must name the fix"
    )


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


@pytest.mark.parametrize("n_scenarios", [1, 8, 13, 20])
def test_the_index_math_is_a_bijection_over_the_grid(n_scenarios):
    """Mirror of the shell arithmetic, proving the ids map onto the
    scenarios x models grid exactly once each — no cell trained twice, none
    skipped — at any catalogue size, since the size is now variable."""
    n_tasks = n_scenarios * len(MODELS)
    cells = {(i // len(MODELS) + 1, MODELS[i % len(MODELS)]) for i in range(n_tasks)}
    assert len(cells) == n_tasks
    assert {scenario for scenario, _ in cells} == set(range(1, n_scenarios + 1))


def test_the_existing_array_script_still_covers_one_model_per_task():
    """train_array.sh must keep working: the single-scenario flow is the
    fallback. Its array is one task per model, so it tracks the shared list."""
    body = (REPO_ROOT / "hpc" / "train_array.sh").read_text(encoding="utf-8")
    assert f"#SBATCH --array=0-{len(MODELS) - 1}" in body
    assert 'COLUMNS_FILE="${COLUMNS_FILE:-config/features.txt}"' in body
    assert "models.sh" in body
    assert "MODELS=(" not in body
