"""Tests for hpc/submit-scenarios.sh, the login-node submit wrapper.

The wrapper exists because `#SBATCH --array=` is parsed by sbatch out of the
script text at submit time, on the login node, before any of the batch script's
own code runs. A batch script therefore cannot size its own array. Getting that
size wrong is silent: run 581402 submitted --array=0-39 against a 13-scenario
catalogue, and scenarios 9-13 were simply never queued while every task that did
run looked healthy.

What is tested here is the arithmetic that turns a scenario range into a task-id
range, and the guards that refuse to submit rather than submit something wrong.
sbatch itself is stubbed on PATH: the tests assert the command line the wrapper
builds, which is the whole contract.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "hpc" / "submit-scenarios.sh"

MODELS = ["xgboost", "lightgbm", "random_forest", "logistic_regression", "naive_bayes", "fcll_net"]

# The real catalogue's size, and the task range it implies. Derived rather than
# written out, so adding a trainer to hpc/models.sh needs no edit here — the
# same reason submit-scenarios.sh computes the array instead of hardcoding it.
N_SCENARIOS = 13


def _array(n_scenarios=N_SCENARIOS, first_scenario=1):
    start = (first_scenario - 1) * len(MODELS)
    return f"--array={start}-{n_scenarios * len(MODELS) - 1}"


def _resolve_bash():
    """Absolute path to Git Bash's bash.

    Bare "bash" is not safe on Windows: CreateProcess searches System32 first,
    and C:\\Windows\\System32\\bash.exe is the WSL launcher, which has a
    different filesystem view and cannot see the script (exit 127).
    """
    for cand in (os.environ.get("SHELL"), shutil.which("bash")):
        if cand and Path(cand).exists():
            return cand
    return "bash"


BASH = _resolve_bash()


def _catalogue(path, n_scenarios):
    """Write a scenario file with `n_scenarios` sections the real parser accepts.

    Each section carries trainDelayed plus one distinguishing column, which is
    all load_scenarios requires: a banner, at least one column, no duplicates
    within the section.
    """
    blocks = []
    for i in range(1, n_scenarios + 1):
        blocks.append(f"# {i} - TEST SCENARIO {i}\ntrainDelayed\ncolumn_{i}")
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return path


def _run(*args, columns_file=None, cwd=None, stub_sbatch=None):
    """Run the wrapper, optionally with a stubbed sbatch first on PATH."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if columns_file is not None:
        env["COLUMNS_FILE"] = str(columns_file)
    if stub_sbatch is not None:
        env["PATH"] = f"{stub_sbatch}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        [BASH, str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd) if cwd else str(REPO_ROOT),
    )


@pytest.fixture
def sbatch_stub(tmp_path):
    """A fake sbatch that records its argv, so a real submission is never made."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "sbatch.argv"
    stub = bindir / "sbatch"
    stub.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > ' f'"{log.as_posix()}"\n'
        'echo "Submitted batch job 999999"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return bindir, log


def test_the_script_is_valid_bash():
    result = subprocess.run(
        [BASH, "-n", str(SCRIPT_PATH)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_the_real_catalogue_sizes_the_array_to_every_scenario():
    """The bug this whole wrapper exists to prevent: a catalogue that has grown
    past the hardcoded header must produce an array covering all of it."""
    result = _run("--dry-run")
    assert result.returncode == 0, result.stderr
    assert _array() in result.stdout, result.stdout


def test_the_task_count_is_scenarios_times_models():
    result = _run("--dry-run")
    assert f"{N_SCENARIOS} scenarios" in result.stdout
    assert f"{len(MODELS)} models" in result.stdout
    assert f"{N_SCENARIOS * len(MODELS)} tasks" in result.stdout


@pytest.mark.parametrize("n_scenarios", [1, 8, 13, 20])
def test_the_array_tracks_the_catalogue_size(tmp_path, n_scenarios):
    """Any catalogue size, not just today's — the point is that no one has to
    edit a header again."""
    catalogue = _catalogue(tmp_path / "scen.txt", n_scenarios)
    result = _run("--dry-run", columns_file=catalogue)
    assert result.returncode == 0, result.stderr
    assert _array(n_scenarios) in result.stdout, result.stdout


@pytest.mark.parametrize("selector,first,last", [
    ("9-13", 9, 13),
    ("9", 9, 9),
    ("1", 1, 1),
    ("1-8", 1, 8),
    ("13", 13, 13),
])
def test_a_scenario_range_maps_to_the_matching_task_range(selector, first, last):
    """scenario is 1-based and scenario-major: first id = (n-1)*models,
    last id = m*models - 1. Off by one here re-runs the wrong feature set."""
    result = _run("--dry-run", "--scenarios", selector)
    assert result.returncode == 0, result.stderr
    assert _array(last, first_scenario=first) in result.stdout, result.stdout


def test_a_scenario_beyond_the_catalogue_is_refused(tmp_path):
    catalogue = _catalogue(tmp_path / "scen.txt", 5)
    result = _run("--dry-run", "--scenarios", "6", columns_file=catalogue)
    assert result.returncode != 0
    assert "5" in result.stderr, result.stderr


def test_a_backwards_range_is_refused():
    result = _run("--dry-run", "--scenarios", "13-9")
    assert result.returncode != 0
    assert "13-9" in result.stderr or "range" in result.stderr.lower()


def test_a_zero_or_negative_scenario_is_refused():
    """Scenario numbers are 1-based; 0 would compute a negative first task id."""
    result = _run("--dry-run", "--scenarios", "0")
    assert result.returncode != 0


def test_a_missing_catalogue_is_refused_with_the_upload_hint(tmp_path):
    result = _run("--dry-run", columns_file=tmp_path / "absent.txt")
    assert result.returncode != 0
    assert "push-features.sh" in result.stderr, result.stderr


def test_an_empty_catalogue_is_refused(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    result = _run("--dry-run", columns_file=empty)
    assert result.returncode != 0


def test_a_catalogue_the_parser_rejects_is_refused(tmp_path):
    """A file of column names with no banners is a features file, not a
    catalogue; submitting it would train scenario 1 of nothing."""
    bad = tmp_path / "features.txt"
    bad.write_text("trainDelayed\nAir temperature\n", encoding="utf-8")
    result = _run("--dry-run", columns_file=bad)
    assert result.returncode != 0


def test_dry_run_does_not_submit(sbatch_stub):
    bindir, log = sbatch_stub
    result = _run("--dry-run", stub_sbatch=bindir)
    assert result.returncode == 0, result.stderr
    assert not log.exists(), "dry run must not invoke sbatch"


def test_a_real_submission_passes_the_computed_array_to_sbatch(sbatch_stub):
    bindir, log = sbatch_stub
    result = _run("--scenarios", "9-13", stub_sbatch=bindir)
    assert result.returncode == 0, result.stderr
    argv = log.read_text(encoding="utf-8").split("\n")
    assert _array(13, first_scenario=9) in argv, argv
    assert any(a.endswith("hpc/train_scenarios.sh") for a in argv), argv


def test_extra_arguments_reach_sbatch(sbatch_stub):
    """So a one-off --time or --partition override needs no edit to the script."""
    bindir, log = sbatch_stub
    result = _run("--time=02:00:00", stub_sbatch=bindir)
    assert result.returncode == 0, result.stderr
    argv = log.read_text(encoding="utf-8").split("\n")
    assert "--time=02:00:00" in argv, argv


def test_it_works_from_any_directory(tmp_path, sbatch_stub):
    """Slurm's own scripts rely on SLURM_SUBMIT_DIR; this one runs before Slurm
    exists, so it must find the repo from its own path."""
    bindir, log = sbatch_stub
    result = _run("--dry-run", cwd=tmp_path, stub_sbatch=bindir)
    assert result.returncode == 0, result.stderr
    assert _array() in result.stdout


def test_the_model_list_comes_from_the_shared_file():
    """Duplicating MODELS is how the wrapper and the batch script drift apart:
    the wrapper sizes the array by count, the batch script indexes by position,
    and a mismatch mislabels every result directory."""
    body = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "models.sh" in body
    assert "MODELS=(" not in body, "the wrapper must source, not redefine, MODELS"


# --- --models: some of the models of the grid ---------------------------------------------------

FOUR = ["xgboost", "lightgbm", "random_forest", "logistic_regression"]


def _ids_to_spec(ids):
    """Compress consecutive task ids the way Slurm writes them: [0,1,2,3,6,7] -> '0-3,6-7'."""
    ids = sorted(ids)
    parts, start, prev = [], ids[0], ids[0]
    for i in ids[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if i is not None:
            start = prev = i
    return ",".join(parts)


def _subset_array(scenarios, models):
    ids = [(s - 1) * len(MODELS) + MODELS.index(m) for s in scenarios for m in models]
    return "--array=" + _ids_to_spec(ids)


def test_four_models_over_the_whole_catalogue_leave_gaps_for_the_other_two():
    """Task id = (scenario - 1) * len(MODELS) + model index, because train_scenarios.sh
    decodes it that way. Four of six models are therefore 0-3, 6-9, ... — not 0-51, which
    would run naive_bayes, fcll_net and the wrong scenarios."""
    result = _run("--dry-run", "--models", ",".join(FOUR))
    assert result.returncode == 0, result.stderr
    assert "--array=0-3,6-9,12-15,18-21,24-27,30-33,36-39,42-45,48-51,54-57,60-63,66-69,72-75" in result.stdout
    assert _subset_array(range(1, 14), FOUR) in result.stdout
    assert "13 scenarios x 4 models" in result.stdout
    assert "52 tasks" in result.stdout


@pytest.mark.parametrize("scenarios,models,expected", [
    ("2-3", ["xgboost", "logistic_regression"], "--array=6,9,12,15"),
    ("13", ["random_forest"], "--array=74"),
    ("1", ["xgboost", "lightgbm", "random_forest"], "--array=0-2"),
    ("4-5", ["lightgbm", "random_forest"], "--array=19-20,25-26"),
])
def test_a_model_subset_of_a_scenario_range_maps_to_the_matching_ids(scenarios, models, expected):
    result = _run("--dry-run", "--scenarios", scenarios, "--models", ",".join(models))
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout, result.stdout


def test_the_order_and_repeats_of_the_model_names_do_not_matter():
    a = _run("--dry-run", "--models", "logistic_regression,xgboost,xgboost,lightgbm,random_forest")
    b = _run("--dry-run", "--models=xgboost,lightgbm,random_forest,logistic_regression")
    assert a.returncode == 0 and b.returncode == 0, (a.stderr, b.stderr)
    assert _subset_array(range(1, 14), FOUR) in a.stdout
    assert _subset_array(range(1, 14), FOUR) in b.stdout


def test_every_model_given_by_name_is_the_same_array_as_no_flag():
    result = _run("--dry-run", "--models", ",".join(MODELS))
    assert result.returncode == 0, result.stderr
    assert _array() in result.stdout, result.stdout


def test_an_unknown_model_is_refused_and_the_valid_names_are_listed():
    result = _run("--dry-run", "--models", "xgboost,catboost")
    assert result.returncode == 2
    assert "catboost" in result.stderr
    for name in MODELS:
        assert name in result.stderr, result.stderr


def test_a_missing_model_value_is_refused():
    result = _run("--dry-run", "--models")
    assert result.returncode == 2


def test_a_model_subset_tells_the_tasks_it_is_deliberate(tmp_path):
    """Without this every first task prints 'scenarios 9-13 will NOT be trained', a
    false alarm that teaches the reader to ignore the real one (run 581402)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    seen = tmp_path / "subset.env"
    stub = bindir / "sbatch"
    stub.write_text(
        f'#!/bin/bash\necho "${{RAILWAY_ARRAY_SUBSET:-unset}}" > "{seen.as_posix()}"\n'
        'echo "Submitted batch job 999999"\n', encoding="utf-8")
    stub.chmod(0o755)

    result = _run("--models", ",".join(FOUR), stub_sbatch=bindir)
    assert result.returncode == 0, result.stderr
    assert seen.read_text(encoding="utf-8").strip() == "1"

    result = _run(stub_sbatch=bindir)  # the full grid is not a subset
    assert result.returncode == 0, result.stderr
    assert seen.read_text(encoding="utf-8").strip() == "unset"
