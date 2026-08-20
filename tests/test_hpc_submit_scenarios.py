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

MODELS = ["xgboost", "lightgbm", "random_forest", "logistic_regression", "naive_bayes"]


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
    assert "--array=0-64" in result.stdout, result.stdout


def test_the_task_count_is_scenarios_times_models():
    result = _run("--dry-run")
    assert "13 scenarios" in result.stdout
    assert f"{len(MODELS)} models" in result.stdout
    assert "65 tasks" in result.stdout


@pytest.mark.parametrize("n_scenarios,expected", [
    (1, "--array=0-4"),
    (8, "--array=0-39"),
    (13, "--array=0-64"),
    (20, "--array=0-99"),
])
def test_the_array_tracks_the_catalogue_size(tmp_path, n_scenarios, expected):
    """Any catalogue size, not just today's — the point is that no one has to
    edit a header again."""
    catalogue = _catalogue(tmp_path / "scen.txt", n_scenarios)
    result = _run("--dry-run", columns_file=catalogue)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout, result.stdout


@pytest.mark.parametrize("selector,expected", [
    ("9-13", "--array=40-64"),
    ("9", "--array=40-44"),
    ("1", "--array=0-4"),
    ("1-8", "--array=0-39"),
    ("13", "--array=60-64"),
])
def test_a_scenario_range_maps_to_the_matching_task_range(selector, expected):
    """scenario is 1-based and scenario-major: first id = (n-1)*models,
    last id = m*models - 1. Off by one here re-runs the wrong feature set."""
    result = _run("--dry-run", "--scenarios", selector)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout, result.stdout


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
    assert "--array=40-64" in argv, argv
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
    assert "--array=0-64" in result.stdout


def test_the_model_list_comes_from_the_shared_file():
    """Duplicating MODELS is how the wrapper and the batch script drift apart:
    the wrapper sizes the array by count, the batch script indexes by position,
    and a mismatch mislabels every result directory."""
    body = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "models.sh" in body
    assert "MODELS=(" not in body, "the wrapper must source, not redefine, MODELS"
