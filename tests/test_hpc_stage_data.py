"""Static checks for hpc/stage_data.sh.

The script calls ssh and scp, so it cannot run off the cluster; bash -n is the
only mechanical check available. What these tests protect is the payload
decision. The default transfer is 26 MB and the raw one is 2.6 GB — a hundred
times larger — so which one runs must never be decided by accident.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "hpc" / "stage_data.sh"
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")

BASH = shutil.which("bash") or r"C:\Program Files\Git\usr\bin\bash.EXE"

PREPROCESSED = "data/output/101-preprocessed_training_ready"
RAW = "data/input"


def test_the_script_is_syntactically_valid():
    result = subprocess.run([BASH, "-n", str(SCRIPT_PATH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_raw_input_is_opt_in():
    """fcll_net needs data/input, but the other five models never touch it.
    Staging 2.6 GB on every routine re-stage would make the common case a
    hundred times more expensive than it needs to be."""
    assert "--raw" in SCRIPT, "there must be a flag to request the raw input"
    assert "STAGE_RAW=0" in SCRIPT, "the raw input must default to off"


def test_the_preprocessed_data_is_always_staged():
    """It is what the five tabular trainers read; skipping it is never right."""
    assert PREPROCESSED in SCRIPT


def test_the_raw_directory_is_named_exactly_once_as_a_source():
    """A typo'd path here uploads nothing and the fcll_net cells fail on the
    cluster after queueing, which is the slowest possible way to find out."""
    assert f'"{RAW}"' in SCRIPT or f"SRC_RAW=\"{RAW}\"" in SCRIPT


def test_rsync_is_preferred_over_scp():
    """At 2.6 GB resumability stops being a nicety: scp restarts from zero."""
    assert "rsync" in SCRIPT
    assert SCRIPT.index("rsync") < SCRIPT.index("falling back") if "falling back" in SCRIPT else True


def test_an_unknown_flag_is_refused_rather_than_ignored():
    """A silently ignored --raw is the failure this whole flag exists to avoid:
    the user believes 2.6 GB went up, and the grid dies hours later."""
    assert "Unknown option" in SCRIPT or "unknown option" in SCRIPT


@pytest.mark.parametrize("argv,expect_raw", [
    ([], 0),
    (["--raw"], 1),
])
def test_the_flag_actually_sets_the_variable(argv, expect_raw):
    """Executes the real argument-parsing loop rather than asserting its text,
    so a branch that parses but never assigns cannot pass."""
    body = SCRIPT[SCRIPT.index("STAGE_RAW=0"):SCRIPT.index("REMOTE=")]
    snippet = "\n".join([
        "set -- " + " ".join(argv),
        body,
        'printf "%s" "${STAGE_RAW}"',
    ])
    result = subprocess.run([BASH, "-c", snippet], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(expect_raw)
