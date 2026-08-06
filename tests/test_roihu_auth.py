"""Guard-clause tests for hpc/roihu-auth.sh.

The signing path itself is not tested here: it requires an interactive MyCSC
device-code login, so any test of it would exercise a mock rather than the
integration. See the spec's Testing section. What is tested is every way the
script should refuse to run, plus the flag plumbing to csc_cert.py.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "hpc" / "roihu-auth.sh"


def _resolve_bash():
    """Absolute path to Git Bash's bash.

    Passing bare "bash" to subprocess is not safe on Windows: CreateProcess
    searches System32 before PATH, and C:\\Windows\\System32\\bash.exe is the
    WSL launcher, which has a different filesystem view and cannot see the
    script (exit 127). Resolve explicitly.
    """
    for cand in (os.environ.get("SHELL"), shutil.which("bash")):
        if cand and Path(cand).exists():
            return cand
    return "bash"


BASH = _resolve_bash()


def _run(home, *args, path_prepend=None, env_extra=None):
    """Run the script with HOME redirected and optional PATH stubs.

    ROIHU_AUTH_NO_AGENT is set throughout: agent handling would otherwise spawn
    real ssh-agent processes and leak them, and none of these tests are about
    the agent.
    """
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["ROIHU_AUTH_NO_AGENT"] = "1"
    env.pop("SSH_AUTH_SOCK", None)
    if path_prepend:
        env["PATH"] = f"{path_prepend}{os.pathsep}{env['PATH']}"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [BASH, str(SCRIPT), *args],
        capture_output=True, text=True, env=env,
    )


def _stub_dir(tmp_path, name, body):
    """Create an executable stub named `name` in its own PATH directory."""
    d = tmp_path / f"stubs_{name}"
    d.mkdir(exist_ok=True)
    stub = d / name
    stub.write_text("#!/bin/bash\n" + body)
    stub.chmod(0o755)
    return d


@pytest.fixture
def home(tmp_path):
    (tmp_path / ".ssh").mkdir()
    return tmp_path


def test_aborts_when_private_key_missing(home):
    result = _run(home)
    assert result.returncode != 0
    assert "id_csc" in result.stderr
    assert "ssh-keygen -t ed25519" in result.stderr


def test_abort_message_does_not_run_the_signing_tool(home):
    """A missing key must fail before any network work is attempted."""
    result = _run(home)
    assert "Certificate" not in result.stdout


def test_aborts_when_vendored_tool_missing(home, tmp_path):
    """REPO_ROOT derives from the script's own location, so a copy outside the
    repo must report the missing tool rather than crash."""
    (home / ".ssh" / "id_csc").write_text("fake private key")
    elsewhere = tmp_path / "no_such_repo" / "hpc"
    elsewhere.mkdir(parents=True)
    script_copy = elsewhere / "roihu-auth.sh"
    script_copy.write_text(SCRIPT.read_text())
    script_copy.chmod(0o755)
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["ROIHU_AUTH_NO_AGENT"] = "1"
    result = subprocess.run(
        [BASH, str(script_copy)], capture_output=True, text=True, env=env
    )
    assert result.returncode != 0
    assert "csc_cert.py" in result.stderr


def _key_and_python_stub(home, tmp_path, body):
    """Set up a usable key pair plus a `python` stub running `body`."""
    (home / ".ssh" / "id_csc").write_text("fake private key")
    (home / ".ssh" / "id_csc.pub").write_text("ssh-ed25519 AAAA fake")
    return _stub_dir(tmp_path, "python", body)


def test_passes_required_flags_to_the_tool(home, tmp_path):
    """-a none and -p must both be passed; neither implies the other."""
    argv_log = tmp_path / "argv.txt"
    stubs = _key_and_python_stub(
        home, tmp_path, f'printf "%s\\n" "$@" > "{argv_log}"\nexit 0\n'
    )
    result = _run(home, path_prepend=str(stubs))
    assert result.returncode == 0, result.stderr
    recorded = argv_log.read_text().splitlines()
    assert "-a" in recorded and "none" in recorded
    assert "-p" in recorded
    assert "-u" in recorded and "vpozzobo" in recorded


def test_refresh_flag_is_forwarded(home, tmp_path):
    argv_log = tmp_path / "argv.txt"
    stubs = _key_and_python_stub(
        home, tmp_path, f'printf "%s\\n" "$@" > "{argv_log}"\nexit 0\n'
    )
    _run(home, "-r", path_prepend=str(stubs))
    assert "-r" in argv_log.read_text().splitlines()


def test_refresh_flag_absent_by_default(home, tmp_path):
    """The empty REFRESH array must expand to nothing under `set -u`."""
    argv_log = tmp_path / "argv.txt"
    stubs = _key_and_python_stub(
        home, tmp_path, f'printf "%s\\n" "$@" > "{argv_log}"\nexit 0\n'
    )
    result = _run(home, path_prepend=str(stubs))
    assert result.returncode == 0, result.stderr
    recorded = argv_log.read_text().splitlines()
    assert "-r" not in recorded
    assert "" not in recorded, "empty array expanded to an empty argument"


def test_csc_user_is_overridable(home, tmp_path):
    argv_log = tmp_path / "argv.txt"
    stubs = _key_and_python_stub(
        home, tmp_path, f'printf "%s\\n" "$@" > "{argv_log}"\nexit 0\n'
    )
    _run(
        home,
        path_prepend=str(stubs),
        env_extra={"CSC_USER": "someone_else"},
    )
    assert "someone_else" in argv_log.read_text().splitlines()


def test_tool_failure_stops_before_reporting_success(home, tmp_path):
    """A signing failure must not fall through to a stale expiry printout."""
    stubs = _key_and_python_stub(
        home, tmp_path, 'echo "Error: boom" >&2\nexit 1\n'
    )
    (home / ".ssh" / "id_csc-cert.pub").write_text("stale cert")
    result = _run(home, path_prepend=str(stubs))
    assert result.returncode != 0
    assert "Valid:" not in result.stdout
    assert "Ready." not in result.stdout
