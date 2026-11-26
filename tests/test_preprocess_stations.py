"""scripts/preprocess_stations.py: every station preprocessed without editing config.

A station used to be preprocessed by editing TARGET_STATION_CODE in
config/const_preprocessing.py and the EXECUTE_* flags in config/const.py, and
that edited const.py was once pushed to the cluster, where array 727792 trained
nothing (c77c574). The driver runs each station in its own interpreter through
--station / --pipeline / --data-root / --input-dir, checks the result belongs
to the station, and files it under "<N> - <ST>" in the archive.
"""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import preprocess_stations as driver  # noqa: E402


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "STATIONS_ROOT", str(tmp_path / "stations"))
    archive = tmp_path / "NUMERIC"
    archive.mkdir()
    return tmp_path, archive


def _args(archive, **kw):
    base = dict(skip_existing=False, overwrite=False, no_archive=False, archive=str(archive))
    base.update(kw)
    return SimpleNamespace(**base)


def _fake_run(calls, code=0):
    """A subprocess.run stand-in that 'preprocesses' by writing the archived folders."""
    def run(cmd, **kwargs):
        calls.append(cmd)
        root = cmd[cmd.index("--data-root") + 1]
        for name in ("100-preprocessed", "101-preprocessed_training_ready", "log"):
            folder = os.path.join(root, "data", "output", name)
            os.makedirs(folder, exist_ok=True)
            Path(folder, "x.parquet").write_text("x", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, code)
    return run


def _passing_check(station, root, input_dir, log=print):
    return True, {"station": station, "files": 96, "rows": 1000, "median_delay": 1.0}


def test_the_command_uses_flags_not_config_edits():
    cmd = driver.preprocess_command("OL", python="py")
    assert cmd[:2] == ["py", "-u"]
    for flag, value in (("--pipeline", "preprocessing"), ("--station", "OL"),
                        ("--input-dir", driver.INPUT_DIR)):
        assert cmd[cmd.index(flag) + 1] == value
    assert cmd[cmd.index("--data-root") + 1].endswith(os.path.join("stations", "OL"))


def test_a_station_is_preprocessed_checked_and_archived_under_its_number(roots):
    _, archive = roots
    calls = []
    ok, line = driver.process_station("OL", _args(archive), driver.load_stations(),
                                      run=_fake_run(calls), check=_passing_check, log=lambda *_: None)
    assert ok, line
    assert len(calls) == 1
    assert (archive / "6 - OL" / "101-preprocessed_training_ready" / "x.parquet").exists()
    assert (archive / "6 - OL" / "100-preprocessed").is_dir()
    assert not (archive / "6 - OL" / "9-process_actual_time_column").exists()  # absent, skipped


def test_a_full_archive_folder_is_refused_before_preprocessing(roots):
    """Half an hour of preprocessing must not be spent only to be refused at the copy."""
    _, archive = roots
    full = archive / "7 - ROI" / "101-preprocessed_training_ready"
    full.mkdir(parents=True)
    (full / "old.parquet").write_text("old", encoding="utf-8")
    calls = []
    ok, line = driver.process_station("ROI", _args(archive), driver.load_stations(),
                                      run=_fake_run(calls), check=_passing_check, log=lambda *_: None)
    assert not ok and "not empty" in line
    assert calls == []
    assert (full / "old.parquet").exists()


def test_overwrite_replaces_the_archived_folder(roots):
    _, archive = roots
    full = archive / "7 - ROI" / "101-preprocessed_training_ready"
    full.mkdir(parents=True)
    (full / "old.parquet").write_text("old", encoding="utf-8")
    ok, _ = driver.process_station("ROI", _args(archive, overwrite=True), driver.load_stations(),
                                   run=_fake_run([]), check=_passing_check, log=lambda *_: None)
    assert ok
    assert not (full / "old.parquet").exists()
    assert (full / "x.parquet").exists()


def test_a_failed_run_or_check_is_not_archived(roots):
    _, archive = roots
    ok, line = driver.process_station("OL", _args(archive), driver.load_stations(),
                                      run=_fake_run([], code=1), check=_passing_check,
                                      log=lambda *_: None)
    assert not ok and "exited 1" in line
    ok, line = driver.process_station(
        "OL", _args(archive), driver.load_stations(), run=_fake_run([]),
        check=lambda *a, **k: (False, {}), log=lambda *_: None)
    assert not ok and "station check failed" in line
    assert not (archive / "6 - OL").exists()


def test_skip_existing_does_not_rerun_a_station_that_passes(roots):
    _, archive = roots
    calls = []
    ok, _ = driver.process_station("OL", _args(archive, skip_existing=True, no_archive=True),
                                   driver.load_stations(), run=_fake_run(calls),
                                   check=_passing_check, log=lambda *_: None)
    assert ok and calls == []


def test_a_neighbouring_stop_with_the_same_delays_does_not_fail_the_check():
    """HKI failed the first all-stations run although its own share was 1.000: a
    gap rule wanted 0.30 over the runner-up, and PSL (the next stop, the delay
    carried over) matched 0.70-0.79 of the rows. Wrong data cannot reach 0.95 for
    any station but its own, so the rule is 'own >= 0.95, every other below'."""
    from check_station import month_passes
    assert month_passes(1.0, 0.786)        # HKI 2022_01
    assert not month_passes(0.40, 0.99)    # the files belong to another station
    assert not month_passes(1.0, 0.97)     # two stations claim the rows: ambiguous
    assert not month_passes(0.90, 0.10)
