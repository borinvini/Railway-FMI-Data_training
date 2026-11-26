"""The multi-station grid: one station list, one tree per station, every model fetched.

Before it, a station was run by hand: edit TARGET_STATION_CODE, preprocess,
stage into the single /scratch/.../data/output/101-..., submit, fetch, archive,
then delete everything on Roihu before the next station, because every path was
fixed. Seven stations meant seven of those cycles. These tests pin the contract
that lets the seven grids run at once:

- hpc/stations.sh is the one station list; scripts/stations.py parses the same
  line, so the archive number (position + 1) cannot differ between the shell and
  Python sides;
- each script that touches Roihu takes the station into its own tree
  (stations/<ST>/), and without a station keeps the old layout;
- ridge_baseline is appended to hpc/models.sh (never inserted: the order is the
  task-id contract) and its 999-* folder is fetched.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stations as stations_module  # noqa: E402

HPC = REPO_ROOT / "hpc"


def _read(name):
    return (HPC / name).read_text(encoding="utf-8")


def test_the_station_list_is_the_binary_archive_order():
    assert stations_module.load_stations() == ["HKI", "PSL", "TPE", "SK", "YV", "OL", "ROI"]


def test_the_shell_and_python_read_the_same_line():
    line = next(l for l in _read("stations.sh").splitlines() if l.startswith("STATIONS=("))
    assert line == "STATIONS=(" + " ".join(stations_module.load_stations()) + ")"


def test_the_archive_folder_is_the_position_plus_one():
    assert stations_module.archive_folder_name("HKI") == "1 - HKI"
    assert stations_module.archive_folder_name("OL") == "6 - OL"
    assert stations_module.archive_folder_name("ROI") == "7 - ROI"


def test_select_stations_keeps_list_order_and_refuses_typos():
    assert stations_module.select_stations(None) == stations_module.load_stations()
    assert stations_module.select_stations("all") == stations_module.load_stations()
    assert stations_module.select_stations("roi, ol") == ["OL", "ROI"]
    with pytest.raises(ValueError, match="OLL"):
        stations_module.select_stations("OL,OLL")


def test_a_broken_station_file_is_refused(tmp_path):
    bad = tmp_path / "stations.sh"
    bad.write_text("STATIONS=(\n  OL\n)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one-line"):
        stations_module.load_stations(str(bad))
    bad.write_text("STATIONS=(OL OL)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        stations_module.load_stations(str(bad))


def test_ridge_baseline_is_appended_not_inserted():
    """The task id decodes by position; inserting would relabel every result folder."""
    line = next(l for l in _read("models.sh").splitlines() if l.startswith("MODELS=("))
    assert line == ("MODELS=(xgboost lightgbm random_forest logistic_regression "
                    "naive_bayes fcll_net ridge_baseline)")


def test_ridge_baseline_is_a_model_main_py_accepts():
    import main
    assert "ridge_baseline" in main.MODEL_STAGES


def test_submit_scenarios_takes_a_station_into_its_own_tree():
    body = _read("submit-scenarios.sh")
    assert 'STATION_ENV=(RAILWAY_STATION="${STATION}")' in body
    assert '"--job-name=railway-${STATION}"' in body
    assert "stations/${STATION}/data/output/101-preprocessed_training_ready" in body
    assert 'source "${REPO_ROOT}/hpc/stations.sh"' in body
    assert '"${STATION_ENV[@]+"${STATION_ENV[@]}"}" "${CMD[@]}"' in body


def test_submit_stations_loops_the_single_station_wrapper():
    body = _read("submit-stations.sh")
    assert 'bash hpc/submit-scenarios.sh --station "${station}"' in body
    assert "source \"${REPO_ROOT}/hpc/stations.sh\"" in body
    assert "STATIONS=(" not in body, "the wrapper must source, not redefine, STATIONS"


def test_stage_data_checks_every_station_before_moving_bytes():
    body = _read("stage_data.sh")
    assert 'src="data/stations/${station}/${SRC_PREPROCESSED}"' in body
    assert 'dest="${SCRATCH}/stations/${station}/${SRC_PREPROCESSED}"' in body
    first_check = body.index("Run scripts/preprocess_stations.py first")
    first_transfer = body.index('transfer "${src}" "${dest}"')
    assert first_check < first_transfer
    # the hash compares the checksum column only (Git Bash prints "hash *name")
    assert "LOCAL_HASH='md5sum *.parquet | cut -c1-32 | md5sum | cut -c1-32'" in body


def test_fetch_stations_refuses_a_full_archive_before_fetching():
    body = _read("fetch-stations.sh")
    refuse = body.index("is not empty. Move it aside first; nothing was fetched.")
    fetch = body.index('bash hpc/fetch-results.sh --scenarios --station "${station}"')
    assert refuse < fetch
    assert 'hpc/grid_summary.py "results/${station}"' in body
    assert "STATIONS=(" not in body


def _bash():
    for cand in (os.environ.get("SHELL"), shutil.which("bash")):
        if cand and Path(cand).exists():
            return cand
    return None


@pytest.mark.parametrize("script", [
    "stations.sh", "submit-stations.sh", "fetch-stations.sh", "stage_data.sh",
    "submit-scenarios.sh", "train_scenarios.sh", "fetch-results.sh"])
def test_the_scripts_are_valid_bash(script):
    bash = _bash()
    if bash is None:
        pytest.skip("no bash on PATH")
    result = subprocess.run([bash, "-n", str(HPC / script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_scripts_have_lf_line_endings():
    """.gitattributes forces LF on commit, but a CRLF working copy run straight from
    Git Bash fails on the first $'\\r'."""
    for script in ("stations.sh", "submit-stations.sh", "fetch-stations.sh"):
        assert b"\r\n" not in (HPC / script).read_bytes(), script


def test_submit_stations_dry_run_calls_one_grid_per_station(tmp_path):
    bash = _bash()
    if bash is None:
        pytest.skip("no bash on PATH")
    catalogue = tmp_path / "scenarios.txt"
    catalogue.write_text("# 1 - A\ntrainDelayed\ncol_a\n\n# 2 - B\ntrainDelayed\ncol_b\n",
                         encoding="utf-8")
    env = dict(os.environ, COLUMNS_FILE=str(catalogue), PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [bash, str(HPC / "submit-stations.sh"), "--stations", "ROI,OL",
         "--models", "xgboost,ridge_baseline", "--dry-run"],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert re.findall(r"^=== (\w+) ===$", result.stdout, re.M) == ["OL", "ROI"]
    assert result.stdout.count("--job-name=railway-OL") == 1
    assert result.stdout.count("--job-name=railway-ROI") == 1
    # 2 scenarios x {xgboost (0), ridge_baseline (6)} of 7 models
    assert result.stdout.count("--array=0,6-7,13") == 2
