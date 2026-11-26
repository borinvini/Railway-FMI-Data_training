"""scripts/roihu_windows.py: the stage and fetch of the multi-station grid without Git Bash.

On the university machine Git Bash cannot start ("Access is denied") and the
Windows ssh-agent service is disabled, so hpc/stage_data.sh and
hpc/fetch-stations.sh cannot run and every ssh connection asks for the
passphrase. The Python version does each job in one connection; its local hash
must equal the remote `md5sum *.parquet | cut -c1-32 | md5sum | cut -c1-32`, or
every station would read DIFFER.
"""
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import roihu_windows as rw  # noqa: E402


def test_the_local_hash_is_the_md5_of_the_md5_column(tmp_path):
    (tmp_path / "training_ready_2018_02.parquet").write_bytes(b"bbb")
    (tmp_path / "training_ready_2018_01.parquet").write_bytes(b"aaa")
    (tmp_path / "notes.json").write_bytes(b"ignored")
    column = "".join(hashlib.md5(x).hexdigest() + "\n" for x in (b"aaa", b"bbb"))  # name order
    assert rw.folder_hash(str(tmp_path)) == (2, hashlib.md5(column.encode()).hexdigest())


def test_stage_replaces_only_the_named_stations_101_folders():
    cmd = rw.stage_remote_command(["OL", "ROI"])
    assert "rm -rf OL/data/output/101-preprocessed_training_ready ROI/data/output/101-preprocessed_training_ready" in cmd
    # from the scp'd file, never from ssh's stdin (Windows OpenSSH dropped the
    # connection after the passphrase prompt when stdin was the tar pipe)
    assert f"tar -xf {rw.UPLOAD}" in cmd and "tar -xf -" not in cmd
    assert cmd.index("rm -rf") < cmd.index("tar -xf") < cmd.index("echo STAGED")
    assert "md5sum *.parquet | cut -c1-32 | md5sum | cut -c1-32" in cmd


def test_fetch_takes_prep_from_xgboost_only_and_includes_the_ridge_baseline():
    cmd = rw.fetch_remote_command(["OL"])
    assert "OL/run_s??_xgboost/data/output/50[0-5]-*" in cmd
    assert "OL/run_s??_*/data/output/50[0-5]-*" not in cmd
    assert "OL/run_s??_*/data/output/100[0-4]-*" in cmd
    assert "OL/run_s??_*/data/output/999-*" in cmd
