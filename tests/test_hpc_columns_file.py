"""Every batch script must gain the columns-file guard and flag together.

These are static checks rather than executions: the scripts call sbatch/srun and
cannot run off the cluster. What they catch is the realistic mistake — updating
two of the three scripts and leaving the third to train on the wrong features.
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_SCRIPTS = ["train_all.sh", "train_array.sh", "smoke_test.sh"]


@pytest.mark.parametrize("name", BATCH_SCRIPTS)
def test_script_defaults_and_passes_the_columns_file(name):
    body = (REPO_ROOT / "hpc" / name).read_text(encoding="utf-8")
    assert 'COLUMNS_FILE="${COLUMNS_FILE:-config/features.txt}"' in body
    assert '--columns-file "${COLUMNS_FILE}"' in body


@pytest.mark.parametrize("name", BATCH_SCRIPTS)
def test_script_fails_before_srun_when_the_file_is_absent(name):
    """A forgotten upload must cost a second, not a full training run."""
    body = (REPO_ROOT / "hpc" / name).read_text(encoding="utf-8")
    assert 'if [ ! -f "${COLUMNS_FILE}" ]; then' in body
    guard_at = body.index('if [ ! -f "${COLUMNS_FILE}" ]; then')
    srun_at = body.index("srun python main.py")
    assert guard_at < srun_at, "the guard must run before srun"


@pytest.mark.parametrize("name", BATCH_SCRIPTS)
def test_script_records_the_feature_set_in_the_log(name):
    body = (REPO_ROOT / "hpc" / name).read_text(encoding="utf-8")
    assert "sha256sum" in body, "the log must identify which feature set ran"


def test_push_features_script_exists_and_targets_the_repo_clone():
    body = (REPO_ROOT / "hpc" / "push-features.sh").read_text(encoding="utf-8")
    assert "scp" in body
    assert "roihu:/projappl/project_2019266/railway-fmi-code/config/features.txt" in body
