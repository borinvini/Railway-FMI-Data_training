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
def test_script_fails_before_srun_when_the_file_is_absent_or_empty(name):
    """A forgotten upload, or a truncated scp leaving a zero-byte file, must cost
    a second, not a full training run. `-s` (not `-f`) so a zero-byte file also
    fails the guard instead of reaching Python."""
    body = (REPO_ROOT / "hpc" / name).read_text(encoding="utf-8")
    assert 'if [ ! -s "${COLUMNS_FILE}" ]; then' in body
    guard_at = body.index('if [ ! -s "${COLUMNS_FILE}" ]; then')
    srun_at = body.index("srun python main.py")
    assert guard_at < srun_at, "the guard must run before srun"


@pytest.mark.parametrize("name", BATCH_SCRIPTS)
def test_script_records_the_feature_set_in_the_log(name):
    body = (REPO_ROOT / "hpc" / name).read_text(encoding="utf-8")
    assert "sha256sum" in body, "the log must identify which feature set ran"


def test_push_features_script_exists_and_targets_the_repo_clone():
    """The destination is the cluster clone's config/ directory, and the
    basename comes from the local file — so features.txt and scenarios.txt each
    land under their own name instead of one silently overwriting the other."""
    body = (REPO_ROOT / "hpc" / "push-features.sh").read_text(encoding="utf-8")
    assert "scp" in body
    assert 'REMOTE_DIR="roihu:/projappl/project_2019266/railway-fmi-code/config"' in body
    assert 'REMOTE="${REMOTE_DIR}/$(basename "${LOCAL}")"' in body


def test_push_features_script_validates_locally_before_scp():
    """A duplicate name or all-commented file must fail on the laptop, not only
    after the scp lands on Roihu, so parse locally with the real parser first.

    Matched on the import prefix rather than the full line: the validator now
    imports three names, and which ones is the concern of
    test_push_features_validates_scenario_files_section_by_section."""
    body = (REPO_ROOT / "hpc" / "push-features.sh").read_text(encoding="utf-8")
    assert "from config.columns_file import" in body
    assert "load_columns" in body
    validate_at = body.index("from config.columns_file import")
    scp_at = body.index('scp "${LOCAL}" "${REMOTE}"')
    assert validate_at < scp_at, "local validation must run before the upload"


def test_push_features_derives_the_remote_name_from_the_local_file():
    """A hardcoded remote basename would land scenarios.txt as features.txt on
    the cluster, and train_scenarios.sh would not find its default file."""
    body = (REPO_ROOT / "hpc" / "push-features.sh").read_text(encoding="utf-8")
    assert 'REMOTE_DIR="roihu:/projappl/project_2019266/railway-fmi-code/config"' in body
    assert '$(basename "${LOCAL}")' in body


def test_push_features_validates_scenario_files_section_by_section():
    """A scenario file parsed with load_columns would fail on the repeated
    trainDelayed, so the validator must branch on the file shape."""
    body = (REPO_ROOT / "hpc" / "push-features.sh").read_text(encoding="utf-8")
    assert "has_scenarios" in body
    assert "load_scenarios" in body
