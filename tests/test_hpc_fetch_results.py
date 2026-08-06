"""Static checks for hpc/fetch-results.sh.

The script calls ssh and scp, so it cannot run off the cluster; bash -n is the
only mechanical check available. What these tests protect is the part that is
easy to break with a careless edit — the globs, which decide whether the fetch
pulls 2 GB or 40 GB, and whether it silently drags the staged input back down.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (REPO_ROOT / "hpc" / "fetch-results.sh").read_text(encoding="utf-8")


def test_both_flags_are_handled_and_anything_else_is_rejected():
    assert "--models)" in SCRIPT
    assert "--train-all)" in SCRIPT
    assert "unknown argument" in SCRIPT


def test_prep_stage_glob_pins_the_stage_numbers():
    """Explicit, so adding a stage 506 is a deliberate edit here rather than a
    silent inclusion."""
    assert "50[0-5]-*" in SCRIPT
    assert "output/5*" not in SCRIPT


def test_model_glob_cannot_match_the_preprocessed_input():
    """10* also matches 101-preprocessed_training_ready, which is pipeline input
    staged up by hpc/stage_data.sh — matching it would pull the whole staged
    dataset back down once per run directory."""
    assert "100[0-4]-*" in SCRIPT
    assert "output/10*" not in SCRIPT


def test_prep_stages_are_taken_from_run_0_only():
    """train_array.sh re-runs stages 500-505 in all five run roots from
    identical code and identical input; fetching all five would be five times
    the transfer for byte-equivalent data."""
    assert "run_0/data/output/50[0-5]-*" in SCRIPT


def test_model_directories_are_taken_from_every_run_root():
    """Each array task enables exactly one trainer, so the five run roots yield
    five distinct model directories."""
    assert "run_*/data/output/100[0-4]-*" in SCRIPT


def test_the_shap_directory_is_included():
    assert "700-*" in SCRIPT


def test_size_is_reported_before_the_transfer_starts():
    """The point of the du call is to learn a pull is 12 GB before it begins,
    not after."""
    assert "du -csh" in SCRIPT
    assert SCRIPT.index("du -csh") < SCRIPT.index("scp -r")


def test_an_unreachable_cluster_points_at_the_auth_script():
    """The certificate lasts 24 h; expiry is the single most common failure and
    looks like a broken setup."""
    assert "hpc/roihu-auth.sh" in SCRIPT
