"""The local pass that turns run_sNN_<model>/ into results/<scenario name>/.

This is the one piece of the fetch that can be exercised off the cluster, and
the one where the awkward characters in scenario names actually land on disk.
"""
from pathlib import Path

from hpc.sort_results import sort_results

SCENARIOS = """\
# 1 - ALL FEATURES
trainDelayed
trainStopping

# 6 - OPERACIONAL + INSTANT WEATHER + ROLLING WINDOWS 72h
trainDelayed
Air temperature (72h max)
"""


def _fixture(tmp_path, slugs_and_models):
    (tmp_path / "scenarios.txt").write_text(SCENARIOS, encoding="utf-8")
    for slug, model, stage in slugs_and_models:
        output = tmp_path / "results" / f"run_{slug}_{model}" / "data" / "output" / stage
        output.mkdir(parents=True)
        (output / "metrics.csv").write_text("ok\n", encoding="utf-8")
    return tmp_path / "results", tmp_path / "scenarios.txt"


def test_the_slug_indexes_by_position_not_by_the_banner_number(tmp_path):
    """Slug sNN means "the NNth section of the file", not "the section whose
    banner reads NN".

    hpc/train_scenarios.sh computes `scenario = id/5 + 1`, passes it as
    `--scenario N`, and slugs that same N. `select_scenario` resolves a numeric
    selector as the Nth section. So position is authoritative end to end.

    The two readings coincide in the real catalogue only because it happens to
    be numbered 1..8 in order, and they diverge the moment a section is
    reordered or removed. Getting it backwards would silently file results
    under the wrong scenario — the failure this whole slug scheme exists to
    prevent. Here the second section's banner says 6, so the correct slug for
    it is s02.
    """
    results, scenarios = _fixture(tmp_path, [
        ("s02", "xgboost", "1000-xgboost_randomized_search"),
    ])

    assert sort_results(results, scenarios, log=lambda _: None) == 1
    assert (results / "6 - OPERACIONAL + INSTANT WEATHER + ROLLING WINDOWS 72h").is_dir()
    assert not (results / "1 - ALL FEATURES").exists()


def test_slugged_roots_become_scenario_named_folders(tmp_path):
    """The scenario name lands on disk verbatim, spaces and + and all."""
    results, scenarios = _fixture(tmp_path, [
        ("s02", "xgboost", "1000-xgboost_randomized_search"),
        ("s02", "lightgbm", "1001-lightgbm_randomized_search"),
    ])

    assert sort_results(results, scenarios, log=lambda _: None) == 2

    target = results / "6 - OPERACIONAL + INSTANT WEATHER + ROLLING WINDOWS 72h"
    assert (target / "1000-xgboost_randomized_search" / "metrics.csv").exists()
    assert (target / "1001-lightgbm_randomized_search" / "metrics.csv").exists()
    assert not list(results.glob("run_s??_*")), "slugged roots must be removed"


def test_two_scenarios_do_not_collide(tmp_path):
    """Both scenarios produce a 1000-* directory; flattening would lose one."""
    results, scenarios = _fixture(tmp_path, [
        ("s01", "xgboost", "1000-xgboost_randomized_search"),
        ("s02", "xgboost", "1000-xgboost_randomized_search"),
    ])

    sort_results(results, scenarios, log=lambda _: None)

    assert (results / "1 - ALL FEATURES" / "1000-xgboost_randomized_search").is_dir()
    assert (results / "6 - OPERACIONAL + INSTANT WEATHER + ROLLING WINDOWS 72h"
            / "1000-xgboost_randomized_search").is_dir()


def test_an_unknown_slug_is_left_in_place(tmp_path):
    """Better a stray directory the user can inspect than a wrong guess."""
    results, scenarios = _fixture(tmp_path, [
        ("s99", "xgboost", "1000-xgboost_randomized_search"),
    ])

    assert sort_results(results, scenarios, log=lambda _: None) == 0
    assert (results / "run_s99_xgboost").is_dir()
