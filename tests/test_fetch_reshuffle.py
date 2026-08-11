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


def test_a_root_with_no_data_output_is_left_untouched(tmp_path):
    """A run root that never produced output (a failed/cancelled task) must
    not be rmtree'd or counted as consumed — that would destroy the only
    evidence the task failed."""
    results, scenarios = _fixture(tmp_path, [])
    root = results / "run_s01_xgboost"
    root.mkdir(parents=True)

    assert sort_results(results, scenarios, log=lambda _: None) == 0
    assert root.is_dir()
    assert not (results / "1 - ALL FEATURES").exists()


def test_a_scenario_folder_holds_one_prep_set_beside_its_five_models(tmp_path):
    """The shape of a finished scenario folder: six prep stages and five model
    directories, all flat, no intermediate level.

    The prep stages arrive from the xgboost root alone — that is what
    hpc/fetch-results.sh selects, and it is the reason flat is safe. The other
    four roots contribute only their differently-numbered 100N-* directory.
    """
    results, scenarios = _fixture(tmp_path, [
        ("s01", "xgboost", stage) for stage in [
            "500-merge_data_files", "501-filter_delay_outliers",
            "502-select_training_cols", "503-split_dataset",
            "504-balance_classes", "505-scale_weather_features",
            "1000-xgboost_randomized_search",
        ]
    ] + [
        ("s01", "lightgbm", "1001-lightgbm_randomized_search"),
        ("s01", "random_forest", "1002-random_forest_randomized_search"),
        ("s01", "logistic_regression", "1003-regularized_regression"),
        ("s01", "naive_bayes", "1004-naive_bayes"),
    ])

    assert sort_results(results, scenarios, log=lambda _: None) == 5

    target = results / "1 - ALL FEATURES"
    assert sorted(p.name for p in target.iterdir()) == [
        "1000-xgboost_randomized_search",
        "1001-lightgbm_randomized_search",
        "1002-random_forest_randomized_search",
        "1003-regularized_regression",
        "1004-naive_bayes",
        "500-merge_data_files",
        "501-filter_delay_outliers",
        "502-select_training_cols",
        "503-split_dataset",
        "504-balance_classes",
        "505-scale_weather_features",
    ]
    assert all(p.is_dir() for p in target.iterdir())


def test_a_second_root_bringing_prep_stages_would_overwrite_the_first(tmp_path):
    """Documents the coupling this flat layout rests on, so widening the fetch
    glob fails visibly here rather than quietly deleting data on a real run.

    Two roots in one scenario carrying the same 500-* name cannot both survive:
    the sort rmtree's the destination before moving. That is fine only because
    hpc/fetch-results.sh takes 500-505 from run_s??_xgboost alone. If that glob
    ever becomes run_s??_*, this function needs a model level again — see
    test_scenarios_prep_comes_from_one_run_root_per_scenario, which pins the
    fetch side of the same invariant.
    """
    results, scenarios = _fixture(tmp_path, [
        ("s01", "xgboost", "500-merge_data_files"),
        ("s01", "lightgbm", "500-merge_data_files"),
    ])

    assert sort_results(results, scenarios, log=lambda _: None) == 2

    target = results / "1 - ALL FEATURES"
    assert (target / "500-merge_data_files" / "metrics.csv").exists()
    assert len(list(target.iterdir())) == 1, "one survivor, by design of the fetch glob"


def test_a_destination_that_is_a_file_is_replaced_not_raised_on(tmp_path):
    """A stray file at the destination path (e.g. left over from a previous,
    differently-shaped fetch) must be replaced like a directory would be,
    not raise NotADirectoryError mid-loop and leave results/ half-sorted."""
    results, scenarios = _fixture(tmp_path, [
        ("s01", "xgboost", "1000-xgboost_randomized_search"),
    ])
    target = results / "1 - ALL FEATURES"
    target.mkdir(parents=True)
    (target / "1000-xgboost_randomized_search").write_text("stale", encoding="utf-8")

    assert sort_results(results, scenarios, log=lambda _: None) == 1
    assert (target / "1000-xgboost_randomized_search" / "metrics.csv").exists()
