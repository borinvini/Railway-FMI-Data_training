"""CLI layer: flags override config defaults; no flags reproduces current behaviour."""
import os

import pytest

import main as main_module


@pytest.fixture(autouse=True)
def _restore_railway_env():
    """Undo RAILWAY_* variables that apply_env_overrides writes directly.

    monkeypatch cannot help here: delenv on an unset variable records nothing to
    restore, and the plain `os.environ[...] = ...` inside apply_env_overrides is
    not tracked at all. Without this fixture, one test's export leaks into every
    later test in the session — which config/const_training.py now consumes at
    import time, breaking the reload-based tests in test_columns_file.py.
    """
    saved = {k: v for k, v in os.environ.items() if k.startswith("RAILWAY_")}
    try:
        yield
    finally:
        for key in [k for k in os.environ if k.startswith("RAILWAY_")]:
            del os.environ[key]
        os.environ.update(saved)


def test_no_args_returns_all_none():
    args = main_module.parse_args([])
    assert args.stages is None
    assert args.model is None
    assert args.data_root is None
    assert args.n_jobs is None
    assert args.search_iterations is None
    assert args.columns_file is None
    assert args.dump_columns is False


def test_no_args_leaves_state_machine_untouched():
    default = {"merge_data_files": True, "train_xgboost_with_randomized_search_cv": True}
    args = main_module.parse_args([])
    assert main_module.build_state_machine(args, default) == default


def test_stages_flag_enables_only_named_stages():
    default = {"merge_data_files": True, "split_dataset": True, "balance_classes": True}
    args = main_module.parse_args(["--stages", "merge_data_files,split_dataset"])
    result = main_module.build_state_machine(args, default)
    assert result["merge_data_files"] is True
    assert result["split_dataset"] is True
    assert result["balance_classes"] is False


def test_stages_flag_rejects_unknown_stage():
    default = {"merge_data_files": True}
    args = main_module.parse_args(["--stages", "not_a_stage"])
    try:
        main_module.build_state_machine(args, default)
    except ValueError as exc:
        assert "not_a_stage" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown stage name")


def test_model_flag_enables_trainer_plus_prerequisites():
    default = {k: True for k in list(main_module.PREREQUISITE_STAGES) + list(main_module.MODEL_STAGES.values())}
    args = main_module.parse_args(["--model", "naive_bayes"])
    result = main_module.build_state_machine(args, default)
    assert result["train_naive_bayes_with_randomized_search_cv"] is True
    for stage in main_module.PREREQUISITE_STAGES:
        assert result[stage] is True, f"prerequisite {stage} must stay enabled"
    assert result["train_xgboost_with_randomized_search_cv"] is False
    assert result["train_lightgbm_with_randomized_search_cv"] is False


def test_every_model_name_maps_to_a_real_stage():
    from config.const_training import TRAINING_STATE_MACHINE
    for short_name, stage in main_module.MODEL_STAGES.items():
        assert stage in TRAINING_STATE_MACHINE, f"{short_name} -> {stage} is not a real stage"


def test_prerequisite_stages_are_all_real_stages():
    from config.const_training import TRAINING_STATE_MACHINE
    for stage in main_module.PREREQUISITE_STAGES:
        assert stage in TRAINING_STATE_MACHINE


def test_apply_env_overrides_sets_variables(monkeypatch):
    monkeypatch.delenv("RAILWAY_DATA_ROOT", raising=False)
    monkeypatch.delenv("RAILWAY_N_JOBS", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("RAILWAY_SEARCH_ITERATIONS", raising=False)
    args = main_module.parse_args(
        ["--data-root", "/scratch/x", "--n-jobs", "40", "--search-iterations", "10"]
    )
    main_module.apply_env_overrides(args)
    assert os.environ["RAILWAY_DATA_ROOT"] == "/scratch/x"
    # --n-jobs must set its own variable, not overwrite Slurm's allocation record.
    assert os.environ["RAILWAY_N_JOBS"] == "40"
    assert "SLURM_CPUS_PER_TASK" not in os.environ
    assert os.environ["RAILWAY_SEARCH_ITERATIONS"] == "10"


def test_apply_env_overrides_is_a_noop_without_flags(monkeypatch):
    monkeypatch.delenv("RAILWAY_DATA_ROOT", raising=False)
    monkeypatch.delenv("RAILWAY_N_JOBS", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("RAILWAY_SEARCH_ITERATIONS", raising=False)
    main_module.apply_env_overrides(main_module.parse_args([]))
    assert "RAILWAY_DATA_ROOT" not in os.environ
    assert "RAILWAY_N_JOBS" not in os.environ
    assert "SLURM_CPUS_PER_TASK" not in os.environ
    assert "RAILWAY_SEARCH_ITERATIONS" not in os.environ


def test_n_jobs_zero_is_rejected():
    try:
        main_module.parse_args(["--n-jobs", "0"])
    except SystemExit:
        pass  # argparse rejects it via parser.error()
    else:
        raise AssertionError("expected --n-jobs 0 to be rejected")


def test_n_jobs_below_minus_one_is_rejected():
    try:
        main_module.parse_args(["--n-jobs", "-2"])
    except SystemExit:
        pass  # argparse rejects it via parser.error()
    else:
        raise AssertionError("expected --n-jobs -2 to be rejected")


def test_n_jobs_minus_one_is_accepted():
    args = main_module.parse_args(["--n-jobs", "-1"])
    assert args.n_jobs == -1


def test_main_defers_config_imports():
    """A module-scope config/src import would bind constants before
    apply_env_overrides() runs, silently breaking --data-root, --n-jobs and
    --search-iterations. See main.py's module docstring."""
    import ast

    tree = ast.parse(open("main.py", encoding="utf-8").read())
    top = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    top |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
    assert top <= {"argparse", "os", "sys"}, f"module-scope import breaks env overrides: {top}"


def test_search_iterations_below_ten_is_rejected():
    """range(10, n+1, 10) is empty below 10, which would silently skip training."""
    try:
        main_module.parse_args(["--search-iterations", "5"])
    except SystemExit:
        pass  # argparse rejects it
    else:
        raise AssertionError("expected --search-iterations 5 to be rejected")


def test_stages_and_model_are_mutually_exclusive():
    try:
        main_module.parse_args(["--stages", "merge_data_files", "--model", "xgboost"])
    except SystemExit:
        pass  # argparse rejects it via parser.error()
    else:
        raise AssertionError("expected --stages combined with --model to be rejected")


def test_dump_columns_flag_parses():
    args = main_module.parse_args(["--dump-columns"])
    assert args.dump_columns is True


def _fake_pipeline_class(calls):
    """Build a stand-in for TrainingPipeline that records which methods were called."""
    import pandas as pd

    class FakePipeline:
        def merge_data_files(self, csv_files):
            calls["merge"] += 1
            return {
                "success": True,
                "data": pd.DataFrame({"alpha": [1], "beta": [2], "trainDelayed": [0]}),
            }

        def execute_training_pipeline_steps(self, csv_files, state_machine):
            calls["train"] += 1
            return {"success": True}

    return FakePipeline


def test_dump_columns_prints_features_txt_lines(capsys):
    calls = {"merge": 0, "train": 0}
    pipeline = _fake_pipeline_class(calls)()

    rc = main_module.dump_columns(pipeline, [])

    out = capsys.readouterr().out
    assert rc == 0
    assert calls["merge"] == 1
    assert calls["train"] == 0, "no trainer may run under --dump-columns"
    assert "alpha" in out and "beta" in out
    # One bare name per line, the format of config/features.txt
    assert "\nalpha\n" in out and "\nbeta\n" in out
    assert "SELECTED_COLUMNS" not in out


def test_dump_columns_reports_merge_failure():
    class FailingPipeline:
        def merge_data_files(self, csv_files):
            return {"success": False, "error": "no training-ready files"}

    assert main_module.dump_columns(FailingPipeline(), []) == 1


@pytest.fixture
def pipeline_flags(monkeypatch):
    """Pin EXECUTE_* to the shape these tests assume.

    _run() imports both flags from config.const at call time, so whatever the
    working copy happens to have set decides what _run does. With preprocessing
    switched on, a test that only means to drive the training block instead runs
    the real 96-month preprocessing pipeline and appears to hang.
    """
    import config.const
    import config.const_training
    monkeypatch.setattr(config.const, "EXECUTE_PREPROCESSING_DATA_PIPELINE", False)
    monkeypatch.setattr(config.const, "EXECUTE_TRAINING_PIPELINE", True)
    # SELECTED_COLUMNS is empty until features.txt is resolved, and _run refuses a
    # training run without columns.
    monkeypatch.setattr(config.const_training, "SELECTED_COLUMNS", ["trainDisrupted"])


def test_run_returns_nonzero_on_training_failure(monkeypatch, pipeline_flags):
    """A failed training pipeline must make _run() (and therefore main()) return a
    non-zero code, so Slurm/sacct records the job as failed instead of COMPLETED."""
    calls = {"merge": 0, "train": 0}
    FakePipeline = _fake_pipeline_class(calls)

    class FailingTrainPipeline(FakePipeline):
        def execute_training_pipeline_steps(self, csv_files, state_machine):
            calls["train"] += 1
            return {"success": False, "errors": ["boom"]}

    # _run() constructs TrainingPipeline() directly (not via _make_pipeline) for the
    # training step, so patch the class where _run() imports it from.
    import src.training_pipeline
    monkeypatch.setattr(src.training_pipeline, "TrainingPipeline", FailingTrainPipeline)

    args = main_module.parse_args([])
    rc = main_module._run(args)

    assert calls["train"] == 1
    assert rc != 0


def test_run_returns_nonzero_when_training_results_is_falsy(monkeypatch, pipeline_flags):
    calls = {"merge": 0, "train": 0}
    FakePipeline = _fake_pipeline_class(calls)

    class NoneResultPipeline(FakePipeline):
        def execute_training_pipeline_steps(self, csv_files, state_machine):
            calls["train"] += 1
            return None

    import src.training_pipeline
    monkeypatch.setattr(src.training_pipeline, "TrainingPipeline", NoneResultPipeline)

    args = main_module.parse_args([])
    rc = main_module._run(args)

    assert calls["train"] == 1
    assert rc != 0


def test_dump_columns_short_circuits_before_training(monkeypatch):
    """--dump-columns must make _run return via dump_columns without ever reaching
    the training block, proven by driving the real _run() end-to-end with fakes."""
    calls = {"merge": 0, "train": 0}
    monkeypatch.setattr(main_module, "_make_pipeline", _fake_pipeline_class(calls))

    args = main_module.parse_args(["--dump-columns"])
    rc = main_module._run(args)

    assert rc == 0
    assert calls["merge"] == 1
    assert calls["train"] == 0, "training must never run under --dump-columns"


def test_columns_file_flag_parses_and_sets_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("RAILWAY_COLUMNS_FILE", raising=False)
    features = tmp_path / "features.txt"
    features.write_text("trainDelayed\nAir temperature (12h max)\n", encoding="utf-8")

    args = main_module.parse_args(["--columns-file", str(features)])
    assert args.columns_file == str(features)

    main_module.apply_env_overrides(args)
    assert os.environ["RAILWAY_COLUMNS_FILE"] == str(features)


def test_no_columns_file_leaves_env_var_unset(monkeypatch):
    monkeypatch.delenv("RAILWAY_COLUMNS_FILE", raising=False)
    main_module.apply_env_overrides(main_module.parse_args([]))
    assert "RAILWAY_COLUMNS_FILE" not in os.environ


def test_missing_columns_file_is_rejected_at_parse_time(tmp_path):
    """Fails in about a second, rather than after the merge stage has run."""
    try:
        main_module.parse_args(["--columns-file", str(tmp_path / "nope.txt")])
    except SystemExit:
        pass  # argparse rejects it via parser.error()
    else:
        raise AssertionError("expected a missing --columns-file to be rejected")


def test_empty_columns_file_is_rejected_at_parse_time(tmp_path):
    features = tmp_path / "features.txt"
    features.write_text("# nothing selected\n", encoding="utf-8")
    try:
        main_module.parse_args(["--columns-file", str(features)])
    except SystemExit:
        pass  # argparse rejects it via parser.error()
    else:
        raise AssertionError("expected an empty --columns-file to be rejected")


def test_parse_args_does_not_import_const_training(tmp_path, monkeypatch):
    """parse_args()'s `from config.columns_file import load_columns` must stay the
    only config import it triggers. If someone moves apply_env_overrides() to run
    inside _run() after the config imports, or a config/__init__.py starts
    importing const_training, parse_args would pull const_training in before
    RAILWAY_COLUMNS_FILE is set and every run would silently use the frozen list
    with the suite staying green. This test pins that config.const_training is
    not yet imported when parse_args() returns."""
    import sys

    features = tmp_path / "features.txt"
    features.write_text("trainDelayed\n", encoding="utf-8")

    # A previously run test in this session may already have imported
    # config.const_training; hide that from this test and restore it after.
    monkeypatch.delitem(sys.modules, "config.const_training", raising=False)

    main_module.parse_args(["--columns-file", str(features)])

    assert "config.const_training" not in sys.modules


# --- default columns file: the IDE run button ----------------------------------
#
# `python main.py` with no flags used to train on a hard-coded SELECTED_COLUMNS list
# that had drifted from config/features.txt (old target, dropped columns). A plain run
# now uses config/features.txt; an explicit choice always wins.

def _args(argv=()):
    return main_module.parse_args(list(argv))


@pytest.fixture
def default_file(monkeypatch, tmp_path):
    path = tmp_path / "features.txt"
    path.write_text("trainDisrupted\ntrainStopping\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "DEFAULT_COLUMNS_FILE", str(path))
    monkeypatch.delenv("RAILWAY_COLUMNS_FILE", raising=False)
    return path


def test_a_plain_run_uses_the_default_features_file(default_file):
    args = _args()
    main_module.resolve_default_columns_file(args)
    assert args.columns_file == str(default_file)
    main_module.apply_env_overrides(args)
    assert os.environ["RAILWAY_COLUMNS_FILE"] == str(default_file)


def test_the_default_file_is_resolved_from_the_repo_not_the_working_directory():
    expected = os.path.join(os.path.dirname(os.path.abspath(main_module.__file__)),
                            "config", "features.txt")
    assert main_module.DEFAULT_COLUMNS_FILE == expected


def test_an_explicit_columns_file_wins(default_file, tmp_path):
    other = tmp_path / "other.txt"
    other.write_text("trainDisrupted\n", encoding="utf-8")
    args = _args(["--columns-file", str(other)])
    main_module.resolve_default_columns_file(args)
    assert args.columns_file == str(other)


def test_an_exported_columns_file_wins(default_file, monkeypatch):
    monkeypatch.setenv("RAILWAY_COLUMNS_FILE", "/somewhere/else.txt")
    args = _args()
    main_module.resolve_default_columns_file(args)
    assert args.columns_file is None


def test_dump_columns_does_not_need_or_use_the_default_file(default_file):
    args = _args(["--dump-columns"])
    main_module.resolve_default_columns_file(args)
    assert args.columns_file is None


def test_a_missing_default_file_is_left_for_run_to_report(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "DEFAULT_COLUMNS_FILE", str(tmp_path / "absent.txt"))
    monkeypatch.delenv("RAILWAY_COLUMNS_FILE", raising=False)
    args = _args()
    main_module.resolve_default_columns_file(args)
    assert args.columns_file is None


def test_a_malformed_default_file_exits_with_its_message(default_file):
    default_file.write_text("# nothing but comments\n", encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        main_module.resolve_default_columns_file(_args())
    assert str(default_file) in str(err.value)


def test_a_scenario_catalogue_as_the_default_file_is_refused(default_file):
    default_file.write_text("# 1 - ONE\ntrainDisrupted\n# 2 - TWO\ntrainStopping\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main_module.resolve_default_columns_file(_args())


def test_training_without_columns_is_refused_up_front():
    machine = {"select_training_cols": True, "build_fcll_sequences": False}
    message = main_module.missing_columns_message([], machine)
    assert "features.example.txt" in message


def test_the_refusal_only_applies_when_a_stage_reads_the_columns():
    assert main_module.missing_columns_message([], {"select_training_cols": False}) is None
    assert main_module.missing_columns_message([], {"build_fcll_sequences": True}) is not None
    assert main_module.missing_columns_message(["a"], {"select_training_cols": True}) is None
    assert main_module.missing_columns_message([], {"select_training_cols": True}, dump_columns=True) is None


def test_run_returns_1_without_columns_before_any_stage_runs(monkeypatch, pipeline_flags):
    import config.const_training
    monkeypatch.setattr(config.const_training, "SELECTED_COLUMNS", [])
    calls = {"merge": 0, "train": 0}
    import src.training_pipeline
    monkeypatch.setattr(src.training_pipeline, "TrainingPipeline", _fake_pipeline_class(calls))

    rc = main_module._run(main_module.parse_args([]))

    assert rc == 1
    assert calls == {"merge": 0, "train": 0}


def test_main_sets_env_var_before_run(monkeypatch, tmp_path):
    """main() must call apply_env_overrides() before _run(), because _run()'s
    deferred config imports bind SELECTED_COLUMNS from RAILWAY_COLUMNS_FILE at
    import time. If that ordering were ever reversed, _run() would import the
    frozen list before the env var existed and this test would catch it."""
    features = tmp_path / "features.txt"
    features.write_text("trainDelayed\n", encoding="utf-8")
    monkeypatch.delenv("RAILWAY_COLUMNS_FILE", raising=False)

    seen = {}

    def fake_run(args):
        seen["value"] = os.environ.get("RAILWAY_COLUMNS_FILE")
        return 0

    monkeypatch.setattr(main_module, "_run", fake_run)

    rc = main_module.main(["--columns-file", str(features)])

    assert rc == 0
    assert seen["value"] == str(features)


import pytest

SCENARIO_TEXT = """\
# 1 - ALL FEATURES
trainDelayed
Air temperature (12h max)

# 2 - ONLY OPERACIONAL FEATURES
trainDelayed
trainStopping
"""


@pytest.fixture
def scenario_file(tmp_path):
    f = tmp_path / "scenarios.txt"
    f.write_text(SCENARIO_TEXT, encoding="utf-8")
    return str(f)


def test_no_args_leaves_the_new_flags_unset():
    args = main_module.parse_args([])
    assert args.scenario is None
    assert args.list_scenarios is False


def test_scenario_flag_is_accepted_with_a_columns_file(scenario_file):
    args = main_module.parse_args(["--columns-file", scenario_file, "--scenario", "2"])
    assert args.scenario == "2"


def test_scenario_without_columns_file_is_rejected():
    with pytest.raises(SystemExit):
        main_module.parse_args(["--scenario", "2"])


def test_unknown_scenario_index_is_rejected_at_parse_time(scenario_file):
    """A bad index must cost a second on the laptop, not a queued allocation."""
    with pytest.raises(SystemExit):
        main_module.parse_args(["--columns-file", scenario_file, "--scenario", "9"])


def test_unknown_scenario_name_is_rejected_at_parse_time(scenario_file):
    with pytest.raises(SystemExit):
        main_module.parse_args(
            ["--columns-file", scenario_file, "--scenario", "9 - NOPE"]
        )


def test_scenario_file_without_scenario_flag_is_rejected(scenario_file):
    """load_columns raises the 'pass --scenario' error; parse_args surfaces it."""
    with pytest.raises(SystemExit):
        main_module.parse_args(["--columns-file", scenario_file])


def test_plain_columns_file_still_validates_without_a_scenario(tmp_path):
    f = tmp_path / "features.txt"
    f.write_text("trainDelayed\ntrainStopping\n", encoding="utf-8")
    args = main_module.parse_args(["--columns-file", str(f)])
    assert args.columns_file == str(f)
    assert args.scenario is None


def test_apply_env_overrides_exports_the_scenario(monkeypatch, scenario_file):
    monkeypatch.delenv("RAILWAY_SCENARIO", raising=False)
    monkeypatch.delenv("RAILWAY_COLUMNS_FILE", raising=False)
    args = main_module.parse_args(
        ["--columns-file", scenario_file, "--scenario", "2"]
    )
    main_module.apply_env_overrides(args)
    assert os.environ["RAILWAY_SCENARIO"] == "2"
    assert os.environ["RAILWAY_COLUMNS_FILE"] == scenario_file


def test_apply_env_overrides_leaves_scenario_unset_without_the_flag(monkeypatch, tmp_path):
    monkeypatch.delenv("RAILWAY_SCENARIO", raising=False)
    monkeypatch.delenv("RAILWAY_COLUMNS_FILE", raising=False)
    f = tmp_path / "features.txt"
    f.write_text("trainDelayed\n", encoding="utf-8")
    main_module.apply_env_overrides(
        main_module.parse_args(["--columns-file", str(f)])
    )
    assert "RAILWAY_SCENARIO" not in os.environ


def test_list_scenarios_prints_the_catalogue_and_exits_zero(scenario_file, capsys):
    """It must not import the pipeline or touch data."""
    code = main_module.main(["--columns-file", scenario_file, "--list-scenarios"])
    assert code == 0
    out = capsys.readouterr().out
    assert "1 - ALL FEATURES" in out
    assert "2 - ONLY OPERACIONAL FEATURES" in out


def test_list_scenarios_without_columns_file_is_rejected():
    with pytest.raises(SystemExit):
        main_module.parse_args(["--list-scenarios"])


def test_fcll_net_is_a_valid_model():
    from main import MODEL_STAGES
    assert MODEL_STAGES["fcll_net"] == "train_fcll_net"


def test_fcll_net_requires_sequences_and_skips_balancing():
    """balance_classes resamples rows; on sequence data it would synthesise
    operating history that never happened."""
    from main import parse_args, build_state_machine
    from config.const_training import TRAINING_STATE_MACHINE

    args = parse_args(["--model", "fcll_net"])
    sm = build_state_machine(args, TRAINING_STATE_MACHINE)
    assert sm["build_fcll_sequences"] is True
    assert sm["train_fcll_net"] is True
    assert sm["balance_classes"] is False


def test_the_tabular_models_do_not_build_sequences():
    """Sequence building is expensive; the other five have no use for it."""
    from main import parse_args, build_state_machine
    from config.const_training import TRAINING_STATE_MACHINE

    for model in ("xgboost", "lightgbm", "random_forest",
                  "logistic_regression", "naive_bayes"):
        sm = build_state_machine(parse_args(["--model", model]), TRAINING_STATE_MACHINE)
        assert sm["build_fcll_sequences"] is False, model
        assert sm["balance_classes"] is False, model
        assert sm["train_fcll_net"] is False, model


def test_fcll_net_does_not_depend_on_the_merged_tabular_parquet():
    """select_training_cols reads stage 500/501, which this route never builds.
    Including it makes the cell fail at 'No merged data files found'."""
    from main import parse_args, build_state_machine
    from config.const_training import TRAINING_STATE_MACHINE

    sm = build_state_machine(parse_args(["--model", "fcll_net"]), TRAINING_STATE_MACHINE)
    for stage in ("merge_data_files", "filter_delay_outliers", "select_training_cols",
                  "split_dataset", "balance_classes", "scale_weather_features"):
        assert sm[stage] is False, stage


def test_a_stage_failure_after_an_earlier_success_still_exits_nonzero(
    monkeypatch, pipeline_flags
):
    """The real shape of a failed run: an early stage sets success=True, a later
    trainer fails and appends to errors. Nothing ever sets success back to False
    - there is no `result["success"] = False` anywhere in training_pipeline.py -
    so keying the exit code on that flag alone reports a dead cell as COMPLETED.
    Across a 78-cell grid that is a silently missing model, not a loud one.
    """
    calls = {"merge": 0, "train": 0}
    FakePipeline = _fake_pipeline_class(calls)

    class PartialFailurePipeline(FakePipeline):
        def execute_training_pipeline_steps(self, csv_files, state_machine):
            calls["train"] += 1
            return {
                "success": True,                       # set by build_fcll_sequences
                "steps_executed": ["build_fcll_sequences"],
                "errors": ["train_fcll_net failed: open file failed"],
            }

    import src.training_pipeline
    monkeypatch.setattr(src.training_pipeline, "TrainingPipeline", PartialFailurePipeline)

    rc = main_module._run(main_module.parse_args([]))

    assert calls["train"] == 1
    assert rc != 0, "a stage that recorded an error must not exit 0"


def test_a_clean_run_still_exits_zero(monkeypatch, pipeline_flags):
    """The other side of the guard: an empty errors list must stay a success,
    or every green run starts failing."""
    calls = {"merge": 0, "train": 0}
    FakePipeline = _fake_pipeline_class(calls)

    class CleanPipeline(FakePipeline):
        def execute_training_pipeline_steps(self, csv_files, state_machine):
            calls["train"] += 1
            return {"success": True, "steps_executed": ["train_fcll_net"], "errors": []}

    import src.training_pipeline
    monkeypatch.setattr(src.training_pipeline, "TrainingPipeline", CleanPipeline)

    assert main_module._run(main_module.parse_args([])) == 0
