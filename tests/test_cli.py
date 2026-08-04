"""CLI layer: flags override config defaults; no flags reproduces current behaviour."""
import os

import main as main_module


def test_no_args_returns_all_none():
    args = main_module.parse_args([])
    assert args.stages is None
    assert args.model is None
    assert args.data_root is None
    assert args.n_jobs is None
    assert args.search_iterations is None
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
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("RAILWAY_SEARCH_ITERATIONS", raising=False)
    args = main_module.parse_args(
        ["--data-root", "/scratch/x", "--n-jobs", "40", "--search-iterations", "10"]
    )
    main_module.apply_env_overrides(args)
    assert os.environ["RAILWAY_DATA_ROOT"] == "/scratch/x"
    assert os.environ["SLURM_CPUS_PER_TASK"] == "40"
    assert os.environ["RAILWAY_SEARCH_ITERATIONS"] == "10"


def test_apply_env_overrides_is_a_noop_without_flags(monkeypatch):
    monkeypatch.delenv("RAILWAY_DATA_ROOT", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("RAILWAY_SEARCH_ITERATIONS", raising=False)
    main_module.apply_env_overrides(main_module.parse_args([]))
    assert "RAILWAY_DATA_ROOT" not in os.environ
    assert "SLURM_CPUS_PER_TASK" not in os.environ
    assert "RAILWAY_SEARCH_ITERATIONS" not in os.environ


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


def test_dump_columns_prints_a_pasteable_block(capsys):
    calls = {"merge": 0, "train": 0}
    pipeline = _fake_pipeline_class(calls)()

    rc = main_module.dump_columns(pipeline, [])

    out = capsys.readouterr().out
    assert rc == 0
    assert calls["merge"] == 1
    assert calls["train"] == 0, "no trainer may run under --dump-columns"
    assert "alpha" in out and "beta" in out
    # The printed block must be pasteable straight into SELECTED_COLUMNS
    assert "SELECTED_COLUMNS = [" in out
    assert "'alpha'," in out


def test_dump_columns_reports_merge_failure():
    class FailingPipeline:
        def merge_data_files(self, csv_files):
            return {"success": False, "error": "no training-ready files"}

    assert main_module.dump_columns(FailingPipeline(), []) == 1


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
