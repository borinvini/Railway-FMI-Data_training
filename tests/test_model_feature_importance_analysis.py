"""Stage 702 (model_feature_importance_analysis, off by default): what a MODEL relies on.

Stage 700 explains a model on the rows it was fitted on and sees a sin/cos pair as two
unrelated columns. 702 asks the questions that matter for the 2025 hold-out: which
feature GROUPS still help on unseen rows (grouped permutation importance), and what is
lost when a group is removed (drop-group ablation, with a noise floor so a difference of
a few hundredths of a minute is not read as a finding). The model-free description of the
data, including the Spearman clusters, is stage 701 (tests/test_data_exploration.py).
These tests pin that behaviour on small synthetic frames.
"""
import hashlib
import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from config.const_training import (MODEL_FEATURE_IMPORTANCE_OUTPUT_FOLDER, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER,
                                   SPLIT_DATASET_OUTPUT_FOLDER, TRAINING_STATE_MACHINE)
from src.feature_analysis import (drop_group_ablation, feature_groups, fit_xgboost,
                                  grouped_permutation_importance, make_score_fn,
                                  protected_columns)
from src.training_pipeline import TrainingPipeline

TARGET = "differenceInMinutes_offset"
XGB_PARAMS = {"subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 40}


def _frame(n, seed, target=TARGET):
    """month_sin/cos carry the signal jointly; the rest is noise or a copy of another column."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, n)
    df = pd.DataFrame({
        "month_sin": np.sin(angle), "month_cos": np.cos(angle),
        "noise_a": rng.normal(size=n), "noise_b": rng.normal(size=n),
        "dup_a": rng.normal(size=n),
    })
    df["dup_b"] = 2.0 * df["dup_a"] + rng.normal(scale=0.05, size=n)
    df["constant"] = 1.0
    df[target] = 6.0 * df["month_sin"] + 3.0 * df["month_cos"] + rng.normal(scale=0.5, size=n)
    return df


def _fit_fn(is_classification=False):
    def fit(X, y_fit, seed):
        return fit_xgboost(X, y_fit, seed, is_classification=is_classification,
                           objective="reg:squarederror", n_jobs=1, params=XGB_PARAMS)
    return fit


# --- groups --------------------------------------------------------------------

def test_encoding_families_are_one_group_and_other_columns_stay_single():
    cols = ["month_sin", "month_cos", "wind_dir_sin", "wind_dir_cos", "lonely_sin",
            "weather_scenario_Blizzard", "weather_scenario_Black_Ice",
            "weather_scenario_12h_Blizzard", "weather_scenario_12h_Black_Ice",
            "wawa_group_rain", "wawa_group_snow", "Air temperature"]
    groups = feature_groups(cols)

    assert groups["month"] == ["month_sin", "month_cos"]
    assert groups["wind_dir"] == ["wind_dir_sin", "wind_dir_cos"]
    assert groups["lonely_sin"] == ["lonely_sin"], "a _sin without its _cos is just a column"
    assert groups["weather_scenario"] == ["weather_scenario_Blizzard", "weather_scenario_Black_Ice"]
    assert groups["weather_scenario_12h"] == ["weather_scenario_12h_Blizzard", "weather_scenario_12h_Black_Ice"]
    assert groups["wawa_group"] == ["wawa_group_rain", "wawa_group_snow"]
    assert groups["Air temperature"] == ["Air temperature"]
    assert list(groups) == ["month", "wind_dir", "lonely_sin", "weather_scenario",
                            "weather_scenario_12h", "wawa_group", "Air temperature"]
    assert sorted(protected_columns(groups)) == sorted(
        [c for c in cols if c not in ("lonely_sin", "Air temperature")])


# --- scoring in minutes --------------------------------------------------------

class _Perfect:
    """Predicts the target exactly, in the units the model was fitted on."""

    def __init__(self, y_minutes, y_shift):
        self.pred = np.asarray(y_minutes, float) if y_shift is None else np.log1p(np.asarray(y_minutes, float) + y_shift)

    def predict(self, X):
        return self.pred


@pytest.mark.parametrize("y_shift", [None, 12.0])
@pytest.mark.parametrize("metric", ["mae", "rmse"])
def test_a_perfect_numeric_prediction_scores_zero_whether_or_not_the_fit_is_logged(metric, y_shift):
    y = np.array([-5.0, 0.0, 10.0, 50.0])
    score = make_score_fn(False, metric, y_shift)(_Perfect(y, y_shift), None, y)
    assert score == pytest.approx(0.0, abs=1e-9)


# --- grouped permutation importance --------------------------------------------

def test_the_informative_pair_matters_and_noise_does_not():
    train, test = _frame(800, 1), _frame(400, 2)
    X_tr, y_tr = train.drop(columns=TARGET), train[TARGET].to_numpy()
    model = _fit_fn()(X_tr, y_tr, 0)
    score = make_score_fn(False, "mae", None)

    out = grouped_permutation_importance(model, test.drop(columns=TARGET), test[TARGET].to_numpy(),
                                         feature_groups(X_tr.columns), score, n_repeats=4, seed=0)

    by_group = out.set_index("group")
    assert out.iloc[0]["group"] == "month" and out.iloc[0]["rank"] == 1
    assert by_group.loc["month", "importance_mean"] > 1.0          # minutes of MAE
    assert not by_group.loc["month", "within_noise"]
    assert abs(by_group.loc["noise_a", "importance_mean"]) < 0.2
    assert by_group.loc["month", "n_columns"] == 2


def test_a_group_is_shuffled_with_one_row_order_so_the_pair_stays_consistent():
    X = _frame(300, 3).drop(columns=TARGET)
    y = np.zeros(len(X))
    original_pairs = set(zip(X["month_sin"].round(12), X["month_cos"].round(12)))
    seen = []

    def spy(model, Xp, yp):
        seen.append(Xp[["month_sin", "month_cos"]].copy())
        return 0.0

    grouped_permutation_importance(object(), X, y, {"month": ["month_sin", "month_cos"]}, spy,
                                   n_repeats=2, seed=5)

    baseline_call, *shuffled = seen
    assert shuffled, "the group must have been scored after shuffling"
    for frame in shuffled:
        pairs = set(zip(frame["month_sin"].round(12), frame["month_cos"].round(12)))
        assert pairs == original_pairs, "each row must still hold a real (sin, cos) pair"
        assert not frame["month_sin"].equals(X["month_sin"]), "and the rows must actually move"


# --- drop-group ablation -------------------------------------------------------

def test_dropping_the_informative_group_hurts_and_dropping_noise_stays_within_the_noise_floor():
    train, test = _frame(800, 1), _frame(400, 2)
    X_tr, y_tr = train.drop(columns=TARGET), train[TARGET].to_numpy()
    evals = {"test": (test.drop(columns=TARGET), test[TARGET].to_numpy())}

    table, baseline, noise = drop_group_ablation(
        _fit_fn(), X_tr, y_tr, evals, feature_groups(X_tr.columns),
        {"all_noise": ["noise_a", "noise_b"], "absent": ["nothing_here"]},
        make_score_fn(False, "mae", None), noise_seeds=3, seed=0)

    by = table.set_index("dropped")
    assert by.loc["month", "test_change"] > 1.0 and not by.loc["month", "test_within_noise"]
    assert abs(by.loc["noise_a", "test_change"]) < 0.3
    assert by.loc["all_noise", "kind"] == "set"
    assert "absent" not in by.index, "a set whose columns are all missing is skipped"
    assert noise["test"] > 0, "subsampling must make the seeds differ, or the floor is meaningless"
    assert baseline["test"] <= 0


# --- the stage -----------------------------------------------------------------

def _make_pipeline(tmp_path):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.project_root = str(tmp_path)
    return pipeline


def _write_splits(folder, target, classification=False):
    folder.mkdir(parents=True, exist_ok=True)
    for name, n, seed in (("train", 700, 1), ("test", 300, 2), ("holdout", 300, 3)):
        df = _frame(n, seed, target)
        if classification:
            df[target] = (df[target] > 0).astype(int)
        df.to_parquet(folder / f"merged_data_x_{name}.parquet", index=False)
    return str(folder)


def _digest(folder):
    return {p: hashlib.sha256(open(os.path.join(folder, p), "rb").read()).hexdigest()
            for p in sorted(os.listdir(folder))}


_FAST = dict(PERMUTATION_REPEATS=3, ABLATION_NOISE_SEEDS=2, SEARCH_N_JOBS=1,
             FEATURE_ANALYSIS_XGB_PARAMS=XGB_PARAMS)


def _patched(target, **extra):
    stack = [patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", target)]
    for name, value in {**_FAST, **extra}.items():
        stack.append(patch(f"src.training_pipeline.{name}", value))
    return stack


def _run(pipeline, data_dir, target, **extra):
    from contextlib import ExitStack
    with ExitStack() as stack:
        for p in _patched(target, **extra):
            stack.enter_context(p)
        return pipeline.model_feature_importance_analysis(data_dir=data_dir)


def test_end_to_end_numeric_writes_every_output_and_leaves_the_input_alone(tmp_path):
    data_dir = _write_splits(tmp_path / "scaled", TARGET)
    before = _digest(data_dir)

    result = _run(_make_pipeline(tmp_path), data_dir, TARGET)

    assert result["success"] is True, result
    assert result["problem_type"] == "regression" and result["primary_split"] == "holdout"
    out = result["output_directory"]
    assert os.path.normpath(out).endswith(os.path.normpath(MODEL_FEATURE_IMPORTANCE_OUTPUT_FOLDER))
    for name in ("permutation_importance.csv", "permutation_importance_holdout.png",
                 "permutation_importance_holdout.pdf", "ablation.csv", "ablation_holdout.png",
                 "ablation_holdout.pdf", "summary.json"):
        assert os.path.isfile(os.path.join(out, name)), name
    assert not os.path.exists(os.path.join(out, "suggested_features.txt")), \
        "the Spearman outputs belong to stage 701 now"

    summary = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
    assert summary["groups"]["month"] == ["month_sin", "month_cos"]
    assert summary["top_permutation_groups"][0]["group"] == "month"
    assert summary["primary_split"] == "holdout" and summary["noise_std"]["holdout"] > 0
    assert "no search" in summary["model"]

    perm = pd.read_csv(os.path.join(out, "permutation_importance.csv"))
    assert set(perm["split"]) == {"test", "holdout"}
    assert _digest(data_dir) == before, "the stage must not modify its input"


def test_end_to_end_classification_and_log_target_paths_run(tmp_path):
    data_dir = _write_splits(tmp_path / "scaled", "trainDelayed", classification=True)
    result = _run(_make_pipeline(tmp_path), data_dir, "trainDelayed")
    assert result["success"] is True, result
    assert result["problem_type"] == "classification"

    numeric_dir = _write_splits(tmp_path / "scaled2", TARGET)
    logged = _run(_make_pipeline(tmp_path / "logged"), numeric_dir, TARGET, REGRESSION_LOG_TARGET=True)
    assert logged["success"] is True, logged
    summary = json.load(open(os.path.join(logged["output_directory"], "summary.json"), encoding="utf-8"))
    assert summary["log_target"] is True


def test_a_missing_train_file_is_reported_not_raised(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _make_pipeline(tmp_path).model_feature_importance_analysis(data_dir=str(empty))
    assert result["success"] is False and "train" in result["error"]


# --- wiring --------------------------------------------------------------------

def test_the_switch_is_off_by_default_and_runs_after_701_and_before_the_trainers():
    keys = list(TRAINING_STATE_MACHINE)
    assert TRAINING_STATE_MACHINE["model_feature_importance_analysis"] is False, \
        "the model-based diagnostic is run on demand, not on every full run"
    assert keys.index("explore_training_data") < keys.index("model_feature_importance_analysis") \
        < keys.index("train_xgboost_with_randomized_search_cv")
    assert MODEL_FEATURE_IMPORTANCE_OUTPUT_FOLDER.startswith("data/output/702-")


def _dispatcher_pipeline(tmp_path):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.script_dir = pipeline.project_root = str(tmp_path)
    pipeline.output_dir = str(tmp_path / "output")
    pipeline.log_dir = str(tmp_path / "log")
    os.makedirs(pipeline.log_dir, exist_ok=True)
    return pipeline


_FI_SUCCESS = {"success": True, "output_directory": "/fake", "num_features": 3, "num_groups": 3,
               "primary_split": "holdout", "top_group": ("month", 1.5)}


def _machine(**flags):
    base = {"merge_data_files": False, "select_training_cols": False, "split_dataset": False,
            "balance_classes": False, "scale_weather_features": False, "shap_correlation_analysis": False,
            "model_feature_importance_analysis": True}
    return {**base, **flags}


@pytest.mark.parametrize("scale,folder", [(True, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER),
                                          (False, SPLIT_DATASET_OUTPUT_FOLDER)])
def test_the_dispatcher_hands_the_stage_the_scaled_or_the_split_folder(tmp_path, scale, folder):
    pipeline = _dispatcher_pipeline(tmp_path)
    with patch.object(TrainingPipeline, "model_feature_importance_analysis", return_value=_FI_SUCCESS) as stage, \
         patch.object(TrainingPipeline, "scale_weather_features",
                      return_value={"success": True, "processed_files": 1, "total_train_rows": 1,
                                    "total_test_rows": 1, "weather_features_scaled": []}):
        result = pipeline.execute_training_pipeline_steps([], state_machine=_machine(scale_weather_features=scale))

    stage.assert_called_once()
    assert stage.call_args.kwargs["data_dir"] == os.path.join(str(tmp_path), folder)
    assert "model_feature_importance_analysis" in result["steps_executed"]


def test_run_alone_the_stage_finds_the_scaled_folder_an_earlier_run_left(tmp_path):
    """`--stages model_feature_importance_analysis` enables only this stage, so the scaling flag is
    off; the first real run looked in 503 and failed although 505 was sitting there."""
    scaled = tmp_path / MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
    scaled.mkdir(parents=True)
    (scaled / "merged_data_x_train.parquet").write_bytes(b"")
    with patch.object(TrainingPipeline, "model_feature_importance_analysis", return_value=_FI_SUCCESS) as stage:
        _dispatcher_pipeline(tmp_path).execute_training_pipeline_steps(
            [], state_machine=_machine(scale_weather_features=False))
    assert os.path.normpath(stage.call_args.kwargs["data_dir"]) == os.path.normpath(str(scaled))


def test_the_dispatcher_skips_the_stage_when_disabled(tmp_path):
    with patch.object(TrainingPipeline, "model_feature_importance_analysis") as stage:
        _dispatcher_pipeline(tmp_path).execute_training_pipeline_steps(
            [], state_machine=_machine(model_feature_importance_analysis=False))
    stage.assert_not_called()


def test_a_failed_stage_stops_the_run_before_the_trainers(tmp_path):
    with patch.object(TrainingPipeline, "model_feature_importance_analysis",
                      return_value={"success": False, "error": "boom"}), \
         patch.object(TrainingPipeline, "train_fcll_net") as later:
        result = _dispatcher_pipeline(tmp_path).execute_training_pipeline_steps(
            [], state_machine=_machine(train_fcll_net=True))
    assert "boom" in result["errors"]
    later.assert_not_called()
