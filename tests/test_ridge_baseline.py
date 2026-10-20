"""Stage 999, the ridge-regression baseline that runs before XGBoost on a numeric target.

The XGBoost numeric run reported MAE / RMSE / R2 in minutes with nothing simple to compare it
with. A reviewer asked for a naive reference (R2.14a) and a scale-free number (R2.14c, MASE or
MAE relative to a reference). Logistic regression was the first idea and cannot serve: it is a
classifier, and a numeric delay has no yes/no label. The baseline is therefore a ridge regression
on the same train / test / hold-out files, scored with the same metrics as XGBoost, including
`mae_skill` (1 - MAE / MAE of always predicting the train median, i.e. 1 - MASE against the median).

Pinned here: it fits the raw minutes (no log1p) inside a scaler + ridge pipeline, tunes alpha with
day-grouped CV scored in minutes, reports test and hold-out like XGBoost, treats the cancelled
rows of `REGRESSION_CANCELLED_POLICY='cap'` the same way, is skipped (not failed) for a
classification target, and sits before XGBoost in the state machine and in the dispatcher.
"""
import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.const_preprocessing import GROUP_KEY_COLUMN
from config.const_training import MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER, TRAINING_STATE_MACHINE
from src.training_pipeline import TrainingPipeline

TARGET = "differenceInMinutes"
SMALL_GRID = [0.1, 10.0]


def _pipeline(tmp_path):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.project_root = str(tmp_path)
    pipeline.script_dir = str(tmp_path)
    pipeline.log_dir = str(tmp_path / "log")
    os.makedirs(pipeline.log_dir, exist_ok=True)
    return pipeline


def _delay_frame(years=(2023, 2024), days_per_year=45, rows_per_day=15, seed=3):
    """f1 is the signal, f2 noise; delay is 3*f1 + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        days = pd.date_range(f"{year}-01-01", periods=days_per_year).strftime("%Y-%m-%d")
        for day in days:
            f1 = rng.normal(size=rows_per_day)
            rows.append(pd.DataFrame({
                GROUP_KEY_COLUMN: day,
                "year": year,
                "f1": f1,
                "f2": rng.normal(size=rows_per_day),
                TARGET: 3.0 * f1 + rng.normal(0, 1.0, rows_per_day),
            }))
    return pd.concat(rows, ignore_index=True)


def _scaled_folder(tmp_path, df):
    """Train / test / hold-out files and the day sidecar, as stage 505 leaves them."""
    folder = tmp_path / "scaled"
    dev = df[df["year"] < df["year"].max()].drop(columns="year")
    holdout = df[df["year"] == df["year"].max()].drop(columns=["year", GROUP_KEY_COLUMN])
    _pipeline(tmp_path).write_splits(dev, output_dir=str(folder), identifier="t",
                                     basename="merged_data_x", target_column=TARGET)
    holdout.to_parquet(folder / "merged_data_x_holdout.parquet", index=False)
    return folder


def _run(tmp_path, folder, **extra):
    patches = {"src.training_pipeline.DEFAULT_TARGET_FEATURE": TARGET,
               "src.training_pipeline.RIDGE_BASELINE_ALPHAS": SMALL_GRID,
               "src.training_pipeline.SEARCH_N_JOBS": 1}
    patches.update(extra)
    managers = [patch(name, value) for name, value in patches.items()]
    for m in managers:
        m.start()
    try:
        return _pipeline(tmp_path).train_ridge_baseline(data_dir=str(folder))
    finally:
        for m in managers:
            m.stop()


def test_a_classification_target_is_skipped_not_failed(tmp_path):
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDisrupted"):
        result = _pipeline(tmp_path).train_ridge_baseline(data_dir=str(tmp_path / "nothing"))
    assert result["success"] is True and result["skipped"] is True
    assert not (tmp_path / "data").exists(), "a skipped stage must not create its output folder"


def test_the_baseline_is_a_scaled_ridge_on_the_raw_minutes_with_day_grouped_cv(tmp_path):
    folder = _scaled_folder(tmp_path, _delay_frame())
    train = pd.read_parquet(folder / "merged_data_x_train.parquet")
    seen = {}

    class Spy:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def fit(self, X, y, **fit_kwargs):
            seen.update(X=X, y=y, fit=fit_kwargs)
            raise RuntimeError("spy: stop after capturing the search arguments")

    _run(tmp_path, folder, **{"src.training_pipeline.GridSearchCV": Spy})

    estimator = seen["init"]["estimator"]
    assert isinstance(estimator, Pipeline)
    assert isinstance(estimator.named_steps["scale"], StandardScaler)
    assert isinstance(estimator.named_steps["ridge"], Ridge)
    assert seen["init"]["param_grid"] == {"ridge__alpha": SMALL_GRID}
    np.testing.assert_array_equal(np.asarray(seen["y"]), train[TARGET].to_numpy())   # no log1p
    assert seen["init"]["scoring"] == "neg_mean_absolute_error"                       # already minutes
    assert isinstance(seen["init"]["cv"], GroupKFold)
    groups = seen["fit"]["groups"]
    assert len(groups) == len(train)
    for fit_idx, val_idx in seen["init"]["cv"].split(seen["X"], seen["y"], groups):
        assert set(groups[fit_idx]).isdisjoint(groups[val_idx])


def test_it_runs_end_to_end_and_reports_what_xgboost_reports(tmp_path):
    folder = _scaled_folder(tmp_path, _delay_frame())
    result = _run(tmp_path, folder)

    assert result["success"] is True, result
    assert result["problem_type"] == "regression"
    results = json.loads(open(result["results_file"], encoding="utf-8").read())

    assert results["best_alpha"] in SMALL_GRID
    assert len(results["alpha_scores"]) == len(SMALL_GRID)
    final = results["final_metrics"]
    cv_mae = -results["cv_score"]
    assert cv_mae == pytest.approx(final["test_mae"], rel=0.5)      # one quantity: minutes
    assert final["baseline_mae_train_median"] > final["test_mae"], "a learnable fixture: ridge must beat the constant"
    assert 0 < final["mae_skill"] < 1
    assert set(final["residual_quantiles"]) == {"q05", "q25", "q50", "q75", "q95"}
    assert final["test_r2"] > 0 and "test_bin_f1" in final

    holdout = results["holdout_metrics"]
    assert holdout["mae"] > 0 and "baseline_mae" in holdout and "mae_skill" in holdout

    # the signal column outweighs the noise column
    coefficients = pd.read_csv(os.path.join(result["output_directory"], "ridge_baseline_coefficients_x.csv"))
    assert set(coefficients["feature"]) == {"f1", "f2"}
    assert coefficients.iloc[0]["feature"] == "f1"

    names = os.listdir(result["output_directory"])
    for expected in ("ridge_baseline_residuals_x.png", "ridge_baseline_residuals_holdout_x.png",
                     "ridge_baseline_best_model_x.pkl", "ridge_baseline_x.json"):
        assert expected in names


def test_the_median_reference_matches_the_one_xgboost_reports(tmp_path):
    """Same train median, same MAE definition: the two models' mae_skill are on one scale."""
    from src.cv_utils import mae_baseline
    folder = _scaled_folder(tmp_path, _delay_frame())
    result = _run(tmp_path, folder)
    results = json.loads(open(result["results_file"], encoding="utf-8").read())
    train = pd.read_parquet(folder / "merged_data_x_train.parquet")
    test = pd.read_parquet(folder / "merged_data_x_test.parquet")
    expected = mae_baseline(train[TARGET], test[TARGET], results["final_metrics"]["test_mae"])
    assert results["final_metrics"]["baseline_mae_train_median"] == pytest.approx(expected["baseline_mae"])
    assert results["final_metrics"]["mae_skill"] == pytest.approx(expected["mae_skill"])


def test_cancelled_rows_filled_by_the_cap_policy_are_reported_separately(tmp_path):
    from src.cv_utils import filled_path
    df = _delay_frame()
    df.loc[df.index[::60], TARGET] = np.nan
    dev = df[df["year"] < 2024].drop(columns="year")
    holdout = df[df["year"] == 2024].drop(columns=["year", GROUP_KEY_COLUMN])
    folder = tmp_path / "scaled"
    with patch("src.training_pipeline.REGRESSION_CANCELLED_POLICY", "cap"):
        _pipeline(tmp_path).write_splits(dev, output_dir=str(folder), identifier="t",
                                         basename="merged_data_x", target_column=TARGET)
    value = float(dev[TARGET].quantile(0.995))
    flags = holdout[TARGET].isna().to_numpy()
    holdout_path = folder / "merged_data_x_holdout.parquet"
    holdout.assign(**{TARGET: holdout[TARGET].fillna(value)}).to_parquet(holdout_path, index=False)
    pd.DataFrame({"filled": flags}).to_parquet(filled_path(str(holdout_path)), index=False)

    result = _run(tmp_path, folder)
    assert result["success"] is True, result
    results = json.loads(open(result["results_file"], encoding="utf-8").read())
    running = results["final_metrics_running_trains"]
    assert running["n_filled_excluded"] > 0
    assert running["n_rows"] + running["n_filled_excluded"] == results["dataset_info"]["test_samples"]
    assert results["holdout_metrics_running_trains"]["n_filled_excluded"] == int(flags.sum())


# --- where the stage sits ------------------------------------------------------

def test_the_stage_sits_right_before_xgboost_in_the_state_machine():
    keys = list(TRAINING_STATE_MACHINE)
    assert keys.index("train_ridge_baseline") == keys.index("train_xgboost_with_randomized_search_cv") - 1
    assert TRAINING_STATE_MACHINE["train_ridge_baseline"] is True


def test_the_dispatcher_runs_the_baseline_before_xgboost_on_the_scaled_files(tmp_path):
    state_machine = {"scale_weather_features": False, "train_ridge_baseline": True,
                     "train_xgboost_with_randomized_search_cv": True}
    calls = []
    ok = {"success": True, "models_trained": 1, "problem_type": "regression", "target_feature": TARGET,
          "cv_score": -1.0, "test_rmse": 1.0, "test_mae": 1.0, "test_r2": 0.1, "mae_skill": 0.1,
          "output_directory": "/fake"}

    def fake(name):
        def _call(self, data_dir=None):
            calls.append((name, data_dir))
            return ok
        return _call

    with patch.object(TrainingPipeline, "train_ridge_baseline", fake("ridge")), \
         patch.object(TrainingPipeline, "train_xgboost_with_randomized_search_cv", fake("xgboost")):
        result = _pipeline(tmp_path).execute_training_pipeline_steps([], state_machine=state_machine)

    assert [name for name, _ in calls] == ["ridge", "xgboost"]
    assert calls[0][1] == calls[1][1], "both models must read the same data folder"
    assert not result["errors"]
    assert result["steps_executed"][-2:] == ["train_ridge_baseline", "train_xgboost_with_randomized_search_cv"]


def test_a_skipped_baseline_does_not_stop_the_run_or_count_as_executed(tmp_path):
    state_machine = {"train_ridge_baseline": True}
    skipped = {"success": True, "skipped": True, "problem_type": "classification", "target_feature": "trainDisrupted"}
    with patch.object(TrainingPipeline, "train_ridge_baseline", return_value=skipped):
        result = _pipeline(tmp_path).execute_training_pipeline_steps([], state_machine=state_machine)
    assert not result["errors"]
    assert "train_ridge_baseline" not in result["steps_executed"]


def test_the_model_flag_runs_only_the_baseline_and_its_prerequisites():
    from main import build_state_machine, parse_args
    sm = build_state_machine(parse_args(["--model", "ridge_baseline"]), TRAINING_STATE_MACHINE)
    assert sm["train_ridge_baseline"] is True
    assert sm["train_xgboost_with_randomized_search_cv"] is False
    assert sm["merge_data_files"] is True and sm["scale_weather_features"] is True
    assert sm["build_fcll_sequences"] is False and sm["balance_classes"] is False
