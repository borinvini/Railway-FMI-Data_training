"""A numeric target (differenceInMinutes) through the training stages, XGBoost only.

Switching DEFAULT_TARGET_FEATURE to a numeric target used to fail, and the numbers it would
have produced were not comparable:

1. 684 of 117,750 OL rows (all cancelled trains) have no delay. Nothing removed them, so the
   XGBoost fit, the shap stage and every metric failed on NaN labels.
2. Stage 501 cut the delay at the 0.5% / 99.5% quantiles over every row, so test and the
   hold-out year lost their extreme delays as well (target std 20.1 -> 12.0 min) and RMSE / R2
   were measured on a trimmed target. The cut now uses the train split's own bounds and only
   touches train rows.
3. Stage 502 only dropped the three label columns, so the two other differenceInMinutes*
   columns stayed if a features file listed them: with a numeric target they are the target.
4. The search scored neg_mean_absolute_error on the log-shifted target while every reported
   metric is in minutes, and the best iteration was picked on the log-unit score.
5. Stage 504 binarised the numeric target and ran SMOTE on it.
"""
import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from config.const_preprocessing import GROUP_KEY_COLUMN
from config.const_training import RANDOM_STATE
from src.cv_utils import NegMaeMinutes, load_train_groups, make_minutes_mae_scorer
from src.day_split import split_days
from src.training_pipeline import TrainingPipeline

TARGET = "differenceInMinutes"
OTHER_DELAYS = ["differenceInMinutes_offset", "differenceInMinutes_eachStation_offset"]


def _pipeline(tmp_path):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.project_root = str(tmp_path)
    pipeline.script_dir = str(tmp_path)
    pipeline.log_dir = str(tmp_path / "log")
    os.makedirs(pipeline.log_dir, exist_ok=True)
    return pipeline


def _delay_frame(years=(2023, 2024), days_per_year=40, rows_per_day=15, seed=3):
    """f1 is the signal, f2 noise; delay is 3*f1 + noise, so it is negative sometimes."""
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


# --- stage 502 -----------------------------------------------------------------

def test_stage_502_drops_every_other_target_like_column(tmp_path):
    df = pd.DataFrame({
        TARGET: [1.0], OTHER_DELAYS[0]: [1.0], OTHER_DELAYS[1]: [1.0],
        "trainDelayed": [True], "trainDisrupted": [True], "cancelled": [False],
        "f1": [0.5], GROUP_KEY_COLUMN: ["2023-01-01"],
    })
    wanted = ["trainDisrupted", OTHER_DELAYS[0], OTHER_DELAYS[1], "f1"]
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET):
        kept = _pipeline(tmp_path).apply_column_selection(df, wanted)
    assert set(kept.columns) == {TARGET, "f1", GROUP_KEY_COLUMN}


def test_stage_502_also_protects_a_classification_run_from_the_delay_columns(tmp_path):
    df = pd.DataFrame({
        TARGET: [1.0], OTHER_DELAYS[0]: [1.0], "trainDisrupted": [True], "f1": [0.5],
    })
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDisrupted"):
        kept = _pipeline(tmp_path).apply_column_selection(df, [TARGET, OTHER_DELAYS[0], "f1"])
    assert set(kept.columns) == {"trainDisrupted", "f1"}


# --- stage 501 -----------------------------------------------------------------

@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/outlier_filtered.parquet")
def test_stage_501_leaves_a_numeric_target_for_the_train_only_cut(mock_save, tmp_path):
    df = pd.DataFrame({TARGET: list(np.linspace(-5, 20, 200)) + [900.0], "f1": np.arange(201.0)})
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET):
        result = _pipeline(tmp_path).filter_delay_outliers(data=df)
    assert result["success"] is True
    assert len(result["data"]) == len(df), "stage 501 must not trim a numeric target"
    assert result["lower_bound"] is None and result["upper_bound"] is None


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/outlier_filtered.parquet")
def test_stage_501_still_trims_a_classification_run(mock_save, tmp_path):
    df = pd.DataFrame({TARGET: list(np.linspace(-5, 20, 200)) + [900.0], "f1": np.arange(201.0)})
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDisrupted"):
        result = _pipeline(tmp_path).filter_delay_outliers(data=df)
    assert len(result["data"]) < len(df)
    assert result["data"][TARGET].max() < 900.0


# --- stage 503 -----------------------------------------------------------------

def test_stage_503_drops_nan_targets_and_trims_the_train_split_only(tmp_path):
    df = _delay_frame()
    src, out = tmp_path / "data", tmp_path / "out"
    src.mkdir()

    # One extreme delay per split, placed on days of that split (the split is deterministic).
    dev_days = sorted(set(df.loc[df["year"] < 2024, GROUP_KEY_COLUMN]))
    train_days, test_days = split_days(dev_days, seed=RANDOM_STATE)
    pick = lambda days, year: df.index[df[GROUP_KEY_COLUMN].isin(days) & (df["year"] == year)][0]
    df.loc[pick(train_days, 2023), TARGET] = 600.0
    df.loc[pick(test_days, 2023), TARGET] = 700.0
    df.loc[df.index[df["year"] == 2024][0], TARGET] = 800.0
    # cancelled trains: no delay
    nan_rows = list(df.index[df["year"] < 2024][::97])
    df.loc[nan_rows, TARGET] = np.nan
    assert len(nan_rows) >= 5
    df.to_parquet(src / "merged_data.parquet", index=False)

    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET), \
         patch("src.training_pipeline.HOLDOUT_LAST_YEAR", True):
        result = _pipeline(tmp_path).split_dataset(data_dir=str(src), output_dir=str(out))

    assert result["success"] is True
    train = pd.read_parquet(out / "merged_data_train.parquet")
    test = pd.read_parquet(out / "merged_data_test.parquet")
    holdout = pd.read_parquet(out / "merged_data_holdout.parquet")
    for name, part in (("train", train), ("test", test), ("holdout", holdout)):
        assert part[TARGET].notna().all(), f"{name} still has NaN targets"

    assert 600.0 not in set(train[TARGET]), "the extreme train delay should have been cut"
    assert 700.0 in set(test[TARGET]), "test must keep its real extremes"
    assert 800.0 in set(holdout[TARGET]), "hold-out must keep its real extremes"

    info = result["split_details"][0]["train_outlier_filter"]
    assert info["train_rows_removed"] >= 1
    assert info["upper_bound"] < 600.0
    # the day sidecar still lines up with the (trimmed) train rows
    assert len(load_train_groups(str(out / "merged_data_train.parquet"), len(train))) == len(train)


def test_stage_503_keeps_the_train_extremes_when_the_option_is_off(tmp_path):
    df = _delay_frame(years=(2023,))
    df.loc[df.index[5], TARGET] = 600.0
    src, out = tmp_path / "data", tmp_path / "out"
    src.mkdir()
    df.to_parquet(src / "merged_data.parquet", index=False)
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET), \
         patch("src.training_pipeline.HOLDOUT_LAST_YEAR", False), \
         patch("src.training_pipeline.REGRESSION_FILTER_TRAIN_ONLY", False):
        result = _pipeline(tmp_path).split_dataset(data_dir=str(src), output_dir=str(out))
    assert result["split_details"][0]["train_outlier_filter"] is None


# --- REGRESSION_CANCELLED_POLICY = 'cap' -----------------------------------------
#
# Filling cancelled trains with 9999 would put 63% of the log-target variance on 0.6% of the
# rows, give RMSE ~780 min and move the 99.5% quantile to 9999, which switches the tail cut
# off. 'cap' fills with the train split's own upper bound instead (bounded effect, flagged rows,
# metrics also reported without them). 'drop' stays the default.

def _cap_frame_and_files(tmp_path, policy="cap"):
    """Split a frame with cancelled rows in every split; returns (df, result, out_dir)."""
    df = _delay_frame()
    dev_days = sorted(set(df.loc[df["year"] < 2024, GROUP_KEY_COLUMN]))
    train_days, test_days = split_days(dev_days, seed=RANDOM_STATE)
    train_rows = df.index[df[GROUP_KEY_COLUMN].isin(train_days) & (df["year"] == 2023)]
    test_rows = df.index[df[GROUP_KEY_COLUMN].isin(test_days) & (df["year"] == 2023)]
    hold_rows = df.index[df["year"] == 2024]
    # real extreme delays in test and hold-out: the fill value must not see them
    df.loc[test_rows[3], TARGET] = 700.0
    df.loc[hold_rows[3], TARGET] = 800.0
    # cancelled trains (no delay) in every split
    cancelled = list(train_rows[::90][:5]) + list(test_rows[::40][:3]) + list(hold_rows[::150][:3])
    assert len(cancelled) == 11
    df.loc[cancelled, TARGET] = np.nan
    src, out = tmp_path / "data", tmp_path / "out"
    src.mkdir()
    df.to_parquet(src / "merged_data.parquet", index=False)
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET), \
         patch("src.training_pipeline.HOLDOUT_LAST_YEAR", True), \
         patch("src.training_pipeline.REGRESSION_CANCELLED_POLICY", policy):
        result = _pipeline(tmp_path).split_dataset(data_dir=str(src), output_dir=str(out))
    return df, result, out


def test_cap_keeps_the_cancelled_rows_and_fills_them_with_the_train_bound(tmp_path):
    df, result, out = _cap_frame_and_files(tmp_path)
    assert result["success"] is True
    fill = result["split_details"][0]["cancelled_fill"]
    assert fill["filled_train"] == 5 and fill["filled_test"] == 3 and fill["filled_holdout"] == 3

    train = pd.read_parquet(out / "merged_data_train.parquet")
    test = pd.read_parquet(out / "merged_data_test.parquet")
    holdout = pd.read_parquet(out / "merged_data_holdout.parquet")
    for part in (train, test, holdout):
        assert part[TARGET].notna().all()

    # the value is the train bound of the real delays: bounded, and blind to the 700 / 800 extremes
    info = result["split_details"][0]["train_outlier_filter"]
    assert fill["value"] == pytest.approx(info["upper_bound"])
    assert fill["value"] < 100.0
    assert 700.0 in set(test[TARGET]) and 800.0 in set(holdout[TARGET])


def test_cap_flags_line_up_with_the_rows_they_describe(tmp_path):
    from src.cv_utils import load_filled_flags
    df, result, out = _cap_frame_and_files(tmp_path)
    fill_value = result["split_details"][0]["cancelled_fill"]["value"]
    for split, expected in (("train", 5), ("test", 3), ("holdout", 3)):
        path = str(out / f"merged_data_{split}.parquet")
        frame = pd.read_parquet(path)
        flags = load_filled_flags(path, len(frame))
        assert flags is not None and flags.sum() == expected
        assert (frame.loc[flags, TARGET] == fill_value).all()
        # f1 is a unique row id: the flagged rows are exactly the cancelled ones of the input
        assert df.set_index("f1").loc[frame.loc[flags, "f1"], TARGET].isna().all()
        assert df.set_index("f1").loc[frame.loc[~flags, "f1"], TARGET].notna().all()


def test_the_tail_cut_keeps_cancelled_rows_when_it_trims_the_train_split():
    train = pd.DataFrame({TARGET: list(np.linspace(-5, 20, 400)) + [900.0, np.nan, np.nan], "f1": np.arange(403.0)})
    trimmed, info = TrainingPipeline.trim_train_tails(train, TARGET)
    assert 900.0 not in set(trimmed[TARGET])
    assert int(trimmed[TARGET].isna().sum()) == 2
    assert info["train_rows_removed"] >= 1


def test_drop_policy_is_the_default_and_removes_the_rows(tmp_path):
    import config.const_training as const
    assert const.REGRESSION_CANCELLED_POLICY == "drop"
    df, result, out = _cap_frame_and_files(tmp_path, policy="drop")
    assert result["split_details"][0]["cancelled_fill"] is None
    assert not list(out.glob("*_filled.parquet"))
    rows = sum(len(pd.read_parquet(p)) for p in out.glob("merged_data_*.parquet")
               if p.stem.endswith(("_train", "_test", "_holdout")))
    trimmed = result["split_details"][0]["train_outlier_filter"]["train_rows_removed"]
    assert rows == len(df) - 11 - trimmed, "11 cancelled rows dropped, plus the train tail cut"


def test_both_policies_split_the_same_days(tmp_path):
    """Dropping the cancelled rows before the day split removed days made only of cancellations
    from the shuffle, so 'drop' and 'cap' had different test days and their metrics could not be
    compared. The policy is now applied after the split."""
    from src.cv_utils import load_filled_flags
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    df_d, res_d, out_d = _cap_frame_and_files(tmp_path / "a", policy="drop")
    df_c, res_c, out_c = _cap_frame_and_files(tmp_path / "b", policy="cap")

    test_drop = pd.read_parquet(out_d / "merged_data_test.parquet")
    test_cap = pd.read_parquet(out_c / "merged_data_test.parquet")
    flags = load_filled_flags(str(out_c / "merged_data_test.parquet"), len(test_cap))
    assert set(test_cap.loc[~flags, "f1"]) == set(test_drop["f1"]), "same test rows apart from the cancelled ones"
    assert res_d["split_details"][0]["dropped_missing_target"] == {"train": 5, "test": 3, "holdout": 3}


def test_a_second_split_run_ignores_the_flag_files(tmp_path):
    src = tmp_path / "data"
    src.mkdir()
    df = _delay_frame(years=(2023,))
    df.loc[df.index[::50], TARGET] = np.nan
    df.to_parquet(src / "merged_data.parquet", index=False)
    pipeline = _pipeline(tmp_path)
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET), \
         patch("src.training_pipeline.HOLDOUT_LAST_YEAR", False), \
         patch("src.training_pipeline.REGRESSION_CANCELLED_POLICY", "cap"):
        first = pipeline.split_dataset(data_dir=str(src), output_dir=str(src))
        second = pipeline.split_dataset(data_dir=str(src), output_dir=str(src))
    assert first["success"] is True and second["success"] is True
    assert (src / "merged_data_train_filled.parquet").exists()


def test_stage_505_carries_the_flags_to_the_trainers_folder(tmp_path):
    from config.const_training import MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER, SPLIT_DATASET_OUTPUT_FOLDER
    from src.cv_utils import load_filled_flags
    df = _delay_frame()
    df.loc[df.index[::40], TARGET] = np.nan
    df["Air temperature"] = np.linspace(-10, 20, len(df))
    dev = df[df["year"] < 2024].drop(columns="year")
    pipeline = _pipeline(tmp_path)
    split_dir = tmp_path / SPLIT_DATASET_OUTPUT_FOLDER
    with patch("src.training_pipeline.REGRESSION_CANCELLED_POLICY", "cap"):
        pipeline.write_splits(dev, output_dir=str(split_dir), identifier="t",
                              basename="merged_data_x", target_column=TARGET)
    assert pipeline.scale_weather_features(data_dir=str(split_dir))["success"] is True
    scaled = tmp_path / MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
    for split in ("train", "test"):
        path = str(scaled / f"merged_data_x_{split}.parquet")
        flags = load_filled_flags(path, len(pd.read_parquet(path)))
        assert flags is not None and flags.any()


def test_the_trainer_reports_running_trains_separately_under_cap(tmp_path):
    df = _delay_frame(days_per_year=45)
    df.loc[df.index[::60], TARGET] = np.nan
    dev = df[df["year"] < 2024].drop(columns="year")
    holdout = df[df["year"] == 2024].drop(columns=["year", GROUP_KEY_COLUMN])
    folder = tmp_path / "scaled"
    pipeline = _pipeline(tmp_path)
    with patch("src.training_pipeline.REGRESSION_CANCELLED_POLICY", "cap"):
        pipeline.write_splits(dev, output_dir=str(folder), identifier="t",
                              basename="merged_data_x", target_column=TARGET)
    # the hold-out file as stage 503 leaves it: filled, with its flags
    from src.cv_utils import filled_path
    value = float(dev[TARGET].quantile(0.995))
    flags = holdout[TARGET].isna().to_numpy()
    holdout.assign(**{TARGET: holdout[TARGET].fillna(value)}).to_parquet(folder / "merged_data_x_holdout.parquet", index=False)
    pd.DataFrame({"filled": flags}).to_parquet(filled_path(str(folder / "merged_data_x_holdout.parquet")), index=False)

    small_space = {"n_estimators": [30], "max_depth": [3], "learning_rate": [0.1]}
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET), \
         patch("src.training_pipeline.XGBOOST_PARAM_DISTRIBUTIONS", small_space), \
         patch("src.training_pipeline.RANDOM_SEARCH_ITERATIONS", 10), \
         patch("src.training_pipeline.SEARCH_N_JOBS", 1):
        result = pipeline.train_xgboost_with_randomized_search_cv(data_dir=str(folder))

    assert result["success"] is True, result
    results = json.loads(open(result["results_file"], encoding="utf-8").read())
    test_running = results["final_metrics_running_trains"]
    assert test_running["n_filled_excluded"] > 0
    assert test_running["n_rows"] + test_running["n_filled_excluded"] == results["dataset_info"]["test_samples"]
    hold_running = results["holdout_metrics_running_trains"]
    assert hold_running["n_filled_excluded"] == int(flags.sum())
    # the cancelled rows carry the cap as their delay, so the all-rows MAE is the worse one
    assert results["final_metrics"]["test_mae"] >= test_running["mae"]


# --- the scorer ----------------------------------------------------------------

class _Echo:
    """Estimator that predicts a fixed array, for scorer checks."""

    def __init__(self, pred):
        self.pred = pred

    def predict(self, X):
        return self.pred


def test_the_scorer_reports_mae_in_minutes_not_in_log_units():
    shift = 21.0
    y = np.array([-5.0, 0.0, 10.0, 50.0])
    pred_minutes = np.array([-3.0, 1.0, 12.0, 40.0])
    y_log = np.log1p(y + shift)
    pred_log = np.log1p(pred_minutes + shift)

    expected = -float(np.mean(np.abs(y - pred_minutes)))   # -3.25 minutes
    scorer = make_minutes_mae_scorer(shift)
    assert scorer(_Echo(pred_log), np.zeros((4, 1)), y_log) == pytest.approx(expected)
    # the old scorer would have said something near -0.1 (log units)
    assert abs(float(np.mean(np.abs(y_log - pred_log)))) < 0.5


def test_the_scorer_can_be_pickled_for_the_parallel_search():
    import pickle
    restored = pickle.loads(pickle.dumps(NegMaeMinutes(21.0)))
    assert restored.y_shift == 21.0


# --- stage 504 -----------------------------------------------------------------

def test_stage_504_has_nothing_to_balance_for_a_numeric_target():
    """Covered with fixtures in tests/test_balance_classes.py::test_regression_target_is_not_resampled;
    here only the guard's presence is pinned, so removing it fails this module too."""
    import inspect
    assert "is_regression" in inspect.getsource(TrainingPipeline.balance_classes)
    assert "nothing to balance" in inspect.getsource(TrainingPipeline.balance_classes)


# --- stage 1000, XGBoost -------------------------------------------------------

class _SearchSpy:
    seen = {}

    def __init__(self, **kwargs):
        _SearchSpy.seen = {"init": kwargs}

    def fit(self, X, y, **fit_kwargs):
        _SearchSpy.seen.update(X=X, y=y, fit=fit_kwargs)
        raise RuntimeError("spy: stop after capturing the search arguments")


def _scaled_folder(tmp_path, df):
    """Train / test / hold-out files and the day sidecar, as stage 505 leaves them."""
    folder = tmp_path / "scaled"
    dev = df[df["year"] < df["year"].max()].drop(columns="year")
    holdout = df[df["year"] == df["year"].max()].drop(columns=["year", GROUP_KEY_COLUMN])
    pipeline = _pipeline(tmp_path)
    pipeline.write_splits(dev, output_dir=str(folder), identifier="t", basename="merged_data_x",
                          target_column=TARGET)
    holdout.to_parquet(folder / "merged_data_x_holdout.parquet", index=False)
    return folder


def test_xgboost_regression_uses_day_grouped_cv_and_a_minutes_scorer(tmp_path):
    folder = _scaled_folder(tmp_path, _delay_frame())
    train = pd.read_parquet(folder / "merged_data_x_train.parquet")
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET), \
         patch("src.training_pipeline.RandomizedSearchCV", _SearchSpy):
        _pipeline(tmp_path).train_xgboost_with_randomized_search_cv(data_dir=str(folder))

    seen = _SearchSpy.seen
    assert isinstance(seen["init"]["cv"], GroupKFold)
    groups = seen["fit"]["groups"]
    assert len(groups) == len(train)
    for fit_idx, val_idx in seen["init"]["cv"].split(seen["X"], seen["y"], groups):
        assert set(groups[fit_idx]).isdisjoint(groups[val_idx])
    assert seen["init"]["scoring"] != "neg_mean_absolute_error"
    # a perfect prediction scores 0 whatever the shift, an error of 1 minute scores -1
    y_log = np.asarray(seen["y"])
    shift = float(np.expm1(y_log.min()) - train[TARGET].min())  # recovers y_shift
    scorer = seen["init"]["scoring"]
    assert scorer(_Echo(y_log), np.zeros((len(y_log), 1)), y_log) == pytest.approx(0.0, abs=1e-9)
    one_minute_off = np.log1p(np.expm1(y_log) + 1.0)
    assert scorer(_Echo(one_minute_off), np.zeros((len(y_log), 1)), y_log) == pytest.approx(-1.0)
    assert shift > 0


def test_xgboost_regression_runs_end_to_end_and_reports_minutes(tmp_path):
    """The whole numeric branch on a learnable fixture: tuning, test, hold-out, plots, JSON."""
    folder = _scaled_folder(tmp_path, _delay_frame(days_per_year=45))
    small_space = {"n_estimators": [30], "max_depth": [3], "learning_rate": [0.1]}
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET), \
         patch("src.training_pipeline.XGBOOST_PARAM_DISTRIBUTIONS", small_space), \
         patch("src.training_pipeline.RANDOM_SEARCH_ITERATIONS", 10), \
         patch("src.training_pipeline.SEARCH_N_JOBS", 1):
        result = _pipeline(tmp_path).train_xgboost_with_randomized_search_cv(data_dir=str(folder))

    assert result["success"] is True, result
    assert result["problem_type"] == "regression"
    results = json.loads(open(result["results_file"], encoding="utf-8").read())

    final = results["final_metrics"]
    assert final["test_mae"] > 0
    # tuning score and test MAE are the same quantity now: minutes, not log units
    cv_mae = -results["iteration_analysis"][0]["cv_score"]
    assert cv_mae == pytest.approx(final["test_mae"], rel=0.5)
    assert final["baseline_mae_train_median"] > final["test_mae"], "the model must beat the constant"
    assert 0 < final["mae_skill"] < 1
    assert set(final["residual_quantiles"]) == {"q05", "q25", "q50", "q75", "q95"}

    summary = results["iteration_metrics_summary"]
    assert len(summary["r2_values"]) == 1 and summary["best_r2"] > 0

    holdout = results["holdout_metrics"]
    assert holdout["mae"] > 0 and "baseline_mae" in holdout and "mae_skill" in holdout
    assert results["holdout_iteration_summary"]["r2_values"]

    out = result["output_directory"]
    names = os.listdir(out)
    assert "xgboost_residuals_x.png" in names
    assert "xgboost_residuals_holdout_x.png" in names
