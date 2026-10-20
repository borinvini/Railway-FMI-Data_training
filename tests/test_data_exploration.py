"""Stage 701 (explore_training_data): charts that describe the data itself, with no model.

The stage exists to explain the data to a person, so these tests pin the things that would
silently mislead: the hour, month, weekday, day of month and wind direction recovered from
their sin/cos pairs must be exact; a heavily repeated value (0 mm of rain for 93% of the
trains, the -1 snow-depth sentinel) must get a bin of its own instead of collapsing the
variable into one bin and vanishing from the charts; the share of late trains, not the
median of whole minutes (which is 1 for almost every group), must lead; and nothing in the
stage may fit a model.
"""
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from config.columns_file import load_columns
from config.const_training import (DATA_EXPLORATION_OUTPUT_FOLDER, MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER,
                                   TRAINING_STATE_MACHINE)
from src.data_exploration import (bin_codes, association_by_year, association_table, data_summary, drift_table,
                                  monthly_series, profile_by, rank_eta_squared, recover_readable_variables,
                                  spearman_matrix, variable_names)
from src.feature_analysis import feature_groups, protected_columns, spearman_clusters
from src.training_pipeline import TrainingPipeline

TARGET = "differenceInMinutes_offset"
TWO_PI = 2 * np.pi


def _sincos(values, period, first=0):
    angle = TWO_PI * (np.asarray(values, float) - first) / period
    return np.sin(angle), np.cos(angle)


# --- recovering readable variables ---------------------------------------------------

def test_hour_month_weekday_day_of_month_and_wind_direction_are_recovered_exactly():
    hours = np.array([0.0, 6.5, 13.25, 23.75])
    months = np.array([1, 4, 9, 12])
    weekdays = np.array([1, 2, 5, 7])           # 1 = Sunday
    days = np.array([1, 10, 28, 31])
    wind = np.array([1.0, 90.0, 270.5, 359.0])
    frame = pd.DataFrame()
    for stem, vals, period, first in (("hour", hours, 24, 0), ("month", months, 12, 1), ("day_week", weekdays, 7, 1),
                                      ("day_month", days, 31, 1), ("wind_dir", wind, 360, 0)):
        frame[f"{stem}_sin"], frame[f"{stem}_cos"] = _sincos(vals, period, first)

    readable, meta = recover_readable_variables(frame)

    np.testing.assert_allclose(readable["hour"], hours, atol=1e-9)
    assert readable["month"].tolist() == months.tolist()
    assert readable["weekday"].tolist() == weekdays.tolist()
    assert readable["day_of_month"].tolist() == days.tolist()
    np.testing.assert_allclose(readable["wind_direction"], wind, atol=1e-9)
    assert not readable["wind_calm"].any()
    assert {meta[n]["kind"] for n in variable_names(meta)} == {"circular"}
    assert meta["wind_direction"]["sources"] == ["wind_dir_sin", "wind_dir_cos"]


def test_calm_wind_is_recovered_as_calm_and_not_as_a_direction():
    frame = pd.DataFrame({"wind_dir_sin": [0.0, 1.0], "wind_dir_cos": [0.0, 0.0]})
    readable, _ = recover_readable_variables(frame)
    assert readable["wind_calm"].tolist() == [True, False]
    assert np.isnan(readable["wind_direction"].iloc[0]) and readable["wind_direction"].iloc[1] == pytest.approx(90.0)


def test_plain_columns_pass_through_flags_become_binary_and_constants_are_reported():
    frame = pd.DataFrame({"Air temperature": np.linspace(-10, 20, 50), "trainStopping": [True, False] * 25,
                          "constant": 1.0, "name": ["x"] * 50})
    readable, meta = recover_readable_variables(frame)
    assert list(readable.columns) == ["Air temperature", "trainStopping"]
    assert meta["Air temperature"]["kind"] == "continuous" and meta["trainStopping"]["kind"] == "binary"
    assert readable["trainStopping"].dtype == float
    assert meta["__constant__"] == ["constant"]
    assert variable_names(meta) == ["Air temperature", "trainStopping"]


# --- bins ----------------------------------------------------------------------------

def test_359_and_1_degrees_fall_in_the_same_compass_sector_and_calm_has_its_own_code():
    frame = pd.DataFrame({"wind_dir_sin": [*_sincos([359.0, 1.0, 180.0], 360)[0], 0.0],
                          "wind_dir_cos": [*_sincos([359.0, 1.0, 180.0], 360)[1], 0.0]})
    readable, meta = recover_readable_variables(frame)
    codes = bin_codes(readable, meta, n_bins=10, n_sectors=16)
    assert codes["wind_direction"].tolist() == [0.0, 0.0, 8.0, -1.0]


def test_a_heavily_repeated_value_gets_its_own_bin_so_the_variable_does_not_collapse():
    """Precipitation intensity is 0 for 93% of the trains: a plain quantile cut gave one bin and
    the variable disappeared from the association charts."""
    rng = np.random.default_rng(0)
    rain = np.where(rng.random(2000) < 0.93, 0.0, rng.gamma(2.0, 1.0, 2000))
    snow = np.where(rng.random(2000) < 0.3, -1.0, np.where(rng.random(2000) < 0.3, 0.0, rng.uniform(1, 60, 2000)))
    frame = pd.DataFrame({"rain": rain, "snow": snow})
    readable, meta = recover_readable_variables(frame)

    codes = bin_codes(readable, meta, n_bins=10, n_sectors=16)

    assert codes["rain"].nunique() >= 3
    assert (codes["rain"][rain == 0.0] == 0.0).all() and (codes["rain"][rain > 0] >= 1.0).all()
    assert codes["snow"][snow == -1.0].nunique() == 1 and codes["snow"][snow == 0.0].nunique() == 1
    assert codes["snow"][snow == -1.0].iloc[0] != codes["snow"][snow == 0.0].iloc[0]


# --- association with the target -----------------------------------------------------

def test_rank_eta_squared_is_one_for_a_determining_variable_and_near_chance_for_noise():
    rng = np.random.default_rng(1)
    group = rng.integers(0, 8, 4000)
    y = group * 10.0 + rng.normal(scale=0.01, size=4000)
    # ranks still vary inside a group, so the ceiling for 8 groups is about 1 - 1/8**2 = 0.984
    assert rank_eta_squared(group, y)["eta2"] > 0.97

    noise = rng.normal(size=4000)
    stat = rank_eta_squared(rng.integers(0, 8, 4000), noise)
    assert stat["eta2"] < 0.01 and stat["chance"] == pytest.approx(7 / 3999)
    assert np.isnan(rank_eta_squared(np.zeros(100), rng.normal(size=100))["eta2"]), "one group: nothing to measure"


def test_a_circular_variable_is_measured_even_though_a_correlation_would_miss_it():
    rng = np.random.default_rng(2)
    hour = rng.uniform(0, 24, 6000)
    y = np.cos(TWO_PI * hour / 24) * 5 + rng.normal(size=6000)        # a pure daily wave: rho ~ 0
    frame = pd.DataFrame({"hour_sin": np.sin(TWO_PI * hour / 24), "hour_cos": np.cos(TWO_PI * hour / 24),
                          "noise": rng.normal(size=6000)})
    readable, meta = recover_readable_variables(frame)
    table = association_table(readable, meta, bin_codes(readable, meta, 10, 16), y).set_index("variable")

    assert table.loc["hour", "eta2"] > 0.5 and table.loc["noise", "eta2"] < 0.02
    assert np.isnan(table.loc["hour", "spearman_rho"]), "a line means nothing on a circle"
    assert table.index[0] == "hour", "sorted by how much is explained"


def test_association_by_year_has_one_column_per_year():
    rng = np.random.default_rng(3)
    codes = pd.DataFrame({"a": rng.integers(0, 4, 600)})
    years = pd.Series(np.repeat([2023, 2024, 2025], 200))
    out = association_by_year(codes, rng.normal(size=600), years)
    assert list(out.columns) == [2023, 2024, 2025] and list(out.index) == ["a"]


# --- profiles, time series, drift ----------------------------------------------------

def test_profile_leads_with_the_share_of_late_trains_not_the_median():
    keys = np.array([0] * 100 + [1] * 100)
    y = np.array([1.0] * 80 + [9.0] * 20 + [1.0] * 40 + [9.0] * 60)     # median is 1 in group 0 and 9 in group 1
    prof = profile_by(keys, y, is_binary=False, threshold=5)
    assert prof["share"].tolist() == [0.2, 0.6]
    assert prof["n"].tolist() == [100, 100]
    assert (prof["share_low"] < prof["share"]).all() and (prof["share"] < prof["share_high"]).all()
    assert prof["share_low"].min() >= 0 and prof["share_high"].max() <= 1
    assert prof["p90"].tolist() == [9.0, 9.0] and prof["center"].tolist() == [1.0, 9.0]


def test_profile_of_a_binary_target_is_its_rate_and_a_numeric_target_needs_the_threshold():
    prof = profile_by([0, 0, 0, 0, 1, 1], [1, 0, 0, 0, 1, 1], is_binary=True)
    assert prof["share"].tolist() == [0.25, 1.0] and prof["center"].tolist() == [0.25, 1.0]
    with pytest.raises(ValueError, match="threshold"):
        profile_by([0, 1], [1.0, 2.0], is_binary=False)


def test_monthly_series_counts_rows_and_the_share_above_the_threshold():
    ym = ["2024-01"] * 4 + ["2024-02"] * 4
    y = [0, 0, 10, 10, 10, 10, 10, 0]
    out = monthly_series(pd.Series(ym), y, is_binary=False, threshold=5)
    assert out["n"].tolist() == [4, 4] and out["share_above"].tolist() == [0.5, 0.75]
    assert out.index.tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]


def test_drift_flags_a_shifted_variable_and_not_an_identical_one():
    rng = np.random.default_rng(4)
    n = 4000
    hold = np.r_[np.zeros(3000, bool), np.ones(1000, bool)]
    readable = pd.DataFrame({"same": rng.normal(size=n), "shifted": np.where(hold, rng.normal(3, 1, n), rng.normal(0, 1, n))})
    meta = {"same": {"kind": "continuous", "sources": ["same"]}, "shifted": {"kind": "continuous", "sources": ["shifted"]},
            "__constant__": []}
    out = drift_table(readable, meta, hold).set_index("variable")
    assert out.index[0] == "shifted" and out.loc["shifted", "ks"] > 0.8 and out.loc["same", "ks"] < 0.1
    assert out.loc["shifted", "median_shift_iqr"] > 1.5


def test_data_summary_reports_missing_zeros_and_the_minus_one_sentinel():
    frame = pd.DataFrame({"snow": [-1.0, -1.0, 0.0, 5.0, np.nan]})
    row = data_summary(frame).iloc[0]
    assert (row["n"], row["missing"]) == (4, 1)
    assert row["share_minus_one"] == pytest.approx(0.5) and row["share_zero"] == pytest.approx(0.25)


# --- Spearman ------------------------------------------------------------------------

def _spearman_frame(n=600, seed=5):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, TWO_PI, n)
    df = pd.DataFrame({"month_sin": np.sin(angle), "month_cos": np.cos(angle), "noise_a": rng.normal(size=n),
                       "noise_b": rng.normal(size=n), "dup_a": rng.normal(size=n)})
    df["dup_b"] = 2.0 * df["dup_a"] + rng.normal(scale=0.05, size=n)
    df["constant"] = 1.0
    return df


def test_duplicates_share_a_cluster_and_families_are_never_split_or_dropped(tmp_path):
    X = _spearman_frame()
    res = spearman_clusters(X, protected_columns(feature_groups(X.columns)), threshold=0.9)

    clusters = res["clusters"].set_index("feature")
    assert clusters.loc["dup_a", "cluster"] == clusters.loc["dup_b", "cluster"]
    assert clusters.loc[["dup_a", "dup_b"], "representative"].sum() == 1
    assert clusters.loc["noise_a", "cluster"] != clusters.loc["noise_b", "cluster"]
    assert "month_sin" not in clusters.index
    assert res["constant_columns"] == ["constant"]
    assert {"month_sin", "month_cos"} <= set(res["suggested"]) and "constant" not in res["suggested"]

    path = tmp_path / "suggested.txt"
    path.write_text("# header\n" + TARGET + "\n" + "\n".join(res["suggested"]) + "\n", encoding="utf-8")
    assert load_columns(str(path)) == [TARGET] + res["suggested"]


def test_clustering_a_single_continuous_column_does_not_crash():
    res = spearman_clusters(pd.DataFrame({"only": np.arange(20.0), "sin_x": 0.0}), protected={"sin_x"}, threshold=0.9)
    assert res["linkage"] is None and res["suggested"] == ["only", "sin_x"]


def test_the_spearman_matrix_puts_similar_columns_next_to_each_other():
    X = _spearman_frame().drop(columns=["constant"])
    corr, Z = spearman_matrix(X)
    order = list(corr.columns)
    assert abs(order.index("dup_a") - order.index("dup_b")) == 1
    assert Z is not None and (corr.to_numpy().diagonal() == 1).all()


# --- the stage -----------------------------------------------------------------------

def _make_pipeline(tmp_path):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.project_root = str(tmp_path)
    return pipeline


def _stage_502_frame(years=(2022, 2023, 2024, 2025), per_year=700, seed=0, binary=False):
    """A 502-style frame: the target, sin/cos features, weather in physical units, year and year_month."""
    rng = np.random.default_rng(seed)
    n = per_year * len(years)
    year = np.repeat(list(years), per_year)
    hour = rng.uniform(0, 24, n)
    month = rng.integers(1, 13, n)
    weekday = rng.integers(1, 8, n)
    wind = rng.uniform(0, 360, n)
    calm = rng.random(n) < 0.02
    df = pd.DataFrame({
        "trainStopping": rng.integers(0, 2, n).astype(float),
        "month_sin": np.sin(TWO_PI * (month - 1) / 12), "month_cos": np.cos(TWO_PI * (month - 1) / 12),
        "hour_sin": np.sin(TWO_PI * hour / 24), "hour_cos": np.cos(TWO_PI * hour / 24),
        "day_week_sin": np.sin(TWO_PI * (weekday - 1) / 7), "day_week_cos": np.cos(TWO_PI * (weekday - 1) / 7),
        "Air temperature": rng.normal(2, 9, n), "Wind speed": rng.gamma(4, 1.2, n),
        "Precipitation intensity": np.where(rng.random(n) < 0.9, 0.0, rng.gamma(1.5, 1.0, n)),
        "Snow depth": np.where(rng.random(n) < 0.3, -1.0, rng.uniform(0, 50, n)),
        "wind_dir_sin": np.where(calm, 0.0, np.sin(np.deg2rad(wind))),
        "wind_dir_cos": np.where(calm, 0.0, np.cos(np.deg2rad(wind))),
    })
    df["Dew-point temperature"] = df["Air temperature"] - 3 + rng.normal(scale=0.3, size=n)
    delay = 4 * np.cos(TWO_PI * (hour - 17) / 24) + rng.normal(scale=3, size=n) + 1
    delay = np.round(delay)
    df[TARGET] = (delay > 5).astype(int) if binary else delay
    if binary:
        df = df.rename(columns={TARGET: "trainDelayed"})
    df["year"] = year
    df["departureDate"] = [f"{y}-{m:02d}-15" for y, m in zip(year, month)]
    df["year_month"] = [f"{y}-{m:02d}" for y, m in zip(year, month)]
    return df


def _write_502(root, frame):
    folder = Path(root) / MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "merged_data_selected.parquet"
    frame.to_parquet(path, index=False)
    return str(folder)


def _digest(folder):
    return {p: hashlib.sha256(open(os.path.join(folder, p), "rb").read()).hexdigest() for p in sorted(os.listdir(folder))}


def _run(pipeline, data_dir, target):
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", target):
        return pipeline.explore_training_data(data_dir=data_dir)


ALL_CHARTS = ["01_target_distribution", "02_target_over_time", "03_delay_by_hour", "04_delay_by_weekday_and_hour",
              "05_delay_by_month", "06_delay_by_wind_direction", "07_delay_vs_weather", "08_feature_distributions",
              "09_spearman_heatmap", "10_spearman_dendrogram", "11_association_with_target", "12_association_by_year",
              "13_drift_holdout_vs_development"]


def test_end_to_end_writes_all_thirteen_charts_and_tables_and_fits_no_model(tmp_path):
    data_dir = _write_502(tmp_path, _stage_502_frame())
    before = _digest(data_dir)

    def forbidden(*args, **kwargs):
        raise AssertionError("stage 701 must not fit a model")

    with patch("xgboost.XGBRegressor", forbidden), patch("xgboost.XGBClassifier", forbidden), \
         patch("sklearn.ensemble.RandomForestRegressor", forbidden):
        result = _run(_make_pipeline(tmp_path), data_dir, TARGET)

    assert result["success"] is True, result
    assert result["charts"] == ALL_CHARTS and result["charts_skipped"] == {}
    out = result["output_directory"]
    assert os.path.normpath(out).endswith(os.path.normpath(DATA_EXPLORATION_OUTPUT_FOLDER))
    for stem in ALL_CHARTS:
        for ext in ("png", "pdf"):
            assert os.path.getsize(os.path.join(out, f"{stem}.{ext}")) > 1000, f"{stem}.{ext}"
    for name in ("data_summary.csv", "association_with_target.csv", "association_by_year.csv", "drift_holdout.csv",
                 "spearman_correlation_matrix.csv", "spearman_clusters.csv", "suggested_features.txt", "summary.json"):
        assert os.path.isfile(os.path.join(out, name)), name

    summary = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
    assert summary["model_fitted"] is False and summary["hold_out_year"] == 2025
    assert summary["years"] == [2022, 2023, 2024, 2025]
    assert summary["variables"]["hour"] == "circular" and summary["variables"]["wind_direction"] == "circular"
    assert summary["recovered_from_sin_cos"]["hour"] == ["hour_sin", "hour_cos"]
    assert summary["top_association_with_target"][0]["variable"] == "hour"
    assert ["Air temperature", "Dew-point temperature"] in summary["spearman_clusters_with_several_members"]

    assoc = pd.read_csv(os.path.join(out, "association_with_target.csv")).set_index("variable")
    assert assoc.loc["Precipitation intensity", "eta2"] == assoc.loc["Precipitation intensity", "eta2"], \
        "a zero-inflated variable must still be measured"
    suggested = load_columns(os.path.join(out, "suggested_features.txt"))
    assert suggested[0] == TARGET and {"hour_sin", "hour_cos", "wind_dir_sin", "wind_dir_cos"} <= set(suggested)
    assert len({"Air temperature", "Dew-point temperature"} & set(suggested)) == 1
    assert _digest(data_dir) == before, "the stage must not modify its input"


def test_the_analysis_modules_cannot_fit_a_model():
    for name in ("data_exploration.py", "exploration_plots.py"):
        source = (Path(__file__).resolve().parent.parent / "src" / name).read_text(encoding="utf-8")
        for forbidden in ("xgboost", "lightgbm", "sklearn", "torch", ".fit("):
            assert forbidden not in source, f"{name} mentions {forbidden}"


def test_a_binary_target_gives_rates_and_the_same_charts(tmp_path):
    data_dir = _write_502(tmp_path, _stage_502_frame(binary=True))
    result = _run(_make_pipeline(tmp_path), data_dir, "trainDelayed")
    assert result["success"] is True, result
    assert result["problem_type"] == "classification" and result["charts"] == ALL_CHARTS


def test_a_single_year_skips_the_charts_that_need_two_periods_and_says_so(tmp_path):
    data_dir = _write_502(tmp_path, _stage_502_frame(years=(2024,), per_year=1500))
    result = _run(_make_pipeline(tmp_path), data_dir, TARGET)
    assert result["success"] is True, result
    assert set(result["charts_skipped"]) == {"12_association_by_year", "13_drift_holdout_vs_development"}
    assert result["hold_out_year"] is None and len(result["charts"]) == 11


def test_rows_without_a_target_are_dropped_and_counted(tmp_path):
    frame = _stage_502_frame()
    frame.loc[:49, TARGET] = np.nan                           # cancelled trains have no delay
    result = _run(_make_pipeline(tmp_path), _write_502(tmp_path, frame), TARGET)
    summary = json.load(open(result["summary_path"], encoding="utf-8"))
    assert summary["rows_dropped_missing_target"] == 50 and summary["rows_used"] == len(frame) - 50


def test_a_missing_input_or_an_all_missing_target_is_reported_not_raised(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = _make_pipeline(tmp_path).explore_training_data(data_dir=str(empty))
    assert missing["success"] is False and "select_training_cols" in missing["error"]

    frame = _stage_502_frame(per_year=50)
    frame[TARGET] = np.nan
    result = _run(_make_pipeline(tmp_path / "b"), _write_502(tmp_path / "b", frame), TARGET)
    assert result["success"] is False and "missing" in result["error"]


# --- wiring --------------------------------------------------------------------------

def test_the_switch_is_on_and_runs_after_shap_and_before_the_model_based_stage():
    keys = list(TRAINING_STATE_MACHINE)
    assert TRAINING_STATE_MACHINE["explore_training_data"] is True
    assert keys.index("shap_correlation_analysis") < keys.index("explore_training_data") \
        < keys.index("model_feature_importance_analysis") < keys.index("train_xgboost_with_randomized_search_cv")
    assert DATA_EXPLORATION_OUTPUT_FOLDER.startswith("data/output/701-")


def _dispatcher_pipeline(tmp_path):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.script_dir = pipeline.project_root = str(tmp_path)
    pipeline.output_dir = str(tmp_path / "output")
    pipeline.log_dir = str(tmp_path / "log")
    os.makedirs(pipeline.log_dir, exist_ok=True)
    return pipeline


_SUCCESS = {"success": True, "output_directory": "/fake", "n_rows": 10, "n_variables": 3, "n_charts": 13,
            "suggested_feature_count": 2}


def _machine(**flags):
    base = {"merge_data_files": False, "select_training_cols": False, "split_dataset": False, "balance_classes": False,
            "scale_weather_features": False, "shap_correlation_analysis": False, "explore_training_data": True,
            "model_feature_importance_analysis": False}
    return {**base, **flags}


def test_the_dispatcher_hands_the_stage_the_502_folder(tmp_path):
    with patch.object(TrainingPipeline, "explore_training_data", return_value=_SUCCESS) as stage:
        result = _dispatcher_pipeline(tmp_path).execute_training_pipeline_steps([], state_machine=_machine())
    stage.assert_called_once()
    assert os.path.normpath(stage.call_args.kwargs["data_dir"]) == os.path.normpath(
        os.path.join(str(tmp_path), MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER))
    assert "explore_training_data" in result["steps_executed"]


def test_the_dispatcher_skips_the_stage_when_disabled(tmp_path):
    with patch.object(TrainingPipeline, "explore_training_data") as stage:
        _dispatcher_pipeline(tmp_path).execute_training_pipeline_steps([], state_machine=_machine(explore_training_data=False))
    stage.assert_not_called()


def test_a_failed_stage_stops_the_run_before_the_trainers(tmp_path):
    with patch.object(TrainingPipeline, "explore_training_data", return_value={"success": False, "error": "boom"}), \
         patch.object(TrainingPipeline, "train_fcll_net") as later:
        result = _dispatcher_pipeline(tmp_path).execute_training_pipeline_steps(
            [], state_machine=_machine(train_fcll_net=True))
    assert "boom" in result["errors"]
    later.assert_not_called()
