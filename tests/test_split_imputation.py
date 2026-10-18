"""Median imputation must be fitted on the training split only.

handle_missing_values filled every NaN with the median of its own month, which
includes days that split_dataset later sends to test or the hold-out year. The
imputation now runs inside the split: medians come from the train frame and are
applied to val, test and hold-out. It touches 0.01% of weather cells at OL, so
this is about the principle, not the score.

Weather columns are filled with the train median of their own monthly file
("year_month", stamped by merge_data_files), then of their calendar month over all
train years (the hold-out year has no train rows of its own), then the global train
median. One all-year median is wrong in winter: at OL the all-year Air temperature
median is +2.9 C against -6.3 C in January, and Snow depth is 0 against 32-55 cm
from January to April. The key is dropped after the split; the fitted imputer is
saved at the split and copied beside the scaler, which must run after it.
"""
import json
import os

import joblib

import numpy as np
import pandas as pd
import pytest

from config.const_preprocessing import GROUP_KEY_COLUMN
from config.const_training import MONTH_KEY_COLUMN
from src.training_pipeline import TrainingPipeline


def _pipeline(tmp_path):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.project_root = str(tmp_path)
    pipeline.log_dir = str(tmp_path / "log")
    os.makedirs(pipeline.log_dir, exist_ok=True)
    return pipeline


def test_fit_train_medians_skips_target_and_keys():
    train = pd.DataFrame({
        "Air temperature": [1.0, 3.0, np.nan],
        "trainDelayed": [0, 1, 1],
        "differenceInMinutes": [1.0, np.nan, 9.0],
        GROUP_KEY_COLUMN: ["a", "b", "c"],
    })
    medians = TrainingPipeline.fit_train_medians(train)
    assert medians == {"Air temperature": 2.0}


def test_a_column_that_is_all_nan_in_train_raises():
    train = pd.DataFrame({"Air temperature": [np.nan, np.nan], "Wind speed": [1.0, 2.0]})
    with pytest.raises(ValueError) as err:
        TrainingPipeline.fit_train_medians(train)
    assert "Air temperature" in str(err.value)


def test_apply_medians_fills_nan_and_leaves_the_rest():
    frame = pd.DataFrame({"a": [np.nan, 5.0], "trainDelayed": [0, 1]})
    out = TrainingPipeline.apply_medians(frame, {"a": 2.0})
    assert out["a"].tolist() == [2.0, 5.0]
    assert out["trainDelayed"].tolist() == [0, 1]


def _days(n_days=20, rows_per_day=5):
    rng = np.random.default_rng(0)
    days = np.repeat([f"2023-01-{d + 1:02d}" for d in range(n_days)], rows_per_day)
    return pd.DataFrame({
        GROUP_KEY_COLUMN: days,
        "Air temperature": rng.normal(0, 5, len(days)),
        "trainDelayed": rng.integers(0, 2, len(days)),
        "differenceInMinutes": rng.normal(2, 3, len(days)),
    })


def test_write_splits_fills_test_with_the_train_median(tmp_path):
    df = _days()
    df.loc[df.index[:], "Air temperature"] = df["Air temperature"]
    pipeline = _pipeline(tmp_path)
    df.loc[0, "Air temperature"] = np.nan
    df.loc[len(df) - 1, "Air temperature"] = np.nan
    df.loc[len(df) // 2, "Air temperature"] = np.nan
    split = pipeline.write_splits(df, output_dir=str(tmp_path / "out"), identifier="t")

    medians = split["medians"]
    train_clean = split["frames"]["train"]["Air temperature"]
    assert medians["Air temperature"] == pytest.approx(train_clean.median())
    for name in ("train", "test"):
        assert split["frames"][name]["Air temperature"].isna().sum() == 0
        on_disk = pd.read_parquet(split["files"][name]["path"])
        assert on_disk["Air temperature"].isna().sum() == 0


def test_a_test_nan_gets_the_train_median_not_the_test_median(tmp_path):
    df = _days(n_days=30)
    df["Air temperature"] = np.where(df[GROUP_KEY_COLUMN] < "2023-01-20", 1.0, 100.0)
    pipeline = _pipeline(tmp_path)
    train, test = pipeline.split_by_group(df, GROUP_KEY_COLUMN)
    # one NaN in the last split's days that carry the 100.0 level
    victim = df.index[df[GROUP_KEY_COLUMN] >= "2023-01-25"][0]
    df.loc[victim, "Air temperature"] = np.nan
    split = pipeline.write_splits(df, output_dir=str(tmp_path / "out"), identifier="t")
    train_median = split["medians"]["Air temperature"]
    part = split["frames"]["test"]
    filled = part.loc[part.index == victim, "Air temperature"]
    if len(filled):
        assert filled.iloc[0] == train_median


def test_target_columns_are_never_imputed(tmp_path):
    df = _days()
    df.loc[3, "trainDelayed"] = np.nan
    df.loc[4, "differenceInMinutes"] = np.nan
    split = _pipeline(tmp_path).write_splits(df, output_dir=str(tmp_path / "out"), identifier="t")
    assert "trainDelayed" not in split["medians"]
    assert "differenceInMinutes" not in split["medians"]
    merged = pd.concat(split["frames"].values())
    assert merged["trainDelayed"].isna().sum() == 1
    assert merged["differenceInMinutes"].isna().sum() == 1


# --- per monthly file ----------------------------------------------------------

LEVELS = {"2022-01": -6.0, "2023-01": -12.0, "2022-07": 17.0}


def _monthly_days(levels=LEVELS, days_per_month=12, rows_per_day=4, seed=0):
    """One row block per day; every monthly file has its own temperature level."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, level in levels.items():
        for d in range(days_per_month):
            for _ in range(rows_per_day):
                rows.append({
                    GROUP_KEY_COLUMN: f"{key}-{d + 1:02d}",
                    MONTH_KEY_COLUMN: key,
                    "Air temperature": level + rng.normal(0, 0.2),
                    "Snow depth (24h max)": 40.0 if key.endswith("-01") else 0.0,
                    "trainStopping": 1,
                    "trainDelayed": int(rng.integers(0, 2)),
                })
    return pd.DataFrame(rows)


def test_a_nan_gets_the_median_of_its_own_monthly_file(tmp_path):
    df = _monthly_days()
    nan_rows = list(df.index[df[MONTH_KEY_COLUMN] == "2023-01"][::5])
    df.loc[nan_rows, "Air temperature"] = np.nan
    df.loc[nan_rows, "Snow depth (24h max)"] = np.nan
    split = _pipeline(tmp_path).write_splits(df, output_dir=str(tmp_path / "out"), identifier="t")

    by_key = split["month_medians"]["year_month"]
    own = by_key["2023-01"]
    assert own["Air temperature"] == pytest.approx(-12.0, abs=0.5)
    # not the neighbouring January, and not the all-year median
    assert abs(own["Air temperature"] - by_key["2022-01"]["Air temperature"]) > 3
    assert abs(own["Air temperature"] - split["medians"]["Air temperature"]) > 3
    seen = 0
    for part in split["frames"].values():
        for idx in part.index.intersection(nan_rows):
            assert part.loc[idx, "Air temperature"] == own["Air temperature"]
            assert part.loc[idx, "Snow depth (24h max)"] == own["Snow depth (24h max)"]
            seen += 1
    assert seen == len(nan_rows)


def test_a_year_month_without_train_rows_uses_the_calendar_month_median():
    frame = pd.DataFrame({MONTH_KEY_COLUMN: ["2024-01"], "Air temperature": [np.nan]})
    month_medians = {"year_month": {"2023-01": {"Air temperature": -12.0}},
                     "calendar_month": {1: {"Air temperature": -9.0}}}
    out = TrainingPipeline.apply_medians(frame, {"Air temperature": 5.0}, month_medians)
    assert out["Air temperature"].tolist() == [-9.0]


def test_a_calendar_month_without_a_train_value_uses_the_global_median():
    frame = pd.DataFrame({MONTH_KEY_COLUMN: ["2024-03"], "Air temperature": [np.nan]})
    month_medians = {"year_month": {"2023-01": {"Air temperature": -12.0}},
                     "calendar_month": {1: {"Air temperature": -9.0}}}
    out = TrainingPipeline.apply_medians(frame, {"Air temperature": 5.0}, month_medians)
    assert out["Air temperature"].tolist() == [5.0]


def test_month_medians_never_see_val_or_test_values(tmp_path):
    df = _monthly_days(levels={"2023-01": -12.0}, days_per_month=30)
    late = df[GROUP_KEY_COLUMN] >= "2023-01-25"
    df.loc[late, "Air temperature"] = 100.0
    victim = df.index[late][0]
    df.loc[victim, "Air temperature"] = np.nan
    split = _pipeline(tmp_path).write_splits(df, output_dir=str(tmp_path / "out"), identifier="t")
    train_index = split["frames"]["train"].index
    expected = df.loc[train_index, "Air temperature"].median()
    assert split["month_medians"]["year_month"]["2023-01"]["Air temperature"] == pytest.approx(expected)
    filled = next(p.loc[victim, "Air temperature"] for p in split["frames"].values() if victim in p.index)
    assert filled == pytest.approx(expected)


def test_without_the_month_key_it_is_the_global_imputation(tmp_path):
    df = _days()
    df.loc[[0, 7, 30], "Air temperature"] = np.nan
    split = _pipeline(tmp_path).write_splits(df, output_dir=str(tmp_path / "out"), identifier="t")
    assert split["month_medians"] == {}
    assert sum(p["Air temperature"].isna().sum() for p in split["frames"].values()) == 0


def test_the_month_key_never_reaches_the_written_splits(tmp_path):
    split = _pipeline(tmp_path).write_splits(
        _monthly_days(), output_dir=str(tmp_path / "out"), identifier="t")
    for name in ("train", "test"):
        assert MONTH_KEY_COLUMN not in split["frames"][name].columns
        assert MONTH_KEY_COLUMN not in pd.read_parquet(split["files"][name]["path"]).columns
    assert MONTH_KEY_COLUMN not in split["medians"]


def test_the_imputer_is_saved_beside_the_scaler(tmp_path):
    from config.const_training import (
        MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER, SPLIT_DATASET_OUTPUT_FOLDER)
    pipeline = _pipeline(tmp_path)
    split_dir = tmp_path / SPLIT_DATASET_OUTPUT_FOLDER
    split_dir.mkdir(parents=True)
    medians = {"Air temperature": 3.0}
    month_medians = {"year_month": {"2023-01": {"Air temperature": -12.0}},
                     "calendar_month": {1: {"Air temperature": -9.0}}}
    report = TrainingPipeline.imputation_report(medians, month_medians)
    (split_dir / "imputation_medians.json").write_text(json.dumps(report), encoding="utf-8")
    train = pd.DataFrame({"Air temperature": np.linspace(-10, 20, 50)})
    train.to_parquet(split_dir / "merged_data_x_train.parquet", index=False)
    train.iloc[:10].to_parquet(split_dir / "merged_data_x_test.parquet", index=False)

    result = pipeline.scale_weather_features(data_dir=str(split_dir))

    assert result["success"] is True
    scaled = tmp_path / MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
    artifact = joblib.load(scaled / "weather_scaler.joblib")
    assert artifact["imputer"]["per_year_month"]["2023-01"] == {"Air temperature": -12.0}
    assert artifact["imputer"]["per_calendar_month"]["1"] == {"Air temperature": -9.0}
    assert artifact["imputer"]["global"] == medians
    assert json.loads((scaled / "imputation_medians.json").read_text(encoding="utf-8")) == report
