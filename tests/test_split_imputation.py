"""Median imputation must be fitted on the training split only.

handle_missing_values filled every NaN with the median of its own month, which
includes days that split_dataset later sends to test or the hold-out year. The
imputation now runs inside the split: medians come from the train frame and are
applied to val, test and hold-out. It touches 0.01% of weather cells at OL, so
this is about the principle, not the score.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from config.const_preprocessing import GROUP_KEY_COLUMN
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


def test_write_splits_fills_val_and_test_with_the_train_median(tmp_path):
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
    for name in ("train", "val", "test"):
        assert split["frames"][name]["Air temperature"].isna().sum() == 0
        on_disk = pd.read_parquet(split["files"][name]["path"])
        assert on_disk["Air temperature"].isna().sum() == 0


def test_a_test_nan_gets_the_train_median_not_the_test_median(tmp_path):
    df = _days(n_days=30)
    df["Air temperature"] = np.where(df[GROUP_KEY_COLUMN] < "2023-01-20", 1.0, 100.0)
    pipeline = _pipeline(tmp_path)
    train, val, test = pipeline.split_by_group(df, GROUP_KEY_COLUMN)
    # one NaN in the last split's days that carry the 100.0 level
    victim = df.index[df[GROUP_KEY_COLUMN] >= "2023-01-25"][0]
    df.loc[victim, "Air temperature"] = np.nan
    split = pipeline.write_splits(df, output_dir=str(tmp_path / "out"), identifier="t")
    train_median = split["medians"]["Air temperature"]
    for name in ("val", "test"):
        part = split["frames"][name]
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
