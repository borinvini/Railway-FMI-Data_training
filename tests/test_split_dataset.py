import inspect
import os
from unittest.mock import patch

import pytest
import pandas as pd

from src.training_pipeline import TrainingPipeline


def _make_pipeline(tmp_path):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.script_dir = str(tmp_path)
    pipeline.project_root = str(tmp_path)
    pipeline.output_dir = str(tmp_path / "output")
    pipeline.preprocessed_dir = str(tmp_path / "preprocessed")
    pipeline.randomized_search_dir = str(tmp_path / "rs")
    pipeline.random_forest_dir = str(tmp_path / "rf")
    pipeline.important_features_randomized_search_dir = str(tmp_path / "if")
    pipeline.xgboost_rs_dir = str(tmp_path / "xgb_rs")
    pipeline.regularized_regression_dir = str(tmp_path / "rr")
    pipeline.log_dir = str(tmp_path / "log")
    pipeline.DATA_FILE_PREFIX_FOR_TRAINING = "preprocessed_data_"
    os.makedirs(pipeline.log_dir, exist_ok=True)
    return pipeline


def test_random_state_constant_exists():
    from config.const_training import RANDOM_STATE
    assert isinstance(RANDOM_STATE, int)
    assert RANDOM_STATE == 42


def test_split_dataset_signature_has_no_csv_files():
    sig = inspect.signature(TrainingPipeline.split_dataset)
    assert "csv_files" not in sig.parameters


def test_split_dataset_random_state_default_uses_constant():
    from config.const_training import RANDOM_STATE
    sig = inspect.signature(TrainingPipeline.split_dataset)
    assert sig.parameters["random_state"].default == RANDOM_STATE


def _make_split_dirs(tmp_path):
    src = tmp_path / "data/output/502-select_training_cols"
    src.mkdir(parents=True)
    out = tmp_path / "data/output/503-split_dataset"
    out.mkdir(parents=True)
    return src, out


def test_classification_target_detected_via_config(tmp_path):
    """Target in CLASSIFICATION_PROBLEM must give is_classification=True."""
    from config.const_preprocessing import CLASSIFICATION_PROBLEM
    pipeline = _make_pipeline(tmp_path)
    src, out = _make_split_dirs(tmp_path)

    # 'trainDelayed' is in CLASSIFICATION_PROBLEM
    target = CLASSIFICATION_PROBLEM[0]
    df = pd.DataFrame({
        target: ([True] * 80) + ([False] * 20),
        "feature_a": range(100),
    })
    df.to_parquet(src / "merged_data.parquet", index=False)

    result = pipeline.split_dataset(data_dir=str(src), output_dir=str(out), stratify_column=target)

    assert result["success"] is True
    assert result["split_details"][0]["is_classification"] is True


def test_regression_target_detected_via_config(tmp_path):
    """Target not in CLASSIFICATION_PROBLEM must give is_classification=False."""
    from config.const_preprocessing import REGRESSION_PROBLEM
    pipeline = _make_pipeline(tmp_path)
    src, out = _make_split_dirs(tmp_path)

    # 'differenceInMinutes' is in REGRESSION_PROBLEM, not CLASSIFICATION_PROBLEM
    target = REGRESSION_PROBLEM[0]
    df = pd.DataFrame({
        target: list(range(100)),
        "feature_a": range(100),
    })
    df.to_parquet(src / "merged_data.parquet", index=False)

    result = pipeline.split_dataset(data_dir=str(src), output_dir=str(out), stratify_column=target)

    assert result["success"] is True
    assert result["split_details"][0]["is_classification"] is False


def test_summary_saved_to_output_dir_not_source_dir(tmp_path):
    """split_summary.txt must appear in 503/ (output_dir), not 502/ (source dir)."""
    from config.const_preprocessing import CLASSIFICATION_PROBLEM
    pipeline = _make_pipeline(tmp_path)
    src, out = _make_split_dirs(tmp_path)

    target = CLASSIFICATION_PROBLEM[0]
    df = pd.DataFrame({
        target: ([True] * 80) + ([False] * 20),
        "feature_a": range(100),
    })
    df.to_parquet(src / "merged_data.parquet", index=False)

    result = pipeline.split_dataset(data_dir=str(src), output_dir=str(out), stratify_column=target)

    assert result["success"] is True
    assert (out / "split_summary.txt").exists(), "Summary must be in output dir (503/)"
    assert not (src / "split_summary.txt").exists(), "Summary must NOT be in source dir (502/)"


def test_holdout_carves_out_max_year_and_drops_year_column(tmp_path):
    """With HOLDOUT_LAST_YEAR on and a 'year' column present, the max year must be
    carved into a separate *_holdout.parquet, train/test must only contain earlier
    years, there must be no row overlap between the three sets, and 'year' must be
    dropped from all three saved files."""
    from config.const_preprocessing import CLASSIFICATION_PROBLEM
    pipeline = _make_pipeline(tmp_path)
    src, out = _make_split_dirs(tmp_path)

    target = CLASSIFICATION_PROBLEM[0]
    years = ([2023] * 40) + ([2024] * 40) + ([2025] * 20)
    df = pd.DataFrame({
        target: [i % 2 == 0 for i in range(100)],
        "feature_a": range(100),
        "year": years,
    })
    df.to_parquet(src / "merged_data.parquet", index=False)

    with patch("src.training_pipeline.HOLDOUT_LAST_YEAR", True):
        result = pipeline.split_dataset(data_dir=str(src), output_dir=str(out), stratify_column=target)

    assert result["success"] is True
    assert result["holdout_year"] == 2025
    assert result["total_holdout_rows"] == 20

    train_df = pd.read_parquet(out / "merged_data_train.parquet")
    test_df = pd.read_parquet(out / "merged_data_test.parquet")
    holdout_df = pd.read_parquet(out / "merged_data_holdout.parquet")

    # 'year' must never reach the saved files — it must not become a training feature
    assert "year" not in train_df.columns
    assert "year" not in test_df.columns
    assert "year" not in holdout_df.columns

    # Hold-out is exactly the 20 rows from 2025; train/test only ever came from 2023-2024
    assert len(holdout_df) == 20
    assert len(train_df) + len(test_df) == 80

    # No row overlap between any of the three sets (feature_a is a unique row id here)
    train_ids = set(train_df["feature_a"])
    test_ids = set(test_df["feature_a"])
    holdout_ids = set(holdout_df["feature_a"])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids.isdisjoint(holdout_ids)
    assert test_ids.isdisjoint(holdout_ids)
    assert holdout_ids == set(range(80, 100))  # rows tagged year=2025


def test_holdout_disabled_falls_back_to_two_way_split(tmp_path):
    """With HOLDOUT_LAST_YEAR off, behavior must be the original 2-way split even
    when a 'year' column is present — no *_holdout.parquet should be written."""
    from config.const_preprocessing import CLASSIFICATION_PROBLEM
    pipeline = _make_pipeline(tmp_path)
    src, out = _make_split_dirs(tmp_path)

    target = CLASSIFICATION_PROBLEM[0]
    years = ([2023] * 50) + ([2024] * 50)
    df = pd.DataFrame({
        target: [i % 2 == 0 for i in range(100)],
        "feature_a": range(100),
        "year": years,
    })
    df.to_parquet(src / "merged_data.parquet", index=False)

    with patch("src.training_pipeline.HOLDOUT_LAST_YEAR", False):
        result = pipeline.split_dataset(data_dir=str(src), output_dir=str(out), stratify_column=target)

    assert result["success"] is True
    assert result["holdout_year"] is None
    assert result["total_holdout_rows"] == 0
    assert not (out / "merged_data_holdout.parquet").exists()
    assert result["total_train_rows"] + result["total_test_rows"] == 100


def test_holdout_missing_year_column_falls_back_gracefully(tmp_path):
    """With HOLDOUT_LAST_YEAR on but no 'year' column in the source data, split_dataset
    must fall back to a 2-way split instead of erroring."""
    from config.const_preprocessing import CLASSIFICATION_PROBLEM
    pipeline = _make_pipeline(tmp_path)
    src, out = _make_split_dirs(tmp_path)

    target = CLASSIFICATION_PROBLEM[0]
    df = pd.DataFrame({
        target: ([True] * 80) + ([False] * 20),
        "feature_a": range(100),
    })
    df.to_parquet(src / "merged_data.parquet", index=False)

    with patch("src.training_pipeline.HOLDOUT_LAST_YEAR", True):
        result = pipeline.split_dataset(data_dir=str(src), output_dir=str(out), stratify_column=target)

    assert result["success"] is True
    assert result["total_holdout_rows"] == 0
    assert not (out / "merged_data_holdout.parquet").exists()


# --- Day-grouped split ------------------------------------------------------

from config.const_preprocessing import GROUP_KEY_COLUMN


def test_no_operating_day_appears_in_two_splits():
    """The whole point. One day's trains share weather, disruptions and
    knock-on delays; splitting them across train and test leaks all three."""
    import numpy as np

    days = [f"2018-01-{d:02d}" for d in range(1, 29)]
    df = pd.DataFrame({
        "trainDelayed": np.resize([0, 1], 28 * 10),
        "Air temperature": np.arange(280, dtype=float),
        GROUP_KEY_COLUMN: np.repeat(days, 10),
    })
    pipeline = TrainingPipeline()
    train, test = pipeline.split_by_group(df, group_column=GROUP_KEY_COLUMN)

    train_days = set(train[GROUP_KEY_COLUMN])
    test_days = set(test[GROUP_KEY_COLUMN])
    assert train_days & test_days == set()
    assert train_days | test_days == set(days)


def test_the_split_is_deterministic():
    """Two runs, same seed, same days — or no result is reproducible."""
    import numpy as np

    days = [f"2018-01-{d:02d}" for d in range(1, 29)]
    df = pd.DataFrame({
        "trainDelayed": np.resize([0, 1], 280),
        GROUP_KEY_COLUMN: np.repeat(days, 10),
    })
    pipeline = TrainingPipeline()
    first = pipeline.split_by_group(df, group_column=GROUP_KEY_COLUMN)
    second = pipeline.split_by_group(df, group_column=GROUP_KEY_COLUMN)
    for a, b in zip(first, second):
        assert set(a[GROUP_KEY_COLUMN]) == set(b[GROUP_KEY_COLUMN])


def test_the_split_fractions_are_80_20_and_there_is_no_validation_split():
    """The old 55/20/25 wrote a validation file that no tabular trainer read, so
    20% of the days were never used. Validation is now day-grouped CV in train."""
    import numpy as np
    import config.const_training as const
    from config.const_training import TRAIN_FRACTION, TEST_FRACTION

    assert not hasattr(const, "VAL_FRACTION")
    assert TRAIN_FRACTION + TEST_FRACTION == 1.0
    days = [f"2018-{m:02d}-{d:02d}" for m in range(1, 5) for d in range(1, 26)]
    df = pd.DataFrame({
        "trainDelayed": np.resize([0, 1], len(days) * 4),
        GROUP_KEY_COLUMN: np.repeat(days, 4),
    })
    pipeline = TrainingPipeline()
    train, test = pipeline.split_by_group(df, group_column=GROUP_KEY_COLUMN)
    n = len(days)
    assert abs(len(set(train[GROUP_KEY_COLUMN])) / n - TRAIN_FRACTION) < 0.05
    assert abs(len(set(test[GROUP_KEY_COLUMN])) / n - TEST_FRACTION) < 0.05


def test_the_group_key_is_dropped_from_written_output(tmp_path):
    """Leaving the date in the feature matrix would let a tree memorise
    specific bad-weather days."""
    import numpy as np

    days = [f"2018-01-{d:02d}" for d in range(1, 29)]
    df = pd.DataFrame({
        "trainDelayed": np.resize([0, 1], 280),
        "Air temperature": np.arange(280, dtype=float),
        GROUP_KEY_COLUMN: np.repeat(days, 10),
    })
    pipeline = TrainingPipeline()
    result = pipeline.write_splits(df, output_dir=str(tmp_path), identifier="test")
    assert result["success"]
    written = pd.read_parquet(tmp_path / "merged_data_test_train.parquet")
    assert GROUP_KEY_COLUMN not in written.columns


def test_split_dataset_writes_two_grouped_files_end_to_end(tmp_path):
    """Exercises the whole stage on data that carries the grouping key — the
    other split_dataset tests predate it and still cover the fallback path."""
    import numpy as np
    from config.const_preprocessing import CLASSIFICATION_PROBLEM

    pipeline = _make_pipeline(tmp_path)
    src, out = _make_split_dirs(tmp_path)

    target = CLASSIFICATION_PROBLEM[0]
    days = [f"2018-01-{d:02d}" for d in range(1, 29)]
    pd.DataFrame({
        target: np.resize([True, False], 280),
        "feature_a": np.arange(280, dtype=float),
        GROUP_KEY_COLUMN: np.repeat(days, 10),
    }).to_parquet(src / "merged_data.parquet", index=False)

    with patch("src.training_pipeline.HOLDOUT_LAST_YEAR", False):
        result = pipeline.split_dataset(data_dir=str(src), output_dir=str(out),
                                        stratify_column=target)

    assert result["success"] is True
    detail = result["split_details"][0]
    assert detail["grouped_by"] == GROUP_KEY_COLUMN
    assert detail["train_rows"] + detail["test_rows"] == 280
    assert not (out / "merged_data_val.parquet").exists()

    frames = {n: pd.read_parquet(out / f"merged_data_{n}.parquet")
              for n in ("train", "test")}
    for frame in frames.values():
        assert GROUP_KEY_COLUMN not in frame.columns


def _training_ready_files():
    import glob
    return sorted(glob.glob("data/output/101-preprocessed_training_ready/*.parquet"))


def test_the_base_rate_survives_the_split():
    """Measured 2026-08-20: 27,387 delayed of 116,733 OL rows = 23.461%.

    Day-grouped assignment is not stratified, so this is the check that it did
    not accidentally concentrate bad-weather days in one split."""
    import pyarrow.parquet as pq

    files = _training_ready_files()
    if not files:
        pytest.skip("requires the local training-ready parquet")
    if GROUP_KEY_COLUMN not in pq.read_schema(files[0]).names:
        pytest.skip(
            f"the local 101-* parquet predates {GROUP_KEY_COLUMN}; "
            "re-run preprocessing to regenerate it"
        )

    df = pd.concat(
        [pd.read_parquet(f, columns=["trainDelayed", GROUP_KEY_COLUMN]) for f in files],
        ignore_index=True,
    )
    pipeline = TrainingPipeline()
    for part in pipeline.split_by_group(df, group_column=GROUP_KEY_COLUMN):
        # (train, test)
        rate = part["trainDelayed"].mean()
        assert 0.21 < rate < 0.26, f"base rate {rate:.3%} drifted from 23.5%"
