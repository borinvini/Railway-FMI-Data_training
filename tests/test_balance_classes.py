from unittest.mock import patch


def test_balanced_folder_constant_exists():
    from config.const_training import MERGED_BALANCED_OUTPUT_FOLDER
    assert "504" in MERGED_BALANCED_OUTPUT_FOLDER
    assert "balance" in MERGED_BALANCED_OUTPUT_FOLDER


def test_balance_classes_in_state_machine():
    from config.const_training import TRAINING_STATE_MACHINE
    assert "balance_classes" in TRAINING_STATE_MACHINE
    assert isinstance(TRAINING_STATE_MACHINE["balance_classes"], bool)


def test_balance_classes_after_split_dataset_in_state_machine():
    from config.const_training import TRAINING_STATE_MACHINE
    keys = list(TRAINING_STATE_MACHINE.keys())
    assert keys.index("balance_classes") == keys.index("split_dataset") + 1


import numpy as np
import pandas as pd
import os

from src.training_pipeline import TrainingPipeline
from config.const_preprocessing import TRAIN_DELAY_MINUTES


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


def _make_imbalanced_df(n_punctual=300, n_delayed=100, seed=42):
    """3:1 imbalanced DataFrame with differenceInMinutes, trainDelayed, and two numeric features."""
    rng = np.random.default_rng(seed)
    punctual = rng.uniform(-4, 5, n_punctual)
    delayed = rng.uniform(6, 60, n_delayed)
    diff = np.concatenate([punctual, delayed])
    feat_a = rng.normal(0, 1, n_punctual + n_delayed)
    feat_b = rng.normal(5, 2, n_punctual + n_delayed)
    return pd.DataFrame({
        "differenceInMinutes": diff,
        "trainDelayed": diff > TRAIN_DELAY_MINUTES,
        "feature_a": feat_a,
        "feature_b": feat_b,
    })


def _make_balanced_df(n=200, seed=42):
    """50/50 split — minority share will exceed IMBALANCE_THRESHOLD (30%)."""
    rng = np.random.default_rng(seed)
    diff = np.concatenate([rng.uniform(-4, 5, n // 2), rng.uniform(6, 60, n // 2)])
    return pd.DataFrame({
        "differenceInMinutes": diff,
        "trainDelayed": diff > TRAIN_DELAY_MINUTES,
        "feature_a": rng.normal(0, 1, n),
    })


def _write_train_test(tmp_path, train_df, test_df, subdir="split"):
    """Write train and test parquet files into tmp_path/subdir, return the dir path."""
    d = tmp_path / subdir
    d.mkdir(parents=True, exist_ok=True)
    train_path = d / "merged_data_split_train.parquet"
    test_path = d / "merged_data_split_test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    return str(d)


def _make_test_df(n=50, seed=99):
    rng = np.random.default_rng(seed)
    diff = rng.uniform(-4, 60, n)
    return pd.DataFrame({
        "differenceInMinutes": diff,
        "trainDelayed": diff > TRAIN_DELAY_MINUTES,
        "feature_a": rng.normal(0, 1, n),
    })


def test_none_data_dir_returns_failure(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    result = pipeline.balance_classes(data_dir=None)
    assert result["success"] is False
    assert "error" in result


def test_missing_train_file_returns_failure(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    d = tmp_path / "empty_split"
    d.mkdir()
    result = pipeline.balance_classes(data_dir=str(d))
    assert result["success"] is False
    assert "error" in result


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "not_a_real_target")
def test_misconfigured_target_returns_failure(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df()
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["success"] is False
    assert "not recognized" in result["error"]


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_skips_when_already_balanced(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_balanced_df(n=200)
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["success"] is True
    assert result["skipped"] is True
    assert result["resampling_method"] == "NONE"


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_smote_tomek_increases_minority_count(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df(n_punctual=300, n_delayed=100)
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["success"] is True
    assert result["skipped"] is False
    assert result["resampling_method"] == "SMOTE_TOMEK"
    assert result["minority_share_after"] > result["minority_share_before"]


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_rows_before_and_after_in_result(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df()
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["rows_before"] == len(train_df)
    assert result["rows_after"] > 0


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_non_numeric_columns_are_dropped(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df()
    train_df["causes"] = "weather"
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["success"] is True
    assert "causes" in result["dropped_non_numeric_cols"]


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_nan_rows_dropped_before_resampling(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df()
    train_df.loc[train_df.index[:10], "feature_a"] = np.nan
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["success"] is True
    assert result["skipped"] is False


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_test_file_is_copied_to_output(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df()
    test_df = _make_test_df(n=80)
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["success"] is True
    test_out = pd.read_parquet(result["test_output_path"])
    assert len(test_out) == len(test_df)


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_result_contains_required_keys(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df()
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    for key in ("success", "rows_before", "rows_after",
                 "minority_share_before", "minority_share_after",
                 "resampling_method", "skipped", "dropped_non_numeric_cols",
                 "dropped_counterpart_col",
                 "train_output_path", "test_output_path"):
        assert key in result, f"Missing key: {key}"


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_counterpart_dropped_on_skip_already_balanced(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_balanced_df(n=200)
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["dropped_counterpart_col"] == "differenceInMinutes"
    saved = pd.read_parquet(result["train_output_path"])
    assert "differenceInMinutes" not in saved.columns
    assert "trainDelayed" in saved.columns


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_counterpart_dropped_on_skip_missing_target(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df().drop(columns=["trainDelayed"])
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["skipped"] is True
    assert result["dropped_counterpart_col"] == "differenceInMinutes"
    saved = pd.read_parquet(result["train_output_path"])
    assert "differenceInMinutes" not in saved.columns


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_counterpart_dropped_on_resample(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df(n_punctual=300, n_delayed=100)
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["dropped_counterpart_col"] == "differenceInMinutes"
    saved = pd.read_parquet(result["train_output_path"])
    assert "differenceInMinutes" not in saved.columns


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "differenceInMinutes")
def test_counterpart_dropped_for_regression_target(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df(n_punctual=300, n_delayed=100)
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["dropped_counterpart_col"] == "trainDelayed"
    saved = pd.read_parquet(result["train_output_path"])
    assert "trainDelayed" not in saved.columns
    assert "differenceInMinutes" in saved.columns


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_result_has_no_data_key(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df()
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert "data" not in result, "balance_classes must not return an in-memory DataFrame"


# ---------------------------------------------------------------------------
# State machine tests: balance_classes runs after split_dataset
# ---------------------------------------------------------------------------

_SPLIT_SUCCESS = {
    "success": True,
    "processed_files": 1,
    "total_train_rows": 320,
    "total_test_rows": 80,
}

_BALANCE_SUCCESS = {
    "success": True,
    "rows_before": 320,
    "rows_after": 380,
    "minority_share_before": 25.0,
    "minority_share_after": 48.0,
    "resampling_method": "SMOTE_TOMEK",
    "skipped": False,
    "dropped_non_numeric_cols": [],
    "train_output_path": "/fake/train.parquet",
    "test_output_path": "/fake/test.parquet",
}


def _base_state_machine(**overrides):
    sm = {
        "merge_data_files": False,
        "filter_delay_outliers": False,
        "select_training_cols": False,
        "split_dataset": True,
        "balance_classes": True,
        "scale_weather_features": False,
        "numeric_correlation_analysis": False,
        "data_distribution_analysis": False,
        "target_feature_analysis": False,
        "train_xgboost_with_randomized_search_cv": False,
    }
    sm.update(overrides)
    return sm


@patch.object(TrainingPipeline, "balance_classes")
@patch.object(TrainingPipeline, "split_dataset", return_value=_SPLIT_SUCCESS)
def test_state_machine_calls_balance_classes_after_split(mock_split, mock_balance, tmp_path):
    from config.const_training import SPLIT_DATASET_OUTPUT_FOLDER
    pipeline = _make_pipeline(tmp_path)
    mock_balance.return_value = _BALANCE_SUCCESS

    result = pipeline.execute_training_pipeline_steps([], state_machine=_base_state_machine())

    mock_balance.assert_called_once()
    _, kwargs = mock_balance.call_args
    assert kwargs.get("data_dir") is not None
    assert SPLIT_DATASET_OUTPUT_FOLDER in kwargs["data_dir"]
    assert "balance_classes" in result.get("steps_executed", [])


@patch.object(TrainingPipeline, "balance_classes")
@patch.object(TrainingPipeline, "split_dataset", return_value=_SPLIT_SUCCESS)
def test_state_machine_skips_balance_classes_when_disabled(mock_split, mock_balance, tmp_path):
    pipeline = _make_pipeline(tmp_path)

    pipeline.execute_training_pipeline_steps(
        [], state_machine=_base_state_machine(balance_classes=False)
    )

    mock_balance.assert_not_called()


@patch.object(TrainingPipeline, "balance_classes")
@patch.object(TrainingPipeline, "split_dataset")
def test_split_dataset_no_longer_routes_based_on_balance(mock_split, mock_balance, tmp_path):
    """split_dataset routing must NOT depend on balance_classes — balance runs after split."""
    pipeline = _make_pipeline(tmp_path)
    mock_split.return_value = _SPLIT_SUCCESS
    mock_balance.return_value = {"success": True, "rows_before": 320, "rows_after": 380, "minority_share_before": 25.0, "minority_share_after": 48.0, "resampling_method": "SMOTE_TOMEK", "skipped": False, "dropped_non_numeric_cols": [], "train_output_path": "/fake/train.parquet", "test_output_path": "/fake/test.parquet"}

    pipeline.execute_training_pipeline_steps(
        [], state_machine=_base_state_machine(balance_classes=True, split_dataset=True)
    )

    mock_split.assert_called_once()
    _, kwargs = mock_split.call_args
    data_dir = kwargs.get("data_dir")
    # data_dir must NOT point to balance_classes folder
    assert data_dir is None or "balance" not in str(data_dir)
