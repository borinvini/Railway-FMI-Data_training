import inspect
import os

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
