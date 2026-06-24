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
