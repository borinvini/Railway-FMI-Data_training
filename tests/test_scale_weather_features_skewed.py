import os
import numpy as np
import pandas as pd
import pytest
import joblib

from src.training_pipeline import TrainingPipeline
from config.const_preprocessing import SKEWED_WEATHER_FEATURES
from config.const_training import MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER


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


def _make_zero_inflated_df(n=200, seed=0):
    """~95% zeros with occasional large rain events, mimicking Precipitation amount."""
    rng = np.random.default_rng(seed)
    precip = np.zeros(n)
    event_idx = rng.choice(n, size=max(n // 20, 1), replace=False)
    precip[event_idx] = rng.uniform(5, 50, size=len(event_idx))
    return pd.DataFrame({
        "Precipitation amount": precip,
        "Air temperature": rng.normal(5, 8, n),
    })


def test_skewed_weather_features_constant_exists():
    assert SKEWED_WEATHER_FEATURES == [
        "Precipitation amount", "Precipitation intensity", "Snow depth",
    ]


def test_scale_weather_features_applies_log1p_to_skewed_columns_only(tmp_path):
    pipeline = _make_pipeline(tmp_path)

    source_dir = tmp_path / "input_data"
    source_dir.mkdir(parents=True)
    train_df = _make_zero_inflated_df(seed=1)
    test_df = _make_zero_inflated_df(n=50, seed=2)
    train_df.to_parquet(source_dir / "merged_data_x_train.parquet", index=False)
    test_df.to_parquet(source_dir / "merged_data_x_test.parquet", index=False)

    result = pipeline.scale_weather_features(data_dir=str(source_dir))

    assert result["success"] is True
    assert result["skewed_features_log1p"] == ["Precipitation amount"]

    scaled_dir = os.path.join(str(tmp_path), MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
    artifact = joblib.load(os.path.join(scaled_dir, "weather_scaler.joblib"))
    assert artifact["skewed_features"] == ["Precipitation amount"]
    assert "Air temperature" in artifact["weather_features"]

    scaled_train = pd.read_parquet(os.path.join(scaled_dir, "merged_data_x_train.parquet"))

    # This column is ~95% zero, so RobustScaler's IQR is 0 either way (center_=0,
    # scale_ falls back to 1) -- the scaled value equals the (log1p-transformed or raw)
    # input directly. That makes the log1p pre-transform directly observable: with it,
    # the scaled max equals log1p(raw max); without it, the scaled max would equal the
    # raw max unchanged.
    raw_train_max = train_df["Precipitation amount"].max()
    scaled_max = scaled_train["Precipitation amount"].max()
    assert scaled_max == pytest.approx(np.log1p(raw_train_max), abs=1e-6)
    assert scaled_max < raw_train_max
