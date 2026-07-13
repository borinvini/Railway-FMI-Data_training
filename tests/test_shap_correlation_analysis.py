def test_shap_correlation_analysis_in_state_machine():
    from config.const_training import TRAINING_STATE_MACHINE
    assert 'shap_correlation_analysis' in TRAINING_STATE_MACHINE
    assert TRAINING_STATE_MACHINE['shap_correlation_analysis'] is True


def test_shap_correlation_analysis_output_folder_is_700():
    from config.const_training import SHAP_CORRELATION_ANALYSIS_OUTPUT_FOLDER
    assert SHAP_CORRELATION_ANALYSIS_OUTPUT_FOLDER.startswith("data/output/700-")


import os
from src.training_pipeline import TrainingPipeline


def _make_pipeline(tmp_path):
    """Instantiate TrainingPipeline without filesystem side-effects."""
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.project_root = str(tmp_path)
    return pipeline


def test_missing_train_file_returns_failure(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    empty_data_dir = tmp_path / "no_data_here"
    empty_data_dir.mkdir()

    result = pipeline.shap_correlation_analysis(data_dir=str(empty_data_dir))

    assert result["success"] is False
    assert "error" in result
    assert "Training file not found" in result["error"]


import numpy as np
import pandas as pd


def _make_synthetic_train_parquet(tmp_path):
    """Small synthetic dataset: numeric features + binary 'trainDelayed' target."""
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()

    rng = np.random.default_rng(42)
    n = 80
    feature_a = rng.normal(0, 1, n)
    feature_b = rng.normal(0, 1, n)
    feature_c = rng.normal(0, 1, n)
    # target correlated with feature_a so the model has real signal to learn
    trainDelayed = (feature_a + rng.normal(0, 0.5, n) > 0).astype(int)

    df = pd.DataFrame({
        "feature_a": feature_a,
        "feature_b": feature_b,
        "feature_c": feature_c,
        "trainDelayed": trainDelayed,
    })
    train_path = data_dir / "merged_data_selected_train.parquet"
    df.to_parquet(train_path)
    return str(data_dir)


def test_end_to_end_produces_csv_and_charts(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    data_dir = _make_synthetic_train_parquet(tmp_path)

    result = pipeline.shap_correlation_analysis(data_dir=data_dir)

    assert result["success"] is True
    assert result["num_features"] == 3
    assert result["target_feature"] == "trainDelayed"
    assert result["problem_type"] == "classification"

    output_dir = result["output_directory"]
    csv_path = os.path.join(output_dir, "shap_correlation_matrix.csv")
    png_path = os.path.join(output_dir, "shap_correlation_heatmap.png")
    pdf_path = os.path.join(output_dir, "shap_correlation_heatmap.pdf")
    top_pairs_png_path = os.path.join(output_dir, "shap_correlation_top_pairs.png")
    top_pairs_pdf_path = os.path.join(output_dir, "shap_correlation_top_pairs.pdf")
    top_features_png_path = os.path.join(output_dir, "shap_correlation_top_features_heatmap.png")
    top_features_pdf_path = os.path.join(output_dir, "shap_correlation_top_features_heatmap.pdf")

    assert os.path.isfile(csv_path)
    assert os.path.isfile(png_path)
    assert os.path.isfile(pdf_path)
    assert os.path.isfile(top_pairs_png_path)
    assert os.path.isfile(top_pairs_pdf_path)
    assert os.path.isfile(top_features_png_path)
    assert os.path.isfile(top_features_pdf_path)

    assert result["top_pairs_png_path"] == top_pairs_png_path
    assert result["top_pairs_pdf_path"] == top_pairs_pdf_path
    assert result["top_features_heatmap_png_path"] == top_features_png_path
    assert result["top_features_heatmap_pdf_path"] == top_features_pdf_path

    corr_df = pd.read_csv(csv_path, index_col=0)
    assert list(corr_df.columns) == ["feature_a", "feature_b", "feature_c"]
    assert list(corr_df.index) == ["feature_a", "feature_b", "feature_c"]
    assert corr_df.shape == (3, 3)

    most_correlated_pair = result["most_correlated_pair"]
    assert most_correlated_pair is not None
    feature_1, feature_2, corr_value = most_correlated_pair
    assert feature_1 != feature_2
    assert {feature_1, feature_2}.issubset({"feature_a", "feature_b", "feature_c"})
    assert -1.0 <= corr_value <= 1.0


from unittest.mock import patch
from config.const_training import (
    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER,
    MERGED_BALANCED_OUTPUT_FOLDER,
)


def _make_dispatcher_pipeline(tmp_path):
    """Instantiate TrainingPipeline with the attributes execute_training_pipeline_steps needs."""
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


_SHAP_SUCCESS = {
    "success": True,
    "output_directory": "/fake/output",
    "num_features": 3,
    "target_feature": "trainDelayed",
    "problem_type": "classification",
    "most_correlated_pair": ("feature_a", "feature_b", 0.42),
    "csv_path": "/fake/output/shap_correlation_matrix.csv",
    "png_path": "/fake/output/shap_correlation_heatmap.png",
    "pdf_path": "/fake/output/shap_correlation_heatmap.pdf",
    "top_pairs_png_path": "/fake/output/shap_correlation_top_pairs.png",
    "top_pairs_pdf_path": "/fake/output/shap_correlation_top_pairs.pdf",
    "top_features_heatmap_png_path": "/fake/output/shap_correlation_top_features_heatmap.png",
    "top_features_heatmap_pdf_path": "/fake/output/shap_correlation_top_features_heatmap.pdf",
}


def _make_dispatcher_state_machine(scale: bool, balance: bool, shap_enabled: bool) -> dict:
    return {
        "merge_data_files": False,
        "select_training_cols": False,
        "split_dataset": False,
        "balance_classes": balance,
        "scale_weather_features": scale,
        "shap_correlation_analysis": shap_enabled,
    }


@patch.object(TrainingPipeline, "shap_correlation_analysis")
@patch.object(TrainingPipeline, "scale_weather_features")
def test_shap_dispatcher_uses_scaled_dir_when_scale_enabled(mock_scale, mock_shap, tmp_path):
    """When scale_weather_features=True, shap_correlation_analysis should receive the 505 (scaled) directory."""
    pipeline = _make_dispatcher_pipeline(tmp_path)
    mock_shap.return_value = _SHAP_SUCCESS
    mock_scale.return_value = {
        "success": True, "processed_files": 1,
        "total_train_rows": 1, "total_test_rows": 1,
        "weather_features_scaled": [],
    }

    pipeline.execute_training_pipeline_steps(
        [], state_machine=_make_dispatcher_state_machine(scale=True, balance=False, shap_enabled=True)
    )

    mock_shap.assert_called_once()
    _, kwargs = mock_shap.call_args
    expected = os.path.join(str(tmp_path), MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )


@patch.object(TrainingPipeline, "shap_correlation_analysis")
@patch.object(TrainingPipeline, "balance_classes")
def test_shap_dispatcher_uses_balanced_dir_when_balance_enabled_and_scale_disabled(mock_balance, mock_shap, tmp_path):
    """When balance_classes=True and scale_weather_features=False, shap_correlation_analysis should receive the 504 (balanced) directory."""
    pipeline = _make_dispatcher_pipeline(tmp_path)
    mock_shap.return_value = _SHAP_SUCCESS
    mock_balance.return_value = {
        "success": True, "rows_before": 400, "rows_after": 450,
        "minority_share_before": 25.0, "minority_share_after": 45.0,
        "resampling_method": "SMOTE_TOMEK", "skipped": False,
        "dropped_non_numeric_cols": [], "dropped_counterpart_col": "differenceInMinutes",
        "train_output_path": "/fake/train.parquet", "test_output_path": "/fake/test.parquet",
    }

    pipeline.execute_training_pipeline_steps(
        [], state_machine=_make_dispatcher_state_machine(scale=False, balance=True, shap_enabled=True)
    )

    mock_shap.assert_called_once()
    _, kwargs = mock_shap.call_args
    expected = os.path.join(str(tmp_path), MERGED_BALANCED_OUTPUT_FOLDER)
    assert kwargs.get("data_dir") == expected, (
        f"Expected data_dir={expected!r}, got {kwargs.get('data_dir')!r}"
    )


def test_shap_dispatcher_skipped_when_disabled(tmp_path):
    """When shap_correlation_analysis=False, the method must not be called."""
    pipeline = _make_dispatcher_pipeline(tmp_path)
    with patch.object(TrainingPipeline, "shap_correlation_analysis") as mock_shap:
        pipeline.execute_training_pipeline_steps(
            [], state_machine=_make_dispatcher_state_machine(scale=True, balance=False, shap_enabled=False)
        )
        mock_shap.assert_not_called()
