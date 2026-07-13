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

    assert os.path.isfile(csv_path)
    assert os.path.isfile(png_path)
    assert os.path.isfile(pdf_path)

    corr_df = pd.read_csv(csv_path, index_col=0)
    assert list(corr_df.columns) == ["feature_a", "feature_b", "feature_c"]
    assert list(corr_df.index) == ["feature_a", "feature_b", "feature_c"]
    assert corr_df.shape == (3, 3)
