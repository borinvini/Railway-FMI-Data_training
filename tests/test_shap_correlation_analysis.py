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
