def test_shap_correlation_analysis_in_state_machine():
    from config.const_training import TRAINING_STATE_MACHINE
    assert 'shap_correlation_analysis' in TRAINING_STATE_MACHINE
    assert TRAINING_STATE_MACHINE['shap_correlation_analysis'] is True


def test_shap_correlation_analysis_output_folder_is_700():
    from config.const_training import SHAP_CORRELATION_ANALYSIS_OUTPUT_FOLDER
    assert SHAP_CORRELATION_ANALYSIS_OUTPUT_FOLDER.startswith("data/output/700-")
