import pytest

def test_quantile_constants_exist():
    from config.const_training import FILTER_LOWER_QUANTILE, FILTER_UPPER_QUANTILE
    assert 0.0 < FILTER_LOWER_QUANTILE < 0.1
    assert 0.9 < FILTER_UPPER_QUANTILE < 1.0
    assert FILTER_LOWER_QUANTILE < FILTER_UPPER_QUANTILE

def test_outlier_folder_constant_exists():
    from config.const_training import MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
    assert "502" in MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
    assert "outlier" in MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER

def test_scaled_folder_renamed_to_503():
    from config.const_training import MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
    assert "503" in MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
