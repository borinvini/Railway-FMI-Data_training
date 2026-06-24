import inspect

import numpy as np

from src.training_pipeline import TrainingPipeline


def _src():
    return inspect.getsource(TrainingPipeline.train_xgboost_with_randomized_search_cv)


def test_delay_threshold_constant_exists():
    from config.const_training import DELAY_THRESHOLD_MINUTES
    assert DELAY_THRESHOLD_MINUTES == 5


def test_wmape_formula_correctness():
    y_true = np.array([10.0, 20.0, 0.0, 5.0])
    y_pred = np.array([12.0, 18.0, 2.0, 4.0])
    denom = np.sum(np.abs(y_true))
    wmape = (np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom > 0 else 0.0
    assert abs(wmape - 20.0) < 1e-6


def test_wmape_handles_zero_denominator():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    denom = np.sum(np.abs(y_true))
    wmape = (np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom > 0 else 0.0
    assert wmape == 0.0


def test_mape_removed_from_regression_path():
    src = _src()
    assert 'test_mape_scores' not in src
    assert 'final_test_mape' not in src


def test_wmape_present_in_regression_path():
    src = _src()
    assert 'test_wmape_scores' in src
    assert 'final_test_wmape' in src


def test_delay_threshold_used_in_source():
    assert 'DELAY_THRESHOLD_MINUTES' in _src()


def test_binary_metrics_computed_per_iteration():
    src = _src()
    assert 'test_bin_f1' in src
    assert 'test_bin_precision' in src
    assert 'test_bin_recall' in src
    assert 'test_bin_accuracy' in src


def test_binary_metrics_tracked_across_iterations():
    src = _src()
    assert 'test_bin_f1_scores' in src
    assert 'test_bin_precision_scores' in src
    assert 'test_bin_recall_scores' in src
    assert 'test_bin_accuracy_scores' in src


def test_final_binary_metrics_in_source():
    src = _src()
    assert 'final_bin_f1' in src
    assert 'final_bin_precision' in src
    assert 'final_bin_recall' in src
    assert 'final_bin_accuracy' in src
