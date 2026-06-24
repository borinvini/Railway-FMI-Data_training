import inspect
from config.const_training import (
    LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    LIGHTGBM_PARAM_DISTRIBUTIONS,
    TRAINING_STATE_MACHINE,
)
from src.training_pipeline import TrainingPipeline


def test_lightgbm_output_folder_constant_exists():
    assert LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER == "data/output/1001-lightgbm_randomized_search"


def test_lightgbm_param_distributions_constant_exists():
    expected_keys = {
        'n_estimators', 'num_leaves', 'learning_rate', 'subsample',
        'colsample_bytree', 'min_child_samples', 'reg_alpha', 'reg_lambda'
    }
    assert set(LIGHTGBM_PARAM_DISTRIBUTIONS.keys()) == expected_keys


def test_lightgbm_in_state_machine():
    assert 'train_lightgbm_with_randomized_search_cv' in TRAINING_STATE_MACHINE


def _source():
    return inspect.getsource(TrainingPipeline.train_lightgbm_with_randomized_search_cv)


def test_n_jobs_is_minus_one():
    assert 'n_jobs=-1' in _source()


def test_subsample_freq_set_in_constructor():
    assert 'subsample_freq=1' in _source()


def test_no_eval_metric_in_source():
    assert 'eval_metric' not in _source()


def test_random_state_uses_constant():
    src = _source()
    assert 'RANDOM_STATE' in src
    assert 'random_state=42' not in src


def test_lightgbm_param_distributions_used():
    assert 'LIGHTGBM_PARAM_DISTRIBUTIONS' in _source()


def test_wmape_in_lightgbm_regression():
    assert 'final_test_wmape' in _source()


def test_binary_metrics_in_lightgbm_regression():
    assert 'final_bin_f1' in _source()
