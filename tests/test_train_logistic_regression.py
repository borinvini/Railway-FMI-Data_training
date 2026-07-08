import inspect
from config.const_training import (
    REGULARIZED_REGRESSION_OUTPUT_FOLDER,
    LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS,
    ELASTICNET_PARAM_DISTRIBUTIONS,
    TRAINING_STATE_MACHINE,
)
from src.training_pipeline import TrainingPipeline


def test_regularized_regression_output_folder_constant_exists():
    assert REGULARIZED_REGRESSION_OUTPUT_FOLDER == "data/output/regularized_regression"


def test_logistic_regression_param_distributions_constant_exists():
    expected_keys = {'penalty', 'C', 'l1_ratio', 'class_weight'}
    assert set(LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS.keys()) == expected_keys
    assert set(LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS['penalty']) == {'l1', 'l2', 'elasticnet'}


def test_elasticnet_param_distributions_constant_exists():
    expected_keys = {'alpha', 'l1_ratio'}
    assert set(ELASTICNET_PARAM_DISTRIBUTIONS.keys()) == expected_keys
    assert min(ELASTICNET_PARAM_DISTRIBUTIONS['l1_ratio']) >= 0.1


def test_logistic_regression_in_state_machine():
    assert 'train_logistic_regression_with_randomized_search_cv' in TRAINING_STATE_MACHINE
    assert TRAINING_STATE_MACHINE['train_logistic_regression_with_randomized_search_cv'] is True
