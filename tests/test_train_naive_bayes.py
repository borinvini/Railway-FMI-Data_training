import inspect
from config.const_training import (
    NAIVE_BAYES_OUTPUT_FOLDER,
    NAIVE_BAYES_PARAM_DISTRIBUTIONS,
    BAYESIAN_RIDGE_PARAM_DISTRIBUTIONS,
    TRAINING_STATE_MACHINE,
)
from src.training_pipeline import TrainingPipeline


def test_naive_bayes_output_folder_constant_exists():
    assert NAIVE_BAYES_OUTPUT_FOLDER == "data/output/1004-naive_bayes"


def test_naive_bayes_param_distributions_constant_exists():
    assert set(NAIVE_BAYES_PARAM_DISTRIBUTIONS.keys()) == {'var_smoothing'}


def test_bayesian_ridge_param_distributions_constant_exists():
    expected_keys = {'alpha_1', 'alpha_2', 'lambda_1', 'lambda_2'}
    assert set(BAYESIAN_RIDGE_PARAM_DISTRIBUTIONS.keys()) == expected_keys


def test_naive_bayes_in_state_machine():
    assert 'train_naive_bayes_with_randomized_search_cv' in TRAINING_STATE_MACHINE
    assert TRAINING_STATE_MACHINE['train_naive_bayes_with_randomized_search_cv'] is True
