import inspect
from config.const_training import (
    RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    RANDOM_FOREST_PARAM_DISTRIBUTIONS,
    RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION,
    TRAINING_STATE_MACHINE,
)
from src.training_pipeline import TrainingPipeline


def test_random_forest_output_folder_constant_exists():
    assert RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER == "data/output/random_forest_randomized_search"


def test_random_forest_param_distributions_constant_exists():
    expected_keys = {
        'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
        'max_features', 'bootstrap', 'criterion', 'min_impurity_decrease',
        'class_weight', 'ccp_alpha'
    }
    assert set(RANDOM_FOREST_PARAM_DISTRIBUTIONS.keys()) == expected_keys


def test_random_forest_param_distributions_regression_constant_exists():
    expected_keys = {
        'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
        'max_features', 'bootstrap', 'criterion', 'min_impurity_decrease',
        'ccp_alpha'
    }
    assert set(RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION.keys()) == expected_keys
    assert 'class_weight' not in RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION
    assert set(RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION['criterion']) == {
        'squared_error', 'absolute_error', 'friedman_mse'
    }


def test_random_forest_in_state_machine():
    assert 'train_random_forest_with_randomized_search_cv' in TRAINING_STATE_MACHINE
    assert TRAINING_STATE_MACHINE['train_random_forest_with_randomized_search_cv'] is True
