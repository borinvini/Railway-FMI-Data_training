import inspect
from config.const_training import (
    RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    RANDOM_FOREST_PARAM_DISTRIBUTIONS,
    RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION,
    TRAINING_STATE_MACHINE,
)
from src.training_pipeline import TrainingPipeline


def test_random_forest_output_folder_constant_exists():
    assert RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER == "data/output/1002-random_forest_randomized_search"


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


def _source():
    return inspect.getsource(TrainingPipeline.train_random_forest_with_randomized_search_cv)


def test_n_jobs_is_minus_one():
    assert 'n_jobs=-1' in _source()


def test_no_lightgbm_or_xgboost_specific_args_in_source():
    src = _source()
    assert 'subsample_freq' not in src
    assert 'verbose=-1' not in src


def test_random_state_uses_constant():
    src = _source()
    assert 'RANDOM_STATE' in src
    assert 'random_state=42' not in src


def test_classifier_and_regressor_classes_used():
    src = _source()
    assert 'RandomForestClassifier(' in src
    assert 'RandomForestRegressor(' in src


def test_param_distributions_selected_by_problem_type():
    src = _source()
    assert 'RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION' in src
    assert 'param_distributions=RANDOM_FOREST_PARAM_DISTRIBUTIONS' in src


def test_wmape_in_random_forest_regression():
    assert 'final_test_wmape' in _source()


def test_binary_metrics_in_random_forest_regression():
    assert 'final_bin_f1' in _source()


def test_output_filenames_use_random_forest_prefix():
    src = _source()
    assert 'random_forest_iteration_analysis_' in src
    assert 'random_forest_feature_importance_' in src
    assert 'random_forest_best_model_' in src


def test_dispatcher_routes_random_forest_after_lightgbm():
    dispatcher_src = inspect.getsource(TrainingPipeline.execute_training_pipeline_steps)
    lightgbm_idx = dispatcher_src.index('train_lightgbm_with_randomized_search_cv')
    random_forest_idx = dispatcher_src.index(
        'train_random_forest_with_randomized_search_cv', lightgbm_idx
    )
    assert random_forest_idx > lightgbm_idx


def test_dispatcher_populates_random_forest_file_info_keys():
    dispatcher_src = inspect.getsource(TrainingPipeline.execute_training_pipeline_steps)
    assert 'random_forest_models_trained' in dispatcher_src
    assert 'random_forest_problem_type' in dispatcher_src
