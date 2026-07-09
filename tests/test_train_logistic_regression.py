import inspect
from config.const_training import (
    REGULARIZED_REGRESSION_OUTPUT_FOLDER,
    LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS,
    ELASTICNET_PARAM_DISTRIBUTIONS,
    TRAINING_STATE_MACHINE,
)
from src.training_pipeline import TrainingPipeline


def test_regularized_regression_output_folder_constant_exists():
    assert REGULARIZED_REGRESSION_OUTPUT_FOLDER == "data/output/1003-regularized_regression"


def test_logistic_regression_param_distributions_constant_exists():
    # A list of two dicts (not one flat dict) so l1_ratio is only ever sampled
    # alongside penalty='elasticnet' — avoids sklearn's UserWarning for l1/l2.
    assert isinstance(LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS, list)
    assert len(LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS) == 2

    l1_l2_dict, elasticnet_dict = LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS
    assert set(l1_l2_dict.keys()) == {'penalty', 'C', 'class_weight'}
    assert set(l1_l2_dict['penalty']) == {'l1', 'l2'}
    assert 'l1_ratio' not in l1_l2_dict

    assert set(elasticnet_dict.keys()) == {'penalty', 'C', 'l1_ratio', 'class_weight'}
    assert elasticnet_dict['penalty'] == ['elasticnet']


def test_elasticnet_param_distributions_constant_exists():
    expected_keys = {'alpha', 'l1_ratio'}
    assert set(ELASTICNET_PARAM_DISTRIBUTIONS.keys()) == expected_keys
    assert min(ELASTICNET_PARAM_DISTRIBUTIONS['l1_ratio']) >= 0.1


def test_logistic_regression_in_state_machine():
    assert 'train_logistic_regression_with_randomized_search_cv' in TRAINING_STATE_MACHINE
    assert TRAINING_STATE_MACHINE['train_logistic_regression_with_randomized_search_cv'] is True


def _source():
    return inspect.getsource(TrainingPipeline.train_logistic_regression_with_randomized_search_cv)


def test_n_jobs_is_minus_one():
    assert 'n_jobs=-1' in _source()


def test_saga_solver_used():
    assert "solver='saga'" in _source()


def test_max_iter_hardcoded():
    assert 'max_iter=5000' in _source()


def test_elasticnet_no_n_jobs_kwarg():
    src = _source()
    # ElasticNet constructor call must not pass n_jobs (raises TypeError at runtime)
    elasticnet_start = src.index('ElasticNet(')
    elasticnet_call = src[elasticnet_start:elasticnet_start + 200]
    assert 'n_jobs' not in elasticnet_call


def test_random_state_uses_constant():
    src = _source()
    assert 'RANDOM_STATE' in src
    assert 'random_state=42' not in src


def test_classifier_and_regressor_classes_used():
    src = _source()
    assert 'LogisticRegression(' in src
    assert 'ElasticNet(' in src


def test_param_distributions_selected_by_problem_type():
    src = _source()
    assert 'LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS' in src
    assert 'ELASTICNET_PARAM_DISTRIBUTIONS' in src


def test_feature_importance_uses_coef_not_feature_importances():
    src = _source()
    assert 'coef_' in src
    assert 'feature_importances_' not in src


def test_coefficient_magnitude_label_used():
    src = _source()
    assert 'Coefficient Magnitude' in src


def test_wmape_in_logistic_regression_regression_branch():
    assert 'final_test_wmape' in _source()


def test_binary_metrics_in_logistic_regression_regression_branch():
    assert 'final_bin_f1' in _source()


def test_output_filenames_use_logistic_regression_prefix():
    src = _source()
    assert 'logistic_regression_iteration_analysis_' in src
    assert 'logistic_regression_feature_importance_' in src
    assert 'logistic_regression_best_model_' in src


def test_dispatcher_routes_logistic_regression_after_random_forest():
    dispatcher_src = inspect.getsource(TrainingPipeline.execute_training_pipeline_steps)
    random_forest_idx = dispatcher_src.index('train_random_forest_with_randomized_search_cv')
    logistic_regression_idx = dispatcher_src.index(
        'train_logistic_regression_with_randomized_search_cv', random_forest_idx
    )
    assert logistic_regression_idx > random_forest_idx


def test_dispatcher_populates_logistic_regression_file_info_keys():
    dispatcher_src = inspect.getsource(TrainingPipeline.execute_training_pipeline_steps)
    assert 'logistic_regression_models_trained' in dispatcher_src
    assert 'logistic_regression_problem_type' in dispatcher_src
