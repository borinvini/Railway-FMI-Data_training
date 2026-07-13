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


def _source():
    return inspect.getsource(TrainingPipeline.train_naive_bayes_with_randomized_search_cv)


def test_random_state_uses_constant():
    src = _source()
    assert 'RANDOM_STATE' in src
    assert 'random_state=42' not in src


def test_classifier_and_regressor_classes_used():
    src = _source()
    assert 'GaussianNB(' in src
    assert 'BayesianRidge(' in src


def test_gaussian_nb_constructed_with_no_kwargs():
    src = _source()
    gnb_start = src.index('GaussianNB(')
    gnb_call = src[gnb_start:gnb_start + 20]
    assert 'GaussianNB()' in gnb_call


def test_bayesian_ridge_no_random_state_kwarg():
    src = _source()
    br_start = src.index('BayesianRidge(')
    br_call = src[br_start:br_start + 200]
    assert 'random_state' not in br_call


def test_param_distributions_selected_by_problem_type():
    src = _source()
    assert 'NAIVE_BAYES_PARAM_DISTRIBUTIONS' in src
    assert 'BAYESIAN_RIDGE_PARAM_DISTRIBUTIONS' in src


def test_feature_importance_uses_theta_and_var_for_classification():
    src = _source()
    assert 'theta_' in src
    assert 'var_' in src
    assert 'feature_importances_' not in src


def test_feature_importance_uses_coef_for_regression():
    assert 'best_model.coef_' in _source()


def test_wmape_in_naive_bayes_regression_branch():
    assert 'final_test_wmape' in _source()


def test_binary_metrics_in_naive_bayes_regression_branch():
    assert 'final_bin_f1' in _source()


def test_output_filenames_use_naive_bayes_prefix():
    src = _source()
    assert 'naive_bayes_iteration_analysis_' in src
    assert 'naive_bayes_feature_importance_' in src
    assert 'naive_bayes_best_model_' in src


def test_dispatcher_routes_naive_bayes_after_logistic_regression():
    dispatcher_src = inspect.getsource(TrainingPipeline.execute_training_pipeline_steps)
    logistic_regression_idx = dispatcher_src.index('train_logistic_regression_with_randomized_search_cv')
    naive_bayes_idx = dispatcher_src.index(
        'train_naive_bayes_with_randomized_search_cv', logistic_regression_idx
    )
    assert naive_bayes_idx > logistic_regression_idx


def test_dispatcher_populates_naive_bayes_file_info_keys():
    dispatcher_src = inspect.getsource(TrainingPipeline.execute_training_pipeline_steps)
    assert 'naive_bayes_models_trained' in dispatcher_src
    assert 'naive_bayes_problem_type' in dispatcher_src
