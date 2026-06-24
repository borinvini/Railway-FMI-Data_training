import inspect

from src.training_pipeline import TrainingPipeline


def _get_source():
    return inspect.getsource(TrainingPipeline.train_xgboost_with_randomized_search_cv)


def test_n_jobs_is_minus_one():
    source = _get_source()
    assert 'n_jobs=1' not in source, "n_jobs=1 still present; change to n_jobs=-1"
    assert 'n_jobs=-1' in source


def test_eval_metric_removed_from_classifier():
    source = _get_source()
    assert "eval_metric='logloss'" not in source, "Dead eval_metric='logloss' still present on XGBClassifier"


def test_eval_metric_removed_from_regressor():
    source = _get_source()
    assert "eval_metric='mae'" not in source, "Dead eval_metric='mae' still present on XGBRegressor"


def test_best_cv_score_replaces_best_metric():
    source = _get_source()
    assert 'best_metric' not in source, "'best_metric' still present; rename to 'best_cv_score'"
    assert 'best_cv_score' in source


def test_random_state_literal_replaced_with_constant():
    source = _get_source()
    assert 'random_state=42' not in source, "random_state=42 literal still present; use random_state=RANDOM_STATE"
    assert 'random_state=RANDOM_STATE' in source


def test_scale_pos_weight_not_in_param_distributions():
    from config.const_training import XGBOOST_PARAM_DISTRIBUTIONS
    assert 'scale_pos_weight' not in XGBOOST_PARAM_DISTRIBUTIONS
