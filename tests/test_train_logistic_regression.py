"""Stage 1003 is the simple linear baseline the tree models are compared with.

For a classification target it used to be a competitor, and a slow one:

1. It randomly searched the penalty (l1 / l2 / elasticnet), C and ``class_weight``. The
   class-weight search replaced the constructor's ``'balanced'`` with ``None`` or a hand-made
   ratio, so the stated "class_weight='balanced'" did not hold.
2. It repeated that search ten times (n_iter 10..100), about 2,750 saga fits, and kept the best
   CV score of the ten. On Roihu it ran past the 8 h limit.
3. It fitted on features that only the weather columns had been scaled (RobustScaler), so the
   penalty and the coefficient chart compared unequal scales.

Now the classification path is StandardScaler + L2 LogisticRegression (lbfgs, balanced), one
day-grouped GridSearchCV over C, and it reports PR-AUC and the F1 of "every train is disrupted"
so the baseline can be read against no skill. The numeric target keeps its ElasticNet path.
"""
import inspect
import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from config.const_training import (
    ELASTICNET_PARAM_DISTRIBUTIONS,
    LOGISTIC_BASELINE_CS,
    REGULARIZED_REGRESSION_OUTPUT_FOLDER,
    TRAINING_STATE_MACHINE,
)
from src.training_pipeline import TrainingPipeline
from tests.test_grouped_cv import GROUP_KEY_COLUMN, TARGET, _frame, _pipeline, _prepared_folder

METHOD = "train_logistic_regression_with_randomized_search_cv"


def test_regularized_regression_output_folder_constant_exists():
    assert REGULARIZED_REGRESSION_OUTPUT_FOLDER == "data/output/1003-regularized_regression"


def test_baseline_c_grid_is_log_spaced():
    cs = np.log10(LOGISTIC_BASELINE_CS)
    assert len(cs) >= 5
    assert np.allclose(np.diff(cs), 1.0)


def test_logistic_search_distributions_are_gone():
    import config.const_training as cfg
    assert not hasattr(cfg, "LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS")


def test_elasticnet_param_distributions_constant_exists():
    assert set(ELASTICNET_PARAM_DISTRIBUTIONS) == {'alpha', 'l1_ratio'}
    assert min(ELASTICNET_PARAM_DISTRIBUTIONS['l1_ratio']) >= 0.1


def test_logistic_regression_in_state_machine():
    assert METHOD in TRAINING_STATE_MACHINE
    assert TRAINING_STATE_MACHINE[METHOD] is True


def _classifier_source():
    return inspect.getsource(TrainingPipeline._train_logistic_baseline_classifier)


def _regression_source():
    return inspect.getsource(TrainingPipeline._train_elasticnet_regression)


def test_classifier_is_a_scaled_l2_lbfgs_pipeline_with_balanced_weights():
    src = _classifier_source()
    assert 'StandardScaler()' in src
    assert "penalty='l2'" in src
    assert "solver='lbfgs'" in src
    assert "class_weight='balanced'" in src
    assert "solver='saga'" not in src


def test_classifier_tunes_only_c_with_one_grid_search():
    src = _classifier_source()
    assert src.count('GridSearchCV(') == 1
    assert 'RandomizedSearchCV' not in src
    assert 'param_grid={"logreg__C"' in src
    assert 'class_weight' not in src.split('param_grid=')[1].split(')')[0]


def test_classifier_search_uses_slurm_aware_n_jobs():
    src = _classifier_source()
    assert 'n_jobs=-1' not in src, "hardcoded n_jobs=-1 ignores the Slurm allocation"
    assert 'n_jobs=SEARCH_N_JOBS' in src
    assert 'n_jobs=MODEL_N_JOBS' in src


def test_random_state_uses_constant():
    for src in (_classifier_source(), _regression_source()):
        assert 'RANDOM_STATE' in src
        assert 'random_state=42' not in src


def test_elasticnet_no_n_jobs_kwarg():
    src = _regression_source()
    start = src.index('ElasticNet(')
    # ElasticNet raises TypeError on n_jobs
    assert 'n_jobs' not in src[start:start + 200]


def test_regression_path_keeps_elasticnet_wmape_and_binary_metrics():
    src = _regression_source()
    assert 'ElasticNet(' in src
    assert 'ELASTICNET_PARAM_DISTRIBUTIONS' in src
    assert 'final_test_wmape' in src
    assert 'final_bin_f1' in src


def test_feature_importance_uses_coef_not_feature_importances():
    for src in (_classifier_source(), _regression_source()):
        assert 'coef_' in src
        assert 'feature_importances_' not in src


def test_output_filenames_use_logistic_regression_prefix():
    src = _classifier_source() + _regression_source()
    assert 'logistic_regression_iteration_analysis_' in src
    assert 'logistic_regression_feature_importance_' in src
    assert 'logistic_regression_best_model_' in src


def test_public_method_dispatches_by_problem_type():
    src = inspect.getsource(TrainingPipeline.train_logistic_regression_with_randomized_search_cv)
    assert '_train_logistic_baseline_classifier' in src
    assert '_train_elasticnet_regression' in src


def test_dispatcher_routes_logistic_regression_after_random_forest():
    dispatcher_src = inspect.getsource(TrainingPipeline.execute_training_pipeline_steps)
    random_forest_idx = dispatcher_src.index('train_random_forest_with_randomized_search_cv')
    assert dispatcher_src.index(METHOD, random_forest_idx) > random_forest_idx


def test_dispatcher_populates_logistic_regression_file_info_keys():
    dispatcher_src = inspect.getsource(TrainingPipeline.execute_training_pipeline_steps)
    assert 'logistic_regression_models_trained' in dispatcher_src
    assert 'logistic_regression_problem_type' in dispatcher_src


# --- end to end on a small synthetic split ------------------------------------

def _separable_split(tmp_path):
    """Split folder where f2 carries signal, plus a hold-out copy of the test rows."""
    df = _frame(n_days=60)
    rng = np.random.default_rng(3)
    df["f2"] = rng.normal(size=len(df)) + 2.0 * df[TARGET]
    folder = tmp_path / "scaled"
    _pipeline(tmp_path).write_splits(df, output_dir=str(folder), identifier="t", basename="merged_data_x")
    test = pd.read_parquet(folder / "merged_data_x_test.parquet")
    test.to_parquet(folder / "merged_data_x_holdout.parquet", index=False)
    return folder


def test_classifier_end_to_end_writes_baseline_results(tmp_path):
    folder = _separable_split(tmp_path)
    pipeline = _pipeline(tmp_path)
    with patch("src.training_pipeline.SEARCH_N_JOBS", 1), \
            patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET):
        result = pipeline.train_logistic_regression_with_randomized_search_cv(data_dir=str(folder))

    assert result["success"] is True, result
    assert result["problem_type"] == "classification"

    out = tmp_path / REGULARIZED_REGRESSION_OUTPUT_FOLDER
    results = json.loads((out / "logistic_regression_iteration_analysis_x.json").read_text())
    assert results["best_C"] in LOGISTIC_BASELINE_CS
    assert len(results["C_scores"]) == len(LOGISTIC_BASELINE_CS)
    final = results["final_metrics"]
    for key in ("test_f1", "test_auc", "test_pr_auc", "positive_rate_test", "f1_always_positive"):
        assert key in final
    assert final["test_auc"] > 0.7, "f2 separates the classes; the baseline should find it"
    assert final["test_f1"] > final["f1_always_positive"]
    assert "pr_auc" in results["holdout_metrics"]
    assert "f1_always_positive" in results["holdout_metrics"]

    importance = pd.read_csv(out / "logistic_regression_feature_importance_x.csv")
    assert set(importance["feature"]) == {"f1", "f2"}
    assert importance.iloc[0]["feature"] == "f2"
    assert os.path.exists(out / "logistic_regression_cv_by_C_x.png")
    assert os.path.exists(out / "logistic_regression_best_model_x.pkl")


def test_always_positive_f1_matches_the_formula(tmp_path):
    folder = _separable_split(tmp_path)
    pipeline = _pipeline(tmp_path)
    with patch("src.training_pipeline.SEARCH_N_JOBS", 1), \
            patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET):
        pipeline.train_logistic_regression_with_randomized_search_cv(data_dir=str(folder))
    out = tmp_path / REGULARIZED_REGRESSION_OUTPUT_FOLDER
    final = json.loads((out / "logistic_regression_iteration_analysis_x.json").read_text())["final_metrics"]
    p = final["positive_rate_test"]
    assert final["f1_always_positive"] == pytest.approx(2 * p / (1 + p))
