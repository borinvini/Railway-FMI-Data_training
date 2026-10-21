"""Every classification trainer saves AUC and PR-AUC for each iteration, on test and hold-out.

The first all-stations XGBoost run saved ROC-AUC only for the final best model on the test
split (hold-out had it per iteration) and no PR-AUC anywhere. At a 0.7-1.9% positive rate
ROC-AUC flatters the model: false alarms are divided by the huge pool of normal trains.
PR-AUC divides real alarms by all alarms, and its no-skill value is the positive rate, so
that rate and the F1 of "every train is disrupted" are saved beside it.

`_probability_metrics` is the one definition of these numbers; `_evaluate_holdout` and the
logistic baseline use it as well, so test and hold-out are scored identically.
"""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from config.const_training import (
    LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    NAIVE_BAYES_OUTPUT_FOLDER,
    RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    REGULARIZED_REGRESSION_OUTPUT_FOLDER,
    XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
)
from src.training_pipeline import TrainingPipeline
from tests.test_grouped_cv import TARGET, _frame, _pipeline


class _FixedScores:
    """Stands in for a fitted classifier that returns given positive-class probabilities."""

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.scores, self.scores])


def test_helper_perfect_ranking_scores_one():
    y = np.array([0, 0, 0, 1, 1])
    out = TrainingPipeline._probability_metrics(_FixedScores([0.1, 0.2, 0.3, 0.8, 0.9]), None, y)
    assert out["auc"] == pytest.approx(1.0)
    assert out["pr_auc"] == pytest.approx(1.0)


def test_helper_no_skill_pr_auc_equals_the_base_rate_and_roc_auc_half():
    y = np.array([0] * 98 + [1] * 2)
    out = TrainingPipeline._probability_metrics(_FixedScores(np.full(100, 0.5)), None, y)
    assert out["positive_rate"] == pytest.approx(0.02)
    assert out["pr_auc"] == pytest.approx(0.02)
    assert out["auc"] == pytest.approx(0.5)


def test_helper_always_positive_f1_matches_the_formula():
    y = np.array([0] * 90 + [1] * 10)
    out = TrainingPipeline._probability_metrics(_FixedScores(np.linspace(0, 1, 100)), None, y)
    p = out["positive_rate"]
    assert out["f1_always_positive"] == pytest.approx(2 * p / (1 + p))


def test_helper_single_class_and_missing_predict_proba_do_not_raise():
    one_class = TrainingPipeline._probability_metrics(_FixedScores([0.2, 0.4]), None, np.array([0, 0]))
    assert one_class["auc"] == 0.0 and one_class["pr_auc"] == 0.0
    assert TrainingPipeline._probability_metrics(object(), None, np.array([0, 1])) == {}


# --- end to end on a small separable split --------------------------------------------

def _separable_split(tmp_path):
    """Split folder where f2 carries signal, plus a hold-out copy of the test rows."""
    df = _frame(n_days=60)
    rng = np.random.default_rng(3)
    df["f2"] = rng.normal(size=len(df)) + 2.0 * df[TARGET]
    folder = tmp_path / "scaled"
    _pipeline(tmp_path).write_splits(df, output_dir=str(folder), identifier="t",
                                     basename="merged_data_x")
    test = pd.read_parquet(folder / "merged_data_x_test.parquet")
    test.to_parquet(folder / "merged_data_x_holdout.parquet", index=False)
    return folder


TRAINERS = [
    ("train_xgboost_with_randomized_search_cv", XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER, "xgboost"),
    ("train_lightgbm_with_randomized_search_cv", LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER, "lightgbm"),
    ("train_random_forest_with_randomized_search_cv", RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER, "random_forest"),
    ("train_naive_bayes_with_randomized_search_cv", NAIVE_BAYES_OUTPUT_FOLDER, "naive_bayes"),
]


@pytest.mark.parametrize("method,folder_const,prefix", TRAINERS)
def test_every_iteration_saves_auc_and_pr_auc_for_test_and_holdout(tmp_path, method, folder_const, prefix):
    folder = _separable_split(tmp_path)
    pipeline = _pipeline(tmp_path)
    with patch("src.training_pipeline.SEARCH_N_JOBS", 1), \
            patch("src.training_pipeline.RANDOM_SEARCH_ITERATIONS", 20), \
            patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET):
        result = getattr(pipeline, method)(data_dir=str(folder))
    assert result["success"] is True, result

    results = json.loads((tmp_path / folder_const / f"{prefix}_iteration_analysis_x.json").read_text())
    n_iter = len(results["iteration_analysis"])
    assert n_iter == 2  # 10 and 20 candidates

    for row in results["iteration_analysis"]:  # test split, per iteration
        for key in ("n_iter", "cv_score", "test_metric", "test_accuracy", "test_precision",
                    "test_recall", "test_confusion_matrix", "test_auc", "test_pr_auc"):
            assert row[key] is not None, key
    for row in results["holdout_iteration_analysis"]:  # hold-out, per iteration
        for key in ("accuracy", "f1", "precision", "recall", "confusion_matrix", "auc",
                    "pr_auc", "positive_rate", "f1_always_positive"):
            assert key in row, key

    summary = results["iteration_metrics_summary"]
    assert len(summary["auc_values"]) == len(summary["pr_auc_values"]) == n_iter
    for key in ("best_auc", "best_pr_auc", "average_auc", "average_pr_auc"):
        assert key in summary
    hold_summary = results["holdout_iteration_summary"]
    assert len(hold_summary["auc_values"]) == len(hold_summary["pr_auc_values"]) == n_iter

    final = results["final_metrics"]
    for key in ("test_auc", "test_pr_auc", "positive_rate_test", "f1_always_positive"):
        assert key in final, key
    for key in ("auc", "pr_auc", "positive_rate", "f1_always_positive"):
        assert key in results["holdout_metrics"], key

    # f2 separates the classes, so the ranking must beat the no-skill PR-AUC (the base rate).
    assert final["test_pr_auc"] > final["positive_rate_test"]
    # The final model is one of the iterations, so its PR-AUC appears in the per-iteration list.
    assert any(final["test_pr_auc"] == pytest.approx(v) for v in summary["pr_auc_values"])


def test_logistic_baseline_uses_the_same_definition(tmp_path):
    folder = _separable_split(tmp_path)
    pipeline = _pipeline(tmp_path)
    with patch("src.training_pipeline.SEARCH_N_JOBS", 1), \
            patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET):
        pipeline.train_logistic_regression_with_randomized_search_cv(data_dir=str(folder))
    results = json.loads((tmp_path / REGULARIZED_REGRESSION_OUTPUT_FOLDER
                          / "logistic_regression_iteration_analysis_x.json").read_text())
    for key in ("test_auc", "test_pr_auc", "positive_rate_test", "f1_always_positive"):
        assert key in results["final_metrics"], key
    for key in ("auc", "pr_auc", "positive_rate", "f1_always_positive"):
        assert key in results["holdout_metrics"], key
