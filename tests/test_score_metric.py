"""The classifiers are tuned on a probability metric, not on F1.

The label is `trainDisrupted` = late by more than TRAIN_DELAY_MINUTES or cancelled. With the
threshold at 0 about 73% of the Oulu trains are positive (62% across Finland). F1 of the positive
class then rewards predicting "disrupted" for almost everything: the first all-Finland XGBoost run
chose its hyperparameters by CV F1 and reached 0.667, below the 0.77 of the trivial "every train is
disrupted" classifier. `SCORE_METRIC` feeds the search of XGBoost, LightGBM and Random Forest and
the grid of the logistic baseline, so one constant decides for all four.
"""
import inspect

import pytest

import config.const_training as cfg
from src.training_pipeline import TrainingPipeline

PROBABILITY_METRICS = {"roc_auc", "average_precision"}


def test_classifiers_are_tuned_on_a_probability_metric():
    assert cfg.SCORE_METRIC in PROBABILITY_METRICS


@pytest.mark.parametrize("method", [
    "train_xgboost_with_randomized_search_cv",
    "train_lightgbm_with_randomized_search_cv",
    "train_random_forest_with_randomized_search_cv",
    "_train_logistic_baseline_classifier",
])
def test_every_classifier_reads_the_one_constant(method):
    src = inspect.getsource(getattr(TrainingPipeline, method))
    assert "SCORE_METRIC" in src, f"{method} does not use SCORE_METRIC"
    assert "scoring='f1'" not in src and 'scoring="f1"' not in src, f"{method} hardcodes F1"
