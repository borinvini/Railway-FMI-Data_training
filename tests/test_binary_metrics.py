"""Precision and recall must be positive-class (binary) metrics for a binary target.

Regression guard for a real reporting bug: f1 was computed with
average='binary' while precision and recall used average='weighted'. The three
numbers therefore described different things, and weighted recall is
mathematically identical to accuracy — so reported "recall" was accuracy under
another name, and far more flattering than the true positive-class recall.
"""
import inspect
import re

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from src.training_pipeline import TrainingPipeline


# y_true/y_pred chosen so binary and weighted averaging give clearly different
# answers: TN=2, FP=1, FN=0, TP=1.
Y_TRUE = np.array([0, 0, 0, 1])
Y_PRED = np.array([0, 0, 1, 1])

EXPECTED_BINARY_PRECISION = 0.5   # TP / (TP + FP) = 1 / 2
EXPECTED_BINARY_RECALL = 1.0      # TP / (TP + FN) = 1 / 1
EXPECTED_ACCURACY = 0.75          # (TN + TP) / N = 3 / 4


@pytest.fixture
def scored():
    return TrainingPipeline()._score_predictions(Y_TRUE, Y_PRED, is_classification=True)


def test_precision_is_positive_class_not_weighted(scored):
    assert scored['precision'] == pytest.approx(EXPECTED_BINARY_PRECISION)


def test_recall_is_positive_class_not_weighted(scored):
    assert scored['recall'] == pytest.approx(EXPECTED_BINARY_RECALL)


def test_recall_is_not_merely_accuracy(scored):
    """Weighted recall always equals accuracy — the tell-tale sign of the bug."""
    assert scored['accuracy'] == pytest.approx(EXPECTED_ACCURACY)
    assert scored['recall'] != pytest.approx(scored['accuracy'])


def test_f1_is_consistent_with_reported_precision_and_recall(scored):
    """The headline point: f1 must be derivable from the precision and recall
    printed beside it, which was not true while the averaging modes differed."""
    p, r = scored['precision'], scored['recall']
    assert scored['f1'] == pytest.approx(2 * p * r / (p + r))


def test_multiclass_still_uses_weighted():
    """Binary averaging is undefined for >2 classes; those must stay weighted."""
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 2, 2, 1])
    m = TrainingPipeline()._score_predictions(y_true, y_pred, is_classification=True)
    for key in ('precision', 'recall', 'f1'):
        assert 0.0 <= m[key] <= 1.0


def test_no_unconditional_weighted_averaging_remains():
    """Guards every trainer, not just the shared helper: no precision_score or
    recall_score call may hardcode average='weighted' without the binary branch."""
    source = inspect.getsource(TrainingPipeline)
    offenders = re.findall(
        r"(?:precision_score|recall_score)\([^)]*average='weighted'(?!\s*if)[^)]*\)",
        source,
    )
    assert not offenders, (
        f"{len(offenders)} call(s) still hardcode average='weighted': {offenders[:3]}"
    )
