"""The 4-class target (`delayClass`) runs through XGBoost without touching the binary or numeric runs.

Several pieces of the tabular trainers only make sense for two classes: `scale_pos_weight` (one
positive class; `neg_pos_ratio` even casts the labels to bool, so classes 1-3 would all count as
"positive"), ROC-AUC on the second probability column, `average='binary'`, and the `roc_auc` search
metric, which raises on four classes. For a target in MULTICLASS_PROBLEM the XGBoost trainer instead
balances the classes with sample weights, tunes `f1_macro`, reports macro metrics, one-vs-rest AUC,
per-class precision/recall/F1 and a 4x4 confusion matrix, and keeps the delay tail at stage 501
(the longest delays are the top class). Trainers and diagnostics that have no multiclass branch yet
refuse or skip with a message, rather than run on a label they would misread. The kind is chosen by
the target alone (`cv_utils.target_kind`), so the binary and numeric runs are unchanged.
"""
import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RandomizedSearchCV

from config.const_preprocessing import DELAY_CLASS_NAMES, GROUP_KEY_COLUMN
from config.const_training import (LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER, MULTICLASS_SCORE_METRIC,
                                   RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
                                   REGULARIZED_REGRESSION_OUTPUT_FOLDER, SCORE_METRIC,
                                   XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
from src.cv_utils import (balanced_sample_weights, classification_average, classification_scoring,
                          is_multiclass, target_kind)
from src.training_pipeline import TrainingPipeline
from tests.test_grouped_cv import _pipeline

TARGET = "delayClass"
K = len(DELAY_CLASS_NAMES)


# --- the rule ------------------------------------------------------------------------------------

def test_the_target_alone_decides_the_kind():
    assert target_kind("trainDisrupted") == "binary"
    assert target_kind("trainDelayed") == "binary"
    assert target_kind("delayClass") == "multiclass"
    assert target_kind("differenceInMinutes") == "regression"
    with pytest.raises(ValueError):
        target_kind("not_a_target")
    assert is_multiclass("delayClass") and not is_multiclass("trainDisrupted")
    assert not is_multiclass("differenceInMinutes")


def test_scoring_and_averaging_follow_the_kind():
    assert classification_scoring("binary") == SCORE_METRIC
    assert classification_scoring("multiclass") == MULTICLASS_SCORE_METRIC == "f1_macro"
    assert classification_average([0, 1, 0, 1]) == "binary"            # unchanged binary rule
    assert classification_average([0, 1, 2, 3]) == "weighted"          # unchanged fallback
    assert classification_average([0, 1, 2, 3], multiclass=True) == "macro"


def test_balanced_weights_make_every_class_weigh_the_same():
    y = np.array([0] * 70 + [1] * 20 + [2] * 8 + [3] * 2)
    w = balanced_sample_weights(y)
    totals = [w[y == c].sum() for c in range(4)]
    assert totals == pytest.approx([totals[0]] * 4)
    assert w[y == 3][0] > w[y == 0][0]                                  # the rare class counts more


# --- the metric helpers -----------------------------------------------------------------------------

def test_perfect_predictions_score_one_and_the_matrix_is_4x4():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    out = TrainingPipeline._score_multiclass(y, y)
    assert out["f1"] == out["precision"] == out["recall"] == out["accuracy"] == 1.0
    assert out["balanced_accuracy"] == 1.0 and out["class_mae"] == 0.0
    assert out["within_one_class_accuracy"] == 1.0
    assert out["confusion_matrix"] == (2 * np.eye(4, dtype=int)).tolist()
    assert out["average"] == "macro" and out["class_names"] == list(DELAY_CLASS_NAMES)


def test_a_missing_class_still_gives_a_4x4_matrix_and_per_class_block():
    y_true = np.array([0, 0, 1, 1, 2, 2])                               # no class 3 in the truth
    out = TrainingPipeline._score_multiclass(y_true, np.array([0, 1, 1, 1, 2, 3]))
    assert np.array(out["confusion_matrix"]).shape == (4, 4)
    assert out["per_class"][DELAY_CLASS_NAMES[3]]["support"] == 0
    assert set(out["per_class"]) == set(DELAY_CLASS_NAMES)


def test_the_ordinal_measures_see_how_far_off_a_miss_is():
    y_true = np.array([0, 0, 0, 0])
    out = TrainingPipeline._score_multiclass(y_true, np.array([0, 1, 2, 3]))
    assert out["class_mae"] == pytest.approx((0 + 1 + 2 + 3) / 4)
    assert out["within_one_class_accuracy"] == pytest.approx(0.5)


def test_the_majority_reference_is_the_macro_f1_of_one_constant_class():
    y_true = np.array([0] * 6 + [1] * 2 + [2] * 1 + [3] * 1)
    out = TrainingPipeline._score_multiclass(y_true, y_true)
    p = 0.6
    assert out["f1_macro_majority"] == pytest.approx((2 * p / (1 + p)) / 4)   # one class has F1, three have 0
    assert out["class_shares"][DELAY_CLASS_NAMES[0]] == pytest.approx(p)


class _FixedProba:
    classes_ = np.arange(4)

    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        return self.proba


def test_probability_metrics_are_one_vs_rest_and_skip_an_absent_class():
    y = np.array([0, 1, 2, 0, 1, 2])                                    # class 3 never occurs
    proba = np.eye(4)[[0, 1, 2, 0, 1, 2]] * 0.7 + 0.075
    out = TrainingPipeline._probability_metrics(_FixedProba(proba), None, y)
    assert out["auc"] == pytest.approx(1.0) and out["pr_auc"] == pytest.approx(1.0)
    assert out["auc_per_class"][DELAY_CLASS_NAMES[3]] is None            # undefined, left out of the mean
    assert "positive_rate" not in out and "f1_always_positive" not in out


def test_the_binary_probability_metrics_are_untouched():
    class Two:
        def predict_proba(self, X):
            return np.column_stack([1 - np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.1, 0.2, 0.8, 0.9])])
    out = TrainingPipeline._probability_metrics(Two(), None, np.array([0, 0, 1, 1]))
    assert out["auc"] == pytest.approx(1.0)
    assert set(out) == {"auc", "pr_auc", "positive_rate", "f1_always_positive"}


# --- stage 501 --------------------------------------------------------------------------------------------

def _delays(n=400):
    rng = np.random.default_rng(0)
    delay = np.concatenate([rng.uniform(-3, 10, n - 1), [900.0]])      # one extreme delay, above the 99.5% quantile
    return pd.DataFrame({"differenceInMinutes": delay, "feature_a": rng.normal(size=n)})


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/outlier_filtered.parquet")
def test_stage_501_keeps_the_delay_tail_for_the_class_target(_save, tmp_path):
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "delayClass"):
        out = _pipeline(tmp_path).filter_delay_outliers(data=_delays())
    assert out["success"] is True
    assert len(out["data"]) == 400 and out["rows_removed_upper"] == 0
    assert out["data"]["differenceInMinutes"].max() == 900.0


@patch("src.training_pipeline.save_dataframe_to_parquet", return_value="/fake/outlier_filtered.parquet")
def test_stage_501_still_cuts_the_tail_for_the_binary_target(_save, tmp_path):
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDisrupted"):
        out = _pipeline(tmp_path).filter_delay_outliers(data=_delays())
    assert len(out["data"]) < 400 and out["rows_removed_upper"] > 0


# --- the XGBoost run -----------------------------------------------------------------------------------------

def _four_class_split(tmp_path):
    """Split folder with a 4-class label that f2 mostly determines, plus a hold-out copy of the test rows."""
    rng = np.random.default_rng(5)
    n_days, per_day = 80, 8
    days = pd.date_range("2023-01-01", periods=n_days).strftime("%Y-%m-%d")
    n = n_days * per_day
    f2 = rng.normal(size=n)
    score = f2 + rng.normal(scale=0.6, size=n)
    label = np.digitize(score, [-0.6, 0.4, 1.2])                          # four unequal classes
    df = pd.DataFrame({GROUP_KEY_COLUMN: np.repeat(days, per_day), "f1": np.arange(n, dtype=float),
                       "f2": f2, TARGET: label.astype("int8")})
    folder = tmp_path / "scaled"
    _pipeline(tmp_path).write_splits(df, output_dir=str(folder), identifier="t", basename="merged_data_x")
    pd.read_parquet(folder / "merged_data_x_test.parquet").to_parquet(
        folder / "merged_data_x_holdout.parquet", index=False)
    return folder


class _SpySearch(RandomizedSearchCV):
    """RandomizedSearchCV that records how it was built and fitted."""
    seen = []

    def fit(self, X, y=None, **fit_params):
        _SpySearch.seen.append({"scoring": self.scoring, "estimator": self.estimator,
                                "sample_weight": fit_params.get("sample_weight")})
        return super().fit(X, y, **fit_params)


@pytest.fixture(scope="module")
def xgb_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("multiclass")
    folder = _four_class_split(tmp_path)
    _SpySearch.seen = []
    with patch("src.training_pipeline.SEARCH_N_JOBS", 1), \
            patch("src.training_pipeline.RANDOM_SEARCH_ITERATIONS", 10), \
            patch("src.training_pipeline.RandomizedSearchCV", _SpySearch), \
            patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET):
        result = _pipeline(tmp_path).train_xgboost_with_randomized_search_cv(data_dir=str(folder))
    path = tmp_path / XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER / "xgboost_iteration_analysis_x.json"
    return result, json.loads(path.read_text()), tmp_path


def test_the_search_is_tuned_on_macro_f1_with_balanced_weights_and_no_scale_pos_weight(xgb_run):
    result, _, _ = xgb_run
    assert result["success"] is True, result
    seen = _SpySearch.seen[0]
    assert seen["scoring"] == "f1_macro"
    assert seen["estimator"].get_params().get("scale_pos_weight") is None
    w = seen["sample_weight"]
    assert w is not None and w.max() > w.min()                          # class-balancing weights were passed


def test_the_results_json_carries_the_multiclass_metrics(xgb_run):
    _, results, _ = xgb_run
    final = results["final_metrics"]
    assert results["problem_type"] == "classification" and results["target_type"] == "multiclass"
    assert final["average"] == "macro" and final["class_names"] == list(DELAY_CLASS_NAMES)
    assert np.array(final["confusion_matrix"]).shape == (4, 4)
    assert set(final["per_class"]) == set(DELAY_CLASS_NAMES)
    for key in ("test_f1", "test_precision", "test_recall", "test_accuracy", "test_auc", "test_pr_auc",
                "balanced_accuracy", "class_mae", "within_one_class_accuracy", "f1_macro_majority",
                "class_shares", "auc_per_class", "pr_auc_per_class"):
        assert key in final, key
    assert "positive_rate_test" not in final and "f1_always_positive" not in final   # binary-only references
    assert 0.5 < final["test_auc"] <= 1.0


def test_f2_carries_the_signal_so_the_model_beats_the_majority_reference(xgb_run):
    _, results, _ = xgb_run
    final, hold = results["final_metrics"], results["holdout_metrics"]
    assert final["test_f1"] > final["f1_macro_majority"]
    assert hold["f1"] > hold["f1_macro_majority"]
    assert np.array(hold["confusion_matrix"]).shape == (4, 4)
    assert sum(map(sum, final["confusion_matrix"])) == sum(c["support"] for c in final["per_class"].values())


def test_every_iteration_row_keeps_the_keys_the_result_readers_use(xgb_run):
    _, results, _ = xgb_run
    for row in results["iteration_analysis"]:
        for key in ("n_iter", "cv_score", "test_metric", "test_accuracy", "test_precision",
                    "test_recall", "test_confusion_matrix", "test_auc", "test_pr_auc"):
            assert row[key] is not None, key
    for row in results["holdout_iteration_analysis"]:
        for key in ("accuracy", "f1", "precision", "recall", "confusion_matrix", "auc", "pr_auc"):
            assert key in row, key
    assert results["iteration_metrics_summary"]["best_f1"] >= results["iteration_metrics_summary"]["average_f1"]


def test_the_confusion_matrix_plot_uses_the_class_names(xgb_run):
    _, _, tmp_path = xgb_run
    folder = tmp_path / XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER
    assert (folder / "xgboost_confusion_matrix_x.png").exists()


# --- LightGBM, Random Forest and the logistic baseline ---------------------------------------------------------------

_TRAINERS = {
    "lightgbm": ("train_lightgbm_with_randomized_search_cv", LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
                 "lightgbm_iteration_analysis_x.json"),
    "random_forest": ("train_random_forest_with_randomized_search_cv",
                      RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER, "random_forest_iteration_analysis_x.json"),
    "logistic_regression": ("train_logistic_regression_with_randomized_search_cv",
                            REGULARIZED_REGRESSION_OUTPUT_FOLDER, "logistic_regression_iteration_analysis_x.json"),
}


@pytest.fixture(scope="module", params=sorted(_TRAINERS))
def other_run(request, tmp_path_factory):
    method, folder_name, json_name = _TRAINERS[request.param]
    tmp_path = tmp_path_factory.mktemp(f"multiclass_{request.param}")
    folder = _four_class_split(tmp_path)
    _SpySearch.seen = []
    with patch("src.training_pipeline.SEARCH_N_JOBS", 1), \
            patch("src.training_pipeline.RANDOM_SEARCH_ITERATIONS", 10), \
            patch("src.training_pipeline.RandomizedSearchCV", _SpySearch), \
            patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET):
        result = getattr(_pipeline(tmp_path), method)(data_dir=str(folder))
    results = json.loads((tmp_path / folder_name / json_name).read_text())
    return request.param, result, results, tmp_path / folder_name


def test_the_other_trainers_train_the_class_target(other_run):
    name, result, results, _ = other_run
    assert result["success"] is True, result
    assert results["problem_type"] == "classification" and results["target_type"] == "multiclass"
    assert results["target_feature"] == TARGET


def test_the_other_trainers_report_the_multiclass_metrics(other_run):
    name, _, results, _ = other_run
    final = results["final_metrics"]
    assert final["average"] == "macro" and final["class_names"] == list(DELAY_CLASS_NAMES)
    assert np.array(final["confusion_matrix"]).shape == (K, K)
    assert set(final["per_class"]) == set(DELAY_CLASS_NAMES)
    for key in ("test_f1", "test_auc", "test_pr_auc", "balanced_accuracy", "f1_macro_majority",
                "class_shares", "auc_per_class", "pr_auc_per_class"):
        assert key in final, key
    assert "positive_rate_test" not in final and "f1_always_positive" not in final
    assert final["test_f1"] > final["f1_macro_majority"], name        # f2 carries the signal
    hold = results["holdout_metrics"]
    assert np.array(hold["confusion_matrix"]).shape == (K, K)
    assert hold["f1"] > hold["f1_macro_majority"]


def test_the_other_trainers_tune_on_macro_f1(other_run):
    name, _, results, _ = other_run
    if name == "logistic_regression":
        assert results["tuning_metric"] == MULTICLASS_SCORE_METRIC
    else:
        assert _SpySearch.seen[0]["scoring"] == MULTICLASS_SCORE_METRIC


def test_lightgbm_balances_with_sample_weights_not_scale_pos_weight(other_run):
    name, _, _, _ = other_run
    if name != "lightgbm":
        pytest.skip("lightgbm only")
    seen = _SpySearch.seen[0]
    assert seen["estimator"].get_params().get("scale_pos_weight") in (None, 1.0)
    assert seen["sample_weight"] is not None and seen["sample_weight"].max() > seen["sample_weight"].min()


def test_random_forest_keeps_class_weight_balanced(other_run):
    name, _, _, _ = other_run
    if name != "random_forest":
        pytest.skip("random_forest only")
    assert _SpySearch.seen[0]["estimator"].get_params()["class_weight"] == "balanced"


def test_the_logistic_importance_is_the_mean_abs_coefficient_with_a_signed_column_per_class(other_run):
    name, _, _, folder = other_run
    if name != "logistic_regression":
        pytest.skip("logistic regression only")
    csv = pd.read_csv(folder / "logistic_regression_feature_importance_x.csv")
    assert "importance" in csv.columns and "coefficient" not in csv.columns
    assert {f"coef_{c}" for c in DELAY_CLASS_NAMES} <= set(csv.columns)
    assert (folder / "logistic_regression_confusion_matrix_x.png").exists()


def test_naive_bayes_still_refuses_the_class_target_with_a_message(tmp_path):
    folder = _four_class_split(tmp_path)
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET):
        result = _pipeline(tmp_path).train_naive_bayes_with_randomized_search_cv(data_dir=str(folder))
    assert result["success"] is False
    assert "multiclass" in result["error"] and "xgboost" in result["error"]


def test_the_guard_helper_is_silent_for_binary_and_numeric_targets():
    for target in ("trainDisrupted", "trainDelayed", "differenceInMinutes"):
        with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", target):
            assert TrainingPipeline._multiclass_unsupported("x") is None
            assert TrainingPipeline._multiclass_unsupported("x", skip=True) is None


def test_diagnostics_skip_with_success_and_fcll_refuses():
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET):
        skipped = TrainingPipeline._multiclass_unsupported("shap", skip=True)
        refused = TrainingPipeline._multiclass_unsupported("fcll")
    assert skipped == {"success": True, "skipped": True, "problem_type": "multiclass"}
    assert refused["success"] is False


def test_fcll_net_stages_refuse_the_class_target(tmp_path):
    pipeline = _pipeline(tmp_path)
    with patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", TARGET):
        built = pipeline.build_fcll_sequences(input_dir=str(tmp_path), output_dir=str(tmp_path / "seq"))
        trained = pipeline.train_fcll_net(data_dir=str(tmp_path / "seq"))
    assert built["success"] is False and trained["success"] is False
    assert not (tmp_path / "seq").exists()                                # refused before any work was done
