"""hpc/grid_summary.py: the scenario x model table of a fetched grid.

The table used to live in the untracked roihu-station-run skill and read only
classification JSON (test_auc, f1_always_positive): a numeric grid crashed on
the first KeyError, and the ridge baseline's 999-* folder was never looked at.
"""
import csv
import json

import pytest

from hpc.grid_summary import collect_rows, main


def _regression(mae, skill):
    return {
        "problem_type": "regression",
        "dataset_info": {"features_count": 7, "holdout_samples": 100},
        "final_metrics": {"test_mae": mae + 1, "mae_skill": skill, "test_rmse": 16.0, "test_r2": 0.05},
        "holdout_metrics": {"mae": mae, "mae_skill": skill, "rmse": 23.6, "r2": 0.01,
                            "baseline_mae": 5.8},
    }


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_a_numeric_grid_reads_the_tree_models_and_the_ridge_baseline(tmp_path):
    s1 = tmp_path / "1 - ONLY OPERACIONAL FEATURES"
    _write(s1 / "1000-xgboost_randomized_search" / "xgboost_iteration_analysis_x.json",
           _regression(5.6, 0.04))
    _write(s1 / "999-ridge_baseline" / "ridge_baseline_x.json", _regression(6.4, -0.10))
    _write(tmp_path / "2 - B" / "1001-lightgbm_randomized_search"
           / "lightgbm_iteration_analysis_x.json", _regression(5.7, 0.02))

    rows = collect_rows(str(tmp_path))
    assert [(r["s"], r["model"]) for r in rows] == [(1, "ridge"), (1, "xgb"), (2, "lgbm")]
    assert rows[1]["h_mae"] == 5.6 and rows[1]["h_skill"] == 0.04

    assert main([str(tmp_path)]) == 0
    with open(tmp_path / "grid_summary.csv", encoding="utf-8") as fh:
        written = list(csv.DictReader(fh))
    assert len(written) == 3 and written[0]["kind"] == "numeric"


def test_a_binary_grid_still_works(tmp_path):
    _write(tmp_path / "1 - A" / "1000-xgboost_randomized_search" / "xgboost_iteration_analysis_x.json", {
        "dataset_info": {"features_count": 7},
        "final_metrics": {"test_auc": 0.7, "test_pr_auc": 0.4, "positive_rate_test": 0.2},
        "holdout_metrics": {"auc": 0.69, "pr_auc": 0.38, "positive_rate": 0.21, "precision": 0.3,
                            "recall": 0.6, "f1": 0.4, "f1_always_positive": 0.35},
    })
    rows = collect_rows(str(tmp_path))
    assert rows[0]["kind"] == "binary" and rows[0]["h_auc"] == 0.69


def test_a_mixed_or_empty_grid_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no results"):
        collect_rows(str(tmp_path))
    _write(tmp_path / "1 - A" / "999-ridge_baseline" / "ridge_baseline_x.json", _regression(6.0, 0.0))
    _write(tmp_path / "1 - A" / "1000-xgboost_randomized_search" / "xgboost_iteration_analysis_x.json", {
        "final_metrics": {"test_auc": 0.7, "test_pr_auc": 0.4, "positive_rate_test": 0.2},
        "holdout_metrics": {"auc": 0.69, "pr_auc": 0.38, "positive_rate": 0.21, "precision": 0.3,
                            "recall": 0.6, "f1": 0.4, "f1_always_positive": 0.35},
    })
    with pytest.raises(ValueError, match="mix"):
        collect_rows(str(tmp_path))
