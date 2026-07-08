# Random Forest Training Stage — Design Spec

**Date:** 2026-07-08
**Stage:** New pipeline stage after `train_lightgbm_with_randomized_search_cv` (last training stage)
**Files touched:** `src/training_pipeline.py`, `config/const_training.py`, `tests/test_train_random_forest.py`

---

## Problem Being Solved

The pipeline currently trains two gradient-boosted tree models (XGBoost, LightGBM). Adding Random Forest as a third parallel stage gives a bagging-based baseline for comparison on the same dataset and metric set. The new stage mirrors the LightGBM method (itself a mirror of XGBoost) exactly in structure and outputs, substituting `sklearn.ensemble.RandomForestClassifier` / `RandomForestRegressor`.

Unlike the XGBoost/LightGBM addition, most of the scaffolding for this stage already exists but is unused: `RANDOM_FOREST_PARAM_DISTRIBUTIONS` and `RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER` in `config/const_training.py`, and `self.random_forest_dir` in `TrainingPipeline.__init__`. This work wires that scaffolding up rather than introducing it.

---

## Design

### New config constants (`config/const_training.py`)

`RANDOM_FOREST_PARAM_DISTRIBUTIONS` and `RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER` already exist and require no changes.

`RANDOM_FOREST_PARAM_DISTRIBUTIONS` contains `class_weight` and `criterion: ['gini', 'entropy']`, both classifier-only in scikit-learn — passing them to `RandomForestRegressor` via `RandomizedSearchCV` raises `ValueError: Invalid parameter`. Add a second, regression-safe distribution immediately after it:

```python
# Parameter distributions for Random Forest regression (no class_weight; regression-valid criterion)
RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 15),
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, None],
    'bootstrap': [True, False],
    'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
    'min_impurity_decrease': [0.0, 0.001, 0.005, 0.01],
    'ccp_alpha': [0.0, 0.001, 0.01, 0.05]
}
```

Add to `TRAINING_STATE_MACHINE` immediately after `train_lightgbm_with_randomized_search_cv`, set to `True` (matches XGBoost/LightGBM defaults):

```python
"train_random_forest_with_randomized_search_cv": True,
```

Reused from existing config (no new constants needed): `RANDOM_SEARCH_ITERATIONS`, `RANDOM_SEARCH_CV_FOLDS`, `SCORE_METRIC`, `DELAY_THRESHOLD_MINUTES`, `RANDOM_STATE`, `WEIGHT_DELAY_COLUMN`, `MAX_SAMPLE_WEIGHT_CLASSIFICATION`, `MAX_SAMPLE_WEIGHT_REGRESSION`, `RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER`.

### Import changes (`src/training_pipeline.py`)

Add alongside existing model imports:

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
```

Add `RANDOM_FOREST_PARAM_DISTRIBUTIONS` and `RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION` to the existing `from config.const_training import (...)` block (`RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER` is already imported there).

### New method: `train_random_forest_with_randomized_search_cv(self, data_dir=None)`

Added to `TrainingPipeline` immediately after `train_lightgbm_with_randomized_search_cv`. Structure identical to the LightGBM method with these substitutions:

#### Model construction

```python
# Classification
base_model = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Regression
base_model = RandomForestRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
```

No LightGBM-style `verbose=-1`/`subsample_freq=1` equivalents are needed — `RandomForestClassifier`/`Regressor` don't emit per-iteration output and have no subsample-frequency concept.

#### Hyperparameter search

Uses `RANDOM_FOREST_PARAM_DISTRIBUTIONS` for classification, `RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION` for regression (selected via the same `is_classification` flag already computed for problem-type detection). All other `RandomizedSearchCV` settings (`n_iter`, `scoring`, `cv`, `random_state`, `n_jobs`, `verbose`) are identical to XGBoost/LightGBM.

#### Metrics

Identical to XGBoost/LightGBM:
- **Regression:** RMSE, MAE, R², WMAPE (`sum(|y-ŷ|)/sum(|y|)*100`, 0.0 on zero denominator); binary Precision, Recall, F1, Accuracy via `(y > DELAY_THRESHOLD_MINUTES).astype(int)`
- **Classification:** F1 (binary or weighted), Accuracy; final evaluation adds Precision, Recall, AUC

#### Output files

All written to `RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER`. Filenames use prefix `random_forest_`:

- `random_forest_iteration_analysis_{identifier}.png` — performance curve (RMSE or F1 vs n_iter, CV score on second axis)
- `random_forest_feature_importance_{identifier}.png` — horizontal bar chart
- `random_forest_feature_importance_{identifier}.csv` — importance scores
- `random_forest_iteration_analysis_{identifier}.json` — full results dict
- `random_forest_best_model_{identifier}.pkl` — serialized best model

#### Return dict

Same shape as XGBoost/LightGBM:
- On success: `{"success": True, "models_trained": 1, "problem_type": ..., "target_feature": ..., "cv_score": ..., + metric keys, "output_directory": ..., "results_file": ...}`
- On failure: `{"success": False, "error": ...}`

### State machine dispatcher (`execute_training_pipeline_steps`)

Add a new block immediately after the LightGBM block (before `return result`), copying the same `_use_scaled` / `_use_filtered` / `_data_folder` routing logic verbatim from the LightGBM block, calling `self.train_random_forest_with_randomized_search_cv(data_dir=...)`, and populating `result["file_info"]["random_forest_models_trained"]` / `result["file_info"]["random_forest_problem_type"]` the same way LightGBM populates its keys. Same print statements, same success/error handling, same `result["success"] = True` / early-return-on-error pattern.

### Data directory routing

`data_dir=None` resolves to `MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER` inside the method, same as XGBoost/LightGBM. The dispatcher passes the same computed `data_dir` that XGBoost/LightGBM receive (routing depends only on which upstream stages are enabled, not on which model consumes the data).

---

## Tests (`tests/test_train_random_forest.py`)

Source-inspection tests using `inspect.getsource(TrainingPipeline.train_random_forest_with_randomized_search_cv)`, mirroring `tests/test_train_lightgbm.py`:

| Test | Asserts |
|------|---------|
| `test_random_forest_output_folder_constant_exists` | `RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER` importable from `config.const_training` |
| `test_random_forest_param_distributions_constant_exists` | `RANDOM_FOREST_PARAM_DISTRIBUTIONS` importable; has keys `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`, `criterion`, `class_weight`, `ccp_alpha` |
| `test_random_forest_param_distributions_regression_constant_exists` | `RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION` importable; does NOT have `class_weight` key; `criterion` values are regression-valid (`squared_error`, `absolute_error`, `friedman_mse`) |
| `test_random_forest_in_state_machine` | `'train_random_forest_with_randomized_search_cv'` in `TRAINING_STATE_MACHINE` |
| `test_n_jobs_is_minus_one` | `'n_jobs=-1'` in source |
| `test_random_state_uses_constant` | `'RANDOM_STATE'` in source; `'random_state=42'` not in source |
| `test_classification_uses_classifier_distributions` | `'RANDOM_FOREST_PARAM_DISTRIBUTIONS'` in source, used conditionally on `is_classification` |
| `test_regression_uses_regression_distributions` | `'RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION'` in source |
| `test_wmape_in_random_forest_regression` | `'final_test_wmape'` in source |
| `test_binary_metrics_in_random_forest_regression` | `'final_bin_f1'` in source |
| `test_dispatcher_routes_random_forest_after_lightgbm` | in `training_pipeline.py` source, the `train_random_forest_with_randomized_search_cv` dispatcher block appears after the `train_lightgbm_with_randomized_search_cv` block |

---

## Out of Scope

- Refactoring the XGBoost or LightGBM methods
- Early stopping / OOB scoring (`oob_score`) as an alternative to CV
- Comparing XGBoost vs LightGBM vs Random Forest results programmatically
- Any change to preprocessing or data-splitting stages
- GPU training (not applicable to scikit-learn's RandomForest implementation)
