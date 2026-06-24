# LightGBM Training Stage — Design Spec

**Date:** 2026-06-24
**Stage:** New pipeline stage after `train_xgboost_with_randomized_search_cv`
**Files touched:** `src/training_pipeline.py`, `config/const_training.py`, `tests/test_train_lightgbm.py`

---

## Problem Being Solved

The pipeline currently has one gradient-boosted tree model (XGBoost). Adding LightGBM as a parallel stage enables direct comparison between the two frameworks on the same dataset and metric set. The new stage mirrors the XGBoost method exactly in structure and outputs but uses LightGBM-specific hyperparameters.

---

## Design

### Installation prerequisite

LightGBM is not installed in the project environment. The implementation plan includes:

```
pip install lightgbm
```

Add `import lightgbm as lgb` to `src/training_pipeline.py` alongside existing model imports.

### New config constants (`config/const_training.py`)

Add after `XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER`:

```python
LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/1001-lightgbm_randomized_search"
```

Add after `XGBOOST_PARAM_DISTRIBUTIONS`:

```python
LIGHTGBM_PARAM_DISTRIBUTIONS = {
    'n_estimators':      randint(100, 500),
    'num_leaves':        randint(20, 150),
    'learning_rate':     [0.01, 0.05, 0.1],
    'subsample':         [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree':  [0.6, 0.7, 0.8, 1.0],
    'min_child_samples': randint(10, 100),
    'reg_alpha':         [0.0, 0.1, 0.5, 1.0],
    'reg_lambda':        [0.0, 0.1, 0.5, 1.0],
}
```

Add to `TRAINING_STATE_MACHINE` immediately after `train_xgboost_with_randomized_search_cv`:

```python
"train_lightgbm_with_randomized_search_cv": True,
```

Reused from existing config (no new constants needed): `RANDOM_SEARCH_ITERATIONS`, `RANDOM_SEARCH_CV_FOLDS`, `SCORE_METRIC`, `DELAY_THRESHOLD_MINUTES`, `RANDOM_STATE`, `WEIGHT_DELAY_COLUMN`, `MAX_SAMPLE_WEIGHT_CLASSIFICATION`, `MAX_SAMPLE_WEIGHT_REGRESSION`.

### New method: `train_lightgbm_with_randomized_search_cv(self, data_dir=None)`

Added to `TrainingPipeline` immediately after `train_xgboost_with_randomized_search_cv`. The structure is identical to the XGBoost method with these substitutions:

#### Model construction

```python
# Classification
base_model = lgb.LGBMClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    subsample_freq=1,   # required for subsample to take effect in LightGBM sklearn API
    verbose=-1,         # suppress LightGBM's per-iteration stdout
)

# Regression
base_model = lgb.LGBMRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    subsample_freq=1,
    verbose=-1,
)
```

`verbose=-1` suppresses LightGBM's default per-iteration output that would clutter the pipeline log.

#### Hyperparameter search

Uses `LIGHTGBM_PARAM_DISTRIBUTIONS` instead of `XGBOOST_PARAM_DISTRIBUTIONS`. All other `RandomizedSearchCV` settings (`n_iter`, `scoring`, `cv`, `random_state`, `n_jobs`, `verbose`) are identical.

#### Metrics

Identical to XGBoost method:
- **Regression:** RMSE, MAE, R², WMAPE (`sum(|y-ŷ|)/sum(|y|)*100`, 0.0 on zero denominator); binary Precision, Recall, F1, Accuracy via `(y > DELAY_THRESHOLD_MINUTES).astype(int)`
- **Classification:** F1 (binary or weighted), Accuracy; final evaluation adds Precision, Recall, AUC

#### Output files

All written to `LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER`. Filenames use prefix `lightgbm_` instead of `xgboost_`:

- `lightgbm_iteration_analysis_{identifier}.png` — performance curve (RMSE or F1 vs n_iter, CV score on second axis)
- `lightgbm_feature_importance_{identifier}.png` — horizontal bar chart
- `lightgbm_feature_importance_{identifier}.csv` — importance scores
- `lightgbm_iteration_analysis_{identifier}.json` — full results dict
- `lightgbm_best_model_{identifier}.pkl` — serialized best model

#### Return dict

Same shape as XGBoost:
- On success: `{"success": True, "models_trained": 1, "problem_type": ..., "target_feature": ..., "cv_score": ..., + metric keys, "output_directory": ..., "results_file": ...}`
- On failure: `{"success": False, "error": ...}`

### State machine dispatcher

The existing dispatcher in `training_pipeline.py` already routes state machine keys to method calls. Adding the key to `TRAINING_STATE_MACHINE` is sufficient — the dispatcher calls `pipeline.train_lightgbm_with_randomized_search_cv()` automatically.

### Data directory routing

`data_dir=None` resolves to `MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER` inside the method, same as XGBoost. The existing state machine router passes `data_dir` overrides when upstream stages (balance, scale, select, filter) are enabled — LightGBM will receive the same `data_dir` as XGBoost via the dispatcher.

---

## Tests (`tests/test_train_lightgbm.py`)

Source-inspection tests using `inspect.getsource(TrainingPipeline.train_lightgbm_with_randomized_search_cv)`:

| Test | Asserts |
|------|---------|
| `test_lightgbm_output_folder_constant_exists` | `LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER` importable from `config.const_training` |
| `test_lightgbm_param_distributions_constant_exists` | `LIGHTGBM_PARAM_DISTRIBUTIONS` importable; has keys `num_leaves`, `min_child_samples`, `n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| `test_lightgbm_in_state_machine` | `'train_lightgbm_with_randomized_search_cv'` in `TRAINING_STATE_MACHINE` |
| `test_n_jobs_is_minus_one` | `'n_jobs=-1'` in source |
| `test_subsample_freq_set_in_constructor` | `'subsample_freq=1'` in source |
| `test_no_eval_metric_in_source` | `'eval_metric'` not in source |
| `test_random_state_uses_constant` | `'RANDOM_STATE'` in source; `'random_state=42'` not in source |
| `test_lightgbm_param_distributions_used` | `'LIGHTGBM_PARAM_DISTRIBUTIONS'` in source |
| `test_wmape_in_lightgbm_regression` | `'final_test_wmape'` in source |
| `test_binary_metrics_in_lightgbm_regression` | `'final_bin_f1'` in source |

---

## Out of Scope

- Refactoring the XGBoost method
- Early stopping (requires eval_set, not used here)
- GPU training
- Comparing XGBoost vs LightGBM results programmatically
- Any change to preprocessing or data-splitting stages
