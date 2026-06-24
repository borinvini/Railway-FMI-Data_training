# train_xgboost_with_randomized_search_cv — Cleanup and Performance Fixes

**Date:** 2026-06-24  
**Stage:** Stage 6 — `train_xgboost_with_randomized_search_cv`  
**Files touched:** `src/training_pipeline.py`, `config/const_training.py`

---

## Issues Being Solved

### Issue 1 — `n_jobs=1` causes unnecessarily slow training

Both the base model and `RandomizedSearchCV` are pinned to a single core with no explanation. With 5 iteration values × up to 50 parameter combinations × 5 CV folds, this runs up to 1,250 fits sequentially. Setting `n_jobs=-1` on both enables parallelism across all available CPU cores with no correctness impact.

### Issue 3 — `eval_metric` on base model has no effect

`XGBClassifier(eval_metric='logloss')` and `XGBRegressor(eval_metric='mae')` pass a parameter that only applies when `eval_set` and `early_stopping_rounds` are provided to `.fit()`. `RandomizedSearchCV` never passes these, so the parameter is dead configuration noise.

### Issue 4 — Commented-out `scale_pos_weight` with unexplained magic numbers

`#'scale_pos_weight': [3.9, 4.9, 5.9]` in `XGBOOST_PARAM_DISTRIBUTIONS` has no explanation of where 3.9/4.9/5.9 came from and why it is disabled. Since `balance_classes` (SMOTE-Tomek) handles class imbalance upstream, this parameter is not needed — that reasoning should be documented and the dead comment removed.

### Issue 5 — `best_metric` variable name is ambiguous

`best_metric` tracks the best CV score seen across all iteration runs, but the name doesn't convey what kind of metric or what direction. `best_cv_score` is unambiguous and consistent with `current_cv_score` used in the same loop.

### Bonus — Remaining `random_state=42` literals in this method

Five `random_state=42` literals remain in `train_xgboost_with_randomized_search_cv`: two in CV splitters (`StratifiedKFold`, `KFold`), two in base model constructors, and one in `RandomizedSearchCV`. `RANDOM_STATE` is already imported from `const_training.py` — replace all five for consistency.

### Issue 2 — Clarifying comment on search budget curve

The multi-iteration loop runs independent searches (each `n_iter` is a fresh random draw with `random_state=42`, not a superset of the previous). The resulting plot shows search-budget sensitivity, not convergence. A one-line comment before the loop makes this explicit.

---

## Design

### Issue 1 — Enable parallelism

In `src/training_pipeline.py`, change `n_jobs=1` to `n_jobs=-1` in three places:
- `XGBClassifier(...)` constructor (~line 2328)
- `XGBRegressor(...)` constructor (~line 2336)
- `RandomizedSearchCV(...)` call (~line 2377)

### Issue 3 — Remove dead `eval_metric` parameters

Remove `eval_metric='logloss'` from `XGBClassifier` and `eval_metric='mae'` from `XGBRegressor`.

### Issue 4 — Clean up `scale_pos_weight` in config

In `config/const_training.py`, replace:
```python
    #'scale_pos_weight': [3.9, 4.9, 5.9]
```
with:
```python
    # scale_pos_weight omitted: class imbalance handled upstream by balance_classes (SMOTE-Tomek)
```

### Issue 5 — Rename `best_metric` → `best_cv_score`

In `src/training_pipeline.py`, rename in three places:
- Initialization: `best_metric = -np.inf` → `best_cv_score = -np.inf`
- Comparison: `if current_cv_score > best_metric:` → `if current_cv_score > best_cv_score:`
- Update: `best_metric = current_cv_score` → `best_cv_score = current_cv_score`

### Bonus — Replace `random_state=42` literals with `RANDOM_STATE`

In `src/training_pipeline.py`, within `train_xgboost_with_randomized_search_cv`, replace all five occurrences of `random_state=42` with `random_state=RANDOM_STATE`. `RANDOM_STATE` is already imported.

### Issue 2 — Add clarifying comment

Before the `for i, n_iter in enumerate(iteration_values):` loop, add:
```python
# Each run is an independent random draw — not cumulative. The curve shows
# search-budget sensitivity, not convergence toward a global optimum.
```

---

## Out of Scope

- Adding `eval_set` / early stopping to the XGBoost fits.
- Restoring `scale_pos_weight` as a tunable parameter.
- The 12 remaining `random_state=42` literals elsewhere in `training_pipeline.py` (outside this method).
- Any changes to the iteration values, CV folds, or scoring metric.
