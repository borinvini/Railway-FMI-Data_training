# XGBoost Training Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up `train_xgboost_with_randomized_search_cv` by enabling CPU parallelism, removing dead config, fixing an ambiguous variable name, and replacing `random_state=42` literals with the `RANDOM_STATE` constant.

**Architecture:** All changes are inside one method in `src/training_pipeline.py` plus one comment change in `config/const_training.py`. No new functions, no interface changes, no new imports needed. Tests use `inspect.getsource` to assert the correct patterns are in place — the only way to test cleanup changes without running the full training loop.

**Tech Stack:** Python, XGBoost, scikit-learn, pytest

## Global Constraints

- `n_jobs=-1` inline — no new constant
- `RANDOM_STATE` is already imported in `src/training_pipeline.py` at line 96 — do not add a new import
- All changes are confined to `train_xgboost_with_randomized_search_cv` (~lines 2201–2600) and `XGBOOST_PARAM_DISTRIBUTIONS` (~line 120) in the config
- Do not change any logic, scoring, iteration values, CV folds, or output format
- All tests go in `tests/test_train_xgboost.py` (new file)

---

### Task 1: Enable parallelism, remove dead config, fix naming, replace literals

**Files:**
- Modify: `src/training_pipeline.py` (lines 2324–2379, 2360, 2366, 2414–2415)
- Modify: `config/const_training.py` (line 120)
- Create: `tests/test_train_xgboost.py`

**Interfaces:**
- Consumes: `RANDOM_STATE` already imported at line 96 of `src/training_pipeline.py`; `XGBOOST_PARAM_DISTRIBUTIONS` dict in `config/const_training.py`
- Produces: no interface changes — all changes are internal to the method

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_xgboost.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_train_xgboost.py -v
```

Expected: `test_n_jobs_is_minus_one`, `test_eval_metric_removed_from_classifier`, `test_eval_metric_removed_from_regressor`, `test_best_cv_score_replaces_best_metric`, and `test_random_state_literal_replaced_with_constant` all FAIL. `test_scale_pos_weight_not_in_param_distributions` will PASS (the commented-out key is already absent from the dict) — that is fine; it pins the correct state after the comment is cleaned up.

- [ ] **Step 3: Change `n_jobs=1` → `n_jobs=-1` and remove `eval_metric` from both model constructors**

In `src/training_pipeline.py`, find the block starting around line 2324. Replace:

```python
            if is_classification:
                cv_splitter = StratifiedKFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=42)
                base_model = xgb.XGBClassifier(
                    random_state=42,
                    n_jobs=1,
                    eval_metric='logloss'
                )
                scoring_metric = SCORE_METRIC
            else:
                cv_splitter = KFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=42)
                base_model = xgb.XGBRegressor(
                    random_state=42,
                    n_jobs=1,
                    eval_metric='mae'
                )
```

with:

```python
            if is_classification:
                cv_splitter = StratifiedKFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                base_model = xgb.XGBClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                scoring_metric = SCORE_METRIC
            else:
                cv_splitter = KFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                base_model = xgb.XGBRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
```

- [ ] **Step 4: Rename `best_metric` → `best_cv_score`**

Find line ~2360:
```python
            best_metric = -np.inf  # CV score: higher is always better (F1 or neg_MAE)
```
Change to:
```python
            best_cv_score = -np.inf  # CV score: higher is always better (F1 or neg_MAE)
```

Find lines ~2414–2415:
```python
                if current_cv_score > best_metric:
                    best_metric = current_cv_score
```
Change to:
```python
                if current_cv_score > best_cv_score:
                    best_cv_score = current_cv_score
```

- [ ] **Step 5: Replace `random_state=42` with `random_state=RANDOM_STATE` in `RandomizedSearchCV`**

Find the `RandomizedSearchCV(...)` call inside the loop (~line 2370). Replace:

```python
                randomized_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=XGBOOST_PARAM_DISTRIBUTIONS,
                    n_iter=n_iter,
                    scoring=scoring_metric,
                    cv=cv_splitter,
                    random_state=42,
                    n_jobs=1,
                    verbose=0  # Reduced verbosity for cleaner output
                )
```

with:

```python
                randomized_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=XGBOOST_PARAM_DISTRIBUTIONS,
                    n_iter=n_iter,
                    scoring=scoring_metric,
                    cv=cv_splitter,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=0
                )
```

- [ ] **Step 6: Add clarifying comment before the iteration loop**

Find the line (~2366):
```python
            for i, n_iter in enumerate(iteration_values):
```

Insert two lines immediately before it:
```python
            # Each run is an independent random draw — not cumulative. The curve shows
            # search-budget sensitivity, not convergence toward a global optimum.
            for i, n_iter in enumerate(iteration_values):
```

- [ ] **Step 7: Clean up the `scale_pos_weight` comment in config**

In `config/const_training.py`, find line ~120:
```python
    #'scale_pos_weight': [3.9, 4.9, 5.9]
```

Replace with:
```python
    # scale_pos_weight omitted: class imbalance handled upstream by balance_classes (SMOTE-Tomek)
```

- [ ] **Step 8: Run the new tests to verify they pass**

```
pytest tests/test_train_xgboost.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 9: Run the full test suite to check for regressions**

```
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 10: Commit**

```
git add src/training_pipeline.py config/const_training.py tests/test_train_xgboost.py
git commit -m "fix: enable n_jobs=-1, remove dead eval_metric, rename best_cv_score, use RANDOM_STATE in xgboost training"
```

---

## Self-Review

**Spec coverage:**
- Issue 1 (n_jobs=1) → Steps 3 and 5 (`n_jobs=-1` on model constructors and `RandomizedSearchCV`)
- Issue 2 (clarifying comment) → Step 6
- Issue 3 (dead eval_metric) → Step 3
- Issue 4 (scale_pos_weight comment) → Step 7
- Issue 5 (best_metric rename) → Step 4
- Bonus (random_state literals) → Steps 3 and 5

**Placeholder scan:** None. All code blocks are complete and exact.

**Type consistency:** `RANDOM_STATE` is `int`, used as `random_state=RANDOM_STATE` in all places — matches the existing `random_state` parameter type for scikit-learn and XGBoost. `best_cv_score` is `float`, initialized to `-np.inf`, compared with `>` — same semantics as `best_metric`.
