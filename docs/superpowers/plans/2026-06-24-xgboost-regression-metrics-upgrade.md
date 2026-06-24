# XGBoost Regression Metrics Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MAPE with WMAPE and add threshold-based binary classification metrics (Precision, Recall, F1, Accuracy) to the regression path of `train_xgboost_with_randomized_search_cv`.

**Architecture:** Add `DELAY_THRESHOLD_MINUTES = 5` to `config/const_training.py` and import it. In the method's regression branch, replace the broken MAPE formula with WMAPE (handles zero targets), and compute binary metrics by binarizing both `y_test` and `y_pred` at the threshold. Track all new metrics per iteration and in the final evaluation; update the JSON results dict, summary print, and return dict accordingly.

**Tech Stack:** Python, numpy, scikit-learn (already imported), pytest, inspect

## Global Constraints

- Only the regression branch (`not is_classification`) is modified — classification path is unchanged
- `DELAY_THRESHOLD_MINUTES = 5` is the threshold constant (delay > 5 min = late = 1, else 0)
- WMAPE formula: `sum(|y_true - y_pred|) / sum(|y_true|) * 100`; returns `0.0` when denominator is 0
- All binary sklearn metrics use `zero_division=0`
- No new imports needed — `precision_score`, `recall_score`, `f1_score`, `accuracy_score` already imported
- `DELAY_THRESHOLD_MINUTES` added to `config/const_training.py` after line 90 (`MAX_SAMPLE_WEIGHT_REGRESSION`)
- `DELAY_THRESHOLD_MINUTES` imported in `src/training_pipeline.py` at line ~116, after `SCHEMA_MISMATCH_STRATEGY`
- All tests in `tests/test_xgboost_regression_metrics.py`

---

### Task 1: WMAPE + binary metrics in regression path

**Files:**
- Modify: `config/const_training.py` (add constant after line 90)
- Modify: `src/training_pipeline.py` (import + 6 locations in the method)
- Create: `tests/test_xgboost_regression_metrics.py`

**Interfaces:**
- Consumes: `DELAY_THRESHOLD_MINUTES: int` from `config.const_training`; `TrainingPipeline.train_xgboost_with_randomized_search_cv` source
- Produces: updated regression return dict with keys `test_wmape`, `test_bin_precision`, `test_bin_recall`, `test_bin_f1`, `test_bin_accuracy`; `results["final_metrics"]` gains those keys plus `delay_threshold_minutes`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_xgboost_regression_metrics.py`:

```python
import inspect

import numpy as np

from src.training_pipeline import TrainingPipeline


def _src():
    return inspect.getsource(TrainingPipeline.train_xgboost_with_randomized_search_cv)


def test_delay_threshold_constant_exists():
    from config.const_training import DELAY_THRESHOLD_MINUTES
    assert DELAY_THRESHOLD_MINUTES == 5


def test_wmape_formula_correctness():
    y_true = np.array([10.0, 20.0, 0.0, 5.0])
    y_pred = np.array([12.0, 18.0, 2.0, 4.0])
    denom = np.sum(np.abs(y_true))
    wmape = (np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom > 0 else 0.0
    assert abs(wmape - 20.0) < 1e-6


def test_wmape_handles_zero_denominator():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    denom = np.sum(np.abs(y_true))
    wmape = (np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom > 0 else 0.0
    assert wmape == 0.0


def test_mape_removed_from_regression_path():
    src = _src()
    assert 'test_mape_scores' not in src
    assert 'final_test_mape' not in src


def test_wmape_present_in_regression_path():
    src = _src()
    assert 'test_wmape_scores' in src
    assert 'final_test_wmape' in src


def test_delay_threshold_used_in_source():
    assert 'DELAY_THRESHOLD_MINUTES' in _src()


def test_binary_metrics_computed_per_iteration():
    src = _src()
    assert 'test_bin_f1' in src
    assert 'test_bin_precision' in src
    assert 'test_bin_recall' in src
    assert 'test_bin_accuracy' in src


def test_binary_metrics_tracked_across_iterations():
    src = _src()
    assert 'test_bin_f1_scores' in src
    assert 'test_bin_precision_scores' in src
    assert 'test_bin_recall_scores' in src
    assert 'test_bin_accuracy_scores' in src


def test_final_binary_metrics_in_source():
    src = _src()
    assert 'final_bin_f1' in src
    assert 'final_bin_precision' in src
    assert 'final_bin_recall' in src
    assert 'final_bin_accuracy' in src
```

- [ ] **Step 2: Run tests to verify they fail (or pass for formula tests)**

```
pytest tests/test_xgboost_regression_metrics.py -v
```

Expected:
- `test_wmape_formula_correctness` and `test_wmape_handles_zero_denominator` — PASS (pure math, no code dependency)
- All others — FAIL (`ImportError` on `DELAY_THRESHOLD_MINUTES`; strings not in source)

- [ ] **Step 3: Add `DELAY_THRESHOLD_MINUTES` to `config/const_training.py`**

In `config/const_training.py`, after line 90 (`MAX_SAMPLE_WEIGHT_REGRESSION = 3.0 ...`), add:

```python
DELAY_THRESHOLD_MINUTES = 5  # delay > 5 min = late (1), else on time (0)
```

- [ ] **Step 4: Import `DELAY_THRESHOLD_MINUTES` in `src/training_pipeline.py`**

In the `from config.const_training import (` block, find:

```python
    SCHEMA_MISMATCH_STRATEGY,
)
```

Replace with:

```python
    SCHEMA_MISMATCH_STRATEGY,
    DELAY_THRESHOLD_MINUTES,
)
```

- [ ] **Step 5: Replace tracking list initializations**

In `src/training_pipeline.py`, find:

```python
            # Initialize tracking lists
            iteration_results = []
            test_f1_scores = []
            cv_scores = []
            test_mae_scores = []
            test_mape_scores = []
```

Replace with:

```python
            # Initialize tracking lists
            iteration_results = []
            test_f1_scores = []
            cv_scores = []
            test_mae_scores = []
            test_wmape_scores = []
            test_bin_precision_scores = []
            test_bin_recall_scores = []
            test_bin_f1_scores = []
            test_bin_accuracy_scores = []
```

- [ ] **Step 6: Replace the regression metrics block inside the iteration loop**

In `src/training_pipeline.py`, inside the `for i, n_iter in enumerate(iteration_values):` loop, find:

```python
                else:
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    test_mae = mean_absolute_error(y_test, y_pred)
                    test_r2 = r2_score(y_test, y_pred)
                    epsilon = 1e-8
                    test_mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), epsilon))) * 100

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, Test RMSE = {test_rmse:.4f}, Test R² = {test_r2:.4f}, Test MAE = {test_mae:.4f}, Test MAPE = {test_mape:.2f}%")
```

Replace with:

```python
                else:
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    test_mae = mean_absolute_error(y_test, y_pred)
                    test_r2 = r2_score(y_test, y_pred)
                    denom = np.sum(np.abs(y_test))
                    test_wmape = (np.sum(np.abs(y_test - y_pred)) / denom * 100) if denom > 0 else 0.0

                    y_test_binary = (y_test > DELAY_THRESHOLD_MINUTES).astype(int)
                    y_pred_binary = (y_pred > DELAY_THRESHOLD_MINUTES).astype(int)
                    test_bin_precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
                    test_bin_recall    = recall_score(y_test_binary, y_pred_binary, zero_division=0)
                    test_bin_f1        = f1_score(y_test_binary, y_pred_binary, zero_division=0)
                    test_bin_accuracy  = accuracy_score(y_test_binary, y_pred_binary)

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}, MAE = {test_mae:.4f}, WMAPE = {test_wmape:.2f}%, Bin F1 = {test_bin_f1:.4f}, Bin Acc = {test_bin_accuracy:.4f}")
```

- [ ] **Step 7: Replace the per-iteration score tracking**

In the same loop, find:

```python
                if is_classification:
                    test_f1_scores.append(test_f1)
                else:
                    test_f1_scores.append(test_rmse)
                    test_mae_scores.append(test_mae)
                    test_mape_scores.append(test_mape)
```

Replace with:

```python
                if is_classification:
                    test_f1_scores.append(test_f1)
                else:
                    test_f1_scores.append(test_rmse)
                    test_mae_scores.append(test_mae)
                    test_wmape_scores.append(test_wmape)
                    test_bin_precision_scores.append(test_bin_precision)
                    test_bin_recall_scores.append(test_bin_recall)
                    test_bin_f1_scores.append(test_bin_f1)
                    test_bin_accuracy_scores.append(test_bin_accuracy)
```

- [ ] **Step 8: Replace the `iteration_results.append(...)` call**

In the same loop, find:

```python
                # Store detailed results
                iteration_results.append({
                    'n_iter': n_iter,
                    'cv_score': current_cv_score,
                    'test_metric': test_f1_scores[-1],
                    'test_mae': test_mae_scores[-1] if not is_classification else None,
                    'test_mape': test_mape_scores[-1] if not is_classification else None,
                    'best_params': randomized_search.best_params_
                })
```

Replace with:

```python
                # Store detailed results
                iteration_results.append({
                    'n_iter': n_iter,
                    'cv_score': current_cv_score,
                    'test_metric': test_f1_scores[-1],
                    'test_mae': test_mae_scores[-1] if not is_classification else None,
                    'test_wmape': test_wmape_scores[-1] if not is_classification else None,
                    'test_bin_f1': test_bin_f1_scores[-1] if not is_classification else None,
                    'test_bin_precision': test_bin_precision_scores[-1] if not is_classification else None,
                    'test_bin_recall': test_bin_recall_scores[-1] if not is_classification else None,
                    'test_bin_accuracy': test_bin_accuracy_scores[-1] if not is_classification else None,
                    'best_params': randomized_search.best_params_
                })
```

- [ ] **Step 9: Replace the final evaluation regression block**

In `src/training_pipeline.py`, find:

```python
            else:  # regression
                final_test_rmse = np.sqrt(mean_squared_error(y_test, final_y_pred))
                final_test_mae = mean_absolute_error(y_test, final_y_pred)
                final_test_r2 = r2_score(y_test, final_y_pred)
                epsilon = 1e-8
                final_test_mape = np.mean(np.abs((y_test - final_y_pred) / np.maximum(np.abs(y_test), epsilon))) * 100
```

Replace with:

```python
            else:  # regression
                final_test_rmse = np.sqrt(mean_squared_error(y_test, final_y_pred))
                final_test_mae = mean_absolute_error(y_test, final_y_pred)
                final_test_r2 = r2_score(y_test, final_y_pred)
                denom = np.sum(np.abs(y_test))
                final_test_wmape = (np.sum(np.abs(y_test - final_y_pred)) / denom * 100) if denom > 0 else 0.0

                y_test_binary = (y_test > DELAY_THRESHOLD_MINUTES).astype(int)
                final_y_pred_binary = (final_y_pred > DELAY_THRESHOLD_MINUTES).astype(int)
                final_bin_precision = precision_score(y_test_binary, final_y_pred_binary, zero_division=0)
                final_bin_recall    = recall_score(y_test_binary, final_y_pred_binary, zero_division=0)
                final_bin_f1        = f1_score(y_test_binary, final_y_pred_binary, zero_division=0)
                final_bin_accuracy  = accuracy_score(y_test_binary, final_y_pred_binary)
```

- [ ] **Step 10: Replace `results["final_metrics"]` for regression**

Find:

```python
                else:
                    results["final_metrics"] = {
                        "test_rmse": float(final_test_rmse),
                        "test_mae": float(final_test_mae),          # ADD THIS
                        "test_mape": float(final_test_mape),        # ADD THIS
                        "test_r2": float(final_test_r2)
                    }
```

Replace with:

```python
                else:
                    results["final_metrics"] = {
                        "test_rmse": float(final_test_rmse),
                        "test_mae": float(final_test_mae),
                        "test_wmape": float(final_test_wmape),
                        "test_r2": float(final_test_r2),
                        "delay_threshold_minutes": DELAY_THRESHOLD_MINUTES,
                        "test_bin_precision": float(final_bin_precision),
                        "test_bin_recall": float(final_bin_recall),
                        "test_bin_f1": float(final_bin_f1),
                        "test_bin_accuracy": float(final_bin_accuracy)
                    }
```

- [ ] **Step 11: Replace `results["iteration_metrics_summary"]`**

Find:

```python
                # Add iteration-wise metrics summary
                if not is_classification:
                    results["iteration_metrics_summary"] = {
                        "rmse_values": [float(x) for x in test_f1_scores],
                        "mae_values": [float(x) for x in test_mae_scores],      # ADD THIS
                        "mape_values": [float(x) for x in test_mape_scores],    # ADD THIS
                        "cv_scores": [float(x) for x in cv_scores],
                        "best_rmse": float(min(test_f1_scores)),
                        "best_mae": float(min(test_mae_scores)),                # ADD THIS
                        "best_mape": float(min(test_mape_scores)),              # ADD THIS
                        "average_rmse": float(np.mean(test_f1_scores)),
                        "average_mae": float(np.mean(test_mae_scores)),         # ADD THIS
                        "average_mape": float(np.mean(test_mape_scores))        # ADD THIS
                    }
```

Replace with:

```python
                # Add iteration-wise metrics summary
                if not is_classification:
                    results["iteration_metrics_summary"] = {
                        "rmse_values": [float(x) for x in test_f1_scores],
                        "mae_values": [float(x) for x in test_mae_scores],
                        "wmape_values": [float(x) for x in test_wmape_scores],
                        "cv_scores": [float(x) for x in cv_scores],
                        "best_rmse": float(min(test_f1_scores)),
                        "best_mae": float(min(test_mae_scores)),
                        "best_wmape": float(min(test_wmape_scores)),
                        "average_rmse": float(np.mean(test_f1_scores)),
                        "average_mae": float(np.mean(test_mae_scores)),
                        "average_wmape": float(np.mean(test_wmape_scores)),
                        "bin_f1_values": [float(x) for x in test_bin_f1_scores],
                        "bin_accuracy_values": [float(x) for x in test_bin_accuracy_scores],
                        "bin_precision_values": [float(x) for x in test_bin_precision_scores],
                        "bin_recall_values": [float(x) for x in test_bin_recall_scores]
                    }
```

- [ ] **Step 12: Update the summary print block**

Find:

```python
            # Print summary
            print(f"      Training Summary:")
            print(f"        Problem Type: {problem_type}")
            print(f"        Best Iteration Count: {best_iteration}")
            print(f"        Best Test Score: {best_cv_score:.4f}")
            print(f"        Score Range: {min(test_f1_scores):.4f} - {max(test_f1_scores):.4f}")
```

Replace with:

```python
            # Print summary
            print(f"      Training Summary:")
            print(f"        Problem Type: {problem_type}")
            print(f"        Best Iteration Count: {best_iteration}")
            print(f"        Best CV Score: {best_cv_score:.4f}")
            print(f"        RMSE Range: {min(test_f1_scores):.4f} - {max(test_f1_scores):.4f}")
            if not is_classification:
                print(f"        Final RMSE: {final_test_rmse:.4f}  MAE: {final_test_mae:.4f}  R²: {final_test_r2:.4f}  WMAPE: {final_test_wmape:.2f}%")
                print(f"        Binary metrics (threshold > {DELAY_THRESHOLD_MINUTES} min):")
                print(f"          Precision: {final_bin_precision:.4f}  Recall: {final_bin_recall:.4f}  F1: {final_bin_f1:.4f}  Accuracy: {final_bin_accuracy:.4f}")
```

- [ ] **Step 13: Update the regression return dict**

Find:

```python
            else:
                return {
                    "success": True,
                    "models_trained": 1,
                    "problem_type": problem_type,
                    "target_feature": target_feature,
                    "cv_score": float(iteration_results[best_iteration_idx]["cv_score"]),
                    "test_rmse": float(final_test_rmse),
                    "test_mae": float(final_test_mae),         
                    "test_mape": float(final_test_mape),        
                    "test_r2": float(final_test_r2),
                    "output_directory": output_dir,
                    "results_file": results_file
                }
```

Replace with:

```python
            else:
                return {
                    "success": True,
                    "models_trained": 1,
                    "problem_type": problem_type,
                    "target_feature": target_feature,
                    "cv_score": float(iteration_results[best_iteration_idx]["cv_score"]),
                    "test_rmse": float(final_test_rmse),
                    "test_mae": float(final_test_mae),
                    "test_wmape": float(final_test_wmape),
                    "test_r2": float(final_test_r2),
                    "test_bin_precision": float(final_bin_precision),
                    "test_bin_recall": float(final_bin_recall),
                    "test_bin_f1": float(final_bin_f1),
                    "test_bin_accuracy": float(final_bin_accuracy),
                    "output_directory": output_dir,
                    "results_file": results_file
                }
```

- [ ] **Step 14: Run tests to verify they all pass**

```
pytest tests/test_xgboost_regression_metrics.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 15: Run full suite to check for regressions**

```
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 16: Commit**

```
git add config/const_training.py src/training_pipeline.py tests/test_xgboost_regression_metrics.py
git commit -m "feat: replace MAPE with WMAPE; add binary classification metrics to regression path"
```

---

## Self-Review

**Spec coverage:**
- `DELAY_THRESHOLD_MINUTES = 5` in const_training → Step 3
- Import in training_pipeline.py → Step 4
- WMAPE replaces MAPE in iteration loop → Step 6
- WMAPE tracking list (`test_wmape_scores`) → Step 5 + Step 7
- Binary metrics computed per iteration → Step 6
- Binary metric tracking lists → Step 5 + Step 7
- `iteration_results` dict updated → Step 8
- WMAPE replaces MAPE in final evaluation → Step 9
- Binary metrics in final evaluation → Step 9
- `results["final_metrics"]` updated → Step 10
- `results["iteration_metrics_summary"]` updated → Step 11
- Summary print updated → Step 12
- Return dict updated → Step 13
- Tests: constant, WMAPE formula, MAPE removed, WMAPE present, threshold used, binary metrics present × 3 levels → Step 1

**Placeholder scan:** None. All code blocks are complete and exact.

**Type consistency:** All new variables follow the existing `test_*` / `final_*` / `*_scores` naming pattern. `test_bin_f1_scores` is consistently referenced in Steps 5, 7, and 11. `final_bin_f1` is consistently referenced in Steps 9, 10, 12, and 13.
