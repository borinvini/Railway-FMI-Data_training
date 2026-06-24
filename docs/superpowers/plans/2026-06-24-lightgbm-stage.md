# LightGBM Training Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `train_lightgbm_with_randomized_search_cv` as a new pipeline stage immediately after the XGBoost stage, mirroring its structure exactly but using LightGBM-specific models and hyperparameters.

**Architecture:** New standalone method in `TrainingPipeline` with its own config constants and dispatcher block. No changes to the existing XGBoost method. The dispatcher uses identical data-folder routing logic as the XGBoost dispatcher.

**Tech Stack:** Python, LightGBM (`lgb.LGBMClassifier` / `lgb.LGBMRegressor`), scikit-learn `RandomizedSearchCV`, existing sklearn metrics.

## Global Constraints

- `LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/1001-lightgbm_randomized_search"` — exact string
- `LIGHTGBM_PARAM_DISTRIBUTIONS` keys: `n_estimators`, `num_leaves`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_samples`, `reg_alpha`, `reg_lambda`
- Model constructor must include `n_jobs=-1`, `subsample_freq=1`, `verbose=-1` — no `eval_metric`
- `random_state=RANDOM_STATE` (constant, never literal `42`)
- All output filenames prefixed `lightgbm_` (not `xgboost_`)
- Metrics: identical to XGBoost — WMAPE for regression, binary metrics via `DELAY_THRESHOLD_MINUTES`
- Test file: `tests/test_train_lightgbm.py`
- Run tests: `conda run -n venv_rail_fmi_training python -m pytest tests/ -v` (75 tests currently passing — new tests bring total to 85)

---

### Task 1: Install LightGBM + Config Constants + Imports

**Files:**
- Install: `lightgbm` package in `venv_rail_fmi_training` conda environment
- Modify: `config/const_training.py`
- Modify: `src/training_pipeline.py` (import block only, lines 1–120)
- Create: `tests/test_train_lightgbm.py`

**Interfaces:**
- Produces: `LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER`, `LIGHTGBM_PARAM_DISTRIBUTIONS` (importable from `config.const_training`); `lgb` (importable from `src.training_pipeline` module level); state machine key `"train_lightgbm_with_randomized_search_cv"` in `TRAINING_STATE_MACHINE`

- [ ] **Step 1: Install lightgbm**

```bash
conda run -n venv_rail_fmi_training pip install lightgbm
```

Expected: `Successfully installed lightgbm-X.Y.Z` (version 4.x or higher)

Verify:
```bash
conda run -n venv_rail_fmi_training python -c "import lightgbm; print(lightgbm.__version__)"
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_train_lightgbm.py`:

```python
import inspect
from config.const_training import (
    LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    LIGHTGBM_PARAM_DISTRIBUTIONS,
    TRAINING_STATE_MACHINE,
)
from src.training_pipeline import TrainingPipeline


def test_lightgbm_output_folder_constant_exists():
    assert LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER == "data/output/1001-lightgbm_randomized_search"


def test_lightgbm_param_distributions_constant_exists():
    expected_keys = {
        'n_estimators', 'num_leaves', 'learning_rate', 'subsample',
        'colsample_bytree', 'min_child_samples', 'reg_alpha', 'reg_lambda'
    }
    assert set(LIGHTGBM_PARAM_DISTRIBUTIONS.keys()) == expected_keys


def test_lightgbm_in_state_machine():
    assert 'train_lightgbm_with_randomized_search_cv' in TRAINING_STATE_MACHINE


def _source():
    return inspect.getsource(TrainingPipeline.train_lightgbm_with_randomized_search_cv)


def test_n_jobs_is_minus_one():
    assert 'n_jobs=-1' in _source()


def test_subsample_freq_set_in_constructor():
    assert 'subsample_freq=1' in _source()


def test_no_eval_metric_in_source():
    assert 'eval_metric' not in _source()


def test_random_state_uses_constant():
    src = _source()
    assert 'RANDOM_STATE' in src
    assert 'random_state=42' not in src


def test_lightgbm_param_distributions_used():
    assert 'LIGHTGBM_PARAM_DISTRIBUTIONS' in _source()


def test_wmape_in_lightgbm_regression():
    assert 'final_test_wmape' in _source()


def test_binary_metrics_in_lightgbm_regression():
    assert 'final_bin_f1' in _source()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
conda run -n venv_rail_fmi_training python -m pytest tests/test_train_lightgbm.py -v
```

Expected: Multiple failures — `ImportError` or `AttributeError` (constants and method don't exist yet). The first three tests (`test_lightgbm_output_folder_constant_exists`, `test_lightgbm_param_distributions_constant_exists`, `test_lightgbm_in_state_machine`) will fail with `ImportError`. Tests 4-10 will fail with `AttributeError` (method doesn't exist).

- [ ] **Step 4: Add output folder constant to `config/const_training.py`**

Find line 51:
```python
XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/1000-xgboost_randomized_search"
```

Add immediately after it:
```python
LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/1001-lightgbm_randomized_search"
```

- [ ] **Step 5: Add param distributions to `config/const_training.py`**

Find the closing `}` of `XGBOOST_PARAM_DISTRIBUTIONS` (around line 123, the block ends with `}`). Add after it:

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

- [ ] **Step 6: Add state machine entry to `config/const_training.py`**

Find in `TRAINING_STATE_MACHINE`:
```python
    "train_xgboost_with_randomized_search_cv": True,
```

Add immediately after it:
```python
    "train_lightgbm_with_randomized_search_cv": True,
```

- [ ] **Step 7: Add `import lightgbm as lgb` to `src/training_pipeline.py`**

Find line 19:
```python
import xgboost as xgb
```

Add immediately after it:
```python
import lightgbm as lgb
```

- [ ] **Step 8: Add new constants to the import block in `src/training_pipeline.py`**

Find in the `from config.const_training import (` block (lines 81-118):
```python
    XGBOOST_PARAM_DISTRIBUTIONS,
```

Add immediately after it:
```python
    LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER,
    LIGHTGBM_PARAM_DISTRIBUTIONS,
```

- [ ] **Step 9: Run the first 3 tests to verify they now pass**

```bash
conda run -n venv_rail_fmi_training python -m pytest tests/test_train_lightgbm.py::test_lightgbm_output_folder_constant_exists tests/test_train_lightgbm.py::test_lightgbm_param_distributions_constant_exists tests/test_train_lightgbm.py::test_lightgbm_in_state_machine -v
```

Expected: 3 PASSED. Tests 4-10 still fail (`AttributeError: type object 'TrainingPipeline' has no attribute 'train_lightgbm_with_randomized_search_cv'`) — that's expected, Task 2 adds the method.

- [ ] **Step 10: Run the full existing test suite to verify no regressions**

```bash
conda run -n venv_rail_fmi_training python -m pytest tests/ -v --ignore=tests/test_train_lightgbm.py
```

Expected: 75 tests PASSED.

- [ ] **Step 11: Commit**

```bash
git add config/const_training.py src/training_pipeline.py tests/test_train_lightgbm.py
git commit -m "feat: add LightGBM config constants, state machine entry, and test stubs"
```

---

### Task 2: LightGBM Method + Dispatcher Block

**Files:**
- Modify: `src/training_pipeline.py` — insert method after line 2768 (end of `train_xgboost_with_randomized_search_cv`); insert dispatcher block after line 851 (end of XGBoost dispatcher `else` clause)
- Test: `tests/test_train_lightgbm.py` (all 10 tests must pass)

**Interfaces:**
- Consumes: `lgb`, `LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER`, `LIGHTGBM_PARAM_DISTRIBUTIONS`, `RANDOM_STATE`, `RANDOM_SEARCH_ITERATIONS`, `RANDOM_SEARCH_CV_FOLDS`, `SCORE_METRIC`, `DELAY_THRESHOLD_MINUTES`, `WEIGHT_DELAY_COLUMN`, `MAX_SAMPLE_WEIGHT_CLASSIFICATION`, `MAX_SAMPLE_WEIGHT_REGRESSION`, `MERGED_BALANCED_OUTPUT_FOLDER`, `SPLIT_DATASET_OUTPUT_FOLDER`, `MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER`, `MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER`, `MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER` — all already imported in Task 1
- Produces: `TrainingPipeline.train_lightgbm_with_randomized_search_cv(self, data_dir=None) -> dict`

- [ ] **Step 1: Add the method to `src/training_pipeline.py`**

Find the `############################` comment block that marks `## OLD METHODS` (line 2769). Insert the entire method **before** that comment (i.e., immediately after the closing `}` of `train_xgboost_with_randomized_search_cv` at line 2767):

```python
    def train_lightgbm_with_randomized_search_cv(self, data_dir=None):
        """
        LightGBM training method that tests different iteration counts and plots performance curve.

        This method:
        1. Tests RANDOM_SEARCH_ITERATIONS from 10 to 100 (step=10)
        2. Tracks test F1 binary scores for each iteration count
        3. Plots test_f1_binary vs n_iter curve
        4. Saves the best model and performance metrics
        """

        try:
            print(f"    train_lightgbm_with_randomized_search_cv: Starting LightGBM training with iteration analysis...")

            # Create output directory
            output_dir = os.path.join(self.project_root, LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)

            print(f"    train_lightgbm_with_randomized_search_cv: Output directory: {output_dir}")

            if data_dir is None:
                data_dir = os.path.join(self.project_root, MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER)
            scaled_data_dir = data_dir

            # Find training and test files in the specified location
            train_pattern = os.path.join(scaled_data_dir, "merged_data_*_train.parquet")
            test_pattern = os.path.join(scaled_data_dir, "merged_data_*_test.parquet")

            train_files = glob.glob(train_pattern)
            test_files = glob.glob(test_pattern)

            if not train_files or not test_files:
                error_msg = f"Training or test files not found in {scaled_data_dir}. Train files: {len(train_files)}, Test files: {len(test_files)}"
                print(f"    train_lightgbm_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            # Check if we have exactly one train/test file pair
            if len(train_files) != 1 or len(test_files) != 1:
                error_msg = f"Expected exactly one train and one test file. Found {len(train_files)} train files and {len(test_files)} test files"
                print(f"    train_lightgbm_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            train_file = train_files[0]
            test_file = test_files[0]

            # Extract identifier from filename
            train_filename = os.path.basename(train_file)
            identifier_match = re.search(r'merged_data_(.+?)_train\.parquet', train_filename)
            if not identifier_match:
                error_msg = f"Could not extract identifier from {train_filename}"
                print(f"    train_lightgbm_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            file_identifier = identifier_match.group(1)
            print(f"    train_lightgbm_with_randomized_search_cv: Processing dataset: {file_identifier}")

            # Determine problem type based on target feature
            target_feature = DEFAULT_TARGET_FEATURE
            is_classification = target_feature in CLASSIFICATION_PROBLEM
            is_regression = target_feature in REGRESSION_PROBLEM

            if not (is_classification or is_regression):
                error_msg = f"Target feature '{target_feature}' not recognized as classification or regression problem"
                print(f"    train_lightgbm_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            problem_type = "classification" if is_classification else "regression"
            print(f"    train_lightgbm_with_randomized_search_cv: Detected {problem_type} problem for target '{target_feature}'")

            # Load training and test data
            print(f"      Loading training data from {train_file}")
            train_df = pd.read_parquet(train_file)

            print(f"      Loading test data from {test_file}")
            test_df = pd.read_parquet(test_file)

            # Check if target feature exists
            if target_feature not in train_df.columns:
                error_msg = f"Target feature '{target_feature}' not found in dataset"
                print(f"    train_lightgbm_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            # Prepare features and target
            feature_columns = [col for col in train_df.columns if col != target_feature]

            X_train = train_df[feature_columns]
            y_train = train_df[target_feature]
            X_test = test_df[feature_columns]
            y_test = test_df[target_feature]

            print(f"      Dataset info - Train: {X_train.shape}, Test: {X_test.shape}")
            print(f"      Features: {len(feature_columns)}")

            # Calculate sample weights if applicable
            sample_weights = None
            if WEIGHT_DELAY_COLUMN in train_df.columns:
                print(f"      Calculating sample weights using '{WEIGHT_DELAY_COLUMN}'...")
                weights = train_df[WEIGHT_DELAY_COLUMN].values

                if is_classification:
                    weights = np.clip(weights, 1, MAX_SAMPLE_WEIGHT_CLASSIFICATION)
                else:
                    weights = np.clip(weights, 1, MAX_SAMPLE_WEIGHT_REGRESSION)

                sample_weights = weights
                print(f"      Sample weights - Min: {weights.min():.2f}, Max: {weights.max():.2f}, Mean: {weights.mean():.2f}")

            # Set up cross-validation strategy
            if is_classification:
                cv_splitter = StratifiedKFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                base_model = lgb.LGBMClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    subsample_freq=1,
                    verbose=-1,
                )
                scoring_metric = SCORE_METRIC
            else:
                cv_splitter = KFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                base_model = lgb.LGBMRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    subsample_freq=1,
                    verbose=-1,
                )
                scoring_metric = 'neg_mean_absolute_error'
                # Shift target so all values are positive, then log-transform to compress
                # the extreme right tail (max 963 min) that deflates R² artificially.
                y_shift = max(0.0, -float(y_train.min())) + 1.0
                y_train_log = np.log1p(y_train + y_shift)
                print(f"      Log-transform target: shift={y_shift:.2f}, "
                      f"log range=[{y_train_log.min():.3f}, {y_train_log.max():.3f}] "
                      f"(original [{y_train.min():.1f}, {y_train.max():.1f}])")

            # Define iteration range: 10 to 100 (step=10)
            iteration_values = list(range(10, RANDOM_SEARCH_ITERATIONS + 1, 10))
            print(f"      Testing iteration values: {iteration_values}")

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

            best_model = None
            best_cv_score = -np.inf
            best_iteration = None

            # Train models with different iteration counts
            print(f"      Starting training with different iteration counts...")

            # Each run is an independent random draw — not cumulative. The curve shows
            # search-budget sensitivity, not convergence toward a global optimum.
            for i, n_iter in enumerate(iteration_values):
                print(f"      Progress: {i+1}/{len(iteration_values)} - Testing {n_iter} iterations...")

                # Create RandomizedSearchCV with current iteration count
                randomized_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=LIGHTGBM_PARAM_DISTRIBUTIONS,
                    n_iter=n_iter,
                    scoring=scoring_metric,
                    cv=cv_splitter,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=0
                )

                # Fit with sample weights if available; use log-transformed y for regression
                y_fit = y_train_log if not is_classification else y_train
                if sample_weights is not None:
                    randomized_search.fit(X_train, y_fit, sample_weight=sample_weights)
                else:
                    randomized_search.fit(X_train, y_fit)

                # Get best model for current iteration count
                current_best_model = randomized_search.best_estimator_
                current_cv_score = randomized_search.best_score_

                # Make predictions on test set; inverse-transform for regression
                y_pred = current_best_model.predict(X_test)
                if not is_classification:
                    y_pred = np.expm1(y_pred) - y_shift

                # Calculate test metrics
                if is_classification:
                    test_f1 = f1_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
                    test_accuracy = accuracy_score(y_test, y_pred)

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, Test F1 = {test_f1:.4f}, Test Accuracy = {test_accuracy:.4f}")

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

                # Select best model on CV score (not on test set) to avoid test-set overfitting
                if current_cv_score > best_cv_score:
                    best_cv_score = current_cv_score
                    best_model = current_best_model
                    best_iteration = n_iter

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

                cv_scores.append(current_cv_score)

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

            # Calculate final metrics once on the best model found across all iterations
            print(f"      Calculating final metrics with best model (iteration {best_iteration})...")
            final_y_pred = best_model.predict(X_test)
            if not is_classification:
                final_y_pred = np.expm1(final_y_pred) - y_shift

            if is_classification:
                final_test_accuracy = accuracy_score(y_test, final_y_pred)
                final_test_f1 = f1_score(y_test, final_y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
                final_test_precision = precision_score(y_test, final_y_pred, average='weighted', zero_division=0)
                final_test_recall = recall_score(y_test, final_y_pred, average='weighted')

                if hasattr(best_model, 'predict_proba'):
                    final_y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                    final_test_auc = roc_auc_score(y_test, final_y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
                else:
                    final_test_auc = 0.0

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

            # Create performance curve plot
            print(f"      Creating performance curve plot...")

            fig, ax1 = plt.subplots(figsize=(14, 8))

            color1 = 'tab:blue'
            ax1.set_xlabel('Number of RandomizedSearch Iterations', fontsize=13)
            ax1.set_ylabel(f'Test {"F1 Binary Score" if is_classification else "RMSE"}', color=color1, fontsize=13)
            line1 = ax1.plot(iteration_values, test_f1_scores, 'o-', color=color1, linewidth=3, markersize=8,
                            label=f'Test {"F1 Binary" if is_classification else "RMSE"}')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            color2 = 'tab:green'
            ax2.set_ylabel('Cross-Validation Score', color=color2, fontsize=13)
            line2 = ax2.plot(iteration_values, cv_scores, 's-', color=color2, linewidth=3, markersize=8,
                            label='CV Score')
            ax2.tick_params(axis='y', labelcolor=color2)

            if best_iteration:
                best_idx = iteration_values.index(best_iteration)

                ax1.plot(best_iteration, test_f1_scores[best_idx], 'o', color='red', markersize=15,
                        markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)

                ax2.plot(best_iteration, cv_scores[best_idx], 's', color='red', markersize=15,
                        markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)

                ax1.annotate(f'Best at {best_iteration} iterations\nTest Score: {test_f1_scores[best_idx]:.4f}\nCV Score: {cv_scores[best_idx]:.4f}',
                            xy=(best_iteration, test_f1_scores[best_idx]),
                            xytext=(20, 30), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='orange'),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='red'),
                            fontsize=11, fontweight='bold')

            plt.title(f'LightGBM Performance vs RandomizedSearch Iterations\nDataset: {file_identifier}',
                    fontsize=16, fontweight='bold', pad=20)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.02, 0.98),
                    fontsize=12, framealpha=0.9)

            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'lightgbm_iteration_analysis_{file_identifier}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      Performance curve saved to: {plot_filename}")

            # CREATE AND SAVE FEATURE IMPORTANCE PLOT
            results = {}
            if best_model is not None:
                print(f"      Creating feature importance plot...")

                feature_importance = best_model.feature_importances_
                feature_names = feature_columns

                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=True)

                plt.figure(figsize=(12, max(8, len(feature_names) * 0.4)))

                bars = plt.barh(range(len(importance_df)), importance_df['importance'])

                plt.yticks(range(len(importance_df)), importance_df['feature'])
                plt.xlabel('Feature Importance', fontsize=12)
                plt.title(f'LightGBM Feature Importance - Best Model ({problem_type.title()})\n'
                        f'Dataset: {file_identifier} | Best Iteration: {best_iteration}', fontsize=14)

                for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
                    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                            f'{importance:.3f}',
                            ha='left', va='center', fontsize=10)

                normalized_importance = (importance_df['importance'] - importance_df['importance'].min()) / \
                                    (importance_df['importance'].max() - importance_df['importance'].min() + 1e-8)

                colors = plt.cm.RdYlGn(normalized_importance)
                for bar, color in zip(bars, colors):
                    bar.set_color(color)

                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()

                importance_plot_filename = os.path.join(output_dir, f'lightgbm_feature_importance_{file_identifier}.png')
                plt.savefig(importance_plot_filename, dpi=300, bbox_inches='tight')
                print(f"      Feature importance plot saved to: {importance_plot_filename}")
                plt.close()

                importance_csv_filename = os.path.join(output_dir, f'lightgbm_feature_importance_{file_identifier}.csv')
                importance_df.sort_values('importance', ascending=False).to_csv(importance_csv_filename, index=False)
                print(f"      Feature importance data saved to: {importance_csv_filename}")

                # Create comprehensive results dictionary
                results = {
                    "file_identifier": file_identifier,
                    "problem_type": problem_type,
                    "target_feature": target_feature,
                    "dataset_info": {
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                        "features_count": len(feature_columns),
                        "train_file": train_file,
                        "test_file": test_file
                    },
                    "best_iteration": best_iteration,
                    "iteration_analysis": iteration_results,
                    "best_model_params": best_model.get_params() if best_model else None,
                    "feature_importance": dict(zip(feature_columns, best_model.feature_importances_)) if best_model else None
                }

                # Add problem-specific metrics
                if is_classification:
                    results["final_metrics"] = {
                        "test_accuracy": float(final_test_accuracy),
                        "test_f1": float(final_test_f1),
                        "test_precision": float(final_test_precision),
                        "test_recall": float(final_test_recall),
                        "test_auc": float(final_test_auc)
                    }
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

            results_file = os.path.join(output_dir, f"lightgbm_iteration_analysis_{file_identifier}.json")
            with open(results_file, 'w') as f:
                results_str = json.loads(json.dumps(results, default=str))
                json.dump(results_str, f, indent=2)

            print(f"      Results saved to: {results_file}")

            # Save the best model
            if best_model:
                model_filename = os.path.join(output_dir, f'lightgbm_best_model_{file_identifier}.pkl')
                joblib.dump(best_model, model_filename)
                print(f"      Best model saved to: {model_filename}")

            # Find the index of the best iteration in the results list
            best_iteration_idx = iteration_values.index(best_iteration) if best_iteration else 0

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

            if is_classification:
                return {
                    "success": True,
                    "models_trained": 1,
                    "problem_type": problem_type,
                    "target_feature": target_feature,
                    "cv_score": float(iteration_results[best_iteration_idx]["cv_score"]),
                    "test_f1": float(final_test_f1),
                    "output_directory": output_dir,
                    "results_file": results_file
                }
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

        except Exception as e:
            error_msg = f"Error in train_lightgbm_with_randomized_search_cv: {str(e)}"
            print(f"    train_lightgbm_with_randomized_search_cv: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }
```

- [ ] **Step 2: Add the dispatcher block to `execute_training_pipeline_steps`**

Find the line (around line 851):
```python
        else:
            print(f"    ⊝ train_xgboost_with_randomized_search_cv (disabled)")
```

Add immediately after it (before the next `if state_machine.get("threshold_optimization_xgboost"` block):

```python
        if state_machine.get("train_lightgbm_with_randomized_search_cv", False):
            try:
                print(f"    → train_lightgbm_with_randomized_search_cv")
                _use_scaled = state_machine.get("scale_weather_features", False)
                _use_filtered = (
                    state_machine.get("filter_delay_outliers", False)
                    and not state_machine.get("select_training_cols", False)
                )
                _data_folder = (
                    MERGED_BALANCED_OUTPUT_FOLDER
                    if state_machine.get("balance_classes", False)
                    else SPLIT_DATASET_OUTPUT_FOLDER
                    if state_machine.get("split_dataset", False)
                    else MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
                    if _use_scaled
                    else MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
                    if _use_filtered
                    else MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
                )
                lightgbm_result = self.train_lightgbm_with_randomized_search_cv(
                    data_dir=os.path.join(self.project_root, _data_folder)
                )

                if lightgbm_result and lightgbm_result.get("success", False):
                    result["steps_executed"].append("train_lightgbm_with_randomized_search_cv")
                    result["file_info"]["lightgbm_models_trained"] = lightgbm_result.get("models_trained", 0)
                    result["file_info"]["lightgbm_problem_type"] = lightgbm_result.get("problem_type", "unknown")
                    print(f"      ✓ Successfully trained LightGBM models")
                    print(f"      ✓ Problem type: {lightgbm_result.get('problem_type', 'N/A')}")
                    print(f"      ✓ Models trained: {lightgbm_result.get('models_trained', 0)}")
                    print(f"      ✓ Target feature: {lightgbm_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ Average CV Score: {lightgbm_result.get('cv_score', 0):.4f}")

                    if lightgbm_result.get('problem_type') == 'classification':
                        print(f"      ✓ Average Test F1: {lightgbm_result.get('test_f1', 0):.4f}")
                    else:
                        print(f"      ✓ Average Test RMSE: {lightgbm_result.get('test_rmse', 0):.4f}")
                        print(f"      ✓ Average Test R²: {lightgbm_result.get('test_r2', 0):.4f}")

                    print(f"      ✓ Results saved to: {lightgbm_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = lightgbm_result.get("error", "train_lightgbm_with_randomized_search_cv returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result

            except Exception as e:
                result["errors"].append(f"train_lightgbm_with_randomized_search_cv failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ train_lightgbm_with_randomized_search_cv (disabled)")
```

- [ ] **Step 3: Run all 10 tests to verify they pass**

```bash
conda run -n venv_rail_fmi_training python -m pytest tests/test_train_lightgbm.py -v
```

Expected: 10 PASSED.

- [ ] **Step 4: Run the full test suite to verify no regressions**

```bash
conda run -n venv_rail_fmi_training python -m pytest tests/ -v
```

Expected: 85 tests PASSED (75 existing + 10 new).

- [ ] **Step 5: Commit**

```bash
git add src/training_pipeline.py
git commit -m "feat: add train_lightgbm_with_randomized_search_cv method and dispatcher"
```
