# Logistic Regression Training Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `train_logistic_regression_with_randomized_search_cv` pipeline stage that mirrors the existing Random Forest stage structurally, using `sklearn.linear_model.LogisticRegression` (classification) paired with `sklearn.linear_model.ElasticNet` (regression), wired into `TRAINING_STATE_MACHINE` after `train_random_forest_with_randomized_search_cv`.

**Architecture:** Copy the Random Forest training method (`src/training_pipeline.py:3076-3597`) almost verbatim, swapping the model classes, their hyperparameter distributions, and the feature-importance computation (linear models expose `.coef_`, not `.feature_importances_`). Wire it into the state-machine dispatcher (`execute_training_pipeline_steps`) the same way Random Forest is wired, immediately after the Random Forest block. Reuses the existing (currently unused) `REGULARIZED_REGRESSION_OUTPUT_FOLDER` constant and `self.regularized_regression_dir` attribute as the output location.

**Tech Stack:** Python, scikit-learn (`LogisticRegression`, `ElasticNet`, `RandomizedSearchCV`), pandas, matplotlib, joblib, pytest.

## Global Constraints

- Design spec: `docs/superpowers/plans/2026-07-08-logistic-regression-stage.md` is this file; the approved design is recorded in the plan-mode transcript — follow the code below exactly.
- New stage must default to `True` in `TRAINING_STATE_MACHINE`, placed immediately after `"train_random_forest_with_randomized_search_cv": True,`.
- `LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS` uses `solver='saga'` fixed in the constructor (not searched) so every sampled `penalty` (`l1`/`l2`/`elasticnet`) is valid — no invalid solver/penalty combo can be drawn.
- `ELASTICNET_PARAM_DISTRIBUTIONS` covers `alpha` and `l1_ratio` only; `l1_ratio` lower bound is `0.1` (not `0.0`, which is numerically unstable/warns in sklearn's `ElasticNet`).
- `max_iter=5000` is hardcoded (not searched) in both `LogisticRegression` and `ElasticNet` constructors, to avoid `ConvergenceWarning` spam.
- `ElasticNet` has **no** `n_jobs` parameter — do not pass it (raises `TypeError`).
- Feature importance must use `np.abs(coef)` where `coef = best_model.coef_[0] if is_classification else best_model.coef_` (binary `LogisticRegression.coef_` shape is `(1, n_features)`; `ElasticNet.coef_` shape is `(n_features,)`). The importance DataFrame keeps a second `coefficient` column with the signed value. Plot/CSV labels say "Coefficient Magnitude", not "Feature Importance".
- No internal feature scaler — every column reaching this stage via the scaled data directory is already comparably scaled (verified against `data/output/505-scale_weather_features/merged_data_selected_train.parquet`).
- Output files use prefix `logistic_regression_`, written to `REGULARIZED_REGRESSION_OUTPUT_FOLDER` (already imported in `src/training_pipeline.py`).
- No refactor of the existing XGBoost/LightGBM/RandomForest methods.
- Tests are source-inspection style (`inspect.getsource`), matching `tests/test_train_random_forest.py` — no live training run required.
- **Git hygiene:** `src/training_pipeline.py` may have unrelated pre-existing uncommitted changes in the working tree (from before this plan). Before committing, run `git diff -- src/training_pipeline.py` and stage ONLY the hunks this task added (use `git add -p` or equivalent targeted staging), verifying with `git diff --cached -- src/training_pipeline.py` that the staged diff contains only this task's own change before committing.

---

### Task 1: Config constants — param distributions and state-machine flag

**Files:**
- Modify: `config/const_training.py` (insert `LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS` and `ELASTICNET_PARAM_DISTRIBUTIONS` after `RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION`; insert `TRAINING_STATE_MACHINE` key after `"train_random_forest_with_randomized_search_cv": True,`)
- Test: `tests/test_train_logistic_regression.py` (create)

**Interfaces:**
- Produces: `LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS` (dict, importable from `config.const_training`), `ELASTICNET_PARAM_DISTRIBUTIONS` (dict), `TRAINING_STATE_MACHINE["train_logistic_regression_with_randomized_search_cv"]` (bool, `True`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_logistic_regression.py`:

```python
import inspect
from config.const_training import (
    REGULARIZED_REGRESSION_OUTPUT_FOLDER,
    LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS,
    ELASTICNET_PARAM_DISTRIBUTIONS,
    TRAINING_STATE_MACHINE,
)
from src.training_pipeline import TrainingPipeline


def test_regularized_regression_output_folder_constant_exists():
    assert REGULARIZED_REGRESSION_OUTPUT_FOLDER == "data/output/regularized_regression"


def test_logistic_regression_param_distributions_constant_exists():
    expected_keys = {'penalty', 'C', 'l1_ratio', 'class_weight'}
    assert set(LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS.keys()) == expected_keys
    assert set(LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS['penalty']) == {'l1', 'l2', 'elasticnet'}


def test_elasticnet_param_distributions_constant_exists():
    expected_keys = {'alpha', 'l1_ratio'}
    assert set(ELASTICNET_PARAM_DISTRIBUTIONS.keys()) == expected_keys
    assert min(ELASTICNET_PARAM_DISTRIBUTIONS['l1_ratio']) >= 0.1


def test_logistic_regression_in_state_machine():
    assert 'train_logistic_regression_with_randomized_search_cv' in TRAINING_STATE_MACHINE
    assert TRAINING_STATE_MACHINE['train_logistic_regression_with_randomized_search_cv'] is True
```

Note: `REGULARIZED_REGRESSION_OUTPUT_FOLDER` already exists in `config/const_training.py:40` — read the file first to confirm its current value before running this test (expected `"data/output/regularized_regression"`; if it differs, use the actual value, do not change the existing constant).

- [ ] **Step 2: Run tests to verify the expected failures**

Run: `pytest tests/test_train_logistic_regression.py -v`
Expected: `test_logistic_regression_param_distributions_constant_exists`, `test_elasticnet_param_distributions_constant_exists`, and `test_logistic_regression_in_state_machine` FAIL (constants/key don't exist yet); `test_regularized_regression_output_folder_constant_exists` PASSES (constant already exists).

- [ ] **Step 3: Add the two param-distribution constants to config/const_training.py**

Read `config/const_training.py` first. Insert immediately after the `RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION` dict's closing `}` (currently ending at line 114):

```python

# Parameter distributions for Logistic Regression (classification branch).
# solver='saga' is fixed in the constructor (NOT searched) because it is the only
# solver supporting l1 / l2 / elasticnet penalties — this guarantees every sampled
# penalty is valid, so no invalid solver/penalty pair can ever be drawn.
# l1_ratio is only consulted by saga when penalty=='elasticnet'; it is silently
# ignored (no error) for l1/l2, so it is safe to search unconditionally.
LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],   # inverse reg strength, log-spaced
    'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],     # used only when penalty=='elasticnet'
    'class_weight': ['balanced', {False: 1, True: 5}, {False: 1, True: 10}, None],
}

# Parameter distributions for ElasticNet (regression branch — the "regularized
# regression" pairing implied by REGULARIZED_REGRESSION_OUTPUT_FOLDER).
# l1_ratio lower bound kept at 0.1 (not 0.0): sklearn ElasticNet warns and is
# numerically unstable at l1_ratio==0 (pure Ridge) — use Ridge for that, not here.
ELASTICNET_PARAM_DISTRIBUTIONS = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],  # reg strength, log-spaced
    'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9, 1.0],    # 0=Ridge .. 1=Lasso
}
```

- [ ] **Step 4: Add the state-machine flag**

In the `TRAINING_STATE_MACHINE` dict, insert a new line immediately after `"train_random_forest_with_randomized_search_cv": True,`:

```python
    "train_logistic_regression_with_randomized_search_cv": True,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_train_logistic_regression.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add config/const_training.py tests/test_train_logistic_regression.py
git commit -m "feat: add logistic regression config constants and state-machine flag"
```

---

### Task 2: `train_logistic_regression_with_randomized_search_cv` method

**Files:**
- Modify: `src/training_pipeline.py` (imports; new method inserted after `train_random_forest_with_randomized_search_cv`, i.e. after line 3597, before `numeric_correlation_analysis` at line 3598)
- Test: `tests/test_train_logistic_regression.py` (append)

**Interfaces:**
- Consumes: `LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS`, `ELASTICNET_PARAM_DISTRIBUTIONS`, `REGULARIZED_REGRESSION_OUTPUT_FOLDER` (from Task 1 / existing config), all other constants already imported in `training_pipeline.py`
- Produces: `TrainingPipeline.train_logistic_regression_with_randomized_search_cv(self, data_dir=None)` returning the same dict shape as `train_random_forest_with_randomized_search_cv`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_train_logistic_regression.py`:

```python
def _source():
    return inspect.getsource(TrainingPipeline.train_logistic_regression_with_randomized_search_cv)


def test_n_jobs_is_minus_one():
    assert 'n_jobs=-1' in _source()


def test_saga_solver_used():
    assert "solver='saga'" in _source()


def test_max_iter_hardcoded():
    assert 'max_iter=5000' in _source()


def test_elasticnet_no_n_jobs_kwarg():
    src = _source()
    # ElasticNet constructor call must not pass n_jobs (raises TypeError at runtime)
    elasticnet_start = src.index('ElasticNet(')
    elasticnet_call = src[elasticnet_start:elasticnet_start + 200]
    assert 'n_jobs' not in elasticnet_call


def test_random_state_uses_constant():
    src = _source()
    assert 'RANDOM_STATE' in src
    assert 'random_state=42' not in src


def test_classifier_and_regressor_classes_used():
    src = _source()
    assert 'LogisticRegression(' in src
    assert 'ElasticNet(' in src


def test_param_distributions_selected_by_problem_type():
    src = _source()
    assert 'LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS' in src
    assert 'ELASTICNET_PARAM_DISTRIBUTIONS' in src


def test_feature_importance_uses_coef_not_feature_importances():
    src = _source()
    assert 'coef_' in src
    assert 'feature_importances_' not in src


def test_coefficient_magnitude_label_used():
    src = _source()
    assert 'Coefficient Magnitude' in src


def test_wmape_in_logistic_regression_regression_branch():
    assert 'final_test_wmape' in _source()


def test_binary_metrics_in_logistic_regression_regression_branch():
    assert 'final_bin_f1' in _source()


def test_output_filenames_use_logistic_regression_prefix():
    src = _source()
    assert 'logistic_regression_iteration_analysis_' in src
    assert 'logistic_regression_feature_importance_' in src
    assert 'logistic_regression_best_model_' in src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train_logistic_regression.py -v`
Expected: FAIL with `AttributeError: type object 'TrainingPipeline' has no attribute 'train_logistic_regression_with_randomized_search_cv'` for all newly added tests.

- [ ] **Step 3: Add the sklearn.linear_model import**

Read `src/training_pipeline.py` first. Immediately after the existing `from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor` import line, add:

```python
from sklearn.linear_model import LogisticRegression, ElasticNet
```

- [ ] **Step 4: Add the new constants to the config.const_training import block**

In the `from config.const_training import (...)` block, immediately after the existing `RANDOM_FOREST_PARAM_DISTRIBUTIONS_REGRESSION,` line, add:

```python
    LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS,
    ELASTICNET_PARAM_DISTRIBUTIONS,
```

(`REGULARIZED_REGRESSION_OUTPUT_FOLDER` is already imported — no change needed there.)

- [ ] **Step 5: Insert the new method after `train_random_forest_with_randomized_search_cv`**

Insert immediately after the end of `train_random_forest_with_randomized_search_cv` (after its closing `return {"success": False, "error": error_msg}` and the method's final blank line, i.e. right before `def numeric_correlation_analysis`):

```python
    def train_logistic_regression_with_randomized_search_cv(self, data_dir=None):
        """
        Logistic Regression training method that tests different iteration counts and plots performance curve.

        Uses LogisticRegression for classification and ElasticNet (regularized linear
        regression) for regression, since sklearn's LogisticRegression has no direct
        regression counterpart.

        This method:
        1. Tests RANDOM_SEARCH_ITERATIONS from 10 to 100 (step=10)
        2. Tracks test F1 binary scores for each iteration count
        3. Plots test_f1_binary vs n_iter curve
        4. Saves the best model and performance metrics
        """

        try:
            print(f"    train_logistic_regression_with_randomized_search_cv: Starting Logistic Regression training with iteration analysis...")

            # Create output directory
            output_dir = os.path.join(self.project_root, REGULARIZED_REGRESSION_OUTPUT_FOLDER)
            os.makedirs(output_dir, exist_ok=True)

            print(f"    train_logistic_regression_with_randomized_search_cv: Output directory: {output_dir}")

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
                print(f"    train_logistic_regression_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            # Check if we have exactly one train/test file pair
            if len(train_files) != 1 or len(test_files) != 1:
                error_msg = f"Expected exactly one train and one test file. Found {len(train_files)} train files and {len(test_files)} test files"
                print(f"    train_logistic_regression_with_randomized_search_cv: {error_msg}")
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
                print(f"    train_logistic_regression_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            file_identifier = identifier_match.group(1)
            print(f"    train_logistic_regression_with_randomized_search_cv: Processing dataset: {file_identifier}")

            # Determine problem type based on target feature
            target_feature = DEFAULT_TARGET_FEATURE
            is_classification = target_feature in CLASSIFICATION_PROBLEM
            is_regression = target_feature in REGRESSION_PROBLEM

            if not (is_classification or is_regression):
                error_msg = f"Target feature '{target_feature}' not recognized as classification or regression problem"
                print(f"    train_logistic_regression_with_randomized_search_cv: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

            problem_type = "classification" if is_classification else "regression"
            print(f"    train_logistic_regression_with_randomized_search_cv: Detected {problem_type} problem for target '{target_feature}'")

            # Load training and test data
            print(f"      Loading training data from {train_file}")
            train_df = pd.read_parquet(train_file)

            print(f"      Loading test data from {test_file}")
            test_df = pd.read_parquet(test_file)

            # Check if target feature exists
            if target_feature not in train_df.columns:
                error_msg = f"Target feature '{target_feature}' not found in dataset"
                print(f"    train_logistic_regression_with_randomized_search_cv: {error_msg}")
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
            y_train_log = None  # assigned in regression branch below
            y_shift = 0.0       # assigned in regression branch below
            if is_classification:
                cv_splitter = StratifiedKFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                base_model = LogisticRegression(
                    solver='saga',        # only solver supporting l1/l2/elasticnet — enables the full penalty search
                    max_iter=5000,        # avoid ConvergenceWarning spam; not searched
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                param_distributions = LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS
                scoring_metric = SCORE_METRIC
            else:
                cv_splitter = KFold(n_splits=RANDOM_SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                base_model = ElasticNet(
                    max_iter=5000,        # coordinate descent headroom; not searched
                    random_state=RANDOM_STATE,
                    # NOTE: ElasticNet has no n_jobs parameter — do not pass it (raises TypeError)
                )
                param_distributions = ELASTICNET_PARAM_DISTRIBUTIONS
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
                    param_distributions=param_distributions,
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
                    test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    test_recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)

                    print(f"        Iteration {n_iter}: CV Score = {current_cv_score:.4f}, Test F1 = {test_f1:.4f}, Test Accuracy = {test_accuracy:.4f}, Precision = {test_precision:.4f}, Recall = {test_recall:.4f}")

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

            plt.title(f'Logistic Regression Performance vs RandomizedSearch Iterations\nDataset: {file_identifier}',
                    fontsize=16, fontweight='bold', pad=20)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.02, 0.98),
                    fontsize=12, framealpha=0.9)

            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'logistic_regression_iteration_analysis_{file_identifier}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      Performance curve saved to: {plot_filename}")

            # CREATE AND SAVE FEATURE IMPORTANCE PLOT
            results = {}
            if best_model is not None:
                print(f"      Creating feature importance plot...")

                # Linear models: use coefficients instead of tree feature_importances_.
                # Binary LogisticRegression.coef_ has shape (1, n_features); ElasticNet.coef_
                # has shape (n_features,) for a single-target regression.
                coef = best_model.coef_[0] if is_classification else best_model.coef_
                feature_importance = np.abs(coef)   # magnitude drives sorting + bar length
                feature_names = feature_columns

                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance,  # |coef|, used for sort + plot
                    'coefficient': coef,                # signed — preserves direction of effect
                }).sort_values('importance', ascending=True)

                plt.figure(figsize=(12, max(8, len(feature_names) * 0.4)))

                bars = plt.barh(range(len(importance_df)), importance_df['importance'])

                plt.yticks(range(len(importance_df)), importance_df['feature'])
                plt.xlabel('Coefficient Magnitude', fontsize=12)
                plt.title(f'Logistic Regression Coefficient Magnitude - Best Model ({problem_type.title()})\n'
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

                importance_plot_filename = os.path.join(output_dir, f'logistic_regression_feature_importance_{file_identifier}.png')
                plt.savefig(importance_plot_filename, dpi=300, bbox_inches='tight')
                print(f"      Feature importance plot saved to: {importance_plot_filename}")
                plt.close()

                importance_csv_filename = os.path.join(output_dir, f'logistic_regression_feature_importance_{file_identifier}.csv')
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
                    "feature_importance": dict(zip(feature_columns, np.abs(coef))) if best_model else None
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

                results_file = os.path.join(output_dir, f"logistic_regression_iteration_analysis_{file_identifier}.json")
                with open(results_file, 'w') as f:
                    results_str = json.loads(json.dumps(results, default=str))
                    json.dump(results_str, f, indent=2)

                print(f"      Results saved to: {results_file}")

                # Save the best model
                model_filename = os.path.join(output_dir, f'logistic_regression_best_model_{file_identifier}.pkl')
                joblib.dump(best_model, model_filename)
                print(f"      Best model saved to: {model_filename}")

                # Find the index of the best iteration in the results list
                best_iteration_idx = iteration_values.index(best_iteration) if best_iteration else 0

                # Print summary
                print(f"      Training Summary:")
                print(f"        Problem Type: {problem_type}")
                print(f"        Best Iteration Count: {best_iteration}")
                print(f"        Best CV Score: {best_cv_score:.4f}")
                print(f"        F1 Range: {min(test_f1_scores):.4f} - {max(test_f1_scores):.4f}")
                if is_classification:
                    print(f"        Final Accuracy: {final_test_accuracy:.4f}  F1: {final_test_f1:.4f}  Precision: {final_test_precision:.4f}  Recall: {final_test_recall:.4f}  AUC: {final_test_auc:.4f}")
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
            else:
                return {
                    "success": False,
                    "error": "No model was trained — all iterations failed to produce a best estimator"
                }

        except Exception as e:
            error_msg = f"Error in train_logistic_regression_with_randomized_search_cv: {str(e)}"
            print(f"    train_logistic_regression_with_randomized_search_cv: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg
            }

```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_train_logistic_regression.py -v`
Expected: all tests PASS (4 from Task 1 + 12 from Task 2, total 16).

- [ ] **Step 7: Commit**

Before committing, run `git diff -- src/training_pipeline.py` and confirm only this task's own changes (the sklearn.linear_model import, the two config import lines, and the new method) are present. If unrelated pre-existing uncommitted hunks are mixed in from before this task started, stage only this task's hunks (`git add -p src/training_pipeline.py`), verify with `git diff --cached -- src/training_pipeline.py`, and leave any unrelated hunks unstaged.

```bash
git add tests/test_train_logistic_regression.py
git commit -m "feat: add train_logistic_regression_with_randomized_search_cv method"
```

---

### Task 3: Wire the dispatcher block into `execute_training_pipeline_steps`

**Files:**
- Modify: `src/training_pipeline.py` (inside `execute_training_pipeline_steps`, immediately after the Random Forest `if state_machine.get(...)` block and its `else` branch, before the method's final `return result`)
- Test: `tests/test_train_logistic_regression.py` (append)

**Interfaces:**
- Consumes: `self.train_logistic_regression_with_randomized_search_cv(data_dir=...)` (Task 2), `TRAINING_STATE_MACHINE` key `"train_logistic_regression_with_randomized_search_cv"` (Task 1)
- Produces: `result["file_info"]["logistic_regression_models_trained"]`, `result["file_info"]["logistic_regression_problem_type"]` populated when the stage runs successfully

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train_logistic_regression.py`:

```python
def test_dispatcher_routes_logistic_regression_after_random_forest():
    dispatcher_src = inspect.getsource(TrainingPipeline.execute_training_pipeline_steps)
    random_forest_idx = dispatcher_src.index('train_random_forest_with_randomized_search_cv')
    logistic_regression_idx = dispatcher_src.index(
        'train_logistic_regression_with_randomized_search_cv', random_forest_idx
    )
    assert logistic_regression_idx > random_forest_idx


def test_dispatcher_populates_logistic_regression_file_info_keys():
    dispatcher_src = inspect.getsource(TrainingPipeline.execute_training_pipeline_steps)
    assert 'logistic_regression_models_trained' in dispatcher_src
    assert 'logistic_regression_problem_type' in dispatcher_src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train_logistic_regression.py -v`
Expected: both new tests FAIL (`'train_logistic_regression_with_randomized_search_cv'` and the file_info keys don't appear anywhere in `execute_training_pipeline_steps` yet).

- [ ] **Step 3: Add the dispatcher block**

Read `src/training_pipeline.py` first. In `execute_training_pipeline_steps`, find the Random Forest block:

```python
        if state_machine.get("train_random_forest_with_randomized_search_cv", False):
            ...
        else:
            print(f"    ⊝ train_random_forest_with_randomized_search_cv (disabled)")

        return result
```

Insert the new block between the Random Forest `else` branch and `return result`:

```python
        if state_machine.get("train_logistic_regression_with_randomized_search_cv", False):
            try:
                print(f"    → train_logistic_regression_with_randomized_search_cv")
                _use_scaled = state_machine.get("scale_weather_features", False)
                _use_filtered = (
                    state_machine.get("filter_delay_outliers", False)
                    and not state_machine.get("select_training_cols", False)
                )
                _data_folder = (
                    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER
                    if _use_scaled
                    else MERGED_BALANCED_OUTPUT_FOLDER
                    if state_machine.get("balance_classes", False)
                    else SPLIT_DATASET_OUTPUT_FOLDER
                    if state_machine.get("split_dataset", False)
                    else MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER
                    if _use_filtered
                    else MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
                )
                logistic_regression_result = self.train_logistic_regression_with_randomized_search_cv(
                    data_dir=os.path.join(self.project_root, _data_folder)
                )

                if logistic_regression_result and logistic_regression_result.get("success", False):
                    result["steps_executed"].append("train_logistic_regression_with_randomized_search_cv")
                    result["file_info"]["logistic_regression_models_trained"] = logistic_regression_result.get("models_trained", 0)
                    result["file_info"]["logistic_regression_problem_type"] = logistic_regression_result.get("problem_type", "unknown")
                    print(f"      ✓ Successfully trained Logistic Regression models")
                    print(f"      ✓ Problem type: {logistic_regression_result.get('problem_type', 'N/A')}")
                    print(f"      ✓ Models trained: {logistic_regression_result.get('models_trained', 0)}")
                    print(f"      ✓ Target feature: {logistic_regression_result.get('target_feature', 'N/A')}")
                    print(f"      ✓ Average CV Score: {logistic_regression_result.get('cv_score', 0):.4f}")

                    if logistic_regression_result.get('problem_type') == 'classification':
                        print(f"      ✓ Average Test F1: {logistic_regression_result.get('test_f1', 0):.4f}")
                    else:
                        print(f"      ✓ Average Test RMSE: {logistic_regression_result.get('test_rmse', 0):.4f}")
                        print(f"      ✓ Average Test R²: {logistic_regression_result.get('test_r2', 0):.4f}")

                    print(f"      ✓ Results saved to: {logistic_regression_result.get('output_directory', 'N/A')}")
                    result["success"] = True
                else:
                    error_msg = logistic_regression_result.get("error", "train_logistic_regression_with_randomized_search_cv returned unsuccessful result")
                    result["errors"].append(error_msg)
                    print(f"      ✗ Failed - {error_msg}")
                    return result

            except Exception as e:
                result["errors"].append(f"train_logistic_regression_with_randomized_search_cv failed: {str(e)}")
                print(f"      ✗ Failed - {str(e)}")
                return result
        else:
            print(f"    ⊝ train_logistic_regression_with_randomized_search_cv (disabled)")

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train_logistic_regression.py -v`
Expected: all tests PASS (18 total).

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `pytest tests/ -v`
Expected: all tests PASS, including pre-existing `tests/test_train_random_forest.py` and all other test files.

- [ ] **Step 6: Commit**

Before committing, run `git diff -- src/training_pipeline.py` and confirm only this task's own dispatcher-block addition is present (no changes inside `train_xgboost_with_randomized_search_cv`, `train_lightgbm_with_randomized_search_cv`, or `train_random_forest_with_randomized_search_cv` method bodies). Stage only this task's hunk if unrelated pre-existing changes are mixed in.

```bash
git add src/training_pipeline.py tests/test_train_logistic_regression.py
git commit -m "feat: wire logistic regression stage into training pipeline dispatcher"
```

---

## Out of Scope

- Refactoring the XGBoost, LightGBM, or RandomForest methods
- An internal feature scaler (confirmed unnecessary — all features reaching this stage via the scaled data directory are already comparably scaled)
- Ridge regression as a separate model (ElasticNet with `l1_ratio` near 0 covers similar territory without adding a fifth model class)
- Comparing results across all four models programmatically
- Any change to preprocessing or data-splitting stages
- Running a live end-to-end training pipeline execution as part of this plan's tests (source-inspection tests only, matching the Random Forest stage's test style)
