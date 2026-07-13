# Naive Bayes Training Stage — Design

## Goal

Add a `train_naive_bayes_with_randomized_search_cv` pipeline stage that mirrors the existing Logistic Regression stage structurally, using `sklearn.naive_bayes.GaussianNB` for classification and `sklearn.linear_model.BayesianRidge` for regression (since Naive Bayes has no direct regression counterpart), wired into `TRAINING_STATE_MACHINE` immediately after `train_logistic_regression_with_randomized_search_cv`.

## Architecture

Copy `train_logistic_regression_with_randomized_search_cv` (`src/training_pipeline.py:3651-4183`) almost verbatim, swapping the model classes, their hyperparameter distributions, and the feature-importance computation for the classification branch. Wire it into the dispatcher (`execute_training_pipeline_steps`) the same way Logistic Regression is wired, immediately after the Logistic Regression block.

Training stage order becomes: XGBoost → LightGBM → Random Forest → Logistic Regression → **Naive Bayes**.

## Why GaussianNB + BayesianRidge

`GaussianNB` is classification-only; sklearn has no direct Naive-Bayes regression counterpart. The Logistic Regression stage already solves an analogous problem by pairing `LogisticRegression` (classification) with `ElasticNet` (regression) under one method — `ElasticNet` isn't a Bayesian classifier either, it's the nearest structural cousin (linear, regularized). Following that precedent, `BayesianRidge` is paired with `GaussianNB` for the regression branch: both are Bayesian in spirit (GaussianNB assumes per-class Gaussian likelihoods; BayesianRidge places Gaussian priors on the regression weights), and `BayesianRidge` exposes a `coef_` array shaped like `ElasticNet`'s, so the existing regression-branch feature-importance code needs no structural change.

## Config additions (`config/const_training.py`)

- `TRAINING_STATE_MACHINE["train_naive_bayes_with_randomized_search_cv"] = True`, inserted immediately after the `train_logistic_regression_with_randomized_search_cv` key.
- `NAIVE_BAYES_OUTPUT_FOLDER = "data/output/1004-naive_bayes"` — next number after `1003-regularized_regression`.
- `NAIVE_BAYES_PARAM_DISTRIBUTIONS = {'var_smoothing': loguniform(1e-12, 1e-6)}` — GaussianNB's one real hyperparameter; log-spaced because it spans many orders of magnitude.
- `BAYESIAN_RIDGE_PARAM_DISTRIBUTIONS = {'alpha_1': [1e-7, 1e-6, 1e-5, 1e-4], 'alpha_2': [1e-7, 1e-6, 1e-5, 1e-4], 'lambda_1': [1e-7, 1e-6, 1e-5, 1e-4], 'lambda_2': [1e-7, 1e-6, 1e-5, 1e-4]}` — BayesianRidge's Gamma-prior hyperparameters, filling the regularization-strength role `alpha`/`l1_ratio` play for ElasticNet. Values are a multiplicative sweep around sklearn's default (1e-6) for each prior.

## Method: `train_naive_bayes_with_randomized_search_cv(self, data_dir=None)`

Structurally identical to the Logistic Regression method, with these substitutions only:

- **Output dir**: `NAIVE_BAYES_OUTPUT_FOLDER` instead of `REGULARIZED_REGRESSION_OUTPUT_FOLDER`.
- **File discovery, target detection, sample-weight calculation, CV-splitter selection (`StratifiedKFold`/`KFold`)**: unchanged, copied verbatim.
- **Classification branch**: `base_model = GaussianNB()` (no `random_state`/`n_jobs` — GaussianNB doesn't accept them), `param_distributions = NAIVE_BAYES_PARAM_DISTRIBUTIONS`, `scoring_metric = SCORE_METRIC`.
- **Regression branch**: `base_model = BayesianRidge(max_iter=5000)`, `param_distributions = BAYESIAN_RIDGE_PARAM_DISTRIBUTIONS`, `scoring_metric = 'neg_mean_absolute_error'`, same target log-transform (`y_shift`, `np.log1p`) as Logistic Regression.
- **Iteration sweep**: same `range(10, RANDOM_SEARCH_ITERATIONS + 1, 10)` loop, `RandomizedSearchCV` call, per-iteration metric computation (F1/accuracy/precision/recall for classification; RMSE/MAE/R²/WMAPE/binary metrics for regression), and "select best model by CV score" logic — copied verbatim.
- **Performance curve plot**: identical dual-axis plot, titles updated to "Naive Bayes Performance vs RandomizedSearch Iterations" (classification) — the underlying model name in the title reflects whichever model was actually used for that problem type, matching how the Logistic Regression plot already titles itself generically.
- **Feature importance**:
  - Classification (`GaussianNB`): `coef = np.abs(best_model.theta_[1] - best_model.theta_[0]) / np.sqrt((best_model.var_[0] + best_model.var_[1]) / 2)` — a Cohen's-d-style standardized mean difference per feature, assuming binary classification (matches the existing binary assumption in the Logistic Regression method, which indexes `coef_[0]`). This array has the same shape as `feature_columns`, so it plugs into the existing bar-chart/CSV/JSON code unchanged.
  - Regression (`BayesianRidge`): `coef = best_model.coef_` — unchanged from the `ElasticNet` case, since `BayesianRidge.coef_` has the same `(n_features,)` shape.
  - Plot/axis labels updated from "Coefficient Magnitude" to reflect the metric (e.g. "Standardized Mean Difference" for classification, "Coefficient Magnitude" retained for regression).
- **Output artifacts**: same file set with `naive_bayes_` prefix instead of `logistic_regression_`: `naive_bayes_iteration_analysis_{id}.png`, `naive_bayes_iteration_analysis_{id}.json`, `naive_bayes_feature_importance_{id}.png`, `naive_bayes_feature_importance_{id}.csv`, `naive_bayes_best_model_{id}.pkl`.
- **Return dict**: same shape as Logistic Regression's (`success`, `models_trained`, `problem_type`, `target_feature`, `cv_score`, `test_f1`/`test_rmse` + regression metrics, `output_directory`, `results_file`).
- **Error handling**: same try/except wrapper, same early-return error shapes for missing files, wrong file count, unrecognized target, missing target column.

## Dispatcher wiring (`execute_training_pipeline_steps`)

New `if state_machine.get("train_naive_bayes_with_randomized_search_cv", False):` block, copied from the Logistic Regression block (`src/training_pipeline.py:611-663`), placed immediately after it:

- Same `_use_scaled`/`_use_filtered`/`_data_folder` resolution logic (unchanged — reused verbatim, not model-specific).
- Calls `self.train_naive_bayes_with_randomized_search_cv(data_dir=...)`.
- Populates `result["file_info"]["naive_bayes_models_trained"]` and `result["file_info"]["naive_bayes_problem_type"]`.
- Same success/failure print statements and `result["errors"]`/early-`return` pattern.

## Testing

Mirror `tests/test_train_logistic_regression.py` as `tests/test_train_naive_bayes.py`, inspection-style (`inspect.getsource`), no live training run:

- `NAIVE_BAYES_OUTPUT_FOLDER`, `NAIVE_BAYES_PARAM_DISTRIBUTIONS`, `BAYESIAN_RIDGE_PARAM_DISTRIBUTIONS` constants exist with expected keys.
- `train_naive_bayes_with_randomized_search_cv` in `TRAINING_STATE_MACHINE`, value `True`.
- Method source uses `GaussianNB` and `BayesianRidge` classes.
- Method source selects param distributions by problem type (`NAIVE_BAYES_PARAM_DISTRIBUTIONS` for classification, `BAYESIAN_RIDGE_PARAM_DISTRIBUTIONS` for regression).
- Feature importance for classification uses `theta_`/`var_`, not `coef_`.
- Feature importance for regression uses `coef_`, matching the ElasticNet pattern.
- Output filenames use `naive_bayes_` prefix.
- Dispatcher routes Naive Bayes immediately after Logistic Regression.
- Dispatcher populates `naive_bayes_models_trained`/`naive_bayes_problem_type` file_info keys.

## Out of scope

- No changes to the Logistic Regression, Random Forest, LightGBM, or XGBoost stages beyond reading them as a pattern.
- No multiclass generalization for the `theta_`/`var_` feature-importance formula — binary-only, matching the existing codebase's binary-classification assumption throughout.
