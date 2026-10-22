from scipy.stats import randint, loguniform
from config.const_preprocessing import TRAIN_DELAY_MINUTES
import numpy as np
import os

# Outer RandomizedSearchCV parallelism, and standalone (non-nested) model fits.
# RAILWAY_N_JOBS (set by `python main.py --n-jobs`) takes priority so the CLI flag
# does not have to overwrite Slurm's own SLURM_CPUS_PER_TASK record, which the
# batch scripts read independently. Falling back to SLURM_CPUS_PER_TASK keeps the
# allocation-aware default when no explicit --n-jobs is given: Slurm sets it to the
# cores actually allocated, whereas joblib's -1 would instead grab every physical
# core on the node and thrash. Off-cluster, with neither variable set, this is -1,
# preserving local behaviour.
SEARCH_N_JOBS = int(
    os.environ.get("RAILWAY_N_JOBS", os.environ.get("SLURM_CPUS_PER_TASK", -1))
)

# Inner estimator threads. Kept at 1 so the cores go to the independent CV fits
# rather than to OpenMP threads competing with them. Batch scripts also export
# OMP_NUM_THREADS=1 for the same reason.
MODEL_N_JOBS = 1

TRAINING_STATE_MACHINE = {
    "merge_data_files": True,
    "filter_delay_outliers": True,
    "select_training_cols": True,
    "split_dataset": True,
    "balance_classes": False,        # off: class weights instead (resampling before CV leaks)
    "scale_weather_features": True,
    "build_fcll_sequences": False,   # gated: only --model fcll_net needs it
    "shap_correlation_analysis": True,
    "explore_training_data": True,
    "model_feature_importance_analysis": False,  # model-based and slower; run it on demand
    "train_ridge_baseline": True,    # numeric targets only; skipped for a classification target
    "train_xgboost_with_randomized_search_cv": True,
    "train_lightgbm_with_randomized_search_cv": True,
    "train_random_forest_with_randomized_search_cv": True,
    "train_logistic_regression_with_randomized_search_cv": True,
    "train_naive_bayes_with_randomized_search_cv": True,
    "train_fcll_net": True,
}

MERGED_TRAINING_READY_OUTPUT_FOLDER = "data/output/500-merge_data_files"
MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER = "data/output/501-filter_delay_outliers"
MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER = "data/output/502-select_training_cols"
SPLIT_DATASET_OUTPUT_FOLDER = "data/output/503-split_dataset"
FCLL_SEQUENCES_OUTPUT_FOLDER = "data/output/506-build_fcll_sequences"
FCLL_NET_OUTPUT_FOLDER = "data/output/1005-fcll_net"

# Sequence lengths, per Huang et al. (2020) section 4.1.

FCLL_H_TRAINS = 5
FCLL_Z_STATIONS = 5
MERGED_BALANCED_OUTPUT_FOLDER = "data/output/504-balance_classes"
MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER = "data/output/505-scale_weather_features"
SHAP_CORRELATION_ANALYSIS_OUTPUT_FOLDER = "data/output/700-shap_correlation_analysis"
SHAP_CORRELATION_TOP_N_PAIRS = 20     # Feature pairs shown in the top-correlated-pairs bar chart
SHAP_CORRELATION_TOP_N_FEATURES = 15  # Features shown in the top-correlated-features sub-heatmap

# Stage 701, explore_training_data: charts and tables that describe the data itself (target,
# calendar and weather patterns, distributions, Spearman correlations and clusters, drift between
# the development years and the hold-out year). No model is fitted. Reads stage 502.
DATA_EXPLORATION_OUTPUT_FOLDER = "data/output/701-explore_training_data"
SPEARMAN_CLUSTER_ABS_CORR = 0.90 # |rho| at or above which two continuous columns share a cluster
EXPLORATION_N_BINS = 10          # quantile bins of a continuous variable in the binned charts
EXPLORATION_WIND_SECTORS = 16    # compass sectors of the wind-direction chart (22.5 degrees each)
EXPLORATION_MIN_CELL_COUNT = 30  # cells with fewer rows are masked in the weekday x hour heatmap
EXPLORATION_ANNOTATE_MAX = 25    # the Spearman heatmap prints its values only up to this many variables
EXPLORATION_MAX_PANELS = 16      # panels in the small-multiple charts (the strongest / first variables)

# Stage 702, model_feature_importance_analysis (off by default): grouped permutation importance and
# drop-group ablation. Fits its own plain XGBoost (no search), so it is a fast screen of what a
# MODEL relies on and not a replacement for the tuned model.
MODEL_FEATURE_IMPORTANCE_OUTPUT_FOLDER = "data/output/702-model_feature_importance"
PERMUTATION_REPEATS = 5          # shuffles per feature group
ABLATION_NOISE_SEEDS = 3         # seeds of the full model; their spread is the noise floor
NOISE_FLOOR_FACTOR = 2.0         # an effect within this many noise std is flagged "within noise"
# Defaults except two, both inside the trainer's own search space (XGBOOST_PARAM_DISTRIBUTIONS).
# A default XGBoost is deterministic whatever the seed, which would make the noise floor of the
# refits exactly zero; row and column subsampling make the seeds differ.
FEATURE_ANALYSIS_XGB_PARAMS = {"subsample": 0.8, "colsample_bytree": 0.8}
# Named sets for questions larger than one group. Columns absent from the data are skipped.
ABLATION_EXTRA_SETS = {
    "all_wind": ["wind_dir_sin", "wind_dir_cos", "Wind speed", "Gust speed"],
    "all_calendar": ["month_sin", "month_cos", "hour_sin", "hour_cos", "day_week_sin",
                     "day_week_cos", "day_month_sin", "day_month_cos", "trainStopping"],
}
# Columns the training run keeps. Empty here on purpose: main.py points
# RAILWAY_COLUMNS_FILE at config/features.txt when no --columns-file is given, so a plain
# `python main.py` (the IDE run button) trains on that file. The target (trainDisrupted),
# the day key and the hold-out year are added by the pipeline itself, and the trainers
# exclude the target from X by name. The differenceInMinutes* columns leak the delay and
# must never be listed. An empty list with training on is refused by main.py (it would
# otherwise fall through to the interactive prompt in select_training_cols).
SELECTED_COLUMNS = []
# `python main.py --columns-file <path>` exports RAILWAY_COLUMNS_FILE before this
# module is first imported (main.apply_env_overrides), letting a run select its
# features from a plain-text file that is never committed: the cluster reads a
# scp'd copy, so a feature change needs no commit, push or pull. Without the flag
# main.py falls back to config/features.txt when that file exists.
#
# The resolved list is printed in full: it is the only record of which features a
# job used, since the file is outside git. Pair it with the sha256 the batch
# scripts echo. Any parse error propagates and aborts the run — main.py has
# already reported it far more cheaply at parse time.
#
# When RAILWAY_SCENARIO is also set, the file is a scenario catalogue and that
# variable names which section to bind, by 1-based index or by exact name. The
# resolved scenario name is printed alongside the column list: with the file
# outside git, that log line plus the sha256 the batch scripts echo is the only
# record of what a job actually trained on.
_COLUMNS_FILE = os.environ.get("RAILWAY_COLUMNS_FILE", "")
_SCENARIO = os.environ.get("RAILWAY_SCENARIO", "")
if _COLUMNS_FILE:
    if _SCENARIO:
        from config.columns_file import select_scenario

        _scenario_name, SELECTED_COLUMNS = select_scenario(_COLUMNS_FILE, _SCENARIO)
        print(f"const_training: SELECTED_COLUMNS loaded from {_COLUMNS_FILE} "
              f"scenario {_scenario_name!r} ({len(SELECTED_COLUMNS)} columns)")
    else:
        from config.columns_file import load_columns

        SELECTED_COLUMNS = load_columns(_COLUMNS_FILE)
        print(f"const_training: SELECTED_COLUMNS loaded from {_COLUMNS_FILE} "
              f"({len(SELECTED_COLUMNS)} columns)")
    for _column_name in SELECTED_COLUMNS:
        print(f"    {_column_name}")
# 'intersect' rather than '' so batch jobs never block on input(): under Slurm
# stdin is /dev/null and the prompt at training_pipeline.py:1109 would raise EOFError.
SCHEMA_MISMATCH_STRATEGY = 'intersect'  # 'intersect' = auto-drop; 'fail' = auto-fail; '' = ask

# Asymmetric quantile thresholds for delay outlier removal
# Lower tail: conservative cut (few implausibly-early arrivals)

# Upper tail: more aggressive cut (heavy right tail has likely data errors)
FILTER_LOWER_QUANTILE = 0.005
FILTER_UPPER_QUANTILE = 0.995

# Numeric targets only (differenceInMinutes ...): cut the delay tails from the TRAIN rows
# only, with bounds taken from the train split (stage 503), instead of from every row at
# stage 501. Test and hold-out keep their real delays, so RMSE / MAE / R2 are measured on
# the target as it really is. A classification target is unaffected and keeps stage 501.
REGRESSION_FILTER_TRAIN_ONLY = True

# Numeric targets only: what to do with rows that have no delay (cancelled trains, 684 at OL).
#   'drop' - remove them. The numeric model answers "how late, given the train runs".
#   'cap'  - keep them with the delay set to the train split's FILTER_UPPER_QUANTILE bound
#            (about 111 min at OL: "a cancellation counts as a worst-case delay"). The bound
#            comes from the train rows only. A huge sentinel such as 9999 would put 63% of the
#            target variance on 0.6% of the rows, make RMSE ~780 min and disable the tail cut.
#            The filled rows are flagged next to the split files, and the XGBoost results also
#            report the metrics on the running trains only.
REGRESSION_CANCELLED_POLICY = 'drop'


# Ridge-regression baseline for a numeric target, run before XGBoost: the simple linear model the
# tree models are compared with (MAE, and mae_skill against always predicting the train median).
# Ridge minimises squared error, so it predicts a conditional mean and its MAE can trail the median.
RIDGE_BASELINE_OUTPUT_FOLDER = "data/output/999-ridge_baseline"
RIDGE_BASELINE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # regularisation strength, log-spaced

# L2 logistic-regression baseline for a classification target (stage 1003): the simple linear model
# the tree models are compared with. Only C is tuned (one day-grouped grid search); penalty, solver
# and class_weight='balanced' are fixed.
LOGISTIC_BASELINE_CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # inverse regularisation strength, log-spaced

XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/1000-xgboost_randomized_search"
LIGHTGBM_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/1001-lightgbm_randomized_search"
RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/1002-random_forest_randomized_search"
REGULARIZED_REGRESSION_OUTPUT_FOLDER = "data/output/1003-regularized_regression"
NAIVE_BAYES_OUTPUT_FOLDER = "data/output/1004-naive_bayes"

THRESHOLD_OPTIMIZATION_CONFIG = {
    "threshold_step": 0.01,  # Step size for threshold scanning
    "min_threshold": 0.1,    # Minimum threshold to test
    "max_threshold": 0.9,    # Maximum threshold to test
    "optimization_metric": "f1",  # Metric to optimize: 'f1', 'precision', 'recall', 'accuracy'
    "plot_roc_curve": True,  # Whether to plot ROC curve
    "plot_precision_recall": True,  # Whether to plot Precision-Recall curve
    "save_optimized_models": True  # Whether to save models with optimized thresholds
}

# BorderlineSMOTE configuration constants
BORDERLINE_SMOTE_CONFIG = {
    "k_neighbors": 5,
    "m_neighbors": 10,
    "kind": "borderline-1",  # Options: "borderline-1", "borderline-2"
    "random_state": 42,
    "sampling_strategy": "auto",  # Options: "auto", "minority", float, dict
}

# Model training parameters
IMPORTANCE_THRESHOLD = 0.05
# Number of top features to select for XGBoost important features
TOP_FEATURES_COUNT = 5

# Test/train split configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Day-grouped split of the years left after the hold-out below: 80% of the operating
# days train, 20% test. TEST_SIZE above is retained only for callers that pass an
# explicit test_size; the grouped path uses these two. There is no validation split:
# the tabular models validate with day-grouped CV inside the train split (see
# src/cv_utils.py), and fcll_net carves its early-stopping days out of the train days.
TRAIN_FRACTION = 0.80
TEST_FRACTION = 0.20

# Share of the TRAIN days that fcll_net holds back for early stopping.
FCLL_VAL_FRACTION = 0.15

# Out-of-time hold-out: reserve the most recent calendar year as a final evaluation set
# that is never trained or tuned on, so reported metrics reflect predicting a genuinely
# future year rather than a random-row split of the same years (which leaks temporally
# adjacent rows between train and test). merge_data_files stamps HOLDOUT_YEAR_COLUMN onto
# every row (parsed from each source file's YYYY_MM name); split_dataset carves out the
# max year into a third *_holdout.parquet file and drops the column before saving.
HOLDOUT_LAST_YEAR = True
HOLDOUT_YEAR_COLUMN = 'year'

# Key of the monthly file a row came from ("2023-01"), stamped by merge_data_files
# like HOLDOUT_YEAR_COLUMN. split_dataset fits the weather imputation medians per
# monthly file (then per calendar month, then globally) on the TRAIN rows and drops
# the key right after, so it is never a feature. It does not need month_sin/month_cos.
MONTH_KEY_COLUMN = 'year_month'

# Sample weighting parameters for delay-based weighting
WEIGHT_DELAY_COLUMN = 'NONE' # Put 'NONE' to disable the weights
MAX_SAMPLE_WEIGHT_CLASSIFICATION = 5.0  # Put 1 to disable the weights for classification
MAX_SAMPLE_WEIGHT_REGRESSION = 3.0      # Put 1 to disable the weights for regression
DELAY_THRESHOLD_MINUTES = TRAIN_DELAY_MINUTES  # delay > this = late (1), else on time (0); one source in const_preprocessing

# Parameter distributions for RandomizedSearchCV
DECISION_TREE_PARAM_DISTRIBUTIONS = {
    'max_depth': randint(5, 12),
    'min_samples_split': randint(10, 30),
    'ccp_alpha': [0.0, 0.001, 0.01],
    'class_weight': ['balanced', {False:1, True:5}, {False:1, True:10}]
}

# Parameter distributions for Random Forest with RandomizedSearchCV
RANDOM_FOREST_PARAM_DISTRIBUTIONS = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 15),
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, None],
    'bootstrap': [True, False],  # Both options available
    'criterion': ['gini', 'entropy'],
    'min_impurity_decrease': [0.0, 0.001, 0.005, 0.01],
    'class_weight': ['balanced', {False:1, True:5}, {False:1, True:10}, None],
    'ccp_alpha': [0.0, 0.001, 0.01, 0.05]
}

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

# Parameter distributions for ElasticNet (regression branch — the "regularized
# regression" pairing implied by REGULARIZED_REGRESSION_OUTPUT_FOLDER).
# l1_ratio lower bound kept at 0.1 (not 0.0): sklearn ElasticNet warns and is
# numerically unstable at l1_ratio==0 (pure Ridge) — use Ridge for that, not here.
ELASTICNET_PARAM_DISTRIBUTIONS = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],  # reg strength, log-spaced
    'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9, 1.0],    # 0=Ridge .. 1=Lasso
}

# Parameter distribution for GaussianNB (classification branch). var_smoothing is
# GaussianNB's one real hyperparameter — a variance-floor added to all feature
# variances for numerical stability. Log-spaced because it spans many orders of
# magnitude around sklearn's default (1e-9).
NAIVE_BAYES_PARAM_DISTRIBUTIONS = {
    'var_smoothing': loguniform(1e-12, 1e-6),
}

# Parameter distributions for BayesianRidge (regression branch — paired with
# GaussianNB the way ElasticNet is paired with LogisticRegression: both models
# are Bayesian in spirit, and BayesianRidge exposes a coef_ array shaped like
# ElasticNet's). alpha_1/alpha_2 and lambda_1/lambda_2 are the Gamma-prior
# hyperparameters on the noise precision and weight precision respectively;
# swept as a multiplicative range around sklearn's default (1e-6) for each.
BAYESIAN_RIDGE_PARAM_DISTRIBUTIONS = {
    'alpha_1': [1e-7, 1e-6, 1e-5, 1e-4],
    'alpha_2': [1e-7, 1e-6, 1e-5, 1e-4],
    'lambda_1': [1e-7, 1e-6, 1e-5, 1e-4],
    'lambda_2': [1e-7, 1e-6, 1e-5, 1e-4],
}

# Parameter distributions for XGBoost with RandomizedSearchCV
XGBOOST_PARAM_DISTRIBUTIONS = {
    'n_estimators': randint(100, 400),
    'max_depth': randint(4, 8),
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 1.0],
    # scale_pos_weight is not searched: the trainer sets it to the train neg/pos ratio
    # (class weights replaced SMOTE-Tomek, see TRAINING_STATE_MACHINE["balance_classes"])
}

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

# ===================================================================================================================
# CHOOSING THE CLASSIFICATOR
# ===================================================================================================================
#
# Scoring Option       | Best For                       | Requires           | Description
# ---------------------|--------------------------------|--------------------|---------------------------------
# 'accuracy'           | Balanced datasets              | predict()          | Standard accuracy (correct/total)
# 'balanced_accuracy'  | **Imbalanced datasets**        | predict()          | Average recall per class
# 'f1'                 | Binary classification          | predict()          | Harmonic mean of precision/recall
# 'f1_weighted'        | Multiclass imbalanced          | predict()          | F1 weighted by class support
# 'roc_auc'            | **Binary + probabilities**     | predict_proba()    | Area under ROC curve
# 'average_precision'  | **Imbalanced + probabilities** | predict_proba()    | Area under PR curve
#
# RECOMMENDATIONS FOR TRAIN DELAY PREDICTION:
# - If imbalanced binary: Use 'roc_auc' or 'balanced_accuracy'
# - If balanced binary: Use 'accuracy' or 'f1'
# - If very rare events: Use 'average_precision'
#
# 'roc_auc' tunes every classifier (XGBoost, LightGBM, Random Forest, Logistic Regression). Not 'f1':
# the label is "late by more than TRAIN_DELAY_MINUTES or cancelled", and with a threshold of 0 about
# 73% of the Oulu trains are positive. F1 of the positive class then rewards predicting "disrupted"
# for almost everything, so the search picks the model that gets closest to that. ROC-AUC does not
# depend on the decision threshold or on the base rate.
# ===================================================================================================================
SCORE_METRIC = 'roc_auc'

# Metric the numeric target (differenceInMinutes) is tuned on, XGBoost only so far.
# All are computed in minutes (the model is fitted on log1p of the delay):
#   'mae'  : -MAE, robust to the long delay tail (default, matches mae_skill)
#   'rmse' : -RMSE, punishes the tail harder
#   'r2'   : R2
REGRESSION_SCORE_METRIC = 'mae'

# XGBoost objective for the numeric target, and whether to fit it on log1p(y + shift).
# On the OL data (hold-out 2025, 50-iteration search) the old log fit with squared error
# lost to always predicting the train median (mae_skill -0.060, MAE 6.14); the median-type
# losses on the raw minutes beat it (reg:absoluteerror +0.040, MAE 5.56;
# reg:pseudohubererror +0.019). The log fit pulls predictions below the mean of a
# right-skewed delay. Set REGRESSION_LOG_TARGET = True to get the old behaviour back
# (with reg:squarederror).
REGRESSION_OBJECTIVE = 'reg:absoluteerror'
REGRESSION_LOG_TARGET = False


# RandomizedSearchCV settings
# Minimum meaningful value is 10: training_pipeline.py builds
# list(range(10, RANDOM_SEARCH_ITERATIONS + 1, 10)), which is empty below 10.
RANDOM_SEARCH_ITERATIONS = int(os.environ.get("RAILWAY_SEARCH_ITERATIONS", 50))
RANDOM_SEARCH_CV_FOLDS = 5

# Resampling configuration
RESAMPLING_METHOD = "SMOTE_TOMEK"
# Options: "SMOTE_TOMEK", "EDITED_NEAREST_NEIGHBORS", "NONE"
# "SMOTE_TOMEK": Apply SMOTE-Tomek for oversampling + cleaning
# "EDITED_NEAREST_NEIGHBORS": Apply EditedNearestNeighbors for undersampling
# "NONE": No resampling applied

# EditedNearestNeighbors specific configuration (used when RESAMPLING_METHOD = "EDITED_NEAREST_NEIGHBORS")
ENN_N_NEIGHBORS = 3  # Number of neighbors for EditedNearestNeighbors

# SMOTE-Tomek configuration
IMBALANCE_THRESHOLD = 30.0  # Apply SMOTE-Tomek if minority class < this %
SMOTE_RANDOM_STATE = 42     # For reproducible resampling


# ===================================================================================================================
# XGBOOST OBJECTIVE FUNCTIONS
# ===================================================================================================================

def stable_weighted_mse(y_pred, dtrain):
    """
    Custom XGBoost objective function for regression with weighted MSE.

    Applies higher weights to samples with larger target values, making the model
    more sensitive to larger delays while maintaining numerical stability.

    Parameters:
    -----------
    y_pred : array-like
        Predicted values from the model
    dtrain : xgboost.DMatrix
        Training data matrix containing true labels

    Returns:
    --------
    tuple
        (gradient, hessian) for XGBoost optimization
    """
    y_true = dtrain.get_label()
    # Use constant from config instead of hardcoded value
    weights = np.minimum(MAX_SAMPLE_WEIGHT_REGRESSION, 1.0 + np.abs(y_true) / (np.abs(y_true).mean() * 2))
    # More stable gradient calculation
    grad = weights * (y_pred - y_true)
    hess = weights
    return grad, hess

# Dictionary of available objective functions for XGBoost
XGBOOST_OBJECTIVE_FUNCTIONS = {
    "standard": None,  # Use XGBoost's default objective
    "weighted_mse": stable_weighted_mse  # Custom weighted MSE for regression
}

# Configuration for which objective functions to try for different problem types
XGBOOST_METHODS_CONFIG = {
    "classification": [
        {"name": "standard", "obj": None}
    ],
    "regression": [
        {"name": "standard", "obj": None},
        {"name": "weighted", "obj": "weighted_mse"}  # Reference to function key
    ]
}
