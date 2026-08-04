from scipy.stats import randint, loguniform
import numpy as np
import os

# Outer RandomizedSearchCV parallelism, and standalone (non-nested) model fits.
# Slurm sets SLURM_CPUS_PER_TASK to the cores actually allocated; joblib's -1 would
# instead grab every physical core on the node and thrash. Off-cluster this is -1,
# preserving local behaviour.
SEARCH_N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", -1))

# Inner estimator threads. Kept at 1 so the cores go to the independent CV fits
# rather than to OpenMP threads competing with them. Batch scripts also export
# OMP_NUM_THREADS=1 for the same reason.
MODEL_N_JOBS = 1

TRAINING_STATE_MACHINE = {
    "merge_data_files": True,
    "filter_delay_outliers": True,
    "select_training_cols": True,
    "split_dataset": True,
    "balance_classes": True,
    "scale_weather_features": True,
    "shap_correlation_analysis": True,
    "train_xgboost_with_randomized_search_cv": True,
    "train_lightgbm_with_randomized_search_cv": True,
    "train_random_forest_with_randomized_search_cv": True,
    "train_logistic_regression_with_randomized_search_cv": True,
    "train_naive_bayes_with_randomized_search_cv": True,
}

MERGED_TRAINING_READY_OUTPUT_FOLDER = "data/output/500-merge_data_files"
MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER = "data/output/501-filter_delay_outliers"
MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER = "data/output/502-select_training_cols"
SPLIT_DATASET_OUTPUT_FOLDER = "data/output/503-split_dataset"
MERGED_BALANCED_OUTPUT_FOLDER = "data/output/504-balance_classes"
MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER = "data/output/505-scale_weather_features"
SHAP_CORRELATION_ANALYSIS_OUTPUT_FOLDER = "data/output/700-shap_correlation_analysis"
SHAP_CORRELATION_TOP_N_PAIRS = 20     # Feature pairs shown in the top-correlated-pairs bar chart
SHAP_CORRELATION_TOP_N_FEATURES = 15  # Features shown in the top-correlated-features sub-heatmap
# Frozen from `python main.py --dump-columns` on 2026-08-04. Explicit rather than
# empty so batch jobs never reach the interactive prompt at training_pipeline.py:1364,
# which sits inside a `while True:` loop and would consume the entire walltime.
#
# The captured merge produced 138 common columns (SCHEMA_MISMATCH_STRATEGY='intersect'
# drops the 10 "Cloud amount" columns absent from the 2018-01..2019-05 files; FMI did
# not supply cloud data before June 2019). From those 138, four are excluded here:
#   - differenceInMinutes, differenceInMinutes_offset,
#     differenceInMinutes_eachStation_offset: direct transformations of the delay
#     outcome; including any as a predictor would leak the answer.
#   - year: HOLDOUT_YEAR_COLUMN, stamped by merge_data_files purely so split_dataset
#     can carve out the out-of-time hold-out. select_training_cols force-keeps it
#     even when absent from this list (training_pipeline.py:1416-1418), and
#     split_dataset drops it again right after the split, so it never reaches
#     balancing/scaling/training — it is not a predictive feature.
#
# trainDelayed (the target) is deliberately KEPT: every trainer stage derives its
# feature matrix as `[c for c in train_df.columns if c != target_feature]`
# (training_pipeline.py e.g. lines 2355, 2698), so the target column must survive
# select_training_cols to be available as the label — dropping it here does not
# reduce leakage (the trainers already exclude it from X by name) and instead
# breaks training outright with "Target feature 'trainDelayed' not found in
# dataset", confirmed by an end-to-end run during this task.
SELECTED_COLUMNS = [
    'trainDelayed',
    'cancelled',
    'trainStopping',
    'commercialStop',
    'month',
    'month_sin',
    'month_cos',
    'hour_sin',
    'hour_cos',
    'hour',
    'day_of_week',
    'day_week_sin',
    'day_week_cos',
    'day_of_month',
    'causes_related_to_weather',
    'Air temperature',
    'Wind speed',
    'Gust speed',
    'Wind direction',
    'Relative humidity',
    'Dew-point temperature',
    'Precipitation intensity',
    'Snow depth',
    'Pressure (msl)',
    'Horizontal visibility',
    'weather_scenario_Normal_Clear',
    'weather_scenario_Blizzard',
    'weather_scenario_Heavy_Snow',
    'weather_scenario_Extreme_Cold',
    'weather_scenario_Heavy_Rain',
    'weather_scenario_Freezing_Rain',
    'weather_scenario_Black_Ice',
    'weather_scenario_Dense_Fog',
    'weather_scenario_High_Winds',
    'weather_scenario_Extreme_Heat',
    'weather_scenario_12h_Normal_Clear',
    'weather_scenario_12h_Blizzard',
    'weather_scenario_12h_Heavy_Snow',
    'weather_scenario_12h_Extreme_Cold',
    'weather_scenario_12h_Heavy_Rain',
    'weather_scenario_12h_Freezing_Rain',
    'weather_scenario_12h_Black_Ice',
    'weather_scenario_12h_Dense_Fog',
    'weather_scenario_12h_High_Winds',
    'weather_scenario_12h_Extreme_Heat',
    'weather_scenario_24h_Normal_Clear',
    'weather_scenario_24h_Blizzard',
    'weather_scenario_24h_Heavy_Snow',
    'weather_scenario_24h_Extreme_Cold',
    'weather_scenario_24h_Heavy_Rain',
    'weather_scenario_24h_Freezing_Rain',
    'weather_scenario_24h_Black_Ice',
    'weather_scenario_24h_Dense_Fog',
    'weather_scenario_24h_High_Winds',
    'weather_scenario_24h_Extreme_Heat',
    'weather_scenario_72h_Normal_Clear',
    'weather_scenario_72h_Blizzard',
    'weather_scenario_72h_Heavy_Snow',
    'weather_scenario_72h_Extreme_Cold',
    'weather_scenario_72h_Heavy_Rain',
    'weather_scenario_72h_Freezing_Rain',
    'weather_scenario_72h_Black_Ice',
    'weather_scenario_72h_Dense_Fog',
    'weather_scenario_72h_High_Winds',
    'weather_scenario_72h_Extreme_Heat',
    'Air temperature (12h max)',
    'Air temperature (12h min)',
    'Air temperature (12h mean)',
    'Air temperature (24h max)',
    'Air temperature (24h min)',
    'Air temperature (24h mean)',
    'Air temperature (72h max)',
    'Air temperature (72h min)',
    'Air temperature (72h mean)',
    'Wind speed (12h max)',
    'Wind speed (12h min)',
    'Wind speed (12h mean)',
    'Wind speed (24h max)',
    'Wind speed (24h min)',
    'Wind speed (24h mean)',
    'Wind speed (72h max)',
    'Wind speed (72h min)',
    'Wind speed (72h mean)',
    'Relative humidity (12h max)',
    'Relative humidity (12h min)',
    'Relative humidity (12h mean)',
    'Relative humidity (24h max)',
    'Relative humidity (24h min)',
    'Relative humidity (24h mean)',
    'Relative humidity (72h max)',
    'Relative humidity (72h min)',
    'Relative humidity (72h mean)',
    'Precipitation intensity (12h max)',
    'Precipitation intensity (12h min)',
    'Precipitation intensity (12h mean)',
    'Precipitation intensity (24h max)',
    'Precipitation intensity (24h min)',
    'Precipitation intensity (24h mean)',
    'Precipitation intensity (72h max)',
    'Precipitation intensity (72h min)',
    'Precipitation intensity (72h mean)',
    'Snow depth (12h max)',
    'Snow depth (12h min)',
    'Snow depth (12h mean)',
    'Snow depth (24h max)',
    'Snow depth (24h min)',
    'Snow depth (24h mean)',
    'Snow depth (72h max)',
    'Snow depth (72h min)',
    'Snow depth (72h mean)',
    'Pressure (msl) (12h max)',
    'Pressure (msl) (12h min)',
    'Pressure (msl) (12h mean)',
    'Pressure (msl) (24h max)',
    'Pressure (msl) (24h min)',
    'Pressure (msl) (24h mean)',
    'Pressure (msl) (72h max)',
    'Pressure (msl) (72h min)',
    'Pressure (msl) (72h mean)',
    'Horizontal visibility (12h max)',
    'Horizontal visibility (12h min)',
    'Horizontal visibility (12h mean)',
    'Horizontal visibility (24h max)',
    'Horizontal visibility (24h min)',
    'Horizontal visibility (24h mean)',
    'Horizontal visibility (72h max)',
    'Horizontal visibility (72h min)',
    'Horizontal visibility (72h mean)',
    'Precipitation amount (12h mean)',
    'Precipitation amount (12h cumulative)',
    'Precipitation amount (24h mean)',
    'Precipitation amount (24h cumulative)',
    'Precipitation amount (72h mean)',
    'Precipitation amount (72h cumulative)',
]
# 'intersect' rather than '' so batch jobs never block on input(): under Slurm
# stdin is /dev/null and the prompt at training_pipeline.py:1109 would raise EOFError.
SCHEMA_MISMATCH_STRATEGY = 'intersect'  # 'intersect' = auto-drop; 'fail' = auto-fail; '' = ask

# Asymmetric quantile thresholds for delay outlier removal
# Lower tail: conservative cut (few implausibly-early arrivals)

# Upper tail: more aggressive cut (heavy right tail has likely data errors)
FILTER_LOWER_QUANTILE = 0.005
FILTER_UPPER_QUANTILE = 0.995


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

# Out-of-time hold-out: reserve the most recent calendar year as a final evaluation set
# that is never trained or tuned on, so reported metrics reflect predicting a genuinely
# future year rather than a random-row split of the same years (which leaks temporally
# adjacent rows between train and test). merge_data_files stamps HOLDOUT_YEAR_COLUMN onto
# every row (parsed from each source file's YYYY_MM name); split_dataset carves out the
# max year into a third *_holdout.parquet file and drops the column before saving.
HOLDOUT_LAST_YEAR = True
HOLDOUT_YEAR_COLUMN = 'year'

# Sample weighting parameters for delay-based weighting
WEIGHT_DELAY_COLUMN = 'NONE' # Put 'NONE' to disable the weights
MAX_SAMPLE_WEIGHT_CLASSIFICATION = 5.0  # Put 1 to disable the weights for classification
MAX_SAMPLE_WEIGHT_REGRESSION = 3.0      # Put 1 to disable the weights for regression
DELAY_THRESHOLD_MINUTES = 5  # delay > 5 min = late (1), else on time (0)

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

# Parameter distributions for Logistic Regression (classification branch).
# solver='saga' is fixed in the constructor (NOT searched) because it is the only
# solver supporting l1 / l2 / elasticnet penalties — this guarantees every sampled
# penalty is valid, so no invalid solver/penalty pair can ever be drawn.
# A list of two dicts (not one flat dict) so l1_ratio is only ever sampled alongside
# penalty='elasticnet' — sklearn's LogisticRegression emits a UserWarning (not an
# error) whenever l1_ratio is set with penalty='l1'/'l2', and RandomizedSearchCV
# supports a list of param-distribution dicts, sampling one dict per draw.
LOGISTIC_REGRESSION_PARAM_DISTRIBUTIONS = [
    {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],   # inverse reg strength, log-spaced
        'class_weight': ['balanced', {False: 1, True: 5}, {False: 1, True: 10}, None],
    },
    {
        'penalty': ['elasticnet'],
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],   # inverse reg strength, log-spaced
        'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
        'class_weight': ['balanced', {False: 1, True: 5}, {False: 1, True: 10}, None],
    },
]

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
    # scale_pos_weight omitted: class imbalance handled upstream by balance_classes (SMOTE-Tomek)
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
# ===================================================================================================================
SCORE_METRIC = 'f1'


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
