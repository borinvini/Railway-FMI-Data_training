from scipy.stats import randint
import numpy as np

TRAINING_STATE_MACHINE = {
    "merge_data_files": True,
    "filter_delay_outliers": True,
    "select_training_cols": True,
    "split_dataset": True,
    "balance_classes": True,
    "scale_weather_features": False,
    "numeric_correlation_analysis": False,
    "data_distribution_analysis": False,
    "target_feature_analysis": False,
    "train_xgboost_with_randomized_search_cv": True,
}

#    "select_time_features": False,
#    "drop_nan_columns": False,
#    "correlation_analysis": False,
#    "non_weather_correlation_analysis": False,
#    "correlation_analysis_by_station": False,
#    "snow_depth_delay_analysis": False,
#    "numeric_correlation_analysis": False,
#    "train_decision_tree": False,
#    "threshold_optimization_decision_tree": False,
#    "generate_borderline_smote_data": False,
#    "train_decision_tree_with_borderline_smote_data": False,
#    "threshold_optimization_decision_tree_borderline_smote": False,
#    "train_xgboost_with_randomized_search_cv": False,
#    "threshold_optimization_xgboost": False,
#    "train_xgboost_selected_features": False

MERGED_TRAINING_READY_OUTPUT_FOLDER = "data/output/500-merge_data_files"
MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER = "data/output/501-filter_delay_outliers"
MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER = "data/output/502-select_training_cols"
SPLIT_DATASET_OUTPUT_FOLDER = "data/output/503-split_dataset"
MERGED_BALANCED_OUTPUT_FOLDER = "data/output/504-balance_classes"
MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER = "data/output/505-scale_weather_features"
SELECTED_COLUMNS = []  # Empty = interactive prompt; populate to skip prompt

# Asymmetric quantile thresholds for delay outlier removal
# Lower tail: conservative cut (few implausibly-early arrivals)
# Upper tail: more aggressive cut (heavy right tail has likely data errors)
FILTER_LOWER_QUANTILE = 0.005
FILTER_UPPER_QUANTILE = 0.995

RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER = "data/output/decision_tree_randomized_search_cv"
RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/random_forest_randomized_search"
IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/decision_tree_important_features_randomized_search"
XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/1000-xgboost_randomized_search"
REGULARIZED_REGRESSION_OUTPUT_FOLDER = "data/output/regularized_regression"
DECISION_TREE_THRESHOLD_OPTIMIZED_OUTPUT_FOLDER = "data/output/decision_tree_threshold_optimized"
BORDERLINE_SMOTE_OUTPUT_FOLDER = "data/output/borderline_smote_synthetic"
DECISION_TREE_BORDERLINE_SMOTE_THRESHOLD_OPTIMIZED_OUTPUT_FOLDER = "data/output/decision_tree_borderline_smote_threshold_optimized"
XGBOOST_THRESHOLD_OPTIMIZED_OUTPUT_FOLDER = "data/output/xgboost_threshold_optimized"
XGBOOST_SELECTED_FEATURES_OUTPUT_FOLDER = "data/output/xgboost_selected_features"

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

# Sample weighting parameters for delay-based weighting
WEIGHT_DELAY_COLUMN = 'NONE' # Put 'NONE' to disable the weights
MAX_SAMPLE_WEIGHT_CLASSIFICATION = 5.0  # Put 1 to disable the weights for classification
MAX_SAMPLE_WEIGHT_REGRESSION = 3.0      # Put 1 to disable the weights for regression

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

# Parameter distributions for XGBoost with RandomizedSearchCV
XGBOOST_PARAM_DISTRIBUTIONS = {
    'n_estimators': randint(100, 400),
    'max_depth': randint(4, 8),
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 1.0],
    # scale_pos_weight omitted: class imbalance handled upstream by balance_classes (SMOTE-Tomek)
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
RANDOM_SEARCH_ITERATIONS = 50
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