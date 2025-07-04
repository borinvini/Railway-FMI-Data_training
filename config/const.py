from scipy.stats import randint
import numpy as np

# Constants for file processing
FOLDER_NAME = "data"
INPUT_FOLDER = "data/input"
OUTPUT_FOLDER = "data/output"
PREPROCESSED_OUTPUT_FOLDER = "data/output/preprocessed"
ALL_PREPROCESSED_OUTPUT_FOLDER = "data/output/all_preprocessed"
RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER = "data/output/decision_tree_randomized_search_cv"
RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/random_forest_randomized_search"
DATA_FILE_PREFIX = "matched_data_"
DATA_FILE_PREFIX_FOR_TRAINING = "preprocessed_data_"
IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/decision_tree_important_features_randomized_search"
XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/xgboost_randomized_search"
REGULARIZED_REGRESSION_OUTPUT_FOLDER = "data/output/regularized_regression"

# Weather column missing value threshold (drop columns with more missing values than this %)
WEATHER_MISSING_THRESHOLD = 30.0

# Target feature to use for prediction
DEFAULT_TARGET_FEATURE = 'trainDelayed'  
# Possible values: 'differenceInMinutes', 'differenceInMinutes_offset', 
# 'differenceInMinutes_eachStation_offset', 'trainDelayed', 'cancelled'

# Target column to use for calculating trainDelayed feature
TRAIN_DELAYED_TARGET_COLUMN = 'differenceInMinutes_eachStation_offset'
# Possible values: 'differenceInMinutes', 'differenceInMinutes_offset', 'differenceInMinutes_eachStation_offset'

# Valid target features for selection
VALID_TARGET_FEATURES = ['differenceInMinutes', 'differenceInMinutes_offset', 
                         'differenceInMinutes_eachStation_offset', 'trainDelayed', 'cancelled']

CLASSIFICATION_PROBLEM = ['trainDelayed', 'cancelled']
REGRESSION_PROBLEM = ['differenceInMinutes', 'differenceInMinutes_offset', 'differenceInMinutes_eachStation_offset']


# Valid prediction features (non-target features used for training)
VALID_PREDICTION_FEATURES = ["weather_conditions", "trainStopping", "commercialStop"]

# Target features that are categoricals for classification problems
CATEGORIAL_TARGET_FEATURES = ['trainDelayed', 'cancelled']

# Boolean features that need to be converted to numeric
NON_NUMERIC_FEATURES = ['trainStopping', 'commercialStop']

# Set to True to drop trainStopping and commercialStop from training
DROP_TRAIN_FEATURES = True

# Set to True to drop trainStopping and commercialStop from training
delay = True

# Value to be considered as a delay (in minutes)
# Long-distance trains: 5 min
# Short trains: 2-3 min
TRAIN_DELAY_MINUTES = 5

# Important weather conditions to check for missing value handling
IMPORTANT_WEATHER_CONDITIONS = [
    'Relative humidity', 
    'Dew-point temperature', 
    'Precipitation amount', 
    'Precipitation intensity', 
    'Snow depth', 
    'Horizontal visibility',
    'Wind speed',
    'Gust speed'
]

# List of weather features thathas 2 cols and need to merge in 1 col
WEATHER_COLS_TO_MERGE = [
    "Snow depth", 
    "Precipitation amount", 
    "Precipitation intensity", 
    "Horizontal visibility", 
    "Wind speed", 
    "Gust speed"
]

# Train filtering configuration
FILTER_TRAINS_BY_STATIONS = False  # Set to True to filter trains by required stations
REQUIRED_STATIONS = ['HKI', 'OL', 'ROI']  # Trains must pass through ALL of these stations

# Model training parameters
IMPORTANCE_THRESHOLD = 0.05
# Number of top features to select for XGBoost important features
TOP_FEATURES_COUNT = 5

# Sample weighting parameters for delay-based weighting
WEIGHT_DELAY_COLUMN = 'differenceInMinutes_eachStation_offset' # Put 'NONE' to disable the weights
MAX_SAMPLE_WEIGHT_CLASSIFICATION = 5.0  # Put 1 to disable the weights for classification
MAX_SAMPLE_WEIGHT_REGRESSION = 3.0      # Put 1 to disable the weights for regression

# Pipeline stages configuration
PIPELINE_STAGES = [
    "merge_weather_columns",
    "clean_missing_values",
    "remove_duplicates", 
    "scale_numeric",
    "add_train_delayed",
    "select_target",
    "save_csv",
    "split_dataset",
    "train_regularized_regression", 
    "train_decision_tree_with_randomized_search_cv", 
    "train_random_forest_with_randomized_search_cv",
    "train_xgboost_with_randomized_search_cv"
]

# Dictionary mapping pipeline stage names to their corresponding method names
PIPELINE_STAGE_TO_METHOD_MAPPING = {
    # Data preprocessing stages
    "merge_weather_columns": "merge_weather_columns",
    "clean_missing_values": "handle_missing_values", 
    "remove_duplicates": "remove_duplicates",
    "scale_numeric": "scale_numeric_columns",
    "add_train_delayed": "add_train_delayed_feature",
    "select_target": "select_target_feature",
    
    # Data saving and splitting stages
    "save_csv": "save_month_df_to_csv",
    "split_dataset": "split_month_dataset",
    
    # Model training stages
    "train_regularized_regression": "train_regularized_regression",
    "train_decision_tree_with_randomized_search_cv": "train_decision_tree_with_randomized_search_cv",
    "train_random_forest_with_randomized_search_cv": "train_random_forest_with_randomized_search_cv", 
    "train_xgboost_with_randomized_search_cv": "train_xgboost_with_randomized_search_cv",
    
    # Combined data processing stages
    "merge_all_preprocessed_files": "merge_all_preprocessed_files",
    "split_combined_dataset": "split_combined_dataset",
    "train_decision_tree_combined_data": "train_decision_tree_combined_data"
}

# Reverse mapping: method names to stage names
METHOD_TO_PIPELINE_STAGE_MAPPING = {
    method: stage for stage, method in PIPELINE_STAGE_TO_METHOD_MAPPING.items()
}



# Parameter distributions for RandomizedSearchCV
DECISION_TREE_PARAM_DISTRIBUTIONS = {
    'max_depth': randint(3, 20),  
    'min_samples_split': randint(10, 50),
    'min_samples_leaf': randint(5, 25),   
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2', 0.5, 0.7],
    'min_impurity_decrease': [0.0, 0.001, 0.005, 0.01],
    'ccp_alpha': [0.0, 0.001, 0.01, 0.05],
    'class_weight': ['balanced', {False:1, True:5}, {False:1, True:10}, {False:1, True:15}, None] 
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
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 10),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'max_delta_step': [0, 1, 3, 5, 7, 10]
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
SCORE_METRIC = 'roc_auc'


# RandomizedSearchCV settings
RANDOM_SEARCH_ITERATIONS = 20
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