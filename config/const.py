from scipy.stats import randint
import numpy as np

# Pipeline execution control flag
EXECUTE_PREPROCESSING_DATA_PIPELINE = False
EXECUTE_TRAINING_PIPELINE = True

# Constants for file processing
FOLDER_NAME = "data"
INPUT_FOLDER = "data/input"
OUTPUT_FOLDER = "data/output"
PREPROCESSED_OUTPUT_FOLDER = "data/output/1-preprocessed"
TRAINING_READY_OUTPUT_FOLDER = "data/output/2-preprocessed_training_ready"
MERGED_TRAINING_READY_OUTPUT_FOLDER = "data/output/3-merged_training_ready"
MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER = "data/output/4-merged_scaled_training_ready"
RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER = "data/output/decision_tree_randomized_search_cv"
RANDOM_FOREST_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/random_forest_randomized_search"
DATA_FILE_PREFIX = "matched_data_"
DATA_FILE_PREFIX_FOR_TRAINING = "preprocessed_data_"
IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/decision_tree_important_features_randomized_search"
XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/xgboost_randomized_search"
REGULARIZED_REGRESSION_OUTPUT_FOLDER = "data/output/regularized_regression"
DECISION_TREE_THRESHOLD_OPTIMIZED_OUTPUT_FOLDER = "data/output/decision_tree_threshold_optimized"
BORDERLINE_SMOTE_OUTPUT_FOLDER = "data/output/borderline_smote_synthetic"
DECISION_TREE_BORDERLINE_SMOTE_THRESHOLD_OPTIMIZED_OUTPUT_FOLDER = "data/output/decision_tree_borderline_smote_threshold_optimized"
XGBOOST_THRESHOLD_OPTIMIZED_OUTPUT_FOLDER = "data/output/xgboost_threshold_optimized"


# Pipeline State Machine Configuration
# Dictionary containing method names in execution order with enable/disable flags
PREPROCESSING_STATE_MACHINE = {
    # Data extraction and preprocessing methods
    "extract_nested_data": True,
    "filter_by_target_station": True,
    "process_causes_column": True,
    "add_train_delayed_feature": True,
    "merge_weather_columns": True,
    "process_actual_time_column": True,
    "filter_columns": True,
    "convert_boolean_to_numeric": True,
    "handle_missing_values": True,
    "save_month_df_to_csv": True,
    "convert_hour_to_sincos": True,
    "convert_month_to_sincos": True,
    "convert_dayofweek_to_sincos": True,
    "drop_original_temporal_columns": True,
    "select_target": True,
    "filter_strong_weather_causes": False,
    "remove_duplicates": True,
    "save_training_ready_csv": True
}

TRAINING_STATE_MACHINE = {
    "merge_data_files": True,
    "drop_nan_columns": True,
    "split_dataset": True,
    "scale_weather_features": True,
    "correlation_analysis": True,
    "non_weather_correlation_analysis": True,
    "train_decision_tree": True,
    "threshold_optimization_decision_tree": True,
    "generate_borderline_smote_data": True,
    "train_decision_tree_with_borderline_smote_data": True,
    "threshold_optimization_decision_tree_borderline_smote": True,
    "train_xgboost_with_randomized_search_cv": True,
    "threshold_optimization_xgboost": True  
}

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
VALID_TARGET_FEATURES = [
    'differenceInMinutes', 
    'differenceInMinutes_offset', 
    'differenceInMinutes_eachStation_offset', 
    'trainDelayed', 
    'cancelled'
]

CLASSIFICATION_PROBLEM = ['trainDelayed', 'cancelled']
REGRESSION_PROBLEM = ['differenceInMinutes', 'differenceInMinutes_offset', 'differenceInMinutes_eachStation_offset']

# Station short code to filter data for - only exact matches will be kept
TARGET_STATION_CODE = 'OL'  # Example: 'OL', 'HKI', 'ROI', etc.


# All columns available in the matched dataset before preprocessing (alphabetical order):
#
# actualTime                              - Actual arrival/departure time
# cancelled                               - Boolean indicating if train was cancelled
# causes                                  - Reasons for delays (nested data)
# causes_related_to_weather               - Weather-related delay indicator/score
# commercialStop                          - Boolean indicating if it's a commercial stop
# commercialTrack                         - Track number for commercial operations
# commuterLineID                          - ID for commuter train lines
# countryCode                             - Country code where station is located
# day_of_week                             - Day of week extracted from date (1-7)
# departureDate                           - Date when the train departed
# differenceInMinutes                     - Difference between scheduled and actual time
# differenceInMinutes_eachStation_offset  - Station-specific offset time difference
# differenceInMinutes_offset              - Offset-adjusted time difference
# hour                                    - Hour extracted from time
# month                                   - Month extracted from date
# operatorShortCode                       - Short code of the train operator
# operatorUICCode                         - UIC code of the train operator
# runningCurrently                        - Boolean indicating if train is currently running
# scheduledTime                           - Originally scheduled time
# stationName                             - Name of the station
# stationShortCode                        - Short code for the station
# stationUICCode                          - UIC code for the station
# timetableAcceptanceDate                 - Date when timetable was accepted
# timetableType                           - Type of timetable used
# trainCategory                           - Category classification of the train
# trainNumber                             - Unique identifier for the train
# trainReady                              - Boolean indicating if train is ready for departure
# trainStopping                           - Boolean indicating if train stops at station
# trainType                               - Type of train (e.g., passenger, freight)
# type                                    - Type of stop (arrival/departure)
# version                                 - Version of the data record
# =============================================================================


# Valid prediction features (non-target features used for training)
VALID_TRAIN_PREDICTION_FEATURES = ["trainStopping", "commercialStop","month","hour","day_of_week", "causes_related_to_weather"]

# Target features that are categoricals for classification problems
CATEGORIAL_TARGET_FEATURES = ['trainDelayed', 'cancelled']

# Boolean features that need to be converted to numeric
BOOLEAN_FEATURES = ['trainStopping', 'commercialStop']

# Multi category features
CATEGORICAL_FEATURES = ["month","hour","day_of_week","causes", "causes_related_to_weather"]

# Set to True to drop trainStopping and commercialStop from training
DROP_TRAIN_FEATURES = True

# Set to True to drop trainStopping and commercialStop from training
delay = True

# Value to be considered as a delay (in minutes)
# Long-distance trains: 5 min
# Short trains: 2-3 min
TRAIN_DELAY_MINUTES = 5

# Important weather conditions to check for missing value handling
ALL_WEATHER_FEATURES = [
    'Air temperature', 
    'Wind speed', 
    'Gust speed', 
    'Wind direction', 
    'Relative humidity', 
    'Dew-point temperature',
    'Precipitation amount',
    'Precipitation intensity',
    'Snow depth',
    'Pressure (msl)',
    'Horizontal visibility',
    'Cloud amount'
]

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


# Weather Indicator Categories for Causes Analysis
# Used to create 'causes_related_to_weather' column from nested 'causes' data
# Extracts 'detailedCategoryCode' and assigns weather likelihood scores:
# - Score 3 (Strong): Direct weather-related delays (I1, I2)
# - Score 2 (Possible): May be weather-influenced (A1, K1, O1, P1, S1, S2, T2, T3, V3)
# - Score 1 (Weak): Other non-empty category codes
# - Score 0 (None): Empty/missing cause data

# Strong weather delay indicators (score: 3)
STRONG_INDICATORS = {'I1', 'I2'}

# Possible weather delay indicators (score: 2)  
POSSIBLE_INDICATORS = {'A1', 'K1', 'O1', 'P1', 'S1', 'S2', 'T2', 'T3', 'V3'}


# Model training parameters
IMPORTANCE_THRESHOLD = 0.05
# Number of top features to select for XGBoost important features
TOP_FEATURES_COUNT = 5

# Test/train split configuration
TEST_SIZE = 0.2

# Sample weighting parameters for delay-based weighting
WEIGHT_DELAY_COLUMN = 'differenceInMinutes_eachStation_offset' # Put 'NONE' to disable the weights
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
SCORE_METRIC = 'f1'


# RandomizedSearchCV settings
RANDOM_SEARCH_ITERATIONS = 100
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