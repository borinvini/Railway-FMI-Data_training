from scipy.stats import randint

# Constants for file processing
FOLDER_NAME = "data"
INPUT_FOLDER = "data/input"
OUTPUT_FOLDER = "data/output"
PREPROCESSED_OUTPUT_FOLDER = "data/output/preprocessed"
RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER = "data/output/decision_tree_randomized_search_cv"
DATA_FILE_PREFIX = "matched_data_"
DATA_FILE_PREFIX_FOR_TRAINING = "preprocessed_data_"
IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/decision_tree_important_features_randomized_search"
XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/xgboost_randomized_search"
REGULARIZED_REGRESSION_OUTPUT_FOLDER = "data/output/regularized_regression"

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
    "merge_snow_depth_columns",
    "clean_missing_values",
    "remove_duplicates", 
    "scale_numeric",
    "add_train_delayed",
    "select_target",
    "save_csv",
    "split_dataset",
    "train_regularized_regression", 
    "train_decision_tree_with_randomized_search_cv", 
    "train_decision_tree_rs_with_important_features", 
    "train_xgboost_with_randomized_search_cv",
    "train_xgboost_rs_with_important_features"
]


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

# Parameter distributions for XGBoost with RandomizedSearchCV
XGBOOST_PARAM_DISTRIBUTIONS = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 10),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4]
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