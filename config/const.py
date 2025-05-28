from scipy.stats import randint

# Constants for file processing
FOLDER_NAME = "data"
INPUT_FOLDER = "data/input"
OUTPUT_FOLDER = "data/output"
PREPROCESSED_OUTPUT_FOLDER = "data/output/preprocessed"
DECISION_TREE_OUTPUT_FOLDER = "data/output/decision_tree"
RANDOMIZED_SEARCH_CV_OUTPUT_FOLDER = "data/output/randomized_search_cv"
DATA_FILE_PREFIX = "matched_data_"
DATA_FILE_PREFIX_FOR_TRAINING = "preprocessed_data_"
IMPORTANT_FEATURES_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/important_features_randomized_search"
IMPORTANT_FEATURES_OUTPUT_FOLDER = "data/output/important_features_decision_tree"
XGBOOST_OUTPUT_FOLDER = "data/output/xgboost"
XGBOOST_RANDOMIZED_SEARCH_OUTPUT_FOLDER = "data/output/xgboost_randomized_search"
REGULARIZED_REGRESSION_OUTPUT_FOLDER = "data/output/regularized_regression"

# Target feature to use for prediction
DEFAULT_TARGET_FEATURE = 'differenceInMinutes_offset'  
# Possible values: 'differenceInMinutes', 'differenceInMinutes_offset', 'trainDelayed', 'cancelled'

# Boolean features that need to be converted to numeric
NON_NUMERIC_FEATURES = ['trainStopping', 'commercialStop']

# Train filtering configuration
FILTER_TRAINS_BY_STATIONS = True  # Set to True to filter trains by required stations
REQUIRED_STATIONS = ['HKI', 'OL']  # Trains must pass through ALL of these stations

# Model training parameters
IMPORTANCE_THRESHOLD = 0.05
# Number of top features to select for XGBoost important features
TOP_FEATURES_COUNT = 5


# Parameter distributions for RandomizedSearchCV
RANDOMIZED_SEARCH_PARAM_DISTRIBUTIONS = {
    'max_depth': randint(3, 30),
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 10),
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2']
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

# XGBoost default parameters for non-tuned model
XGBOOST_DEFAULT_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'binary:logistic'
}

# RandomizedSearchCV settings
RANDOM_SEARCH_ITERATIONS = 50
RANDOM_SEARCH_CV_FOLDS = 5